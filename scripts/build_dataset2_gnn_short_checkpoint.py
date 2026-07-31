from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import (
    ContestCheckpointWriter,
    load_checkpoint_dataset,
    load_checkpoint_metadata,
)
from jgrec.rankers.hybrid.fusion import FusionResult
from jgrec.rankers.hybrid.gnn_short_checkpoint import (
    install_gnn_short_setwise_fusion,
)
from jgrec.rankers.registry import create_ranker

ALLOWED_DATASET2_CHANGES = frozenset(
    {
        "lgbm_result",
        "setwise_fusion_result",
        "setwise_fusion_state",
        "setwise_hidden_dim",
    }
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install the authorized short_none 50/40k Setwise head into an "
            "independent, loadable contest checkpoint."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset1-checkpoint", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--setwise-model", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if args.output_checkpoint.exists():
        raise FileExistsError(
            f"refusing to overwrite checkpoint: {args.output_checkpoint}"
        )
    temporary = args.output_checkpoint.with_suffix(
        f"{args.output_checkpoint.suffix}.tmp"
    )
    if temporary.exists():
        raise FileExistsError(
            f"refusing to resume unknown partial checkpoint: {temporary}"
        )
    started = time.time()

    evaluation = _read_json(args.evaluation_report)
    if (
        evaluation.get("status") != "passed"
        or not evaluation.get("gate_passed")
        or not evaluation.get("package_authorized")
        or evaluation.get("variant") != "short_none"
        or evaluation.get("weighting") != "none"
        or int(evaluation.get("graph_epochs", 0)) != 50
        or int(evaluation.get("graph_max_train_edges", 0)) != 40_000
    ):
        raise RuntimeError(
            "evaluation does not authorize short_none 50/40k integration"
        )
    _require_hash(
        args.source_checkpoint,
        evaluation["source_checkpoint_sha256"],
        "source checkpoint",
    )
    _require_hash(
        args.setwise_model,
        evaluation["setwise_model_sha256"],
        "Setwise model",
    )

    source_metadata = load_checkpoint_metadata(args.source_checkpoint)
    dataset1_metadata = load_checkpoint_metadata(args.dataset1_checkpoint)
    if (
        dataset1_metadata["model_name"] != source_metadata["model_name"]
        or tuple(dataset1_metadata["datasets"])
        != tuple(source_metadata["datasets"])
    ):
        raise ValueError("Dataset1 and Dataset2 checkpoints are incompatible")
    source_dataset1 = load_checkpoint_dataset(
        args.dataset1_checkpoint,
        "dataset1",
    )
    source_dataset2 = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset2",
    )
    setwise_result, hidden_dim = _load_setwise_result(
        args.setwise_model,
        evaluation,
        source_feature_count=len(source_dataset2["feature_names"]),
    )
    candidate_dataset2 = install_gnn_short_setwise_fusion(
        source_dataset2,
        setwise_result=setwise_result,
        hidden_dim=hidden_dim,
        setwise_weight=float(evaluation["setwise_weight"]),
    )
    state_audit = _audit_dataset2_changes(
        source_dataset2,
        candidate_dataset2,
    )

    metadata = {
        key: value
        for key, value in dataset1_metadata.items()
        if key.startswith("dataset1_")
    }
    metadata.update(
        {
            key: value
            for key, value in source_metadata.items()
            if key == "dataset2_lgbm_tuning_report"
        }
    )
    metadata.update(
        {
            "derived_from": str(args.source_checkpoint.resolve()),
            "derived_from_sha256": _sha256(args.source_checkpoint),
            "dataset1_derived_from": str(
                args.dataset1_checkpoint.resolve()
            ),
            "dataset1_derived_from_sha256": _sha256(
                args.dataset1_checkpoint
            ),
            "dataset2_integration": "gnn_short_none_e50_edges40000_setwise",
            "dataset2_evaluation_report": str(
                args.evaluation_report.resolve()
            ),
            "dataset2_evaluation_report_sha256": _sha256(
                args.evaluation_report
            ),
            "dataset2_setwise_model": str(args.setwise_model.resolve()),
            "dataset2_setwise_model_sha256": _sha256(args.setwise_model),
            "dataset2_setwise_weight": float(evaluation["setwise_weight"]),
            "dataset2_encoder_retrained": False,
            "dataset2_allowed_state_changes": tuple(
                sorted(ALLOWED_DATASET2_CHANGES)
            ),
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(source_metadata["model_name"]),
        expected_datasets=tuple(source_metadata["datasets"]),
        metadata=metadata,
    )
    try:
        writer.add_dataset("dataset1", source_dataset1)
        writer.add_dataset("dataset2", candidate_dataset2)
        writer.finalize()
    except BaseException:
        writer.abort()
        raise

    reloaded_dataset1 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset1",
    )
    reloaded_dataset2 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset2",
    )
    if _pickle_sha256(reloaded_dataset1) != _pickle_sha256(source_dataset1):
        raise RuntimeError("Dataset1 changed during checkpoint integration")
    reload_audit = _audit_dataset2_changes(
        source_dataset2,
        reloaded_dataset2,
    )
    if reload_audit != state_audit:
        raise RuntimeError("reloaded Dataset2 state audit differs")

    import jittor as jt  # noqa: PLC0415

    jt.flags.use_cuda = 1
    ranker = create_ranker("hybrid", None)
    ranker.hydrate(reloaded_dataset2)
    if ranker.impl.setwise_fusion is None:
        raise RuntimeError("standard checkpoint hydrate omitted Setwise fusion")
    if ranker.impl.setwise_fusion_result is None:
        raise RuntimeError("standard checkpoint hydrate omitted Setwise result")
    if ranker.impl.lgbm_result is None:
        raise RuntimeError("standard checkpoint hydrate omitted LightGBM")
    if abs(
        float(ranker.impl.lgbm_result.mlp_weight)
        - float(evaluation["setwise_weight"])
    ) > 1e-12:
        raise RuntimeError("hydrated Setwise blend weight differs")
    del ranker, reloaded_dataset1, reloaded_dataset2
    gc.collect()

    report = {
        "status": "complete",
        "integration": "gnn_short_none_e50_edges40000_setwise",
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "dataset1_checkpoint": str(args.dataset1_checkpoint.resolve()),
        "dataset1_checkpoint_sha256": _sha256(args.dataset1_checkpoint),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "evaluation_report": str(args.evaluation_report.resolve()),
        "evaluation_report_sha256": _sha256(args.evaluation_report),
        "setwise_model": str(args.setwise_model.resolve()),
        "setwise_model_sha256": _sha256(args.setwise_model),
        "setwise_weight": float(evaluation["setwise_weight"]),
        "dataset1_pickle_sha256": _pickle_sha256(source_dataset1),
        "dataset2_state_audit": state_audit,
        "standard_hydrate_passed": True,
        "encoder_retrained": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "checkpoint-integration-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _load_setwise_result(
    path: Path,
    evaluation: dict[str, Any],
    *,
    source_feature_count: int,
) -> tuple[FusionResult, int]:
    with np.load(path, allow_pickle=False) as archive:
        hidden_dim = int(archive["hidden_dim"][0])
        archive_source_features = int(archive["source_feature_count"][0])
        transform_version = int(archive["context_transform_version"][0])
        if archive_source_features != source_feature_count:
            raise ValueError("Setwise archive source feature count differs")
        if transform_version != 1:
            raise ValueError("unsupported Setwise context transform")
        state = {
            key.removeprefix("state__"): np.asarray(
                archive[key],
                dtype=np.float32,
            )
            for key in archive.files
            if key.startswith("state__")
        }
        if not state:
            raise ValueError("Setwise archive has no model state")
        result = FusionResult(
            best_val_ap=float(evaluation["setwise_best_val_ap"]),
            best_val_mrr=float(evaluation["setwise_best_val_mrr"]),
            state=state,
            mean=np.asarray(archive["mean"], dtype=np.float32),
            std=np.asarray(archive["std"], dtype=np.float32),
            feature_indices=tuple(
                int(value)
                for value in np.asarray(archive["feature_indices"]).tolist()
            ),
            candidate_name="dataset2_gnn_short_none_e50_edges40000",
        )
    return result, hidden_dim


def _audit_dataset2_changes(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    removed = set(source) - set(candidate)
    if removed:
        raise RuntimeError(
            "Dataset2 checkpoint fields were removed: "
            + ", ".join(sorted(removed))
        )
    added = set(candidate) - set(source)
    unexpected_added = added - ALLOWED_DATASET2_CHANGES
    if unexpected_added:
        raise RuntimeError(
            "Dataset2 checkpoint fields were added outside allowlist: "
            + ", ".join(sorted(unexpected_added))
        )
    changed_existing = tuple(
        key
        for key in source
        if _pickle_sha256(source[key]) != _pickle_sha256(candidate[key])
    )
    changed = (*changed_existing, *sorted(added))
    unexpected = set(changed) - ALLOWED_DATASET2_CHANGES
    if unexpected:
        raise RuntimeError(
            "Dataset2 state changed outside allowlist: "
            + ", ".join(sorted(unexpected))
        )
    if set(changed) != ALLOWED_DATASET2_CHANGES:
        missing = ALLOWED_DATASET2_CHANGES - set(changed)
        raise RuntimeError(
            "Dataset2 integration did not change all required fields: "
            + ", ".join(sorted(missing))
        )
    protected_hashes = {
        key: _pickle_sha256(candidate[key])
        for key in candidate
        if key not in ALLOWED_DATASET2_CHANGES
    }
    return {
        "changed_top_level_keys": list(changed),
        "added_top_level_keys": sorted(added),
        "allowed_top_level_keys": sorted(ALLOWED_DATASET2_CHANGES),
        "protected_top_level_sha256": protected_hashes,
        "source_encoder_sha256": _pickle_sha256(source["encoder"]),
        "candidate_encoder_sha256": _pickle_sha256(candidate["encoder"]),
        "encoder_hash_stable": (
            _pickle_sha256(source["encoder"])
            == _pickle_sha256(candidate["encoder"])
        ),
        "source_lgbm_model_text_sha256": _text_sha256(
            source["lgbm_result"].model_text
        ),
        "candidate_lgbm_model_text_sha256": _text_sha256(
            candidate["lgbm_result"].model_text
        ),
    }


def _pickle_sha256(value: Any) -> str:
    sink = _HashWriter()
    pickle.dump(value, sink, protocol=pickle.HIGHEST_PROTOCOL)
    return sink.hexdigest()


class _HashWriter:
    def __init__(self) -> None:
        self._digest = hashlib.sha256()

    def write(self, value: bytes | pickle.PickleBuffer) -> int:
        view = memoryview(value)
        self._digest.update(view)
        return view.nbytes

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
