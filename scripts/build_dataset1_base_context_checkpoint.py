from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jittor as jt

from jgrec.contest_checkpoint import (
    ContestCheckpointWriter,
    load_checkpoint_dataset,
    load_checkpoint_metadata,
)
from jgrec.core.cuda import require_jittor_cuda
from jgrec.rankers.hybrid.base_context_gate import (
    BASE_CONTEXT_INTEGRATION_ID,
    authorize_base_context_package,
)
from jgrec.rankers.hybrid.base_context_head import (
    load_base_context_head,
)
from jgrec.rankers.hybrid.fusion import FusionResult
from jgrec.rankers.registry import create_ranker

_RESERVED_METADATA = {"format", "version", "model_name", "datasets"}
_ALLOWED_DATASET1_CHANGES = {
    "fusion_state",
    "fusion_result",
    "fusion_hidden_dim",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Dataset1 base-context v1 checkpoint authorized by "
            "the locked rolling and one-shot external gates."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--external-result", required=True, type=Path)
    parser.add_argument(
        "--external-evaluation",
        required=True,
        type=Path,
    )
    parser.add_argument("--candidate-head", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()

    require_jittor_cuda(jt)
    if args.output_checkpoint.exists():
        raise FileExistsError(
            f"refusing to overwrite: {args.output_checkpoint}"
        )
    if args.output_report.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_report}")
    started = time.time()

    external_result = _read_json(args.external_result)
    external_evaluation = _read_json(args.external_evaluation)
    authorization = authorize_base_context_package(
        external_result=external_result,
        external_evaluation=external_evaluation,
        actual_external_evaluation_sha256=_sha256(
            args.external_evaluation
        ),
        actual_candidate_head_sha256=_sha256(args.candidate_head),
        actual_source_checkpoint_sha256=_sha256(
            args.source_checkpoint
        ),
    )
    head = load_base_context_head(
        args.candidate_head,
        expected_context_transform_version=1,
    )
    metadata = load_checkpoint_metadata(args.source_checkpoint)
    dataset1_source = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset1",
    )
    dataset2_source = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset2",
    )
    _validate_source_dataset1(dataset1_source)
    source_indices = tuple(
        int(value)
        for value in dataset1_source["fusion_result"].feature_indices
    )
    if head.feature_indices != source_indices:
        raise ValueError(
            "candidate head feature mask differs from source checkpoint"
        )

    dataset1_candidate = dict(dataset1_source)
    dataset1_candidate["fusion_state"] = head.state
    dataset1_candidate["fusion_result"] = FusionResult(
        best_val_ap=head.best_val_ap,
        best_val_mrr=head.best_val_mrr,
        state=head.state,
        mean=head.mean,
        std=head.std,
        feature_indices=head.feature_indices,
        candidate_name=head.candidate_name,
    )
    dataset1_candidate["fusion_hidden_dim"] = head.hidden_dim
    changed_keys = sorted(
        key
        for key in dataset1_candidate
        if dataset1_candidate[key] is not dataset1_source[key]
    )
    if (
        not {"fusion_state", "fusion_result"}.issubset(changed_keys)
        or not set(changed_keys).issubset(_ALLOWED_DATASET1_CHANGES)
    ):
        raise RuntimeError(
            "candidate checkpoint changed unexpected Dataset1 keys: "
            + ", ".join(changed_keys)
        )

    extra_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in _RESERVED_METADATA
    }
    extra_metadata.update(
        {
            "dataset1_base_context_integration_id": (
                BASE_CONTEXT_INTEGRATION_ID
            ),
            "dataset1_base_context_external_result": str(
                args.external_result.resolve()
            ),
            "dataset1_base_context_external_result_sha256": _sha256(
                args.external_result
            ),
            "dataset1_base_context_external_evaluation_sha256": (
                authorization.external_evaluation_sha256
            ),
            "dataset1_base_context_selection_lock_sha256": (
                authorization.selection_lock_sha256
            ),
            "dataset1_base_context_candidate_head_sha256": (
                authorization.candidate_head_sha256
            ),
            "dataset1_base_context_transform_version": 1,
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(metadata["model_name"]),
        expected_datasets=tuple(metadata["datasets"]),
        metadata=extra_metadata,
    )
    try:
        writer.add_dataset("dataset1", dataset1_candidate)
        writer.add_dataset("dataset2", dataset2_source)
        writer.finalize()
    except BaseException:
        writer.abort()
        raise

    del dataset1_candidate, dataset1_source, dataset2_source
    gc.collect()
    reloaded = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset1",
    )
    ranker = create_ranker("hybrid", None)
    ranker.hydrate(reloaded)
    actual_result = ranker.impl.fusion_result
    if (
        ranker.impl.fusion is None
        or actual_result is None
        or tuple(actual_result.feature_indices) != head.feature_indices
        or int(actual_result.mean.shape[0]) != int(head.mean.shape[0])
        or int(ranker.impl._fusion_hidden_dim) != head.hidden_dim
    ):
        raise RuntimeError("reloaded base-context checkpoint differs")

    report = {
        "status": "complete",
        "package_authorized": True,
        "integration_id": BASE_CONTEXT_INTEGRATION_ID,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": (
            authorization.source_checkpoint_sha256
        ),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "candidate_head": str(args.candidate_head.resolve()),
        "candidate_head_sha256": authorization.candidate_head_sha256,
        "selection_lock_sha256": (
            authorization.selection_lock_sha256
        ),
        "external_result": str(args.external_result.resolve()),
        "external_result_sha256": _sha256(args.external_result),
        "external_evaluation": str(
            args.external_evaluation.resolve()
        ),
        "external_evaluation_sha256": (
            authorization.external_evaluation_sha256
        ),
        "dataset1_changed_top_level_keys": changed_keys,
        "dataset1_allowed_top_level_keys": sorted(
            _ALLOWED_DATASET1_CHANGES
        ),
        "dataset2_state_retrained": False,
        "standard_hydrate_passed": True,
        "elapsed_seconds": time.time() - started,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_report, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def _validate_source_dataset1(state: dict[str, Any]) -> None:
    time_ramp = state.get("time_ramp_config")
    if (
        state.get("fusion_result") is None
        or state.get("fusion_state") is None
        or state.get("lgbm_result") is None
        or state.get("time_ramp_setwise_result") is None
        or state.get("time_ramp_setwise_fusion_state") is None
        or time_ramp is None
        or float(time_ramp["power"]) != 0.5
    ):
        raise ValueError(
            "source checkpoint is not the Dataset1 gamma=0.5 champion"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
