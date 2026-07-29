from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.cooccur_lift_bugfixed_v1 import (
    validate_bugfixed_v1_materialization_inputs,
)
from jgrec.core.io import read_interactions, read_test_queries
from jgrec.rankers.hybrid.cooccur_lift import (
    CooccurLiftAugmentedView,
    load_frozen_cooccur_lift_config,
)
from jgrec.rankers.hybrid.cooccur_lift_native import (
    materialize_compact_cooccur_lift,
)
from jgrec.rankers.hybrid.fusion import (
    FusionResult,
    build_fusion_from_state,
    predict_logits,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView
from jgrec.rankers.registry import create_ranker


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the externally accepted cooccur-lift auxiliary "
            "probabilities for Dataset2 test candidates without changing "
            "the production checkpoint."
        )
    )
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--external-report", type=Path)
    parser.add_argument("--candidate-contract", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--auxiliary-model", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--query-order",
        choices=("chronological", "source_grouped"),
        default="source_grouped",
        help=(
            "Feature scoring may group future-only test rows by source to "
            "reuse exact source caches; outputs are restored to CSV order."
        ),
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    config = load_frozen_cooccur_lift_config(args.frozen_config)
    lock = _read_json(args.selection_lock)
    lock_sha256 = _sha256(args.selection_lock)
    evidence = _validate_evidence(
        args=args,
        config=config,
        lock=lock,
        lock_sha256=lock_sha256,
    )
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    interactions = read_interactions(args.train_csv).sort_by_time()
    queries = read_test_queries(args.test_csv)
    if len(queries) == 0 or queries.candidate_count != 100:
        raise ValueError("Dataset2 test queries must be nonempty full100 rows")
    if not np.all(queries.time[1:] >= queries.time[:-1]):
        raise ValueError("Dataset2 test queries must be chronological")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    time_span = int(interactions.time[-1]) - int(interactions.time[0])
    short_window = float(time_span) * config.short_window_ratio
    lift_path = args.output_dir / "test-lift-features.npy"
    popularity_path = args.output_dir / "test-candidate0-causal-popularity.npy"
    native_contract = materialize_compact_cooccur_lift(
        interactions=interactions,
        sources=queries.src,
        candidates=queries.candidates,
        destinations=queries.candidates[:, 0],
        event_time=queries.time,
        short_window=short_window,
        lift_path=lift_path,
        positive_popularity_path=popularity_path,
        progress_path=args.output_dir / "test-materialization-progress.json",
        work_dir=args.output_dir,
    )
    lift = np.load(lift_path, mmap_mode="r", allow_pickle=False)

    checkpoint_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset2",
    )
    feature_names = tuple(
        str(name) for name in checkpoint_state["feature_names"]
    )
    gnn_column = feature_names.index("gnn_short")
    ranker = create_ranker("hybrid", None)
    ranker.hydrate(checkpoint_state)
    del checkpoint_state
    model, result = _load_auxiliary_model(args.auxiliary_model)
    jt.flags.use_cuda = 1
    probabilities_path = args.output_dir / "test-auxiliary-probabilities.npy"
    probabilities = np.lib.format.open_memmap(
        probabilities_path,
        mode="w+",
        dtype=np.float64,
        shape=queries.candidates.shape,
    )
    if args.query_order == "source_grouped":
        score_order = np.argsort(queries.src, kind="stable")
    else:
        score_order = np.arange(len(queries), dtype=np.int64)
    for start in range(0, len(queries), args.batch_size):
        stop = min(start + args.batch_size, len(queries))
        rows = score_order[start:stop]
        batch_queries = queries[rows]
        base = ranker.impl.encoder.features_for_query_array(batch_queries)
        augmented = CooccurLiftAugmentedView(
            base,
            short_none_scores=base[..., gnn_column],
            gnn_short_column=gnn_column,
            lift_features=lift[rows],
        )
        view = SetwiseFeatureView(augmented, transform_version=1)
        selected = np.asarray(view[:], dtype=np.float32)
        if result.feature_indices != tuple(range(selected.shape[-1])):
            selected = selected[..., result.feature_indices]
        logits = predict_logits(
            model,
            selected,
            result.mean,
            result.std,
        )
        probabilities[rows] = _softmax(logits)
        if stop % 5_000 < args.batch_size or stop == len(queries):
            print(
                json.dumps(
                    {
                        "status": "scoring_test",
                        "completed_rows": stop,
                        "total_rows": len(queries),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del base, augmented, view, selected, logits
        if stop % 8_192 < args.batch_size:
            gc.collect()
            jt.sync_all()
            jt.clean()
    probabilities.flush()
    del probabilities
    persisted = np.load(
        probabilities_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if not np.all(np.isfinite(persisted)):
        raise ValueError("test auxiliary probabilities are non-finite")
    maximum_row_sum_error = float(
        np.max(np.abs(persisted.sum(axis=1) - 1.0))
    )
    if maximum_row_sum_error > 5e-6:
        raise ValueError("test auxiliary probabilities are not normalized")
    report = {
        "schema_version": 1,
        "status": "complete_online_candidate_materialization",
        "integration_id": config.integration_id,
        "candidate_id": evidence["candidate_id"],
        "evidence_mode": evidence["mode"],
        "selected_weight": 0.5,
        "selection_lock_sha256": lock_sha256,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "auxiliary_model": str(args.auxiliary_model.resolve()),
        "auxiliary_model_sha256": _sha256(args.auxiliary_model),
        "test_csv": str(args.test_csv.resolve()),
        "test_csv_sha256": _sha256(args.test_csv),
        "test_candidate_fingerprint": hashlib.sha256(
            np.ascontiguousarray(queries.candidates).tobytes(order="C")
        ).hexdigest(),
        "probabilities": str(probabilities_path.resolve()),
        "probabilities_sha256": _sha256(probabilities_path),
        "shape": list(persisted.shape),
        "maximum_row_sum_error": maximum_row_sum_error,
        "feature_scoring_query_order": args.query_order,
        "feature_scoring_output_order": "original_test_csv_row_order",
        "native_materializer": native_contract,
        "scoring_device": "cuda",
        "production_checkpoint_modified": False,
        "external_metric_read": False,
        "elapsed_seconds": time.time() - started,
    }
    report.update(evidence["report_fields"])
    _write_json(args.output_dir / "test-materialization-report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _validate_evidence(
    *,
    args: argparse.Namespace,
    config: Any,
    lock: dict[str, Any],
    lock_sha256: str,
) -> dict[str, Any]:
    bugfixed_paths = (
        args.candidate_contract,
        args.training_report,
    )
    if any(path is not None for path in bugfixed_paths):
        if not all(path is not None for path in bugfixed_paths):
            raise ValueError(
                "bugfixed v1 scoring requires candidate and training reports"
            )
        if args.external_report is not None:
            raise ValueError(
                "bugfixed v1 scoring must not reuse historical external evidence"
            )
        contract = _read_json(args.candidate_contract)
        training = _read_json(args.training_report)
        evidence = validate_bugfixed_v1_materialization_inputs(
            contract=contract,
            contract_sha256=_sha256(args.candidate_contract),
            training_report=training,
            actual_model_sha256=_sha256(args.auxiliary_model),
            actual_source_checkpoint_sha256=_sha256(
                args.source_checkpoint
            ),
        )
        training_assets = contract["training_assets"]
        if (
            lock_sha256 != training_assets["selection_lock_sha256"]
            or _sha256(args.frozen_config)
            != training_assets["frozen_config_sha256"]
            or lock.get("integration_id") != config.integration_id
            or float(lock.get("selected_weight", -1.0))
            != evidence["selected_weight"]
        ):
            raise ValueError("bugfixed v1 frozen scoring evidence differs")
        return {
            **evidence,
            "mode": "bugfixed_v1_training_evidence",
            "report_fields": {
                "candidate_contract": str(
                    args.candidate_contract.resolve()
                ),
                "candidate_contract_sha256": _sha256(
                    args.candidate_contract
                ),
                "training_report": str(args.training_report.resolve()),
                "training_report_sha256": _sha256(args.training_report),
                "external_report_reused": False,
            },
        }

    if args.external_report is None:
        raise ValueError(
            "historical scoring requires an accepted external report"
        )
    external = _read_json(args.external_report)
    if external.get("status") != "accepted":
        raise ValueError("external evaluation is not accepted")
    if (
        lock.get("integration_id") != config.integration_id
        or external.get("integration_id") != config.integration_id
    ):
        raise ValueError("accepted evidence integration_id differs")
    if (
        float(lock.get("selected_weight")) != 0.5
        or float(external.get("selected_weight")) != 0.5
    ):
        raise ValueError("accepted evidence is not bound to weight 0.50")
    if external.get("selection_lock_sha256") != lock_sha256:
        raise ValueError("external report selection lock hash differs")
    return {
        "candidate_id": config.integration_id,
        "mode": "historical_accepted_external_evidence",
        "report_fields": {
            "external_report": str(args.external_report.resolve()),
            "external_report_sha256": _sha256(args.external_report),
        },
    }


def _load_auxiliary_model(path: Path) -> tuple[Any, FusionResult]:
    payload = np.load(path, allow_pickle=False)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    feature_indices = tuple(
        int(value) for value in payload["feature_indices"]
    )
    hidden_dim = int(np.asarray(payload["hidden_dim"]).reshape(-1)[0])
    state = {
        key.removeprefix("state__"): np.asarray(
            payload[key],
            dtype=np.float32,
        )
        for key in payload.files
        if key.startswith("state__")
    }
    payload.close()
    model = build_fusion_from_state(
        input_dim=len(mean),
        hidden_dim=hidden_dim,
        state=state,
    )
    result = FusionResult(
        best_val_ap=0.0,
        best_val_mrr=0.0,
        state=state,
        mean=mean,
        std=std,
        feature_indices=feature_indices,
        candidate_name="cooccur_lift_aux_expert_v1_full_origin",
    )
    return model, result


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
