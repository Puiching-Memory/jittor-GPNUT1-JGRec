from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import (
    ContestCheckpointWriter,
    load_checkpoint_dataset,
    load_checkpoint_metadata,
)
from jgrec.core.io import discover_datasets, read_interactions, read_test_queries
from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetResult, TrainingReport
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex
from jgrec.rankers.registry import create_ranker
from jgrec.submission import expected_test_rows, validate_submission_file, write_zip

DECAY_FEATURE_NAME = "cooccur_time_decay_score"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a Dataset2 two-hop-decay checkpoint and package after a passed frozen gate."
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--lgbm-model", required=True, type=Path)
    parser.add_argument("--champion-dataset1", required=True, type=Path)
    parser.add_argument("--dataset2-train", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {args.output_dir}")
    if args.output_checkpoint.exists() or args.output_checkpoint.with_suffix(
        f"{args.output_checkpoint.suffix}.tmp"
    ).exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {args.output_checkpoint}")
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    if not evaluation.get("gate_passed") or not evaluation.get("package_authorized"):
        raise ValueError("full-100 temporal validation gate did not authorize packaging")
    if not evaluation["gate"]["full_delta_passed"] or not evaluation["gate"][
        "all_slices_non_decreasing"
    ]:
        raise ValueError("evaluation report gate details are inconsistent")
    model_text = args.lgbm_model.read_text(encoding="utf-8")
    if _sha256(args.lgbm_model) != evaluation["candidate"]["model_sha256"]:
        raise ValueError("LightGBM model hash does not match the passed evaluation")

    frozen = evaluation["frozen_config"]
    decay_config = frozen["new_feature"]
    interactions = read_interactions(args.dataset2_train).sort_by_time()
    source_dataset2 = load_checkpoint_dataset(args.source_checkpoint, "dataset2")
    source_config = source_dataset2["config"]
    if source_config.max_fit_events > 0 and len(interactions) > source_config.max_fit_events:
        interactions = interactions.tail(source_config.max_fit_events)
    del source_dataset2
    gc.collect()

    started = time.time()
    print("[two-hop-package] building full-history decay-only sparse map", flush=True)
    decay_index = TemporalInteractionIndex()
    decay_index.fit(
        interactions,
        build_transitions=False,
        build_cooccurs=False,
        cooccur_history_limit=int(decay_config["cooccur_history_limit"]),
        future_only_transition_cooccur=True,
        cooccur_time_decay_ratio=float(decay_config["decay_ratio"]),
    )
    decay_map = decay_index.future_cooccur_decay_maps
    decay_anchor_time = int(decay_index.cooccur_decay_anchor_time)
    decay_tau = float(decay_index.cooccur_decay_tau)
    decay_nnz = int(decay_map.nnz())
    del decay_index, interactions
    gc.collect()
    print(
        f"[two-hop-package] decay map ready nnz={decay_nnz} tau={decay_tau:.3f}",
        flush=True,
    )

    source_metadata = load_checkpoint_metadata(args.source_checkpoint)
    extra_metadata = {
        key: value
        for key, value in source_metadata.items()
        if key not in {"format", "version", "model_name", "datasets"}
    }
    extra_metadata.update(
        {
            "derived_from": str(args.source_checkpoint.resolve()),
            "dataset2_two_hop_decay_evaluation": str(args.evaluation_report.resolve()),
            "dataset2_two_hop_decay_model_sha256": _sha256(args.lgbm_model),
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(source_metadata["model_name"]),
        expected_datasets=tuple(source_metadata["datasets"]),
        metadata=extra_metadata,
    )
    dataset2_state: dict[str, Any] | None = None
    try:
        dataset1_state = load_checkpoint_dataset(args.source_checkpoint, "dataset1")
        writer.add_dataset("dataset1", dataset1_state)
        del dataset1_state
        gc.collect()

        dataset2_state = load_checkpoint_dataset(args.source_checkpoint, "dataset2")
        current_names = tuple(str(name) for name in dataset2_state["feature_names"])
        if len(current_names) != 63 or DECAY_FEATURE_NAME in current_names:
            raise ValueError("source Dataset2 feature schema is not the frozen 63-column champion")
        current_lgbm = dataset2_state.get("lgbm_result")
        if current_lgbm is None:
            raise ValueError("source Dataset2 state has no LightGBM result")
        dataset2_state["config"] = replace(
            dataset2_state["config"],
            structure_cooccur_time_decay_enabled=True,
            structure_cooccur_time_decay_ratio=float(decay_config["decay_ratio"]),
            structure_cooccur_time_decay_source_history_limit=int(
                decay_config["source_history_limit"]
            ),
        )
        dataset2_state["feature_names"] = (*current_names, DECAY_FEATURE_NAME)
        structure_index = dataset2_state["encoder"]["structure"]["index"]
        structure_index.future_cooccur_decay_maps = decay_map
        structure_index.cooccur_decay_enabled = True
        structure_index.cooccur_decay_anchor_time = decay_anchor_time
        structure_index.cooccur_decay_tau = decay_tau
        structure_index.future_only = True
        dataset2_state["lgbm_result"] = replace(
            current_lgbm,
            best_val_mrr=float(evaluation["candidate"]["lgbm"]["full"]),
            model_text=model_text,
            feature_indices=tuple(range(64)),
            candidate_name="lgbm_two_hop_decay_lr003_fixed308",
            mlp_weight=float(frozen["mlp_weight"]),
        )
        writer.add_dataset("dataset2", dataset2_state)
        writer.finalize()
    except Exception:
        writer.abort()
        raise
    del decay_map
    gc.collect()

    if dataset2_state is None:
        raise RuntimeError("Dataset2 candidate state was not created")
    test_queries = read_test_queries(args.data_dir / "dataset2" / "test.csv")
    smoke_queries = test_queries[: min(8, len(test_queries))]
    before_ranker = create_ranker("hybrid", None)
    before_ranker.hydrate(dataset2_state)
    before_scores = before_ranker.predict_batch(smoke_queries)
    del before_ranker, dataset2_state
    gc.collect()
    reloaded_state = load_checkpoint_dataset(args.output_checkpoint, "dataset2")
    after_ranker = create_ranker("hybrid", None)
    after_ranker.hydrate(reloaded_state)
    after_scores = after_ranker.predict_batch(smoke_queries)
    np.testing.assert_allclose(after_scores, before_scores, rtol=1e-7, atol=1e-9)
    del reloaded_state, before_scores, after_scores
    gc.collect()

    datasets = {dataset.name: dataset for dataset in discover_datasets(args.data_dir)}
    dataset1 = datasets["dataset1"]
    dataset2 = datasets["dataset2"]
    csv_dir = args.output_dir / "csv"
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    zip_path = args.output_dir / "result.zip"
    csv_dir.mkdir(parents=True, exist_ok=False)

    source_dataset1_hash = _sha256(args.champion_dataset1)
    shutil.copyfile(args.champion_dataset1, dataset1_output)
    validate_submission_file(dataset1_output, expected_rows=expected_test_rows(dataset1))
    copied_dataset1_hash = _sha256(dataset1_output)
    if copied_dataset1_hash != source_dataset1_hash:
        raise RuntimeError("Dataset1 copy is not byte-identical to the online champion")

    dataset2_result = build_dataset_submission(
        dataset=dataset2,
        ranker=after_ranker,
        output_dir=csv_dir,
        batch_size=args.batch_size,
        verbose=True,
        fit_ranker=False,
    )
    validate_submission_file(dataset2_output, expected_rows=expected_test_rows(dataset2))
    dataset1_result = DatasetResult(
        name="dataset1",
        rows=expected_test_rows(dataset1),
        output_path=dataset1_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    write_zip([dataset1_result, dataset2_result], zip_path)

    report = {
        "status": "packaged_after_passed_gate",
        "validation_gate_passed": True,
        "validation_full_delta": evaluation["candidate"]["blend_full_delta"],
        "validation_slice_deltas": evaluation["candidate"]["blend_slice_deltas"],
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "checkpoint_reload_prediction_equal": True,
        "decay_map_nnz": decay_nnz,
        "decay_anchor_time": decay_anchor_time,
        "decay_tau": decay_tau,
        "dataset1_mode": "byte_copy_from_online_champion",
        "dataset1_rows": dataset1_result.rows,
        "dataset1_sha256": copied_dataset1_hash,
        "dataset2_mode": "frozen_mlp_plus_two_hop_decay_lgbm",
        "dataset2_rows": dataset2_result.rows,
        "dataset2_sha256": _sha256(dataset2_output),
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "candidate-report.json", report)
    shutil.copyfile(args.evaluation_report, args.output_dir / "full100-validation-report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
