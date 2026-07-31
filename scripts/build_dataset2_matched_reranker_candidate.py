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
from jgrec.core.cuda import require_jittor_cuda
from jgrec.core.io import discover_datasets, read_interactions
from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetResult, TrainingReport
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.fusion import FusionResult
from jgrec.rankers.hybrid.fusion_analysis import authorized_setwise_weight
from jgrec.rankers.hybrid.ranker import (
    TemporalHybridRanker,
    _config_for_selected_features,
)
from jgrec.rankers.registry import create_ranker
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
    write_zip,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Dataset2 matched-reranker package only from a passing "
            "three-slice evaluation report."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--lgbm-model", required=True, type=Path)
    parser.add_argument("--setwise-model", required=True, type=Path)
    parser.add_argument("--champion-dataset1", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    import jittor as jt  # noqa: PLC0415

    require_jittor_cuda(jt)
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite candidate directory: {args.output_dir}"
        )
    if args.output_checkpoint.exists() or args.output_checkpoint.with_suffix(
        f"{args.output_checkpoint.suffix}.tmp"
    ).exists():
        raise FileExistsError(
            f"refusing to overwrite candidate checkpoint: {args.output_checkpoint}"
        )
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    winner = evaluation.get("winner")
    if (
        evaluation.get("status") != "passed"
        or not evaluation.get("gate_passed")
        or not evaluation.get("package_authorized")
        or winner not in {"lightgbm", "setwise"}
    ):
        raise RuntimeError(
            "evaluation did not authorize a LightGBM or Setwise package"
        )
    winner_report = evaluation[winner]
    if not winner_report.get("gate_passed"):
        raise RuntimeError("selected winner did not pass its own metric gate")

    started = time.time()
    source_metadata = load_checkpoint_metadata(args.source_checkpoint)
    dataset2_state = load_checkpoint_dataset(args.source_checkpoint, "dataset2")
    config = dataset2_state["config"]
    fusion_result = dataset2_state["fusion_result"]
    feature_indices = tuple(int(index) for index in fusion_result.feature_indices)
    interactions = read_interactions(args.data_dir / "dataset2" / "train.csv")
    interactions = interactions.sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)

    final_config = _config_for_selected_features(config, feature_indices)
    dataset_profile = dataset2_state["dataset_profile"]
    final_future_only = bool(
        dataset_profile is not None
        and dataset_profile.test_min_time > dataset_profile.train_max_time > 0
    )
    if final_future_only:
        final_config = replace(
            final_config,
            structure_future_only_transition_cooccur=True,
        )
    ranker = TemporalHybridRanker(
        recent_window=int(dataset2_state["recent_window"])
    )
    ranker.id_map = NodeIdMap.from_interactions(interactions)
    ranker.dataset_profile = dataset_profile
    encoder_cache = ranker._encoder_state_cache(
        interactions,
        final_config,
        verbose=True,
    )
    final_snapshot = (
        encoder_cache.snapshot_for_prefix(len(interactions))
        if encoder_cache is not None
        else None
    )
    rng = np.random.default_rng(config.seed + 10_000)
    final_encoder = ranker._timed_fit_encoder(
        "matched_candidate_final_encoder",
        interactions,
        final_config,
        rng,
        verbose=True,
        deterministic_snapshot=final_snapshot,
    )
    if encoder_cache is not None:
        encoder_cache.clear()
    del final_snapshot, encoder_cache, ranker
    if final_future_only:
        final_encoder.compact_for_future_queries()
    if tuple(final_encoder.feature_names) != tuple(dataset2_state["feature_names"]):
        raise RuntimeError("final encoder feature schema differs from checkpoint")
    dataset2_state["encoder"] = final_encoder.snapshot()
    dataset2_state["id_map"] = {
        "src_values": final_encoder.id_map.src_values,
        "dst_values": final_encoder.id_map.dst_values,
    }
    dataset2_state["segment_gate_result"] = None
    del final_encoder
    gc.collect()

    if winner == "lightgbm":
        model_text = args.lgbm_model.read_text(encoding="utf-8")
        if not model_text.strip():
            raise ValueError("matched LightGBM model is empty")
        current_lgbm = dataset2_state.get("lgbm_result")
        if current_lgbm is None:
            raise ValueError("source checkpoint has no Dataset2 LightGBM expert")
        dataset2_state["lgbm_result"] = replace(
            current_lgbm,
            best_val_mrr=float(winner_report["expert"]["full"]),
            model_text=model_text,
            candidate_name="lgbm_recent200k_full100_matched_mrr",
            mlp_weight=0.07,
        )
        dataset2_state["setwise_fusion_state"] = None
        dataset2_state["setwise_fusion_result"] = None
        dataset2_state["setwise_hidden_dim"] = 64
    else:
        setwise_weight = authorized_setwise_weight(evaluation)
        setwise_payload = np.load(args.setwise_model, allow_pickle=False)
        hidden_dim = int(setwise_payload["hidden_dim"][0])
        source_feature_count = int(setwise_payload["source_feature_count"][0])
        if source_feature_count != len(dataset2_state["feature_names"]):
            raise ValueError("Setwise model source feature count differs")
        if int(setwise_payload["context_transform_version"][0]) != 1:
            raise ValueError("unsupported Setwise context transform")
        setwise_state = {
            key.removeprefix("state__"): np.asarray(
                setwise_payload[key],
                dtype=np.float32,
            )
            for key in setwise_payload.files
            if key.startswith("state__")
        }
        setwise_result = FusionResult(
            best_val_ap=float(winner_report["best_val_ap"]),
            best_val_mrr=float(winner_report["best_val_mrr"]),
            state=setwise_state,
            mean=np.asarray(setwise_payload["mean"], dtype=np.float32),
            std=np.asarray(setwise_payload["std"], dtype=np.float32),
            feature_indices=tuple(
                int(value) for value in setwise_payload["feature_indices"]
            ),
            candidate_name="setwise_recent200k_full100_matched_mrr",
        )
        dataset2_state["setwise_fusion_state"] = setwise_state
        dataset2_state["setwise_fusion_result"] = setwise_result
        dataset2_state["setwise_hidden_dim"] = hidden_dim
        current_lgbm = dataset2_state.get("lgbm_result")
        if current_lgbm is None:
            raise ValueError("Setwise blend requires the champion LightGBM expert")
        dataset2_state["lgbm_result"] = replace(
            current_lgbm,
            mlp_weight=setwise_weight,
        )

    extra_metadata = {
        key: value
        for key, value in source_metadata.items()
        if key not in {"format", "version", "model_name", "datasets"}
    }
    extra_metadata.update(
        {
            "derived_from": str(args.source_checkpoint.resolve()),
            "dataset2_matched_evaluation": str(
                args.evaluation_report.resolve()
            ),
            "dataset2_matched_winner": winner,
            "dataset2_setwise_weight": (
                setwise_weight if winner == "setwise" else None
            ),
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(source_metadata["model_name"]),
        expected_datasets=tuple(source_metadata["datasets"]),
        metadata=extra_metadata,
    )
    try:
        dataset1_state = load_checkpoint_dataset(
            args.source_checkpoint,
            "dataset1",
        )
        writer.add_dataset("dataset1", dataset1_state)
        del dataset1_state
        gc.collect()
        writer.add_dataset("dataset2", dataset2_state)
        writer.finalize()
    except BaseException:
        writer.abort()
        raise

    reloaded_dataset2 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset2",
    )
    reloaded_ranker = create_ranker("hybrid", None)
    reloaded_ranker.hydrate(reloaded_dataset2)
    del reloaded_dataset2, dataset2_state
    gc.collect()

    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    dataset1 = datasets["dataset1"]
    dataset2 = datasets["dataset2"]
    csv_dir = args.output_dir / "csv"
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    zip_path = args.output_dir / "result.zip"
    csv_dir.mkdir(parents=True, exist_ok=False)

    source_dataset1_hash = _sha256(args.champion_dataset1)
    shutil.copyfile(args.champion_dataset1, dataset1_output)
    validate_submission_file(
        dataset1_output,
        expected_rows=expected_test_rows(dataset1),
    )
    copied_dataset1_hash = _sha256(dataset1_output)
    if copied_dataset1_hash != source_dataset1_hash:
        raise RuntimeError("copied Dataset1 CSV differs from champion")

    dataset2_result = build_dataset_submission(
        dataset=dataset2,
        ranker=reloaded_ranker,
        output_dir=csv_dir,
        batch_size=args.batch_size,
        verbose=True,
        fit_ranker=False,
    )
    validate_submission_file(
        dataset2_output,
        expected_rows=expected_test_rows(dataset2),
    )
    dataset1_result = DatasetResult(
        name="dataset1",
        rows=expected_test_rows(dataset1),
        output_path=dataset1_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    write_zip([dataset1_result, dataset2_result], zip_path)

    report = {
        "status": "complete",
        "winner": winner,
        "setwise_weight": (
            float(reloaded_ranker.impl.lgbm_result.mlp_weight)
            if winner == "setwise"
            else None
        ),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "evaluation_report": str(args.evaluation_report.resolve()),
        "evaluation_report_sha256": _sha256(args.evaluation_report),
        "offline_gate": winner_report,
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "dataset1_mode": "byte_copy_from_online_champion",
        "dataset1_rows": dataset1_result.rows,
        "dataset1_sha256": copied_dataset1_hash,
        "dataset2_mode": f"matched_{winner}_with_rebuilt_final_encoder",
        "dataset2_rows": dataset2_result.rows,
        "dataset2_sha256": _sha256(dataset2_output),
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(args.output_dir / "candidate-report.json", report)
    shutil.copyfile(
        args.evaluation_report,
        args.output_dir / "evaluation-report.json",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
