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

FROZEN_DATASET2_SHA256 = (
    "d7e0d574789aa6b507e592fffb5da054839aba663f63ed58134ea2d951b1ae1e"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Dataset1 full-100 Setwise candidate only from a passing "
            "forward-held evaluation report."
        )
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--setwise-model", required=True, type=Path)
    parser.add_argument("--champion-dataset2", required=True, type=Path)
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

    evaluation = _read_json(args.evaluation_report)
    setwise_weight = authorized_setwise_weight(evaluation)
    setwise_report = evaluation["setwise"]
    if int(setwise_report["training_rows"]) not in {100_000, 200_000}:
        raise ValueError("evaluation selected an unsupported training scale")
    if _sha256(args.setwise_model) != setwise_report["model_sha256"]:
        raise ValueError("selected Setwise model hash differs from evaluation")
    if _sha256(args.champion_dataset2) != FROZEN_DATASET2_SHA256:
        raise ValueError("Dataset2 source is not the frozen online champion")

    started = time.time()
    source_metadata = load_checkpoint_metadata(args.source_checkpoint)
    dataset1_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset1",
    )
    config = dataset1_state["config"]
    fusion_result = dataset1_state["fusion_result"]
    feature_indices = tuple(
        int(index) for index in fusion_result.feature_indices
    )
    interactions = read_interactions(
        args.data_dir / "dataset1" / "train.csv"
    ).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)

    final_config = _config_for_selected_features(config, feature_indices)
    dataset_profile = dataset1_state["dataset_profile"]
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
        recent_window=int(dataset1_state["recent_window"])
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
        "dataset1_full100_setwise_final_encoder",
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
    if tuple(final_encoder.feature_names) != tuple(
        dataset1_state["feature_names"]
    ):
        raise RuntimeError("final Dataset1 encoder feature schema differs")
    dataset1_state["encoder"] = final_encoder.snapshot()
    dataset1_state["id_map"] = {
        "src_values": final_encoder.id_map.src_values,
        "dst_values": final_encoder.id_map.dst_values,
    }
    dataset1_state["segment_gate_result"] = None
    dataset1_state["multi_interest_proxy_state"] = None
    del final_encoder
    gc.collect()

    with np.load(args.setwise_model, allow_pickle=False) as payload:
        hidden_dim = int(payload["hidden_dim"][0])
        source_feature_count = int(payload["source_feature_count"][0])
        training_rows = int(payload["training_rows"][0])
        if source_feature_count != len(dataset1_state["feature_names"]):
            raise ValueError("Setwise model source feature count differs")
        if training_rows != int(setwise_report["training_rows"]):
            raise ValueError("Setwise model training scale differs")
        if int(payload["context_transform_version"][0]) != 1:
            raise ValueError("unsupported Setwise context transform")
        setwise_state = {
            key.removeprefix("state__"): np.asarray(
                payload[key],
                dtype=np.float32,
            )
            for key in payload.files
            if key.startswith("state__")
        }
        setwise_result = FusionResult(
            best_val_ap=float(setwise_report["best_val_ap"]),
            best_val_mrr=float(setwise_report["best_val_mrr"]),
            state=setwise_state,
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            feature_indices=tuple(
                int(value) for value in payload["feature_indices"]
            ),
            candidate_name=(
                f"dataset1_setwise_recent{training_rows}_full100"
            ),
        )
    dataset1_state["setwise_fusion_state"] = setwise_state
    dataset1_state["setwise_fusion_result"] = setwise_result
    dataset1_state["setwise_hidden_dim"] = hidden_dim
    current_lgbm = dataset1_state.get("lgbm_result")
    if current_lgbm is None:
        raise ValueError(
            "Dataset1 Setwise blend requires the champion LightGBM expert"
        )
    dataset1_state["lgbm_result"] = replace(
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
            "dataset1_full100_setwise_evaluation": str(
                args.evaluation_report.resolve()
            ),
            "dataset1_setwise_training_rows": training_rows,
            "dataset1_setwise_weight": setwise_weight,
        }
    )
    writer = ContestCheckpointWriter(
        args.output_checkpoint,
        model_name=str(source_metadata["model_name"]),
        expected_datasets=tuple(source_metadata["datasets"]),
        metadata=extra_metadata,
    )
    try:
        writer.add_dataset("dataset1", dataset1_state)
        dataset2_state = load_checkpoint_dataset(
            args.source_checkpoint,
            "dataset2",
        )
        writer.add_dataset("dataset2", dataset2_state)
        writer.finalize()
    except BaseException:
        writer.abort()
        raise
    finally:
        del dataset1_state
        if "dataset2_state" in locals():
            del dataset2_state
        gc.collect()

    reloaded_dataset1 = load_checkpoint_dataset(
        args.output_checkpoint,
        "dataset1",
    )
    reloaded_ranker = create_ranker("hybrid", None)
    reloaded_ranker.hydrate(reloaded_dataset1)
    if (
        reloaded_ranker.impl.setwise_fusion_result is None
        or reloaded_ranker.impl.lgbm_result is None
        or abs(
            float(reloaded_ranker.impl.lgbm_result.mlp_weight)
            - setwise_weight
        )
        > 1e-12
    ):
        raise RuntimeError("reloaded Dataset1 Setwise checkpoint differs")
    del reloaded_dataset1
    gc.collect()

    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    dataset1 = datasets["dataset1"]
    dataset2 = datasets["dataset2"]
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=False)
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    zip_path = args.output_dir / "result.zip"

    dataset1_result = build_dataset_submission(
        dataset=dataset1,
        ranker=reloaded_ranker,
        output_dir=csv_dir,
        batch_size=args.batch_size,
        verbose=True,
        fit_ranker=False,
    )
    validate_submission_file(
        dataset1_output,
        expected_rows=expected_test_rows(dataset1),
    )
    shutil.copyfile(args.champion_dataset2, dataset2_output)
    validate_submission_file(
        dataset2_output,
        expected_rows=expected_test_rows(dataset2),
    )
    copied_dataset2_hash = _sha256(dataset2_output)
    if copied_dataset2_hash != FROZEN_DATASET2_SHA256:
        raise RuntimeError("copied Dataset2 CSV differs from the champion")
    dataset2_result = DatasetResult(
        name="dataset2",
        rows=expected_test_rows(dataset2),
        output_path=dataset2_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    write_zip([dataset1_result, dataset2_result], zip_path)

    report = {
        "status": "complete",
        "winner": "setwise",
        "setwise_weight": setwise_weight,
        "training_rows": training_rows,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "evaluation_report": str(args.evaluation_report.resolve()),
        "evaluation_report_sha256": _sha256(args.evaluation_report),
        "offline_gate": setwise_report,
        "output_checkpoint": str(args.output_checkpoint.resolve()),
        "output_checkpoint_bytes": args.output_checkpoint.stat().st_size,
        "output_checkpoint_sha256": _sha256(args.output_checkpoint),
        "dataset1_mode": "full100_setwise_with_rebuilt_final_encoder",
        "dataset1_rows": dataset1_result.rows,
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_mode": "byte_copy_from_online_setwise_champion",
        "dataset2_rows": dataset2_result.rows,
        "dataset2_sha256": copied_dataset2_hash,
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
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
