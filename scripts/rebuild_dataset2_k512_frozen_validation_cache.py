from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from build_dataset2_full100_train_cache import (
    _candidate_checksum,
    _close_parallel_encoder,
    _configure_cuda,
    _features_with_parallel_gate,
    _jsonable,
    _sha256,
    _validation_paths,
    _write_json_atomic,
)
from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.full100_training import (
    build_frozen_candidate_queries,
    sample_chronological_events,
    validate_frozen_validation_alignment,
    validate_full100_cache_arrays,
)
from jgrec.rankers.hybrid.parallel_structure import (
    ForkedStructureFeatureTower,
)
from jgrec.rankers.hybrid.ranker import TemporalHybridRanker

EXPECTED_ROWS = 20_000
EXPECTED_CANDIDATES = 100
EXPECTED_FEATURES = 63


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    _configure_cuda()

    paths = _validation_paths(args.output_prefix)
    protected = (*paths.values(), args.report)
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite frozen-validation artifacts: "
            f"{existing}"
        )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_report = _read_json(args.train_cache_report)
    train_binding = _validate_train_report(
        train_report,
        checkpoint=args.checkpoint,
    )
    state = load_checkpoint_dataset(args.checkpoint, args.dataset_name)
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    if len(feature_names) != EXPECTED_FEATURES:
        raise ValueError("checkpoint feature count differs from K512 contract")

    interactions = read_interactions(args.train_csv).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    val_size = max(1, int(len(interactions) * config.val_ratio))
    train_end = max(2, len(interactions) - val_size)
    if train_end != int(train_report["split"]["train_end"]):
        raise ValueError("train_end differs from the completed training cache")

    row_rng = np.random.default_rng(config.seed)
    sampled_val, validation_row_indices = sample_chronological_events(
        interactions[train_end:],
        max_events=config.max_val_events,
        rng=row_rng,
        row_offset=train_end,
    )
    reference_paths = _validation_paths(
        args.frozen_validation_prefix
    )
    frozen_sidecars = (
        "candidates",
        "src",
        "dst",
        "time",
        "row_indices",
    )
    reference = {
        name: np.load(
            reference_paths[name],
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in frozen_sidecars
    }
    initial_frozen_alignment = validate_frozen_validation_alignment(
        positives=sampled_val,
        row_indices=validation_row_indices,
        candidates=reference["candidates"],
        reference_src=reference["src"],
        reference_dst=reference["dst"],
        reference_time=reference["time"],
        reference_row_indices=reference["row_indices"],
        reference_candidates=reference["candidates"],
    )
    validation_shape = (
        len(sampled_val),
        EXPECTED_CANDIDATES,
        len(feature_names),
    )
    if validation_shape != (
        EXPECTED_ROWS,
        EXPECTED_CANDIDATES,
        EXPECTED_FEATURES,
    ):
        raise ValueError(
            f"frozen validation shape differs: {validation_shape}"
        )

    supervised_config = replace(
        config,
        structure_future_only_transition_cooccur=True,
        supervised_feature_cache_dir=None,
        verbose=True,
    )
    ranker = TemporalHybridRanker(
        recent_window=int(state["recent_window"])
    )
    ranker.id_map = NodeIdMap.from_interactions(interactions)
    ranker.dataset_profile = state["dataset_profile"]
    encoder_cache = ranker._encoder_state_cache(
        interactions,
        supervised_config,
        verbose=True,
    )
    val_snapshot = (
        encoder_cache.snapshot_for_prefix(train_end)
        if encoder_cache is not None
        else None
    )
    encoder_rng = np.random.default_rng()
    encoder_rng.bit_generator.state = copy.deepcopy(
        train_report["fusion_rng_state_after_build"]
    )
    encoder_rng_state_before_fit = _jsonable(
        encoder_rng.bit_generator.state
    )
    val_encoder = ranker._timed_fit_encoder(
        "frozen_joint_full100_train_end_encoder",
        interactions[:train_end],
        supervised_config,
        encoder_rng,
        verbose=True,
        deterministic_snapshot=val_snapshot,
    )
    if tuple(val_encoder.feature_names) != feature_names:
        raise RuntimeError(
            "frozen validation encoder feature schema differs"
        )
    if encoder_cache is not None:
        encoder_cache.release_except()
    del val_snapshot, state
    gc.collect()

    temp_feature_path = paths["features"].with_name(
        f".{paths['features'].name}.part"
    )
    if temp_feature_path.exists():
        raise FileExistsError(
            "partial frozen-validation cache exists; inspect it first"
        )
    features = np.lib.format.open_memmap(
        temp_feature_path,
        mode="w+",
        dtype=np.float32,
        shape=validation_shape,
    )
    total_batches = (
        len(sampled_val) + args.batch_rows - 1
    ) // args.batch_rows
    candidate_checksum = 0
    parallel_structure: ForkedStructureFeatureTower | None = None
    parallel_parity: dict[str, Any] | None = None
    parallel_trial: dict[str, Any] | None = None
    try:
        for start in range(0, len(sampled_val), args.batch_rows):
            end = min(start + args.batch_rows, len(sampled_val))
            batch_id = start // args.batch_rows + 1
            batch_started = time.time()
            batch_events = sampled_val[start:end]
            frozen_candidates = reference["candidates"][start:end]
            queries = build_frozen_candidate_queries(
                batch_events,
                frozen_candidates,
            )
            (
                batch_features,
                parallel_structure,
                parallel_parity,
                parallel_trial,
            ) = _features_with_parallel_gate(
                encoder=val_encoder,
                queries=queries,
                worker_count=args.structure_workers,
                minimum_speedup=args.minimum_parallel_speedup,
                comparison_worker_count=(
                    args.comparison_structure_workers
                ),
                minimum_comparison_speedup=(
                    args.minimum_comparison_speedup
                ),
                minimum_memory_reserve_bytes=int(
                    args.minimum_memory_reserve_gib * 1024**3
                ),
                parallel_structure=parallel_structure,
                parallel_parity=parallel_parity,
                parallel_trial=parallel_trial,
                label="frozen-joint-val-cache",
            )
            expected_shape = (
                end - start,
                EXPECTED_CANDIDATES,
                len(feature_names),
            )
            if batch_features.shape != expected_shape:
                raise RuntimeError(
                    "frozen validation encoder returned an unexpected shape"
                )
            if not np.all(np.isfinite(batch_features)):
                raise ValueError(
                    "non-finite frozen validation feature in "
                    f"batch {batch_id}"
                )
            features[start:end] = batch_features
            candidate_checksum += _candidate_checksum(
                queries.candidates
            )
            del batch_features, queries
            val_encoder.clear_batch_caches()
            features.flush()
            gc.collect()
            progress = {
                "status": "building",
                "protocol": (
                    "frozen_candidate_validation_recovery_v1"
                ),
                "candidate_sampling_performed": False,
                "completed_rows": end,
                "total_rows": len(sampled_val),
                "batch": batch_id,
                "total_batches": total_batches,
                "candidate_checksum": candidate_checksum,
                "requested_structure_workers": (
                    args.structure_workers
                ),
                "selected_structure_workers": (
                    1
                    if parallel_structure is None
                    else parallel_structure.worker_count
                ),
                "parallel_parity": parallel_parity,
                "parallel_trial": parallel_trial,
                "joint_build_id": train_binding["joint_build_id"],
                "recovery_process_id": os.getpid(),
                "elapsed_seconds": time.time() - started,
                "last_batch_seconds": time.time() - batch_started,
            }
            _write_json_atomic(paths["progress"], progress)
            print(
                f"[frozen-joint-val-cache] "
                f"batch={batch_id}/{total_batches} "
                f"rows={end}/{len(sampled_val)} "
                f"batch_seconds={progress['last_batch_seconds']:.1f} "
                f"elapsed={progress['elapsed_seconds']:.1f}",
                flush=True,
            )
        features.flush()
    except BaseException:
        features.flush()
        _close_parallel_encoder(
            val_encoder,
            parallel_structure,
            terminate=True,
        )
        raise
    else:
        _close_parallel_encoder(val_encoder, parallel_structure)
        del features
        os.replace(temp_feature_path, paths["features"])

    for name in frozen_sidecars:
        _copy_file_atomic(reference_paths[name], paths[name])

    output = {
        name: np.load(path, mmap_mode="r", allow_pickle=False)
        for name, path in paths.items()
        if name != "progress"
    }
    cache_contract = validate_full100_cache_arrays(
        features=output["features"],
        candidates=output["candidates"],
        src=output["src"],
        dst=output["dst"],
        time=output["time"],
        row_indices=output["row_indices"],
        expected_train_rows=EXPECTED_ROWS,
        expected_candidate_count=EXPECTED_CANDIDATES,
        expected_feature_count=EXPECTED_FEATURES,
    )
    final_alignment = validate_frozen_validation_alignment(
        positives=sampled_val,
        row_indices=output["row_indices"],
        candidates=output["candidates"],
        reference_src=reference["src"],
        reference_dst=reference["dst"],
        reference_time=reference["time"],
        reference_row_indices=reference["row_indices"],
        reference_candidates=reference["candidates"],
    )
    reference_descriptors = {
        name: _descriptor(reference_paths[name])
        for name in frozen_sidecars
    }
    artifacts = {
        name: _descriptor(path)
        for name, path in paths.items()
        if name != "progress"
    }
    for name in ("candidates", "src", "dst", "time", "row_indices"):
        if artifacts[name]["sha256"] != reference_descriptors[name][
            "sha256"
        ]:
            raise ValueError(
                f"frozen validation {name} byte hash differs"
            )

    elapsed = time.time() - started
    recovery = {
        "protocol": "frozen_candidate_validation_recovery_v1",
        "recovery_process_id": os.getpid(),
        "train_joint_build_id": train_binding["joint_build_id"],
        "train_joint_process_id": train_binding["joint_process_id"],
        "train_feature_sha256": train_binding["train_feature_sha256"],
        "candidate_sampling_performed": False,
        "frozen_query_alignment_exact": True,
        "reference_sidecars_exact": final_alignment,
        "prebuild_reference_alignment": initial_frozen_alignment,
        "reference_artifacts": reference_descriptors,
    }
    report = {
        "status": "complete",
        "dataset_name": args.dataset_name,
        "protocol": (
            "K512 validation-only feature rebuild over the frozen "
            "20k x 100 query contract; candidate sampling is disabled"
        ),
        "joint_build": {
            "id": train_binding["joint_build_id"],
            "pid": os.getpid(),
            "role": "validation",
        },
        "recovery": recovery,
        "train_replay": {
            "matched": True,
            "method": (
                "training cache hash binding plus frozen validation "
                "query replay"
            ),
        },
        "train_cache_report": str(
            args.train_cache_report.resolve()
        ),
        "train_cache_report_sha256": _sha256(
            args.train_cache_report
        ),
        "train_feature_sha256": train_binding[
            "train_feature_sha256"
        ],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "feature_names": list(feature_names),
        "prediction_limits": {
            "structure_predict_neighbor_limit": int(
                config.structure_predict_neighbor_limit
            ),
            "source_profile_predict_history_limit": int(
                config.source_profile_predict_history_limit
            ),
        },
        "validation_shape": list(validation_shape),
        "candidate_shape": list(validation_shape[:2]),
        "cache_contract": cache_contract,
        "candidate_checksum": candidate_checksum,
        "requested_structure_workers": args.structure_workers,
        "selected_structure_workers": (
            1
            if parallel_structure is None
            else parallel_structure.worker_count
        ),
        "parallel_parity": parallel_parity,
        "parallel_trial": parallel_trial,
        "split": {
            "train_end": train_end,
            "validation_row_start": int(validation_row_indices[0]),
            "validation_row_stop_exclusive": (
                int(validation_row_indices[-1]) + 1
            ),
            "validation_rows_strictly_increasing": bool(
                np.all(
                    validation_row_indices[1:]
                    > validation_row_indices[:-1]
                )
            ),
            "validation_time_min": int(sampled_val.time.min()),
            "validation_time_max": int(sampled_val.time.max()),
        },
        "artifacts": artifacts,
        "encoder_rng_state_before_fit": encoder_rng_state_before_fit,
        "post_validation_rng_state": _jsonable(
            encoder_rng.bit_generator.state
        ),
        "elapsed_seconds": elapsed,
    }
    _write_json_atomic(args.report, report)
    _write_json_atomic(
        paths["progress"],
        {
            "status": "complete",
            "protocol": "frozen_candidate_validation_recovery_v1",
            "candidate_sampling_performed": False,
            "completed_rows": len(sampled_val),
            "total_rows": len(sampled_val),
            "joint_build_id": train_binding["joint_build_id"],
            "recovery_process_id": os.getpid(),
            "elapsed_seconds": elapsed,
        },
    )
    print(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        flush=True,
    )
    del val_encoder, encoder_cache, ranker
    gc.collect()
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild only the dataset2 K512 validation features while "
            "reusing a byte-exact frozen validation candidate contract."
        )
    )
    parser.add_argument(
        "--dataset-name",
        choices=("dataset2",),
        default="dataset2",
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument(
        "--train-cache-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--frozen-validation-prefix",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--structure-workers", type=int, default=8)
    parser.add_argument(
        "--minimum-parallel-speedup",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--comparison-structure-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--minimum-comparison-speedup",
        type=float,
        default=1.10,
    )
    parser.add_argument(
        "--minimum-memory-reserve-gib",
        type=float,
        default=8.0,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_rows <= 0:
        raise ValueError("batch rows must be positive")
    if args.structure_workers <= 1:
        raise ValueError("structure workers must be greater than one")
    if args.minimum_parallel_speedup <= 1.0:
        raise ValueError(
            "minimum parallel speedup must be greater than one"
        )
    if args.comparison_structure_workers < 2:
        raise ValueError(
            "comparison structure workers must be at least two"
        )
    if args.structure_workers <= args.comparison_structure_workers:
        raise ValueError(
            "trial structure workers must exceed comparison workers"
        )
    if args.minimum_comparison_speedup <= 1.0:
        raise ValueError(
            "minimum comparison speedup must be greater than one"
        )
    if args.minimum_memory_reserve_gib <= 0.0:
        raise ValueError("minimum memory reserve must be positive")


def _validate_train_report(
    report: dict[str, Any],
    *,
    checkpoint: Path,
) -> dict[str, Any]:
    if report.get("status") != "complete":
        raise ValueError("training cache report is incomplete")
    if report.get("dataset_name") != "dataset2":
        raise ValueError("training cache report is not dataset2")
    if report.get("train_selection") != "recent":
        raise ValueError("training cache selection is not recent")
    if tuple(report.get("train_shape", ())) != (
        200_000,
        EXPECTED_CANDIDATES,
        EXPECTED_FEATURES,
    ):
        raise ValueError("training cache shape differs")
    limits = report.get("prediction_limits")
    if (
        not isinstance(limits, dict)
        or limits.get("structure_predict_neighbor_limit") != 512
        or limits.get("source_profile_predict_history_limit") != 512
    ):
        raise ValueError("training cache is not K512")
    if report.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("training cache checkpoint differs")
    joint = report.get("joint_build")
    if (
        not isinstance(joint, dict)
        or joint.get("role") != "train"
        or not str(joint.get("id", ""))
        or int(joint.get("pid", -1)) <= 0
    ):
        raise ValueError("training cache joint provenance differs")
    artifacts = report.get("artifacts")
    feature = (
        artifacts.get("features")
        if isinstance(artifacts, dict)
        else None
    )
    if not isinstance(feature, dict):
        raise ValueError("training feature artifact is missing")
    feature_path = Path(str(feature.get("path", "")))
    if (
        not feature_path.is_file()
        or feature_path.stat().st_size != int(feature.get("bytes", -1))
        or _sha256(feature_path) != feature.get("sha256")
    ):
        raise ValueError("training feature artifact differs")
    if not isinstance(report.get("fusion_rng_state_after_build"), dict):
        raise ValueError("training cache lacks its post-build RNG state")
    split = report.get("split")
    if not isinstance(split, dict) or int(split.get("train_end", -1)) <= 1:
        raise ValueError("training cache split differs")
    return {
        "joint_build_id": str(joint["id"]),
        "joint_process_id": int(joint["pid"]),
        "train_feature_sha256": str(feature["sha256"]),
    }


def _descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _copy_file_atomic(source: Path, destination: Path) -> None:
    temp_path = destination.with_name(f".{destination.name}.part")
    if temp_path.exists():
        raise FileExistsError(
            f"partial frozen sidecar exists: {temp_path}"
        )
    with source.open("rb") as source_handle, temp_path.open(
        "xb"
    ) as destination_handle:
        shutil.copyfileobj(
            source_handle,
            destination_handle,
            length=4 * 1024 * 1024,
        )
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    os.replace(temp_path, destination)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
