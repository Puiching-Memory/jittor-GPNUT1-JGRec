from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.full100_training import (
    resolve_training_context_end,
    sample_chronological_events,
    select_recent_events,
    validate_candidate_matrix,
    validate_full100_cache_arrays,
)
from jgrec.rankers.hybrid.parallel_structure import (
    ForkedStructureFeatureTower,
    select_parallel_worker_trial,
    validate_exact_parallel_features,
)
from jgrec.rankers.hybrid.ranker import (
    SupervisedFeatureBuilder,
    TemporalHybridRanker,
)


def main(*, default_dataset_name: str = "dataset2") -> int:
    parser = argparse.ArgumentParser(
        description="Build a full-100 supervised training-feature cache."
    )
    parser.add_argument(
        "--dataset-name",
        choices=("dataset1", "dataset2"),
        default=default_dataset_name,
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-cache-prefix", type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--validation-output-prefix", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--train-rows", type=int, default=50_000)
    parser.add_argument("--validation-rows", type=int, default=20_000)
    parser.add_argument(
        "--train-selection",
        choices=("sampled", "recent"),
        default="sampled",
    )
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--structure-workers", type=int, default=1)
    parser.add_argument(
        "--minimum-parallel-speedup",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--comparison-structure-workers",
        type=int,
        default=0,
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
    args = parser.parse_args()

    if args.candidate_count != 100:
        raise ValueError("the frozen experiment requires exactly 100 training candidates")
    if args.train_rows <= 0:
        raise ValueError("training rows must be positive")
    if args.validation_rows <= 0:
        raise ValueError("validation rows must be positive")
    if args.batch_rows <= 0:
        raise ValueError("batch rows must be positive")
    if args.structure_workers <= 0:
        raise ValueError("structure workers must be positive")
    if (
        args.structure_workers > 1
        and args.minimum_parallel_speedup <= 1.0
    ):
        raise ValueError(
            "minimum parallel speedup must be greater than one"
        )
    if args.comparison_structure_workers < 0:
        raise ValueError("comparison structure workers must not be negative")
    if args.comparison_structure_workers:
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
    replay_values = (args.source_cache_prefix, args.replay_report)
    if any(value is not None for value in replay_values) and not all(
        value is not None for value in replay_values
    ):
        raise ValueError(
            "source replay requires both --source-cache-prefix "
            "and --replay-report"
        )
    joint_values = (
        args.validation_output_prefix,
        args.validation_report,
    )
    if any(value is not None for value in joint_values) and not all(
        value is not None for value in joint_values
    ):
        raise ValueError(
            "joint validation requires both --validation-output-prefix "
            "and --validation-report"
        )
    joint_build = all(value is not None for value in joint_values)
    replay = (
        _require_replay_evidence(args)
        if all(value is not None for value in replay_values)
        else {
            "status": "not_required_for_fresh_joint_build",
            "feature_report": None,
        }
    )
    _configure_cuda()

    paths = _output_paths(args.output_prefix)
    validation_paths = (
        _validation_paths(args.validation_output_prefix)
        if joint_build
        else {}
    )
    protected = (
        *paths.values(),
        *validation_paths.values(),
        args.report,
        *([args.validation_report] if args.validation_report is not None else []),
    )
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite full-100 artifacts: {existing}")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.validation_output_prefix is not None:
        args.validation_output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.validation_report is not None:
        args.validation_report.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    joint_build_id = uuid.uuid4().hex if joint_build else None
    state = load_checkpoint_dataset(args.checkpoint, args.dataset_name)
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    source_manifest_path: Path | None = None
    source_val_path: Path | None = None
    source_manifest: dict[str, Any] | None = None
    source_val: np.ndarray | None = None
    if args.source_cache_prefix is not None:
        source_manifest_path = args.source_cache_prefix.with_suffix(".json")
        source_val_path = args.source_cache_prefix.with_suffix(".val.npy")
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        source_val = np.load(source_val_path, mmap_mode="r", allow_pickle=False)
        if list(source_val.shape) != source_manifest["val"]["shape"]:
            raise ValueError("source validation shape does not match its manifest")
        if source_val.shape != (20_000, 100, len(feature_names)):
            raise ValueError(
                "source validation tensor is not the frozen 20k x 100 tensor"
            )

    interactions = read_interactions(args.train_csv).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    val_size = max(1, int(len(interactions) * config.val_ratio))
    train_end = max(2, len(interactions) - val_size)
    configured_context_end = max(
        1,
        min(train_end - 1, int(train_end * config.context_ratio)),
    )
    context_end = resolve_training_context_end(
        train_end=train_end,
        configured_context_ratio=config.context_ratio,
        requested_train_rows=args.train_rows,
    )
    rng = np.random.default_rng(config.seed)
    train_pool = interactions[context_end:train_end]
    if args.train_selection == "recent":
        sampled_train, relative_row_indices = select_recent_events(
            train_pool,
            requested_rows=args.train_rows,
        )
    else:
        if len(train_pool) < args.train_rows:
            raise ValueError(
                f"training event pool has only {len(train_pool)} rows; "
                f"{args.train_rows} were requested"
            )
        relative_row_indices = np.sort(
            rng.choice(len(train_pool), size=args.train_rows, replace=False)
        )
        sampled_train = train_pool.take(relative_row_indices)
    train_row_indices = relative_row_indices.astype(np.int64, copy=False) + context_end
    sampled_val, validation_row_indices = sample_chronological_events(
        interactions[train_end:],
        max_events=config.max_val_events,
        rng=rng,
        row_offset=train_end,
    )
    train_shape = (len(sampled_train), args.candidate_count, len(feature_names))
    candidate_shape = train_shape[:2]
    if train_shape != (args.train_rows, 100, 63):
        raise ValueError(f"frozen full-100 training shape mismatch: {train_shape}")
    if source_val is not None and len(sampled_val) != source_val.shape[0]:
        raise ValueError("sampled validation rows do not align with the frozen validation tensor")
    if joint_build and len(sampled_val) != args.validation_rows:
        raise ValueError(
            f"joint validation rows differ: actual={len(sampled_val)} "
            f"expected={args.validation_rows}"
        )

    required_bytes = (
        int(np.prod(train_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
        + int(np.prod(candidate_shape, dtype=np.int64)) * np.dtype(np.int32).itemsize
        + len(sampled_train)
        * (
            3 * np.dtype(np.int32).itemsize
            + np.dtype(np.int64).itemsize
        )
    )
    if joint_build:
        validation_shape = (
            len(sampled_val),
            args.candidate_count,
            len(feature_names),
        )
        required_bytes += (
            int(np.prod(validation_shape, dtype=np.int64))
            * np.dtype(np.float32).itemsize
            + int(np.prod(validation_shape[:2], dtype=np.int64))
            * np.dtype(np.int32).itemsize
            + len(sampled_val)
            * (
                3 * np.dtype(np.int32).itemsize
                + np.dtype(np.int64).itemsize
            )
        )
    free_bytes = shutil.disk_usage(args.output_prefix.parent).free
    if free_bytes < required_bytes + 2 * 1024**3:
        raise OSError(
            f"not enough free disk for atomic full-100 build: free={free_bytes} required={required_bytes}"
        )

    supervised_config = replace(
        config,
        structure_future_only_transition_cooccur=True,
        supervised_feature_cache_dir=None,
        verbose=True,
    )
    ranker = TemporalHybridRanker(recent_window=int(state["recent_window"]))
    ranker.id_map = NodeIdMap.from_interactions(interactions)
    ranker.dataset_profile = state["dataset_profile"]
    encoder_cache = ranker._encoder_state_cache(
        interactions,
        supervised_config,
        verbose=True,
    )
    train_snapshot = (
        encoder_cache.snapshot_for_prefix(context_end)
        if encoder_cache is not None
        else None
    )
    encoder = ranker._timed_fit_encoder(
        "full100_train_context_encoder",
        interactions[:context_end],
        supervised_config,
        rng,
        verbose=True,
        deterministic_snapshot=train_snapshot,
    )
    if tuple(encoder.feature_names) != feature_names:
        raise RuntimeError("full-100 encoder feature schema differs from the checkpoint")
    if encoder_cache is not None:
        encoder_cache.release_except()
    del train_snapshot, state
    gc.collect()

    build_config = replace(
        supervised_config,
        num_negatives=args.candidate_count - 1,
        supervised_feature_batch_size=args.batch_rows,
    )
    builder = SupervisedFeatureBuilder(
        encoder=encoder,
        dst_pool=np.unique(interactions.dst).astype(np.int64, copy=False),
        config=build_config,
        label="full100_train_features",
    )
    temp_feature_path = paths["features"].with_name(f".{paths['features'].name}.part")
    temp_candidate_path = paths["candidates"].with_name(
        f".{paths['candidates'].name}.part"
    )
    if temp_feature_path.exists() or temp_candidate_path.exists():
        raise FileExistsError("a partial full-100 cache already exists; inspect it before retrying")
    features = np.lib.format.open_memmap(
        temp_feature_path,
        mode="w+",
        dtype=np.float32,
        shape=train_shape,
    )
    candidates = np.lib.format.open_memmap(
        temp_candidate_path,
        mode="w+",
        dtype=np.int32,
        shape=candidate_shape,
    )

    total_batches = (len(sampled_train) + args.batch_rows - 1) // args.batch_rows
    candidate_checksum = 0
    nested_control_checksum = 0
    parallel_structure: ForkedStructureFeatureTower | None = None
    parallel_parity: dict[str, Any] | None = None
    parallel_trial: dict[str, Any] | None = None
    try:
        for start in range(0, len(sampled_train), args.batch_rows):
            end = min(start + args.batch_rows, len(sampled_train))
            batch_id = start // args.batch_rows + 1
            batch_events = sampled_train[start:end]
            batch_started = time.time()
            queries = builder.batch_for_events(batch_events, rng)
            candidate_report = validate_candidate_matrix(
                batch_events.dst,
                queries.candidates,
                expected_candidate_count=args.candidate_count,
            )
            nested_control_report = validate_candidate_matrix(
                batch_events.dst,
                queries.candidates[:, :32],
                expected_candidate_count=32,
            )
            (
                batch_features,
                parallel_structure,
                parallel_parity,
                parallel_trial,
            ) = _features_with_parallel_gate(
                encoder=encoder,
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
                label="full100-cache",
            )
            if batch_features.shape != (end - start, args.candidate_count, len(feature_names)):
                raise RuntimeError("encoder returned an unexpected full-100 feature shape")
            if not np.all(np.isfinite(batch_features)):
                raise ValueError(f"non-finite feature value in batch {batch_id}")
            features[start:end] = batch_features
            candidates[start:end] = queries.candidates
            candidate_checksum += _candidate_checksum(queries.candidates)
            nested_control_checksum += _candidate_checksum(queries.candidates[:, :32])
            del batch_features, queries
            encoder.clear_batch_caches()
            if batch_id % 2 == 0 or end == len(sampled_train):
                features.flush()
                candidates.flush()
            gc.collect()
            progress = {
                "status": "building",
                "completed_rows": end,
                "total_rows": len(sampled_train),
                "batch": batch_id,
                "total_batches": total_batches,
                "last_batch_candidate_contract": candidate_report,
                "last_nested32_candidate_contract": nested_control_report,
                "candidate_checksum": candidate_checksum,
                "nested32_candidate_checksum": nested_control_checksum,
                "requested_structure_workers": args.structure_workers,
                "selected_structure_workers": (
                    1
                    if parallel_structure is None
                    else parallel_structure.worker_count
                ),
                "parallel_parity": parallel_parity,
                "parallel_trial": parallel_trial,
                "elapsed_seconds": time.time() - started,
                "last_batch_seconds": time.time() - batch_started,
            }
            _write_json_atomic(paths["progress"], progress)
            print(
                f"[full100-cache] batch={batch_id}/{total_batches} rows={end}/{len(sampled_train)} "
                f"batch_seconds={progress['last_batch_seconds']:.1f} "
                f"elapsed={progress['elapsed_seconds']:.1f}",
                flush=True,
            )
        features.flush()
        candidates.flush()
    except BaseException:
        features.flush()
        candidates.flush()
        _close_parallel_encoder(
            encoder,
            parallel_structure,
            terminate=True,
        )
        raise
    else:
        _close_parallel_encoder(encoder, parallel_structure)
        del features, candidates
        os.replace(temp_feature_path, paths["features"])
        os.replace(temp_candidate_path, paths["candidates"])

    _save_array_atomic(paths["src"], sampled_train.src.astype(np.int32, copy=False))
    _save_array_atomic(paths["dst"], sampled_train.dst.astype(np.int32, copy=False))
    _save_array_atomic(paths["time"], sampled_train.time.astype(np.int64, copy=False))
    _save_array_atomic(paths["row_indices"], train_row_indices)
    cache_contract = validate_full100_cache_arrays(
        features=np.load(paths["features"], mmap_mode="r", allow_pickle=False),
        candidates=np.load(paths["candidates"], mmap_mode="r", allow_pickle=False),
        src=np.load(paths["src"], mmap_mode="r", allow_pickle=False),
        dst=np.load(paths["dst"], mmap_mode="r", allow_pickle=False),
        time=np.load(paths["time"], mmap_mode="r", allow_pickle=False),
        row_indices=np.load(paths["row_indices"], mmap_mode="r", allow_pickle=False),
        expected_train_rows=args.train_rows,
        expected_candidate_count=args.candidate_count,
        expected_feature_count=len(feature_names),
    )
    report = {
        "status": "complete",
        "dataset_name": args.dataset_name,
        "protocol": (
            "one full100 feature cache; matched32 control is the nested candidate "
            "and feature view at positions [:, :32]"
        ),
        "replay_status": replay["status"],
        "replay_feature_report": replay["feature_report"],
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
        "train_shape": list(train_shape),
        "candidate_shape": list(candidate_shape),
        "train_selection": args.train_selection,
        "requested_train_rows": args.train_rows,
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
        "nested32_control": {
            "train_shape": [len(sampled_train), 32, len(feature_names)],
            "candidate_shape": [len(sampled_train), 32],
            "candidate_checksum": nested_control_checksum,
            "derived_from_full100_positions": [0, 32],
        },
        "sampled_validation_rows": len(sampled_val),
        "split": {
            "configured_context_end": configured_context_end,
            "context_end": context_end,
            "context_backoff_rows": configured_context_end - context_end,
            "train_end": train_end,
            "interaction_rows": len(interactions),
            "training_pool_rows": len(train_pool),
            "selected_sorted_row_start": int(train_row_indices[0]),
            "selected_sorted_row_stop_exclusive": int(train_row_indices[-1]) + 1,
        },
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
            if name != "progress"
        },
        "fusion_rng_state_after_build": _jsonable(rng.bit_generator.state),
        "elapsed_seconds": time.time() - started,
    }
    if (
        source_manifest is not None
        and source_manifest_path is not None
        and source_val is not None
        and source_val_path is not None
    ):
        report["source_cache_key"] = source_manifest["key"]
        report["source_cache_manifest_sha256"] = _sha256(source_manifest_path)
        report["source_validation_path"] = str(source_val_path.resolve())
        report["source_validation_shape"] = list(source_val.shape)
        report["source_validation_sha256"] = _sha256(source_val_path)
    if joint_build_id is not None:
        report["joint_build"] = {
            "id": joint_build_id,
            "pid": os.getpid(),
            "role": "train",
        }
    _write_json_atomic(args.report, report)
    _write_json_atomic(
        paths["progress"],
        {
            "status": "complete",
            "completed_rows": len(sampled_train),
            "total_rows": len(sampled_train),
            "elapsed_seconds": report["elapsed_seconds"],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    del builder, encoder
    gc.collect()
    if joint_build:
        if (
            args.validation_output_prefix is None
            or args.validation_report is None
            or joint_build_id is None
        ):
            raise AssertionError("joint validation arguments were not normalized")
        _build_joint_validation_cache(
            ranker=ranker,
            dataset_name=args.dataset_name,
            encoder_cache=encoder_cache,
            interactions=interactions,
            supervised_config=supervised_config,
            rng=rng,
            train_end=train_end,
            sampled_val=sampled_val,
            validation_row_indices=validation_row_indices,
            dst_pool=np.unique(interactions.dst).astype(np.int64, copy=False),
            feature_names=feature_names,
            candidate_count=args.candidate_count,
            batch_rows=args.batch_rows,
            structure_workers=args.structure_workers,
            minimum_parallel_speedup=args.minimum_parallel_speedup,
            comparison_structure_workers=(
                args.comparison_structure_workers
            ),
            minimum_comparison_speedup=(
                args.minimum_comparison_speedup
            ),
            minimum_memory_reserve_bytes=int(
                args.minimum_memory_reserve_gib * 1024**3
            ),
            output_prefix=args.validation_output_prefix,
            report_path=args.validation_report,
            train_report_path=args.report,
            train_report=report,
            joint_build_id=joint_build_id,
            started=started,
        )
    del encoder_cache, ranker
    gc.collect()
    return 0


def _build_joint_validation_cache(
    *,
    ranker: TemporalHybridRanker,
    dataset_name: str,
    encoder_cache: Any,
    interactions: Any,
    supervised_config: Any,
    rng: np.random.Generator,
    train_end: int,
    sampled_val: Any,
    validation_row_indices: np.ndarray,
    dst_pool: np.ndarray,
    feature_names: tuple[str, ...],
    candidate_count: int,
    batch_rows: int,
    structure_workers: int,
    minimum_parallel_speedup: float,
    comparison_structure_workers: int,
    minimum_comparison_speedup: float,
    minimum_memory_reserve_bytes: int,
    output_prefix: Path,
    report_path: Path,
    train_report_path: Path,
    train_report: dict[str, Any],
    joint_build_id: str,
    started: float,
) -> None:
    paths = _validation_paths(output_prefix)
    val_snapshot = (
        encoder_cache.snapshot_for_prefix(train_end)
        if encoder_cache is not None
        else None
    )
    val_encoder = ranker._timed_fit_encoder(
        "joint_full100_train_end_encoder",
        interactions[:train_end],
        supervised_config,
        rng,
        verbose=True,
        deterministic_snapshot=val_snapshot,
    )
    if tuple(val_encoder.feature_names) != feature_names:
        raise RuntimeError("joint validation encoder feature schema differs")
    if encoder_cache is not None:
        encoder_cache.release_except()
    del val_snapshot
    gc.collect()

    val_shape = (len(sampled_val), candidate_count, len(feature_names))
    candidate_shape = val_shape[:2]
    builder = SupervisedFeatureBuilder(
        encoder=val_encoder,
        dst_pool=dst_pool,
        config=replace(
            supervised_config,
            num_negatives=candidate_count - 1,
            supervised_feature_batch_size=batch_rows,
        ),
        label="joint_full100_validation_features",
    )
    temp_feature_path = paths["features"].with_name(
        f".{paths['features'].name}.part"
    )
    temp_candidate_path = paths["candidates"].with_name(
        f".{paths['candidates'].name}.part"
    )
    if temp_feature_path.exists() or temp_candidate_path.exists():
        raise FileExistsError(
            "a partial joint validation cache exists; inspect it before retrying"
        )
    features = np.lib.format.open_memmap(
        temp_feature_path,
        mode="w+",
        dtype=np.float32,
        shape=val_shape,
    )
    candidates = np.lib.format.open_memmap(
        temp_candidate_path,
        mode="w+",
        dtype=np.int32,
        shape=candidate_shape,
    )

    total_batches = (len(sampled_val) + batch_rows - 1) // batch_rows
    candidate_checksum = 0
    parallel_structure: ForkedStructureFeatureTower | None = None
    parallel_parity: dict[str, Any] | None = None
    parallel_trial: dict[str, Any] | None = None
    try:
        for start in range(0, len(sampled_val), batch_rows):
            end = min(start + batch_rows, len(sampled_val))
            batch_id = start // batch_rows + 1
            batch_started = time.time()
            batch_events = sampled_val[start:end]
            queries = builder.batch_for_events(batch_events, rng)
            candidate_report = validate_candidate_matrix(
                batch_events.dst,
                queries.candidates,
                expected_candidate_count=candidate_count,
            )
            (
                batch_features,
                parallel_structure,
                parallel_parity,
                parallel_trial,
            ) = _features_with_parallel_gate(
                encoder=val_encoder,
                queries=queries,
                worker_count=structure_workers,
                minimum_speedup=minimum_parallel_speedup,
                comparison_worker_count=comparison_structure_workers,
                minimum_comparison_speedup=minimum_comparison_speedup,
                minimum_memory_reserve_bytes=(
                    minimum_memory_reserve_bytes
                ),
                parallel_structure=parallel_structure,
                parallel_parity=parallel_parity,
                parallel_trial=parallel_trial,
                label="joint-val-cache",
            )
            expected_shape = (
                end - start,
                candidate_count,
                len(feature_names),
            )
            if batch_features.shape != expected_shape:
                raise RuntimeError(
                    "joint validation encoder returned an unexpected shape"
                )
            if not np.all(np.isfinite(batch_features)):
                raise ValueError(
                    f"non-finite joint validation feature in batch {batch_id}"
                )
            features[start:end] = batch_features
            candidates[start:end] = queries.candidates
            candidate_checksum += _candidate_checksum(queries.candidates)
            del batch_features, queries
            val_encoder.clear_batch_caches()
            features.flush()
            candidates.flush()
            gc.collect()
            progress = {
                "status": "building",
                "completed_rows": end,
                "total_rows": len(sampled_val),
                "batch": batch_id,
                "total_batches": total_batches,
                "last_candidate_contract": candidate_report,
                "candidate_checksum": candidate_checksum,
                "requested_structure_workers": structure_workers,
                "selected_structure_workers": (
                    1
                    if parallel_structure is None
                    else parallel_structure.worker_count
                ),
                "parallel_parity": parallel_parity,
                "parallel_trial": parallel_trial,
                "joint_build_id": joint_build_id,
                "process_id": os.getpid(),
                "elapsed_seconds": time.time() - started,
                "last_batch_seconds": time.time() - batch_started,
            }
            _write_json_atomic(paths["progress"], progress)
            print(
                f"[joint-val-cache] batch={batch_id}/{total_batches} "
                f"rows={end}/{len(sampled_val)} "
                f"batch_seconds={progress['last_batch_seconds']:.1f} "
                f"elapsed={progress['elapsed_seconds']:.1f}",
                flush=True,
            )
        features.flush()
        candidates.flush()
    except BaseException:
        features.flush()
        candidates.flush()
        _close_parallel_encoder(
            val_encoder,
            parallel_structure,
            terminate=True,
        )
        raise
    else:
        _close_parallel_encoder(val_encoder, parallel_structure)
        del features, candidates
        os.replace(temp_feature_path, paths["features"])
        os.replace(temp_candidate_path, paths["candidates"])

    _save_array_atomic(
        paths["src"],
        sampled_val.src.astype(np.int32, copy=False),
    )
    _save_array_atomic(
        paths["dst"],
        sampled_val.dst.astype(np.int32, copy=False),
    )
    _save_array_atomic(
        paths["time"],
        sampled_val.time.astype(np.int64, copy=False),
    )
    _save_array_atomic(paths["row_indices"], validation_row_indices)
    cache_contract = validate_full100_cache_arrays(
        features=np.load(paths["features"], mmap_mode="r", allow_pickle=False),
        candidates=np.load(
            paths["candidates"],
            mmap_mode="r",
            allow_pickle=False,
        ),
        src=np.load(paths["src"], mmap_mode="r", allow_pickle=False),
        dst=np.load(paths["dst"], mmap_mode="r", allow_pickle=False),
        time=np.load(paths["time"], mmap_mode="r", allow_pickle=False),
        row_indices=np.load(
            paths["row_indices"],
            mmap_mode="r",
            allow_pickle=False,
        ),
        expected_train_rows=len(sampled_val),
        expected_candidate_count=candidate_count,
        expected_feature_count=len(feature_names),
    )
    validation_report = {
        "status": "complete",
        "dataset_name": dataset_name,
        "protocol": (
            "one Python process and one live prefix-state cache built the "
            f"recent-{train_report['requested_train_rows']} context_end "
            "training cache followed by the train_end full-100 validation cache"
        ),
        "joint_build": {
            "id": joint_build_id,
            "pid": os.getpid(),
            "role": "validation",
        },
        "train_replay": {
            "matched": True,
            "method": "same_process_joint_build",
        },
        "train_cache_report": str(train_report_path.resolve()),
        "train_cache_report_sha256": _sha256(train_report_path),
        "train_feature_sha256": train_report["artifacts"]["features"]["sha256"],
        "checkpoint": train_report["checkpoint"],
        "checkpoint_sha256": train_report["checkpoint_sha256"],
        "feature_names": list(feature_names),
        "prediction_limits": {
            "structure_predict_neighbor_limit": int(
                supervised_config.structure_predict_neighbor_limit
            ),
            "source_profile_predict_history_limit": int(
                supervised_config.source_profile_predict_history_limit
            ),
        },
        "validation_shape": list(val_shape),
        "candidate_shape": list(candidate_shape),
        "cache_contract": cache_contract,
        "candidate_checksum": candidate_checksum,
        "requested_structure_workers": structure_workers,
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
            "validation_row_stop_exclusive": int(validation_row_indices[-1]) + 1,
            "validation_rows_strictly_increasing": bool(
                np.all(validation_row_indices[1:] > validation_row_indices[:-1])
            ),
            "validation_time_min": int(sampled_val.time.min()),
            "validation_time_max": int(sampled_val.time.max()),
        },
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
            if name != "progress"
        },
        "post_validation_rng_state": _jsonable(rng.bit_generator.state),
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(report_path, validation_report)
    _write_json_atomic(
        paths["progress"],
        {
            "status": "complete",
            "completed_rows": len(sampled_val),
            "total_rows": len(sampled_val),
            "joint_build_id": joint_build_id,
            "process_id": os.getpid(),
            "elapsed_seconds": validation_report["elapsed_seconds"],
        },
    )
    print(
        json.dumps(
            validation_report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    del builder, val_encoder
    gc.collect()


def _features_with_parallel_gate(
    *,
    encoder: Any,
    queries: Any,
    worker_count: int,
    minimum_speedup: float,
    comparison_worker_count: int,
    minimum_comparison_speedup: float,
    minimum_memory_reserve_bytes: int,
    parallel_structure: ForkedStructureFeatureTower | None,
    parallel_parity: dict[str, Any] | None,
    parallel_trial: dict[str, Any] | None,
    label: str,
) -> tuple[
    np.ndarray,
    ForkedStructureFeatureTower | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    if worker_count <= 1:
        return (
            encoder.features_for_query_array(queries),
            parallel_structure,
            parallel_parity,
            parallel_trial,
        )
    if parallel_structure is not None:
        return (
            encoder.features_for_query_array(queries),
            parallel_structure,
            parallel_parity,
            parallel_trial,
        )

    sequential_started = time.perf_counter()
    sequential_features = encoder.features_for_query_array(queries)
    sequential_seconds = time.perf_counter() - sequential_started
    encoder.clear_batch_caches()
    source_structure = encoder.structure
    source_profile = encoder.source_profile
    try:
        if comparison_worker_count:
            (
                comparison_features,
                comparison_tower,
                comparison_parity,
            ) = _evaluate_parallel_arm(
                encoder=encoder,
                queries=queries,
                source_structure=source_structure,
                source_profile=source_profile,
                worker_count=comparison_worker_count,
                sequential_features=sequential_features,
                sequential_seconds=sequential_seconds,
                minimum_speedup=minimum_speedup,
            )
            _close_parallel_encoder(encoder, comparison_tower)
            encoder.clear_batch_caches()
            (
                trial_features,
                trial_tower,
                trial_parity,
            ) = _evaluate_parallel_arm(
                encoder=encoder,
                queries=queries,
                source_structure=source_structure,
                source_profile=source_profile,
                worker_count=worker_count,
                sequential_features=sequential_features,
                sequential_seconds=sequential_seconds,
                minimum_speedup=minimum_speedup,
            )
            try:
                available_memory_bytes = _available_memory_bytes()
                decision = select_parallel_worker_trial(
                    baseline_report=comparison_parity,
                    trial_report=trial_parity,
                    baseline_worker_count=comparison_worker_count,
                    trial_worker_count=worker_count,
                    minimum_incremental_speedup=(
                        minimum_comparison_speedup
                    ),
                    available_memory_bytes=available_memory_bytes,
                    minimum_memory_reserve_bytes=(
                        minimum_memory_reserve_bytes
                    ),
                )
            except BaseException:
                _close_parallel_encoder(
                    encoder,
                    trial_tower,
                    terminate=True,
                )
                del trial_features, comparison_features
                raise
            trial_report = {
                "status": "selected",
                "baseline": comparison_parity,
                "trial": trial_parity,
                "decision": decision,
            }
            if decision["selected_worker_count"] == worker_count:
                del comparison_features
                print(
                    f"[{label}] worker trial selected "
                    f"{worker_count} workers "
                    f"incremental_speedup="
                    f"{decision['incremental_speedup']:.3f}x "
                    f"available_gib="
                    f"{available_memory_bytes / 1024**3:.2f}",
                    flush=True,
                )
                return (
                    trial_features,
                    trial_tower,
                    trial_parity,
                    trial_report,
                )

            _close_parallel_encoder(encoder, trial_tower)
            del trial_features, comparison_features
            encoder.clear_batch_caches()
            (
                fallback_features,
                fallback_tower,
                fallback_parity,
            ) = _evaluate_parallel_arm(
                encoder=encoder,
                queries=queries,
                source_structure=source_structure,
                source_profile=source_profile,
                worker_count=comparison_worker_count,
                sequential_features=sequential_features,
                sequential_seconds=sequential_seconds,
                minimum_speedup=minimum_speedup,
            )
            trial_report["fallback_replay"] = fallback_parity
            print(
                f"[{label}] worker trial fell back to "
                f"{comparison_worker_count} workers "
                f"reason={decision['fallback_reason']} "
                f"incremental_speedup="
                f"{decision['incremental_speedup']:.3f}x",
                flush=True,
            )
            return (
                fallback_features,
                fallback_tower,
                fallback_parity,
                trial_report,
            )

        parallel_features, tower, parity = _evaluate_parallel_arm(
            encoder=encoder,
            queries=queries,
            source_structure=source_structure,
            source_profile=source_profile,
            worker_count=worker_count,
            sequential_features=sequential_features,
            sequential_seconds=sequential_seconds,
            minimum_speedup=minimum_speedup,
        )
    except BaseException:
        encoder.structure = source_structure
        encoder.source_profile = source_profile
        raise
    finally:
        del sequential_features
        gc.collect()
    print(
        f"[{label}] exact parallel parity passed "
        f"workers={parity['worker_count_observed']} "
        f"speedup={parity['speedup']:.3f}x",
        flush=True,
    )
    return parallel_features, tower, parity, parallel_trial


def _evaluate_parallel_arm(
    *,
    encoder: Any,
    queries: Any,
    source_structure: Any,
    source_profile: Any,
    worker_count: int,
    sequential_features: np.ndarray,
    sequential_seconds: float,
    minimum_speedup: float,
) -> tuple[
    np.ndarray,
    ForkedStructureFeatureTower,
    dict[str, Any],
]:
    tower = ForkedStructureFeatureTower(
        source_structure,
        worker_count=worker_count,
        source_profile=source_profile,
    )
    encoder.structure = tower
    encoder.source_profile = tower.source_profile
    try:
        parallel_started = time.perf_counter()
        parallel_features = encoder.features_for_query_array(queries)
        parallel_seconds = time.perf_counter() - parallel_started
        parity = validate_exact_parallel_features(
            sequential_features,
            parallel_features,
            sequential_seconds=sequential_seconds,
            parallel_seconds=parallel_seconds,
            minimum_speedup=minimum_speedup,
            worker_pids=tower.active_worker_pids,
        )
    except BaseException:
        encoder.structure = source_structure
        encoder.source_profile = source_profile
        tower.close(terminate=True)
        raise
    return parallel_features, tower, parity


def _available_memory_bytes() -> int:
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable for worker trial")


def _close_parallel_encoder(
    encoder: Any,
    parallel_structure: ForkedStructureFeatureTower | None,
    *,
    terminate: bool = False,
) -> None:
    if parallel_structure is None:
        return
    encoder.structure = parallel_structure.source
    encoder.source_profile = parallel_structure.source_profile_source
    parallel_structure.close(terminate=terminate)


def _require_replay_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.replay_report is None or args.source_cache_prefix is None:
        raise ValueError("source replay paths are required")
    report = json.loads(args.replay_report.read_text(encoding="utf-8"))
    if report.get("status") not in {"passed", "rejected"}:
        raise RuntimeError("bounded champion replay did not finish")
    contract = report.get("candidate_report", {})
    if (
        contract.get("rows") != report.get("replay_rows")
        or contract.get("candidate_count") != 32
        or contract.get("positive_mismatches") != 0
        or contract.get("duplicate_rows") != 0
    ):
        raise RuntimeError("bounded replay candidate contract is invalid")
    if Path(report["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("replay checkpoint differs from the full-100 checkpoint")
    if Path(report["cache_prefix"]).resolve() != args.source_cache_prefix.resolve():
        raise ValueError("replay source cache differs from the requested source cache")
    return report


def _output_paths(prefix: Path) -> dict[str, Path]:
    base = str(prefix)
    return {
        "features": Path(f"{base}.train.npy"),
        "candidates": Path(f"{base}.train-candidates.npy"),
        "src": Path(f"{base}.train-src.npy"),
        "dst": Path(f"{base}.train-dst.npy"),
        "time": Path(f"{base}.train-time.npy"),
        "row_indices": Path(f"{base}.train-row-indices.npy"),
        "progress": Path(f"{base}.progress.json"),
    }


def _validation_paths(prefix: Path) -> dict[str, Path]:
    base = str(prefix)
    return {
        "features": Path(f"{base}.val.npy"),
        "candidates": Path(f"{base}.val-candidates.npy"),
        "src": Path(f"{base}.val-src.npy"),
        "dst": Path(f"{base}.val-dst.npy"),
        "time": Path(f"{base}.val-time.npy"),
        "row_indices": Path(f"{base}.val-row-indices.npy"),
        "progress": Path(f"{base}.progress.json"),
    }


def _candidate_checksum(candidates: np.ndarray) -> int:
    values = np.asarray(candidates, dtype=np.int64)
    weights = np.arange(1, values.shape[1] + 1, dtype=np.int64)
    return int((values * weights).sum(dtype=np.int64))


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    temp_path = path.with_name(f".{path.name}.part")
    with temp_path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _configure_cuda() -> None:
    import jittor as jt  # noqa: PLC0415

    if not jt.has_cuda:
        raise RuntimeError("CUDA is required for the full-100 champion encoder")
    jt.flags.use_cuda = 1


if __name__ == "__main__":
    raise SystemExit(main())
