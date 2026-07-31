from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.full100_training import (
    matched_cache_replay_report,
    sample_chronological_events,
    select_recent_events,
    validate_candidate_matrix,
)
from jgrec.rankers.hybrid.ranker import SupervisedFeatureBuilder, TemporalHybridRanker


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the recent-200k Dataset2 encoder and build a matched "
            "train_end/full-100 validation cache."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--validation-rows", type=int, default=20_000)
    parser.add_argument("--replay-rows", type=int, default=4096)
    parser.add_argument("--batch-rows", type=int, default=4096)
    args = parser.parse_args()

    if args.candidate_count != 100:
        raise ValueError("the matched experiment requires exactly 100 candidates")
    if args.validation_rows <= 0 or args.replay_rows <= 0 or args.batch_rows <= 0:
        raise ValueError("validation, replay, and batch rows must be positive")
    _configure_cuda()

    paths = _output_paths(args.output_prefix)
    protected = (*paths.values(), args.report)
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite matched validation artifacts: {existing}")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    train_report = json.loads(args.train_cache_report.read_text(encoding="utf-8"))
    if train_report.get("status") != "complete":
        raise RuntimeError("recent-200k training cache is incomplete")
    if train_report.get("train_selection") != "recent":
        raise ValueError("training cache was not built from the recent event window")
    expected_train_rows = int(train_report["requested_train_rows"])
    if expected_train_rows != 200_000:
        raise ValueError("matched validation requires the frozen recent-200k cache")

    train_paths = _train_paths(args.train_cache_prefix)
    for name, path in train_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing training cache artifact {name}: {path}")
    feature_names = tuple(str(name) for name in train_report["feature_names"])
    train_features = np.load(train_paths["features"], mmap_mode="r", allow_pickle=False)
    train_candidates = np.load(train_paths["candidates"], mmap_mode="r", allow_pickle=False)
    train_src = np.load(train_paths["src"], mmap_mode="r", allow_pickle=False)
    train_dst = np.load(train_paths["dst"], mmap_mode="r", allow_pickle=False)
    train_time = np.load(train_paths["time"], mmap_mode="r", allow_pickle=False)
    train_row_indices = np.load(
        train_paths["row_indices"],
        mmap_mode="r",
        allow_pickle=False,
    )
    if train_features.shape != (expected_train_rows, 100, len(feature_names)):
        raise ValueError("recent training feature shape differs from its report")
    if train_candidates.shape != train_features.shape[:2]:
        raise ValueError("recent training candidates do not align with features")
    for name, values in {
        "src": train_src,
        "dst": train_dst,
        "time": train_time,
        "row_indices": train_row_indices,
    }.items():
        if values.shape != (expected_train_rows,):
            raise ValueError(f"recent training {name} sidecar does not align")
    _require_hash(
        train_paths["features"],
        train_report["artifacts"]["features"]["sha256"],
        "recent training features",
    )

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    if tuple(str(name) for name in state["feature_names"]) != feature_names:
        raise ValueError("checkpoint feature schema differs from recent cache")
    interactions = read_interactions(args.train_csv).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    val_size = max(1, int(len(interactions) * config.val_ratio))
    train_end = max(2, len(interactions) - val_size)
    context_end = max(1, min(train_end - 1, int(train_end * config.context_ratio)))
    if {
        "context_end": context_end,
        "train_end": train_end,
        "interaction_rows": len(interactions),
    } != {
        key: int(train_report["split"][key])
        for key in ("context_end", "train_end", "interaction_rows")
    }:
        raise ValueError("current temporal split differs from recent cache report")

    rng = np.random.default_rng(config.seed)
    recent_train, relative_train_rows = select_recent_events(
        interactions[context_end:train_end],
        requested_rows=expected_train_rows,
    )
    expected_train_row_indices = relative_train_rows + context_end
    sampled_val, validation_row_indices = sample_chronological_events(
        interactions[train_end:],
        max_events=config.max_val_events,
        rng=rng,
        row_offset=train_end,
    )
    if len(sampled_val) != args.validation_rows:
        raise ValueError(
            f"validation row count differs: actual={len(sampled_val)} "
            f"expected={args.validation_rows}"
        )
    _require_equal(train_row_indices, expected_train_row_indices, "training row indices")
    _require_equal(train_src, recent_train.src, "training src")
    _require_equal(train_dst, recent_train.dst, "training dst")
    _require_equal(train_time, recent_train.time, "training time")
    if not np.all(validation_row_indices[1:] > validation_row_indices[:-1]):
        raise ValueError("validation row indices are not strictly chronological")
    if not np.all(sampled_val.time[1:] >= sampled_val.time[:-1]):
        raise ValueError("sampled validation events are not time sorted")

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
    train_encoder = ranker._timed_fit_encoder(
        "matched_validation_train_replay_encoder",
        interactions[:context_end],
        supervised_config,
        rng,
        verbose=True,
        deterministic_snapshot=train_snapshot,
    )
    if tuple(train_encoder.feature_names) != feature_names:
        raise RuntimeError("replayed training encoder feature schema differs")
    if encoder_cache is not None:
        encoder_cache.release_except()
    del train_snapshot

    replay_rows = min(args.replay_rows, expected_train_rows)
    replay_builder = SupervisedFeatureBuilder(
        encoder=train_encoder,
        dst_pool=np.unique(interactions.dst).astype(np.int64, copy=False),
        config=replace(
            supervised_config,
            num_negatives=args.candidate_count - 1,
            supervised_feature_batch_size=args.batch_rows,
        ),
        label="matched_validation_train_replay",
    )
    replay_queries = replay_builder.batch_for_events(recent_train[:replay_rows], rng)
    replay_features = train_encoder.features_for_query_array(replay_queries)
    replay_report = matched_cache_replay_report(
        expected_candidates=train_candidates[:replay_rows],
        actual_candidates=replay_queries.candidates,
        expected_features=train_features[:replay_rows],
        actual_features=replay_features,
    )
    replay_report["rows"] = replay_rows
    replay_report["expected_candidate_sha256"] = _sha256_array(
        train_candidates[:replay_rows]
    )
    replay_report["actual_candidate_sha256"] = _sha256_array(
        replay_queries.candidates
    )
    if not replay_report["matched"]:
        rejected = {
            "status": "rejected",
            "reason": "recent training cache replay mismatch",
            "checkpoint": str(args.checkpoint.resolve()),
            "train_cache_report": str(args.train_cache_report.resolve()),
            "replay": replay_report,
            "elapsed_seconds": time.time() - started,
        }
        _write_json_atomic(args.report, rejected)
        print(json.dumps(rejected, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return 2

    del replay_features, replay_queries, replay_builder, train_encoder
    gc.collect()
    rng.bit_generator.state = train_report["fusion_rng_state_after_build"]
    val_snapshot = (
        encoder_cache.snapshot_for_prefix(train_end)
        if encoder_cache is not None
        else None
    )
    val_encoder = ranker._timed_fit_encoder(
        "matched_validation_train_end_encoder",
        interactions[:train_end],
        supervised_config,
        rng,
        verbose=True,
        deterministic_snapshot=val_snapshot,
    )
    if tuple(val_encoder.feature_names) != feature_names:
        raise RuntimeError("validation encoder feature schema differs")
    if encoder_cache is not None:
        encoder_cache.release_except()
    del val_snapshot, encoder_cache, ranker, state
    gc.collect()

    val_shape = (len(sampled_val), args.candidate_count, len(feature_names))
    candidate_shape = val_shape[:2]
    required_bytes = (
        int(np.prod(val_shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
        + int(np.prod(candidate_shape, dtype=np.int64)) * np.dtype(np.int32).itemsize
    )
    free_bytes = shutil.disk_usage(args.output_prefix.parent).free
    if free_bytes < required_bytes + 2 * 1024**3:
        raise OSError(
            f"not enough free disk for validation cache: "
            f"free={free_bytes} required={required_bytes}"
        )

    build_config = replace(
        supervised_config,
        num_negatives=args.candidate_count - 1,
        supervised_feature_batch_size=args.batch_rows,
    )
    builder = SupervisedFeatureBuilder(
        encoder=val_encoder,
        dst_pool=np.unique(interactions.dst).astype(np.int64, copy=False),
        config=build_config,
        label="matched_full100_validation_features",
    )
    temp_feature_path = paths["features"].with_name(f".{paths['features'].name}.part")
    temp_candidate_path = paths["candidates"].with_name(
        f".{paths['candidates'].name}.part"
    )
    if temp_feature_path.exists() or temp_candidate_path.exists():
        raise FileExistsError("partial validation cache exists; inspect before retrying")
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

    total_batches = (len(sampled_val) + args.batch_rows - 1) // args.batch_rows
    candidate_checksum = 0
    try:
        for start in range(0, len(sampled_val), args.batch_rows):
            end = min(start + args.batch_rows, len(sampled_val))
            batch_id = start // args.batch_rows + 1
            batch_started = time.time()
            batch_events = sampled_val[start:end]
            queries = builder.batch_for_events(batch_events, rng)
            candidate_report = validate_candidate_matrix(
                batch_events.dst,
                queries.candidates,
                expected_candidate_count=args.candidate_count,
            )
            batch_features = val_encoder.features_for_query_array(queries)
            if batch_features.shape != (
                end - start,
                args.candidate_count,
                len(feature_names),
            ):
                raise RuntimeError("validation encoder returned an unexpected shape")
            if not np.all(np.isfinite(batch_features)):
                raise ValueError(f"non-finite validation feature in batch {batch_id}")
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
                "elapsed_seconds": time.time() - started,
                "last_batch_seconds": time.time() - batch_started,
            }
            _write_json_atomic(paths["progress"], progress)
            print(
                f"[matched-val-cache] batch={batch_id}/{total_batches} "
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
        raise
    else:
        del features, candidates
        os.replace(temp_feature_path, paths["features"])
        os.replace(temp_candidate_path, paths["candidates"])

    _save_array_atomic(paths["src"], sampled_val.src.astype(np.int32, copy=False))
    _save_array_atomic(paths["dst"], sampled_val.dst.astype(np.int32, copy=False))
    _save_array_atomic(paths["time"], sampled_val.time.astype(np.int64, copy=False))
    _save_array_atomic(paths["row_indices"], validation_row_indices)
    report = {
        "status": "complete",
        "protocol": (
            "recent-200k context_end replay followed by the matching train_end "
            "temporal encoder and full-100 validation candidates"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_cache_report": str(args.train_cache_report.resolve()),
        "train_cache_report_sha256": _sha256(args.train_cache_report),
        "train_feature_sha256": train_report["artifacts"]["features"]["sha256"],
        "feature_names": list(feature_names),
        "train_replay": replay_report,
        "validation_shape": list(val_shape),
        "candidate_shape": list(candidate_shape),
        "candidate_checksum": candidate_checksum,
        "validation_candidate_seed_protocol": (
            "continue from train cache fusion_rng_state_after_build; fit the "
            "train_end encoder, then sample validation candidates"
        ),
        "split": {
            "context_end": context_end,
            "train_end": train_end,
            "interaction_rows": len(interactions),
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
    _write_json_atomic(args.report, report)
    _write_json_atomic(
        paths["progress"],
        {
            "status": "complete",
            "completed_rows": len(sampled_val),
            "total_rows": len(sampled_val),
            "elapsed_seconds": report["elapsed_seconds"],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def _train_paths(prefix: Path) -> dict[str, Path]:
    base = str(prefix)
    return {
        "features": Path(f"{base}.train.npy"),
        "candidates": Path(f"{base}.train-candidates.npy"),
        "src": Path(f"{base}.train-src.npy"),
        "dst": Path(f"{base}.train-dst.npy"),
        "time": Path(f"{base}.train-time.npy"),
        "row_indices": Path(f"{base}.train-row-indices.npy"),
    }


def _output_paths(prefix: Path) -> dict[str, Path]:
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


def _require_equal(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError(f"{label} differ from the recent cache contract")


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: actual={actual} expected={expected}")


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


def _sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    array = np.ascontiguousarray(values)
    digest.update(memoryview(array).cast("B"))
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
        raise RuntimeError("CUDA is required for the matched validation encoder")
    jt.flags.use_cuda = 1


if __name__ == "__main__":
    raise SystemExit(main())
