from __future__ import annotations

import argparse
import copy
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
from jgrec.rankers.hybrid.full100_training import validate_candidate_matrix
from jgrec.rankers.hybrid.ranker import (
    SupervisedFeatureBuilder,
    TemporalHybridRanker,
    _sample_events,
)

CONTROL_CANDIDATE_COUNT = 32
FULL_CANDIDATE_COUNT = 100


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build matched Dataset2 32- and 100-candidate training caches."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--control-prefix", required=True, type=Path)
    parser.add_argument("--full100-prefix", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch-rows", type=int, default=4096)
    args = parser.parse_args()

    if args.batch_rows <= 0:
        raise ValueError("batch rows must be positive")
    if args.control_prefix.resolve() == args.full100_prefix.resolve():
        raise ValueError("control and full-100 cache prefixes must differ")
    replay = _require_replay_evidence(args)
    _configure_cuda()

    control_paths = _output_paths(args.control_prefix)
    full_paths = _output_paths(args.full100_prefix)
    protected = (*control_paths.values(), *full_paths.values(), args.report)
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite matched-cache artifacts: {existing}")
    args.control_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.full100_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    source_manifest_path = args.source_cache_prefix.with_suffix(".json")
    source_val_path = args.source_cache_prefix.with_suffix(".val.npy")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_val = np.load(source_val_path, mmap_mode="r", allow_pickle=False)
    if list(source_val.shape) != source_manifest["val"]["shape"]:
        raise ValueError("source validation shape does not match its manifest")
    if source_val.shape != (20_000, 100, len(feature_names)):
        raise ValueError("source validation tensor is not the frozen 20k x 100 tensor")

    interactions = read_interactions(args.train_csv).sort_by_time()
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    val_size = max(1, int(len(interactions) * config.val_ratio))
    train_end = max(2, len(interactions) - val_size)
    context_end = max(1, min(train_end - 1, int(train_end * config.context_ratio)))
    encoder_rng = np.random.default_rng(config.seed)
    sampled_train = _sample_events(
        interactions[context_end:train_end],
        config.max_train_events,
        encoder_rng,
    )
    sampled_val = _sample_events(
        interactions[train_end:],
        config.max_val_events,
        encoder_rng,
    )
    control_shape = (len(sampled_train), CONTROL_CANDIDATE_COUNT, len(feature_names))
    full_shape = (len(sampled_train), FULL_CANDIDATE_COUNT, len(feature_names))
    if control_shape != (50_000, 32, 63) or full_shape != (50_000, 100, 63):
        raise ValueError(
            f"frozen matched-cache shapes differ: control={control_shape} full={full_shape}"
        )
    if len(sampled_val) != source_val.shape[0]:
        raise ValueError("sampled validation rows do not align with the frozen validation tensor")

    required_bytes = sum(
        int(np.prod(shape, dtype=np.int64))
        * (
            np.dtype(np.float32).itemsize
            if len(shape) == 3
            else np.dtype(np.int32).itemsize
        )
        for shape in (control_shape, control_shape[:2], full_shape, full_shape[:2])
    )
    free_bytes = shutil.disk_usage(args.full100_prefix.parent).free
    if free_bytes < required_bytes + 2 * 1024**3:
        raise OSError(
            f"not enough free disk for atomic matched build: free={free_bytes} "
            f"required={required_bytes}"
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
        "matched_train_context_encoder",
        interactions[:context_end],
        supervised_config,
        encoder_rng,
        verbose=True,
        deterministic_snapshot=train_snapshot,
    )
    if tuple(encoder.feature_names) != feature_names:
        raise RuntimeError("matched encoder feature schema differs from the checkpoint")
    if encoder_cache is not None:
        encoder_cache.release_except()
    del train_snapshot, encoder_cache, ranker, state
    gc.collect()

    sampling_state = copy.deepcopy(encoder_rng.bit_generator.state)
    control_rng = _clone_rng(sampling_state)
    full_rng = _clone_rng(sampling_state)
    dst_pool = np.unique(interactions.dst).astype(np.int64, copy=False)
    control_builder = SupervisedFeatureBuilder(
        encoder=encoder,
        dst_pool=dst_pool,
        config=replace(
            supervised_config,
            num_negatives=CONTROL_CANDIDATE_COUNT - 1,
            supervised_feature_batch_size=args.batch_rows,
        ),
        label="matched32_train_features",
    )
    full_builder = SupervisedFeatureBuilder(
        encoder=encoder,
        dst_pool=dst_pool,
        config=replace(
            supervised_config,
            num_negatives=FULL_CANDIDATE_COUNT - 1,
            supervised_feature_batch_size=args.batch_rows,
        ),
        label="full100_train_features",
    )
    control_arrays, control_temp = _open_memmaps(control_paths, control_shape)
    full_arrays, full_temp = _open_memmaps(full_paths, full_shape)
    control_features, control_candidates = control_arrays
    full_features, full_candidates = full_arrays

    total_batches = (len(sampled_train) + args.batch_rows - 1) // args.batch_rows
    control_checksum = 0
    full_checksum = 0
    try:
        for start in range(0, len(sampled_train), args.batch_rows):
            end = min(start + args.batch_rows, len(sampled_train))
            batch_id = start // args.batch_rows + 1
            batch_events = sampled_train[start:end]
            batch_started = time.time()
            control_batch, control_contract = _encode_batch(
                builder=control_builder,
                encoder=encoder,
                events=batch_events,
                rng=control_rng,
                candidate_count=CONTROL_CANDIDATE_COUNT,
                feature_count=len(feature_names),
            )
            control_features[start:end] = control_batch[0]
            control_candidates[start:end] = control_batch[1]
            control_checksum += _candidate_checksum(control_batch[1])
            del control_batch
            encoder.clear_batch_caches()

            full_batch, full_contract = _encode_batch(
                builder=full_builder,
                encoder=encoder,
                events=batch_events,
                rng=full_rng,
                candidate_count=FULL_CANDIDATE_COUNT,
                feature_count=len(feature_names),
            )
            full_features[start:end] = full_batch[0]
            full_candidates[start:end] = full_batch[1]
            full_checksum += _candidate_checksum(full_batch[1])
            del full_batch
            encoder.clear_batch_caches()

            if batch_id % 2 == 0 or end == len(sampled_train):
                _flush_arrays(control_arrays)
                _flush_arrays(full_arrays)
            gc.collect()
            progress = {
                "status": "building",
                "completed_rows": end,
                "total_rows": len(sampled_train),
                "batch": batch_id,
                "total_batches": total_batches,
                "last_control_contract": control_contract,
                "last_full100_contract": full_contract,
                "control_candidate_checksum": control_checksum,
                "full100_candidate_checksum": full_checksum,
                "elapsed_seconds": time.time() - started,
                "last_batch_seconds": time.time() - batch_started,
            }
            _write_json_atomic(control_paths["progress"], progress)
            _write_json_atomic(full_paths["progress"], progress)
            print(
                f"[matched-cache] batch={batch_id}/{total_batches} "
                f"rows={end}/{len(sampled_train)} "
                f"batch_seconds={progress['last_batch_seconds']:.1f} "
                f"elapsed={progress['elapsed_seconds']:.1f}",
                flush=True,
            )
        _flush_arrays(control_arrays)
        _flush_arrays(full_arrays)
    except BaseException:
        _flush_arrays(control_arrays)
        _flush_arrays(full_arrays)
        raise
    else:
        del control_features, control_candidates, full_features, full_candidates
        del control_arrays, full_arrays
        _publish_memmaps(control_temp, control_paths)
        _publish_memmaps(full_temp, full_paths)

    for paths in (control_paths, full_paths):
        _save_array_atomic(paths["src"], sampled_train.src.astype(np.int32, copy=False))
        _save_array_atomic(paths["dst"], sampled_train.dst.astype(np.int32, copy=False))
        _save_array_atomic(paths["time"], sampled_train.time.astype(np.int32, copy=False))

    report = {
        "status": "complete",
        "protocol": "one newly fitted encoder; cloned post-fit NumPy RNG; matched 32 vs 100 groups",
        "replay_status": replay["status"],
        "replay_feature_report": replay["feature_report"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "source_cache_key": source_manifest["key"],
        "source_cache_manifest_sha256": _sha256(source_manifest_path),
        "source_validation_path": str(source_val_path.resolve()),
        "source_validation_shape": list(source_val.shape),
        "source_validation_sha256": _sha256(source_val_path),
        "feature_names": list(feature_names),
        "control32": {
            "train_shape": list(control_shape),
            "candidate_shape": list(control_shape[:2]),
            "candidate_checksum": control_checksum,
            "artifacts": _artifact_report(control_paths),
            "rng_state_after_build": _jsonable(control_rng.bit_generator.state),
        },
        "full100": {
            "train_shape": list(full_shape),
            "candidate_shape": list(full_shape[:2]),
            "candidate_checksum": full_checksum,
            "artifacts": _artifact_report(full_paths),
            "rng_state_after_build": _jsonable(full_rng.bit_generator.state),
        },
        "sampled_validation_rows": len(sampled_val),
        "split": {
            "context_end": context_end,
            "train_end": train_end,
            "interaction_rows": len(interactions),
        },
        "post_encoder_sampling_rng_state": _jsonable(sampling_state),
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(args.report, report)
    complete_progress = {
        "status": "complete",
        "completed_rows": len(sampled_train),
        "total_rows": len(sampled_train),
        "elapsed_seconds": report["elapsed_seconds"],
    }
    _write_json_atomic(control_paths["progress"], complete_progress)
    _write_json_atomic(full_paths["progress"], complete_progress)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def _require_replay_evidence(args: argparse.Namespace) -> dict[str, Any]:
    report = json.loads(args.replay_report.read_text(encoding="utf-8"))
    if report.get("status") not in {"passed", "rejected"}:
        raise RuntimeError("bounded replay did not finish")
    contract = report.get("candidate_report", {})
    if (
        contract.get("rows") != report.get("replay_rows")
        or contract.get("candidate_count") != CONTROL_CANDIDATE_COUNT
        or contract.get("positive_mismatches") != 0
        or contract.get("duplicate_rows") != 0
    ):
        raise RuntimeError("bounded replay candidate contract is invalid")
    if Path(report["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("replay checkpoint differs from the matched experiment checkpoint")
    if Path(report["cache_prefix"]).resolve() != args.source_cache_prefix.resolve():
        raise ValueError("replay source cache differs from the requested source cache")
    return report


def _encode_batch(
    *,
    builder: SupervisedFeatureBuilder,
    encoder,
    events,
    rng: np.random.Generator,
    candidate_count: int,
    feature_count: int,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, int]]:
    queries = builder.batch_for_events(events, rng)
    contract = validate_candidate_matrix(
        events.dst,
        queries.candidates,
        expected_candidate_count=candidate_count,
    )
    features = encoder.features_for_query_array(queries)
    expected_shape = (len(events), candidate_count, feature_count)
    if features.shape != expected_shape:
        raise RuntimeError(f"encoder feature shape differs: {features.shape} != {expected_shape}")
    if not np.all(np.isfinite(features)):
        raise ValueError("encoder produced non-finite features")
    return (features, queries.candidates), contract


def _open_memmaps(
    paths: dict[str, Path],
    feature_shape: tuple[int, int, int],
) -> tuple[tuple[np.memmap, np.memmap], dict[str, Path]]:
    temp = {
        name: paths[name].with_name(f".{paths[name].name}.part")
        for name in ("features", "candidates")
    }
    if any(path.exists() for path in temp.values()):
        raise FileExistsError("a partial matched cache exists; inspect it before retrying")
    arrays = (
        np.lib.format.open_memmap(
            temp["features"],
            mode="w+",
            dtype=np.float32,
            shape=feature_shape,
        ),
        np.lib.format.open_memmap(
            temp["candidates"],
            mode="w+",
            dtype=np.int32,
            shape=feature_shape[:2],
        ),
    )
    return arrays, temp


def _flush_arrays(arrays: tuple[np.memmap, np.memmap]) -> None:
    for array in arrays:
        array.flush()


def _publish_memmaps(temp: dict[str, Path], paths: dict[str, Path]) -> None:
    for name in ("features", "candidates"):
        os.replace(temp[name], paths[name])


def _clone_rng(state: dict[str, Any]) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state)
    return rng


def _output_paths(prefix: Path) -> dict[str, Path]:
    base = str(prefix)
    return {
        "features": Path(f"{base}.train.npy"),
        "candidates": Path(f"{base}.train-candidates.npy"),
        "src": Path(f"{base}.train-src.npy"),
        "dst": Path(f"{base}.train-dst.npy"),
        "time": Path(f"{base}.train-time.npy"),
        "progress": Path(f"{base}.progress.json"),
    }


def _artifact_report(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
        if name != "progress"
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
        raise RuntimeError("CUDA is required for the matched champion encoder")
    jt.flags.use_cuda = 1


if __name__ == "__main__":
    raise SystemExit(main())
