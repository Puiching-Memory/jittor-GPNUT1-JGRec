from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.core.io import read_interactions
from jgrec.rankers.hybrid.source_sequence_cache import (
    SourceSequenceRows,
    build_causal_source_sequences,
    expanding_timestamp_abcd_folds,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build strictly causal source histories for Dataset2 CST A/B/C/D."
        )
    )
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    manifest_path = args.output_dir / "fold-manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        _verify_existing(manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"sequence cache directory is not empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    base = str(args.train_cache_prefix)
    paths = {
        "features": Path(f"{base}.train.npy"),
        "candidates": Path(f"{base}.train-candidates.npy"),
        "dst": Path(f"{base}.train-dst.npy"),
        "row_indices": Path(f"{base}.train-row-indices.npy"),
        "src": Path(f"{base}.train-src.npy"),
        "time": Path(f"{base}.train-time.npy"),
    }
    features = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
    candidates = np.load(
        paths["candidates"],
        mmap_mode="r",
        allow_pickle=False,
    )
    dst = np.load(paths["dst"], mmap_mode="r", allow_pickle=False)
    row_indices = np.load(
        paths["row_indices"],
        mmap_mode="r",
        allow_pickle=False,
    )
    sources = np.load(paths["src"], mmap_mode="r", allow_pickle=False)
    query_times = np.load(
        paths["time"],
        mmap_mode="r",
        allow_pickle=False,
    )
    _validate_sidecars(
        features,
        candidates,
        dst,
        row_indices,
        sources,
        query_times,
    )
    interactions = read_interactions(args.train_csv).sort_by_time()
    if int(row_indices[-1]) >= len(interactions):
        raise ValueError("training cache row index exceeds train.csv")
    np.testing.assert_array_equal(
        interactions.src[row_indices],
        np.asarray(sources),
    )
    np.testing.assert_array_equal(
        interactions.dst[row_indices],
        np.asarray(dst),
    )
    np.testing.assert_array_equal(
        interactions.time[row_indices],
        np.asarray(query_times, dtype=np.int32),
    )
    folds = expanding_timestamp_abcd_folds(query_times)

    print(
        f"[sequence-cache] build causal rows={len(sources)} "
        f"max_length={args.max_length}",
        flush=True,
    )
    causal = build_causal_source_sequences(
        interactions,
        query_src=sources,
        query_time=query_times,
        max_length=args.max_length,
    )
    artifacts: dict[str, dict[str, Any]] = {}
    artifacts.update(
        _save_sequence_rows(
            args.output_dir,
            "train-causal",
            causal,
        )
    )
    del causal

    for fold in folds:
        score_start, score_stop = fold.score_rows
        print(
            f"[sequence-cache] fold={fold.index} "
            f"score=[{score_start},{score_stop}) "
            f"origin={fold.score_time_min}",
            flush=True,
        )
        score_rows = build_causal_source_sequences(
            interactions,
            query_src=sources[score_start:score_stop],
            query_time=query_times[score_start:score_stop],
            max_length=args.max_length,
            history_time_limit=fold.score_time_min,
        )
        artifacts.update(
            _save_sequence_rows(
                args.output_dir,
                f"fold-{fold.index}-score-frozen",
                score_rows,
            )
        )
        del score_rows

    manifest = {
        "status": "complete",
        "protocol": "dataset2_source_conditioned_abcd_sequence_cache_v1",
        "strict_history_rule": "event_time < query_time",
        "score_history_rule": (
            "event_time < min(query_time, fold_score_time_min)"
        ),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "max_length": int(args.max_length),
        "num_items": int(
            max(
                int(np.max(interactions.dst)),
                int(np.max(candidates)),
            )
        ),
        "train_rows": int(features.shape[0]),
        "candidate_count": int(features.shape[1]),
        "feature_count": int(features.shape[2]),
        "positive_candidate_matches_dst": bool(
            np.array_equal(np.asarray(candidates[:, 0]), np.asarray(dst))
        ),
        "folds": [asdict(fold) for fold in folds],
        "source_cache_report": str(args.train_cache_report.resolve()),
        "source_cache_report_sha256": _sha256(args.train_cache_report),
        "source_artifacts": {
            key: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for key, path in paths.items()
        },
        "artifacts": artifacts,
        "sequence_statistics": _sequence_statistics(
            args.output_dir,
            folds,
        ),
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


def _validate_sidecars(
    features: Any,
    candidates: Any,
    dst: Any,
    row_indices: Any,
    sources: Any,
    query_times: Any,
) -> None:
    rows = int(features.shape[0])
    if (
        len(features.shape) != 3
        or features.shape[1:] != (100, 63)
        or candidates.shape != features.shape[:2]
        or dst.shape != (rows,)
        or row_indices.shape != (rows,)
        or sources.shape != (rows,)
        or query_times.shape != (rows,)
    ):
        raise ValueError("Dataset2 full-100 sidecars do not align")
    if not np.all(np.diff(query_times.astype(np.int64)) >= 0):
        raise ValueError("Dataset2 training cache is not chronological")
    if not np.array_equal(np.asarray(candidates[:, 0]), np.asarray(dst)):
        raise ValueError("positive candidate is not column zero")


def _save_sequence_rows(
    output_dir: Path,
    prefix: str,
    rows: SourceSequenceRows,
) -> dict[str, dict[str, Any]]:
    arrays = {
        f"{prefix}-items": np.asarray(rows.items, dtype=np.int32),
        f"{prefix}-time-buckets": np.asarray(
            rows.time_buckets,
            dtype=np.int32,
        ),
        f"{prefix}-lengths": np.asarray(rows.lengths, dtype=np.int32),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, values in arrays.items():
        path = output_dir / f"{name}.npy"
        _save_array_atomic(path, values)
        result[name] = {
            "path": str(path.resolve()),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": _sha256(path),
        }
    return result


def _sequence_statistics(
    output_dir: Path,
    folds: Any,
) -> dict[str, Any]:
    lengths = np.load(
        output_dir / "train-causal-lengths.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    stats: dict[str, Any] = {
        "train_causal": _length_stats(lengths),
    }
    for fold in folds:
        values = np.load(
            output_dir
            / f"fold-{fold.index}-score-frozen-lengths.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        stats[f"fold_{fold.index}_score_frozen"] = _length_stats(values)
    return stats


def _length_stats(lengths: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "rows": int(values.size),
        "empty_rows": int(np.sum(values == 0)),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": int(values.max()),
    }


def _verify_existing(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "complete":
        raise ValueError("existing sequence cache is incomplete")
    for artifact in manifest["artifacts"].values():
        path = Path(artifact["path"])
        if not path.exists() or _sha256(path) != artifact["sha256"]:
            raise ValueError(f"sequence cache artifact differs: {path}")


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
