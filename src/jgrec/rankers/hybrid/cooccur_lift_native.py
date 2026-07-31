from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.core.types import InteractionTable


def compile_cooccur_lift_materializer(
    output_dir: Path,
    *,
    source_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = Path(__file__).resolve().parents[4]
    source = source_path or root / "scripts" / "cooccur_lift_materializer.cpp"
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required for compact cooccurrence materialization")
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = output_dir / (
        "cooccur-lift-materializer.exe"
        if os.name == "nt"
        else "cooccur-lift-materializer"
    )
    command = [
        compiler,
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-fopenmp",
        str(source),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True)
    return executable, {
        "backend": "native_dense_triangular_uint8_with_exact_overflow",
        "compiler": compiler,
        "compile_command": command,
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "executable": str(executable.resolve()),
        "executable_sha256": _sha256(executable),
    }


def materialize_compact_cooccur_lift(
    *,
    interactions: InteractionTable,
    sources: np.ndarray,
    candidates: np.ndarray,
    destinations: np.ndarray,
    event_time: np.ndarray,
    availability_time: np.ndarray | None = None,
    short_window: float,
    lift_path: Path,
    positive_popularity_path: Path,
    progress_path: Path,
    work_dir: Path,
) -> dict[str, Any]:
    if len(interactions) == 0:
        raise ValueError("training interactions are empty")
    query_sources = _as_int32("sources", sources)
    query_candidates = _as_int32("candidates", candidates)
    query_destinations = _as_int32("destinations", destinations)
    query_times = _as_int32("event_time", event_time)
    query_availability_times = _as_int32(
        "availability_time",
        query_times if availability_time is None else availability_time,
    )
    train_sources = _as_int32("train src", interactions.src)
    train_destinations = _as_int32("train dst", interactions.dst)
    train_times = _as_int32("train time", interactions.time)
    if query_candidates.ndim != 2:
        raise ValueError("candidate matrix must be two-dimensional")
    query_rows, candidate_count = query_candidates.shape
    if query_rows == 0 or candidate_count == 0:
        raise ValueError("candidate matrix must be nonempty")
    for label, values in (
        ("sources", query_sources),
        ("destinations", query_destinations),
        ("event_time", query_times),
        ("availability_time", query_availability_times),
    ):
        if values.shape != (query_rows,):
            raise ValueError(f"{label} must match candidate rows")
    if not np.all(train_times[1:] >= train_times[:-1]):
        raise ValueError("training interactions must be chronological")
    if not np.all(query_times[1:] >= query_times[:-1]):
        raise ValueError("query rows must be chronological")
    if not np.all(
        query_availability_times[1:] >= query_availability_times[:-1]
    ):
        raise ValueError("query availability times must be chronological")
    if np.any(query_availability_times > query_times):
        raise ValueError("availability_time must not exceed event_time")
    if not np.isfinite(short_window) or short_window <= 0:
        raise ValueError("short_window must be positive and finite")

    maximum_source = int(max(train_sources.max(), query_sources.max()))
    maximum_destination = int(
        max(
            train_destinations.max(),
            query_destinations.max(),
            query_candidates.max(),
        )
    )
    minimum_identifier = int(
        min(
            train_sources.min(),
            train_destinations.min(),
            query_sources.min(),
            query_destinations.min(),
            query_candidates.min(),
        )
    )
    if minimum_identifier < 0 or maximum_source < 0 or maximum_destination < 1:
        raise ValueError("source and destination identifiers must be nonnegative")

    native_dir = work_dir / "native-materializer"
    raw_dir = native_dir / "raw"
    if native_dir.exists():
        raise FileExistsError(f"refusing to overwrite native work dir: {native_dir}")
    raw_dir.mkdir(parents=True)
    executable, contract = compile_cooccur_lift_materializer(native_dir)
    inputs = {
        "train-src.raw": train_sources,
        "train-dst.raw": train_destinations,
        "train-time.raw": train_times,
        "query-src.raw": query_sources,
        "query-candidates.raw": query_candidates,
        "query-dst.raw": query_destinations,
        "query-time.raw": query_times,
        "query-availability-time.raw": query_availability_times,
    }
    for filename, values in inputs.items():
        np.asarray(values, dtype=np.int32).tofile(raw_dir / filename)

    lift_raw = raw_dir / "lift.raw"
    popularity_raw = raw_dir / "positive-popularity.raw"
    command = [
        str(executable),
        str(raw_dir / "train-src.raw"),
        str(raw_dir / "train-dst.raw"),
        str(raw_dir / "train-time.raw"),
        str(len(train_sources)),
        str(raw_dir / "query-src.raw"),
        str(raw_dir / "query-candidates.raw"),
        str(raw_dir / "query-dst.raw"),
        str(raw_dir / "query-time.raw"),
        str(raw_dir / "query-availability-time.raw"),
        str(query_rows),
        str(candidate_count),
        format(float(short_window), ".17g"),
        str(maximum_source),
        str(maximum_destination),
        str(lift_raw),
        str(popularity_raw),
        str(progress_path),
    ]
    subprocess.run(command, check=True)

    _raw_to_npy(
        lift_raw,
        lift_path,
        dtype=np.float32,
        shape=(query_rows, candidate_count, 2),
    )
    _raw_to_npy(
        popularity_raw,
        positive_popularity_path,
        dtype=np.int32,
        shape=(query_rows,),
    )
    for path in raw_dir.iterdir():
        path.unlink()
    raw_dir.rmdir()
    contract.update(
        {
            "history_limit": 64,
            "cooccur_history_limit": 256,
            "strict_upper_cutoff": 'searchsorted(side="left") equivalent',
            "strict_short_lower_cutoff": 'searchsorted(side="right") equivalent',
            "train_rows_read": len(train_sources),
            "query_rows_materialized": query_rows,
            "candidate_count": candidate_count,
            "maximum_source_id": maximum_source,
            "maximum_destination_id": maximum_destination,
            "short_window": float(short_window),
            "separate_availability_time": availability_time is not None,
            "collapsed_short_rows": int(
                np.count_nonzero(
                    query_times.astype(np.float64)
                    - query_availability_times.astype(np.float64)
                    >= float(short_window)
                )
            ),
            "full_train_consumed": True,
            "future_events_used_in_query_features": False,
            "cooccur_time_decay_score_reused": False,
        }
    )
    return contract


def _as_int32(label: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{label} must contain integers")
    if array.size:
        info = np.iinfo(np.int32)
        minimum = int(array.min())
        maximum = int(array.max())
        if minimum < info.min or maximum > info.max:
            raise ValueError(f"{label} exceeds native int32 range")
    return np.ascontiguousarray(array, dtype=np.int32)


def _raw_to_npy(
    raw_path: Path,
    output_path: Path,
    *,
    dtype: np.dtype[Any] | type[Any],
    shape: tuple[int, ...],
) -> None:
    source = np.memmap(raw_path, mode="r", dtype=dtype, shape=shape)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )
    row_count = shape[0]
    for start in range(0, row_count, 2_000):
        stop = min(start + 2_000, row_count)
        output[start:stop] = source[start:stop]
    output.flush()
    del output, source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
