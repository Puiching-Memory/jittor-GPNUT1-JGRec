from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from jgrec.core.types import TestQuery, TestQueryArray

from .source_profile import SOURCE_PROFILE_FEATURE_DIM, SourceProfileTower
from .structure import STRUCTURE_FEATURE_DIM, StructureFeatureTower

_FORKED_STRUCTURE_TOWER: StructureFeatureTower | None = None
_FORKED_SOURCE_PROFILE_TOWER: SourceProfileTower | None = None


def _features_for_partition(
    payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    row_indices, src, query_time, candidates = payload
    tower = _FORKED_STRUCTURE_TOWER
    if tower is None:
        raise RuntimeError("forked structure worker was not initialized")
    queries = TestQueryArray(
        src=src,
        time=query_time,
        candidates=candidates,
    )
    return row_indices, tower.features_for_query_array(queries), os.getpid()


def _profile_scores_for_partition(
    payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    row_indices, src, query_time, candidates = payload
    tower = _FORKED_SOURCE_PROFILE_TOWER
    if tower is None:
        raise RuntimeError("forked source-profile worker was not initialized")
    queries = TestQueryArray(
        src=src,
        time=query_time,
        candidates=candidates,
    )
    return row_indices, tower.scores_for_query_array(queries), os.getpid()


class _ForkedSourceProfileTower:
    def __init__(
        self,
        owner: ForkedStructureFeatureTower,
        source: SourceProfileTower,
    ) -> None:
        self._owner = owner
        self._source = source

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._source.feature_names

    def scores_for_queries(
        self,
        queries: TestQueryArray | Sequence[TestQuery],
    ) -> np.ndarray:
        query_array = (
            queries
            if isinstance(queries, TestQueryArray)
            else TestQueryArray.from_queries(tuple(queries))
        )
        return self.scores_for_query_array(query_array)

    def scores_for_query_array(
        self,
        queries: TestQueryArray,
    ) -> np.ndarray:
        return self._owner._evaluate_partitions(
            queries,
            worker_function=_profile_scores_for_partition,
            feature_dim=SOURCE_PROFILE_FEATURE_DIM,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


class ForkedStructureFeatureTower:
    """Evaluate a fixed structure tower in source-stable Linux workers."""

    def __init__(
        self,
        source: StructureFeatureTower,
        *,
        worker_count: int,
        source_profile: SourceProfileTower | None = None,
    ) -> None:
        if worker_count < 2:
            raise ValueError("forked structure evaluation requires at least two workers")
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("forked structure evaluation requires Linux fork")
        self._source = source
        self._source_profile_source = source_profile
        self._worker_count = int(worker_count)
        self._closed = False
        self._active_worker_pids: set[int] = set()

        global _FORKED_SOURCE_PROFILE_TOWER, _FORKED_STRUCTURE_TOWER
        _FORKED_STRUCTURE_TOWER = source
        _FORKED_SOURCE_PROFILE_TOWER = source_profile
        context = mp.get_context("fork")
        self._pool = context.Pool(processes=self._worker_count)
        self._worker_pids = {
            int(process.pid)
            for process in self._pool._pool
            if process.pid is not None
        }
        self._parallel_source_profile = (
            None
            if source_profile is None
            else _ForkedSourceProfileTower(self, source_profile)
        )

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def worker_pids(self) -> set[int]:
        return set(self._worker_pids)

    @property
    def active_worker_pids(self) -> set[int]:
        return set(self._active_worker_pids)

    @property
    def source(self) -> StructureFeatureTower:
        return self._source

    @property
    def source_profile(self) -> _ForkedSourceProfileTower:
        if self._parallel_source_profile is None:
            raise RuntimeError("forked source-profile evaluation is not enabled")
        return self._parallel_source_profile

    @property
    def source_profile_source(self) -> SourceProfileTower | None:
        return self._source_profile_source

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._source.feature_names

    def features_for_queries(
        self,
        queries: TestQueryArray | Sequence[TestQuery],
    ) -> np.ndarray:
        query_array = (
            queries
            if isinstance(queries, TestQueryArray)
            else TestQueryArray.from_queries(tuple(queries))
        )
        return self.features_for_query_array(query_array)

    def features_for_query_array(
        self,
        queries: TestQueryArray,
    ) -> np.ndarray:
        return self._evaluate_partitions(
            queries,
            worker_function=_features_for_partition,
            feature_dim=STRUCTURE_FEATURE_DIM,
        )

    def _evaluate_partitions(
        self,
        queries: TestQueryArray,
        *,
        worker_function: Any,
        feature_dim: int,
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("forked structure tower is closed")
        if not queries:
            return np.empty(
                (0, 0, feature_dim),
                dtype=np.float32,
            )

        worker_slots = np.mod(
            queries.src.astype(np.int64, copy=False),
            self._worker_count,
        )
        payloads = []
        for worker_slot in range(self._worker_count):
            row_indices = np.flatnonzero(worker_slots == worker_slot)
            if row_indices.size == 0:
                continue
            payloads.append(
                (
                    row_indices,
                    np.ascontiguousarray(queries.src[row_indices]),
                    np.ascontiguousarray(queries.time[row_indices]),
                    np.ascontiguousarray(queries.candidates[row_indices]),
                )
            )
        results = self._pool.map(
            worker_function,
            payloads,
            chunksize=1,
        )
        output = np.empty(
            (
                len(queries),
                queries.candidate_count,
                feature_dim,
            ),
            dtype=np.float32,
        )
        for row_indices, features, worker_pid in results:
            output[row_indices] = features
            self._active_worker_pids.add(int(worker_pid))
        return output

    def time_decay_features_for_queries(
        self,
        queries: TestQueryArray | Sequence[TestQuery],
    ) -> np.ndarray:
        return self._source.time_decay_features_for_queries(queries)

    def close(self, *, terminate: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if terminate:
            self._pool.terminate()
        else:
            self._pool.close()
        self._pool.join()

        global _FORKED_SOURCE_PROFILE_TOWER, _FORKED_STRUCTURE_TOWER
        _FORKED_STRUCTURE_TOWER = None
        _FORKED_SOURCE_PROFILE_TOWER = None

    def __enter__(self) -> ForkedStructureFeatureTower:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(terminate=exc_type is not None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


def validate_exact_parallel_features(
    sequential: np.ndarray,
    parallel: np.ndarray,
    *,
    sequential_seconds: float,
    parallel_seconds: float,
    minimum_speedup: float,
    worker_pids: set[int],
) -> dict[str, Any]:
    sequential_array = np.asarray(sequential)
    parallel_array = np.asarray(parallel)
    if sequential_array.shape != parallel_array.shape:
        raise ValueError("parallel feature shapes are not exactly equal")
    if sequential_array.dtype != parallel_array.dtype:
        raise ValueError("parallel feature dtypes are not exactly equal")
    if not np.array_equal(sequential_array, parallel_array):
        raise ValueError("parallel features are not exactly equal")
    if not np.isfinite(sequential_seconds) or sequential_seconds <= 0.0:
        raise ValueError("sequential feature time must be positive and finite")
    if not np.isfinite(parallel_seconds) or parallel_seconds <= 0.0:
        raise ValueError("parallel feature time must be positive and finite")
    if not np.isfinite(minimum_speedup) or minimum_speedup <= 1.0:
        raise ValueError("minimum parallel speedup must be greater than one")
    if len(worker_pids) < 2:
        raise ValueError("parallel feature gate requires multiple worker processes")
    speedup = float(sequential_seconds / parallel_seconds)
    if speedup < minimum_speedup:
        raise ValueError(
            "parallel feature speedup "
            f"{speedup:.6f} is below {minimum_speedup:.6f}"
        )
    sequential_contiguous = np.ascontiguousarray(sequential_array)
    parallel_contiguous = np.ascontiguousarray(parallel_array)
    sequential_sha256 = hashlib.sha256(
        sequential_contiguous.view(np.uint8)
    ).hexdigest()
    parallel_sha256 = hashlib.sha256(
        parallel_contiguous.view(np.uint8)
    ).hexdigest()
    if sequential_sha256 != parallel_sha256:
        raise ValueError("parallel feature byte hashes are not exactly equal")
    return {
        "status": "passed",
        "exact_equal": True,
        "shape": list(sequential_array.shape),
        "dtype": str(sequential_array.dtype),
        "sequential_seconds": float(sequential_seconds),
        "parallel_seconds": float(parallel_seconds),
        "speedup": speedup,
        "minimum_speedup": float(minimum_speedup),
        "worker_count_observed": len(worker_pids),
        "worker_pids": sorted(int(pid) for pid in worker_pids),
        "sequential_sha256": sequential_sha256,
        "parallel_sha256": parallel_sha256,
    }


def select_parallel_worker_trial(
    *,
    baseline_report: Mapping[str, Any],
    trial_report: Mapping[str, Any],
    baseline_worker_count: int,
    trial_worker_count: int,
    minimum_incremental_speedup: float,
    available_memory_bytes: int,
    minimum_memory_reserve_bytes: int,
) -> dict[str, Any]:
    """Select a larger exact worker arm only when speed and memory pass."""
    if baseline_worker_count < 2:
        raise ValueError("baseline worker count must be at least two")
    if trial_worker_count <= baseline_worker_count:
        raise ValueError("trial worker count must exceed baseline")
    if (
        not np.isfinite(minimum_incremental_speedup)
        or minimum_incremental_speedup <= 1.0
    ):
        raise ValueError(
            "minimum incremental speedup must be greater than one"
        )
    if minimum_memory_reserve_bytes <= 0:
        raise ValueError("minimum memory reserve must be positive")
    if available_memory_bytes < 0:
        raise ValueError("available memory must not be negative")
    for name, report, expected_workers in (
        ("baseline", baseline_report, baseline_worker_count),
        ("trial", trial_report, trial_worker_count),
    ):
        if report.get("status") != "passed":
            raise ValueError(f"{name} parallel parity report did not pass")
        observed_workers = int(report.get("worker_count_observed", 0))
        if observed_workers < expected_workers:
            raise ValueError(
                f"{name} parallel arm did not use every worker"
            )
    hashes = {
        str(baseline_report.get("sequential_sha256", "")),
        str(baseline_report.get("parallel_sha256", "")),
        str(trial_report.get("sequential_sha256", "")),
        str(trial_report.get("parallel_sha256", "")),
    }
    if "" in hashes or len(hashes) != 1:
        raise ValueError("parallel trial byte hashes differ")
    baseline_seconds = float(baseline_report["parallel_seconds"])
    trial_seconds = float(trial_report["parallel_seconds"])
    if (
        not np.isfinite(baseline_seconds)
        or baseline_seconds <= 0.0
        or not np.isfinite(trial_seconds)
        or trial_seconds <= 0.0
    ):
        raise ValueError("parallel trial times must be positive and finite")
    incremental_speedup = baseline_seconds / trial_seconds
    fallback_reason: str | None = None
    if incremental_speedup < minimum_incremental_speedup:
        fallback_reason = "incremental_speedup"
    elif available_memory_bytes < minimum_memory_reserve_bytes:
        fallback_reason = "memory_reserve"
    selected_worker_count = (
        baseline_worker_count
        if fallback_reason is not None
        else trial_worker_count
    )
    return {
        "status": "selected",
        "baseline_worker_count": baseline_worker_count,
        "trial_worker_count": trial_worker_count,
        "selected_worker_count": selected_worker_count,
        "incremental_speedup": float(incremental_speedup),
        "minimum_incremental_speedup": float(
            minimum_incremental_speedup
        ),
        "available_memory_bytes": int(available_memory_bytes),
        "minimum_memory_reserve_bytes": int(
            minimum_memory_reserve_bytes
        ),
        "fallback_reason": fallback_reason,
        "exact_sha256": hashes.pop(),
    }
