from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray

from .config import TARGET_WINDOW_FEATURE_NAMES, TargetWindowConfig

TARGET_WINDOW_FEATURE_DIM = len(TARGET_WINDOW_FEATURE_NAMES)
FEATURES_PER_WINDOW = 4
DENSE_TIME_NODE_LIMIT = 2_000_000


class TargetWindowTower:
    def __init__(self, config: TargetWindowConfig | None = None) -> None:
        self.config = config or TargetWindowConfig()
        self.window_fractions = _normalize_fractions(self.config.window_fractions)
        self.dst_event_times: dict[int, np.ndarray] = {}
        self.event_times = np.empty(0, dtype=np.int64)
        self.max_time = 0
        self.min_time = 0
        self.graph_span = 1
        self.dst_event_times_dense: list[np.ndarray] | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return TARGET_WINDOW_FEATURE_NAMES

    def fit(self, interactions: InteractionTable) -> None:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")
        interactions = interactions.sort_by_time()
        self.event_times = _compact_int_array(interactions.time.astype(int).tolist())
        self.min_time = int(interactions.time[0])
        self.max_time = int(interactions.time[-1])
        self.graph_span = max(self.max_time - self.min_time, 1)
        dst_times: dict[int, list[int]] = defaultdict(list)
        for dst, time in zip(interactions.dst, interactions.time):
            dst_times[int(dst)].append(int(time))
        self.dst_event_times = {
            int(dst): _compact_int_array(times)
            for dst, times in dst_times.items()
        }
        self._build_dense_dst_times()

    def fit_from_grouped(
        self,
        *,
        dst_event_times: dict[int, list[int]],
        event_times: list[int],
        min_time: int,
        max_time: int,
    ) -> None:
        if not event_times:
            raise ValueError("training interactions are empty")
        self.event_times = _compact_int_array(event_times)
        self.min_time = int(min_time)
        self.max_time = int(max_time)
        self.graph_span = max(self.max_time - self.min_time, 1)
        self.dst_event_times = {
            int(dst): _compact_int_array(times)
            for dst, times in dst_event_times.items()
        }
        self._build_dense_dst_times()

    def snapshot(self) -> dict[str, Any]:
        return {
            "window_fractions": tuple(self.window_fractions),
            "dst_event_times": dict(self.dst_event_times),
            "event_times": self.event_times,
            "max_time": self.max_time,
            "min_time": self.min_time,
            "graph_span": self.graph_span,
            "dst_event_times_dense": self.dst_event_times_dense,
        }

    def hydrate(self, snapshot: dict[str, Any]) -> None:
        self.window_fractions = _normalize_fractions(snapshot.get("window_fractions", self.config.window_fractions))
        self.dst_event_times = dict(snapshot["dst_event_times"])
        self.event_times = snapshot["event_times"]
        self.max_time = int(snapshot["max_time"])
        self.min_time = int(snapshot["min_time"])
        self.graph_span = int(snapshot["graph_span"])
        self.dst_event_times_dense = snapshot.get("dst_event_times_dense")

    def compact_for_future_queries(self) -> None:
        return

    def features_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, TARGET_WINDOW_FEATURE_DIM), dtype=np.float32)
        if isinstance(queries, TestQueryArray):
            return self.features_for_query_array(queries)
        return self.features_for_query_array(TestQueryArray.from_queries(queries))

    def features_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, TARGET_WINDOW_FEATURE_DIM), dtype=np.float32)
        features = np.zeros((len(queries), queries.candidate_count, TARGET_WINDOW_FEATURE_DIM), dtype=np.float32)
        if not self.config.enabled:
            return features

        for row_idx in range(len(queries)):
            query_time = int(queries.time[row_idx])
            candidate_ids = queries.candidates[row_idx].astype(np.int64, copy=False)
            for window_idx, fraction in enumerate(self.window_fractions):
                offset = window_idx * FEATURES_PER_WINDOW
                width = max(int(math.ceil(float(fraction) * self.graph_span)), 1)
                cutoff_time = min(query_time, self.max_time + 1)
                start_time = self.min_time if fraction >= 1.0 else cutoff_time - width
                window_total = _count_in_window(self.event_times, start_time, cutoff_time)
                denominator = math.log1p(max(window_total, 1))
                counts = np.zeros(candidate_ids.shape[0], dtype=np.float32)
                recency = np.zeros(candidate_ids.shape[0], dtype=np.float32)
                for col_idx, candidate in enumerate(candidate_ids):
                    times = self._dst_times(int(candidate))
                    if times.size == 0:
                        continue
                    left = int(np.searchsorted(times, start_time, side="left"))
                    right = int(np.searchsorted(times, cutoff_time, side="left"))
                    count = right - left
                    if count <= 0:
                        continue
                    counts[col_idx] = float(count)
                    last_time = int(times[right - 1])
                    recency[col_idx] = math.exp(-max(cutoff_time - last_time, 0) / max(float(width), 1.0))
                pop = np.log1p(counts) / denominator
                features[row_idx, :, offset] = pop.astype(np.float32, copy=False)
                features[row_idx, :, offset + 1] = (counts / max(float(window_total), 1.0)).astype(np.float32, copy=False)
                features[row_idx, :, offset + 2] = recency
                features[row_idx, :, offset + 3] = _row_rank_feature(pop)
        return features

    def _dst_times(self, dst: int) -> np.ndarray:
        if self.dst_event_times_dense is not None and 0 <= dst < len(self.dst_event_times_dense):
            return self.dst_event_times_dense[dst]
        return self.dst_event_times.get(int(dst), np.empty(0, dtype=np.int32))

    def _build_dense_dst_times(self) -> None:
        self.dst_event_times_dense = None
        if not self.dst_event_times:
            return
        max_id = max(self.dst_event_times)
        min_id = min(self.dst_event_times)
        if min_id < 0 or max_id > DENSE_TIME_NODE_LIMIT:
            return
        empty = np.empty(0, dtype=np.int32)
        dense = [empty] * (max_id + 1)
        for dst, times in self.dst_event_times.items():
            dense[int(dst)] = times
        self.dst_event_times_dense = dense


def _normalize_fractions(values: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    fractions = tuple(float(value) for value in values)
    if len(fractions) != len(TARGET_WINDOW_FEATURE_NAMES) // FEATURES_PER_WINDOW:
        raise ValueError("target window fractions must contain exactly four values")
    if any(value <= 0.0 for value in fractions):
        raise ValueError("target window fractions must be positive")
    return fractions


def _count_in_window(times: np.ndarray, start_time: int, cutoff_time: int) -> int:
    if times.size == 0:
        return 0
    left = int(np.searchsorted(times, start_time, side="left"))
    right = int(np.searchsorted(times, cutoff_time, side="left"))
    return max(right - left, 0)


def _row_rank_feature(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float32)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float32)
    return (1.0 / ranks).astype(np.float32)


def _compact_int_array(values: list[int]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.int32)
    min_value = min(values)
    max_value = max(values)
    int32 = np.iinfo(np.int32)
    dtype = np.int32 if int32.min <= min_value <= max_value <= int32.max else np.int64
    return np.asarray(values, dtype=dtype)
