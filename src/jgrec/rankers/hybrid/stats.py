from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from jgrec.core.types import Interaction, TestQuery

STAT_FEATURE_NAMES = (
    "pair_strength",
    "repeat_rate",
    "pair_recency",
    "dst_popularity",
    "dst_recency",
    "recent_hit",
    "src_activity",
    "src_recency",
)
STAT_FEATURE_DIM = len(STAT_FEATURE_NAMES)
DENSE_NODE_LIMIT = 10_000_000


@dataclass
class SrcHistory:
    total: int = 0
    last_time: int = 0
    activity: float = 0.0
    dst_counts: Counter[int] = field(default_factory=Counter)
    pair_recent_time: dict[int, int] = field(default_factory=dict)
    recent_dsts: deque[int] = field(default_factory=lambda: deque(maxlen=32))
    recent_ranks: dict[int, int] = field(default_factory=dict)


class TemporalStats:
    def __init__(self, recent_window: int = 32) -> None:
        self.recent_window = recent_window
        self.src_histories: dict[int, SrcHistory] = {}
        self.src_event_times: dict[int, np.ndarray] = {}
        self.src_event_dsts: dict[int, np.ndarray] = {}
        self.dst_event_times: dict[int, np.ndarray] = {}
        self.pair_event_times: dict[tuple[int, int], np.ndarray] = {}
        self.dst_counts: Counter[int] = Counter()
        self.dst_recent_time: dict[int, int] = {}
        self.dst_popularity: dict[int, float] = {}
        self.dst_popularity_dense: np.ndarray | None = None
        self.dst_recent_time_dense: np.ndarray | None = None
        self.event_times = np.empty(0, dtype=np.int64)
        self.total_edges = 0
        self.min_time = 0
        self.max_time = 0
        self.graph_span = 1
        self.log_total_edges = 1.0

    def fit(self, interactions: list[Interaction]) -> None:
        if not interactions:
            raise ValueError("training interactions are empty")

        interactions = _ensure_time_order(interactions)
        self.src_histories = {}
        self.src_event_times = {}
        self.src_event_dsts = {}
        self.dst_event_times = {}
        self.pair_event_times = {}
        self.dst_counts = Counter()
        self.dst_recent_time = {}
        self.dst_popularity = {}
        self.dst_popularity_dense = None
        self.dst_recent_time_dense = None
        self.event_times = _compact_int_array([item.time for item in interactions])
        self.total_edges = len(interactions)
        self.min_time = interactions[0].time
        self.max_time = interactions[-1].time
        self.graph_span = max(self.max_time - self.min_time, 1)
        self.log_total_edges = math.log1p(max(self.total_edges, 1))

        histories: dict[int, SrcHistory] = defaultdict(lambda: SrcHistory(recent_dsts=deque(maxlen=self.recent_window)))
        src_times: dict[int, list[int]] = defaultdict(list)
        src_dsts: dict[int, list[int]] = defaultdict(list)
        dst_times: dict[int, list[int]] = defaultdict(list)
        pair_times: dict[tuple[int, int], list[int]] = defaultdict(list)
        for item in interactions:
            history = histories[item.src]
            history.total += 1
            history.last_time = item.time
            history.dst_counts[item.dst] += 1
            history.pair_recent_time[item.dst] = item.time
            history.recent_dsts.append(item.dst)

            self.dst_counts[item.dst] += 1
            self.dst_recent_time[item.dst] = item.time
            src_times[item.src].append(item.time)
            src_dsts[item.src].append(item.dst)
            dst_times[item.dst].append(item.time)
            pair_times[(item.src, item.dst)].append(item.time)

        self.src_histories = dict(histories)
        self.src_event_times = {
            src: _compact_int_array(times)
            for src, times in src_times.items()
        }
        self.src_event_dsts = {
            src: _compact_int_array(dsts)
            for src, dsts in src_dsts.items()
        }
        self.dst_event_times = {
            dst: _compact_int_array(times)
            for dst, times in dst_times.items()
        }
        self.pair_event_times = {
            pair: _compact_int_array(times)
            for pair, times in pair_times.items()
        }
        for history in self.src_histories.values():
            history.activity = math.log1p(history.total) / self.log_total_edges
            ranks: dict[int, int] = {}
            for rank, dst in enumerate(reversed(history.recent_dsts), start=1):
                ranks.setdefault(dst, rank)
            history.recent_ranks = ranks

        self.dst_popularity = {
            dst: math.log1p(count) / self.log_total_edges
            for dst, count in self.dst_counts.items()
        }
        self._build_dense_dst_features()

    def compact_for_future_queries(self) -> None:
        self.src_event_times = {}
        self.src_event_dsts = {}
        self.dst_event_times = {}
        self.pair_event_times = {}
        self.event_times = np.empty(0, dtype=np.int64)

    def features_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, STAT_FEATURE_DIM), dtype=np.float32)

        candidate_count = len(queries[0].candidates)
        features = np.empty((len(queries), candidate_count, STAT_FEATURE_DIM), dtype=np.float32)
        candidate_ids = np.empty((len(queries), candidate_count), dtype=np.int64)
        query_times = np.empty(len(queries), dtype=np.int64)
        for row_idx, query in enumerate(queries):
            if len(query.candidates) != candidate_count:
                raise ValueError("all queries in a batch must have the same candidate count")
            candidate_ids[row_idx] = query.candidates
            query_times[row_idx] = query.time
        if np.all(query_times > self.max_time):
            for row_idx, query in enumerate(queries):
                self.fill_features(query, features[row_idx], candidate_ids=candidate_ids[row_idx], fill_dst=False)
            self._fill_dst_features_batch(candidate_ids, query_times, features)
            return features

        for row_idx, query in enumerate(queries):
            self.fill_features(query, features[row_idx], candidate_ids=candidate_ids[row_idx], fill_dst=True)
        return features

    def fill_features(
        self,
        query: TestQuery,
        output: np.ndarray,
        candidate_ids: np.ndarray | None = None,
        fill_dst: bool = True,
    ) -> None:
        if query.time <= self.max_time:
            self._fill_features_cutoff(query, output, candidate_ids=candidate_ids, fill_dst=fill_dst)
            return

        history = self.src_histories.get(query.src)
        src_total = history.total if history is not None else 0
        graph_span = self.graph_span

        if history is None:
            src_activity = 0.0
            src_recency = 0.0
        else:
            src_activity = history.activity
            src_recency = math.exp(-max(query.time - history.last_time, 0) / graph_span)

        if candidate_ids is None:
            candidate_ids = np.fromiter(query.candidates, dtype=np.int64, count=len(query.candidates))
        output[:, 0] = 0.0
        output[:, 1] = 0.0
        output[:, 2] = 0.0
        output[:, 3] = 0.0
        output[:, 4] = 0.0
        output[:, 5] = 0.0
        output[:, 6] = src_activity
        output[:, 7] = src_recency
        if history is not None:
            self._fill_pair_features(candidate_ids, query.time, history, src_total, output)
        if fill_dst:
            self._fill_dst_features(candidate_ids, query.time, output)

    def _fill_features_cutoff(
        self,
        query: TestQuery,
        output: np.ndarray,
        candidate_ids: np.ndarray | None = None,
        fill_dst: bool = True,
    ) -> None:
        if candidate_ids is None:
            candidate_ids = np.fromiter(query.candidates, dtype=np.int64, count=len(query.candidates))

        output[:, 0] = 0.0
        output[:, 1] = 0.0
        output[:, 2] = 0.0
        output[:, 3] = 0.0
        output[:, 4] = 0.0
        output[:, 5] = 0.0
        output[:, 6] = 0.0
        output[:, 7] = 0.0

        query_time = int(query.time)
        graph_span = self._graph_span_before(query_time)
        log_total_edges = self._log_total_edges_before(query_time)
        src_total = 0
        recent_ranks: dict[int, int] = {}

        src_times = self.src_event_times.get(query.src)
        src_dsts = self.src_event_dsts.get(query.src)
        if src_times is not None and src_dsts is not None:
            src_cutoff = int(np.searchsorted(src_times, query_time, side="left"))
            src_total = src_cutoff
            if src_cutoff > 0:
                src_last_time = int(src_times[src_cutoff - 1])
                output[:, 6] = math.log1p(src_total) / log_total_edges
                output[:, 7] = math.exp(-max(query_time - src_last_time, 0) / graph_span)
                recent_start = max(0, src_cutoff - self.recent_window)
                recent_dsts = src_dsts[recent_start:src_cutoff]
                for rank, dst in enumerate(reversed(recent_dsts), start=1):
                    recent_ranks.setdefault(int(dst), rank)

        self._fill_pair_features_cutoff(
            query.src,
            candidate_ids,
            query_time,
            graph_span,
            src_total,
            recent_ranks,
            output,
        )
        if fill_dst:
            self._fill_dst_features_cutoff(candidate_ids, query_time, graph_span, log_total_edges, output)

    def _build_dense_dst_features(self) -> None:
        if not self.dst_counts:
            return
        max_dst = max(self.dst_counts)
        min_dst = min(self.dst_counts)
        if min_dst < 0 or max_dst > DENSE_NODE_LIMIT:
            return

        popularity = np.zeros(max_dst + 1, dtype=np.float32)
        recent_time = np.full(max_dst + 1, -1, dtype=np.int64)
        for dst, value in self.dst_popularity.items():
            popularity[dst] = value
        for dst, value in self.dst_recent_time.items():
            recent_time[dst] = value
        self.dst_popularity_dense = popularity
        self.dst_recent_time_dense = recent_time

    def _fill_dst_features(self, candidate_ids: np.ndarray, query_time: int, output: np.ndarray) -> None:
        if self.dst_popularity_dense is not None and self.dst_recent_time_dense is not None:
            valid = (candidate_ids >= 0) & (candidate_ids < self.dst_popularity_dense.shape[0])
            if np.any(valid):
                valid_candidates = candidate_ids[valid]
                output[valid, 3] = self.dst_popularity_dense[valid_candidates]
                last_times = self.dst_recent_time_dense[valid_candidates]
                seen = last_times >= 0
                if np.any(seen):
                    recency = np.exp(-np.maximum(query_time - last_times[seen], 0) / self.graph_span)
                    valid_rows = np.flatnonzero(valid)
                    output[valid_rows[seen], 4] = recency.astype(np.float32)
            return

        for idx, dst in enumerate(candidate_ids):
            dst_int = int(dst)
            output[idx, 3] = self.dst_popularity.get(dst_int, 0.0)
            dst_last_time = self.dst_recent_time.get(dst_int)
            if dst_last_time is not None:
                output[idx, 4] = math.exp(-max(query_time - dst_last_time, 0) / self.graph_span)

    def _fill_pair_features(
        self,
        candidate_ids: np.ndarray,
        query_time: int,
        history: SrcHistory,
        src_total: int,
        output: np.ndarray,
    ) -> None:
        for idx, dst in enumerate(candidate_ids):
            dst_int = int(dst)
            pair_count = history.dst_counts.get(dst_int, 0)
            if pair_count:
                output[idx, 0] = math.log1p(pair_count)
                output[idx, 1] = pair_count / max(src_total, 1)
                pair_last_time = history.pair_recent_time[dst_int]
                output[idx, 2] = math.exp(-max(query_time - pair_last_time, 0) / self.graph_span)

            rank = history.recent_ranks.get(dst_int)
            if rank is not None:
                output[idx, 5] = 1.0 / rank

    def _fill_pair_features_cutoff(
        self,
        src: int,
        candidate_ids: np.ndarray,
        query_time: int,
        graph_span: int,
        src_total: int,
        recent_ranks: dict[int, int],
        output: np.ndarray,
    ) -> None:
        for idx, dst in enumerate(candidate_ids):
            dst_int = int(dst)
            pair_times = self.pair_event_times.get((src, dst_int))
            if pair_times is not None:
                pair_cutoff = int(np.searchsorted(pair_times, query_time, side="left"))
                if pair_cutoff > 0:
                    output[idx, 0] = math.log1p(pair_cutoff)
                    output[idx, 1] = pair_cutoff / max(src_total, 1)
                    pair_last_time = int(pair_times[pair_cutoff - 1])
                    output[idx, 2] = math.exp(-max(query_time - pair_last_time, 0) / graph_span)

            rank = recent_ranks.get(dst_int)
            if rank is not None:
                output[idx, 5] = 1.0 / rank

    def _fill_dst_features_cutoff(
        self,
        candidate_ids: np.ndarray,
        query_time: int,
        graph_span: int,
        log_total_edges: float,
        output: np.ndarray,
    ) -> None:
        for idx, dst in enumerate(candidate_ids):
            dst_times = self.dst_event_times.get(int(dst))
            if dst_times is None:
                continue
            dst_cutoff = int(np.searchsorted(dst_times, query_time, side="left"))
            if dst_cutoff <= 0:
                continue
            output[idx, 3] = math.log1p(dst_cutoff) / log_total_edges
            dst_last_time = int(dst_times[dst_cutoff - 1])
            output[idx, 4] = math.exp(-max(query_time - dst_last_time, 0) / graph_span)

    def _fill_dst_features_batch(
        self,
        candidate_ids: np.ndarray,
        query_times: np.ndarray,
        output: np.ndarray,
    ) -> None:
        if self.dst_popularity_dense is not None and self.dst_recent_time_dense is not None:
            valid = (candidate_ids >= 0) & (candidate_ids < self.dst_popularity_dense.shape[0])
            if np.any(valid):
                valid_rows, valid_cols = np.nonzero(valid)
                valid_candidates = candidate_ids[valid_rows, valid_cols]
                output[valid_rows, valid_cols, 3] = self.dst_popularity_dense[valid_candidates]
                last_times = self.dst_recent_time_dense[valid_candidates]
                seen = last_times >= 0
                if np.any(seen):
                    recency = np.exp(-np.maximum(query_times[valid_rows[seen]] - last_times[seen], 0) / self.graph_span)
                    output[valid_rows[seen], valid_cols[seen], 4] = recency.astype(np.float32)
            return

        for row_idx, row_candidates in enumerate(candidate_ids):
            self._fill_dst_features(row_candidates, int(query_times[row_idx]), output[row_idx])

    def _log_total_edges_before(self, query_time: int) -> float:
        edge_count = int(np.searchsorted(self.event_times, query_time, side="left"))
        return math.log1p(max(edge_count, 1))

    def _graph_span_before(self, query_time: int) -> int:
        edge_count = int(np.searchsorted(self.event_times, query_time, side="left"))
        if edge_count <= 0:
            return 1
        return max(int(self.event_times[edge_count - 1] - self.event_times[0]), 1)


def _compact_int_array(values: list[int]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.int32)
    min_value = min(values)
    max_value = max(values)
    int32 = np.iinfo(np.int32)
    dtype = np.int32 if int32.min <= min_value <= max_value <= int32.max else np.int64
    return np.asarray(values, dtype=dtype)


def _ensure_time_order(interactions: list[Interaction]) -> list[Interaction]:
    if all(left.time <= right.time for left, right in zip(interactions, interactions[1:])):
        return interactions
    return sorted(interactions, key=lambda item: item.time)
