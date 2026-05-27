from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .data import Interaction, TestQuery


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
        self.dst_counts: Counter[int] = Counter()
        self.dst_recent_time: dict[int, int] = {}
        self.dst_popularity: dict[int, float] = {}
        self.dst_popularity_dense: np.ndarray | None = None
        self.dst_recent_time_dense: np.ndarray | None = None
        self.total_edges = 0
        self.min_time = 0
        self.max_time = 0
        self.graph_span = 1
        self.log_total_edges = 1.0

    def fit(self, interactions: list[Interaction]) -> None:
        if not interactions:
            raise ValueError("training interactions are empty")

        self.src_histories = {}
        self.dst_counts = Counter()
        self.dst_recent_time = {}
        self.dst_popularity = {}
        self.dst_popularity_dense = None
        self.dst_recent_time_dense = None
        self.total_edges = len(interactions)
        self.min_time = interactions[0].time
        self.max_time = interactions[-1].time
        self.graph_span = max(self.max_time - self.min_time, 1)
        self.log_total_edges = math.log1p(max(self.total_edges, 1))

        histories: dict[int, SrcHistory] = defaultdict(lambda: SrcHistory(recent_dsts=deque(maxlen=self.recent_window)))
        for item in interactions:
            history = histories[item.src]
            history.total += 1
            history.last_time = item.time
            history.dst_counts[item.dst] += 1
            history.pair_recent_time[item.dst] = item.time
            history.recent_dsts.append(item.dst)

            self.dst_counts[item.dst] += 1
            self.dst_recent_time[item.dst] = item.time

        self.src_histories = dict(histories)
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

    def features_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, STAT_FEATURE_DIM), dtype=np.float32)

        candidate_count = len(queries[0].candidates)
        features = np.empty((len(queries), candidate_count, STAT_FEATURE_DIM), dtype=np.float32)
        for row_idx, query in enumerate(queries):
            if len(query.candidates) != candidate_count:
                raise ValueError("all queries in a batch must have the same candidate count")
            self.fill_features(query, features[row_idx])
        return features

    def fill_features(self, query: TestQuery, output: np.ndarray) -> None:
        history = self.src_histories.get(query.src)
        src_total = history.total if history is not None else 0
        graph_span = self.graph_span

        if history is None:
            src_activity = 0.0
            src_recency = 0.0
            recent_rank: dict[int, int] = {}
        else:
            src_activity = history.activity
            src_recency = math.exp(-max(query.time - history.last_time, 0) / graph_span)
            recent_rank = history.recent_ranks

        candidate_ids = np.fromiter(query.candidates, dtype=np.int64, count=len(query.candidates))
        output[:, 0] = 0.0
        output[:, 1] = 0.0
        output[:, 2] = 0.0
        output[:, 3] = 0.0
        output[:, 4] = 0.0
        output[:, 5] = 0.0
        output[:, 6] = src_activity
        output[:, 7] = src_recency
        self._fill_dst_features(candidate_ids, query.time, output)

        for idx, dst in enumerate(query.candidates):
            pair_count = 0 if history is None else history.dst_counts.get(dst, 0)
            pair_last_time = None if history is None else history.pair_recent_time.get(dst)

            repeat_rate = pair_count / max(src_total, 1)
            pair_strength = math.log1p(pair_count)
            pair_recency = 0.0 if pair_last_time is None else math.exp(-max(query.time - pair_last_time, 0) / graph_span)
            rank = recent_rank.get(dst)
            recent_hit = 0.0 if rank is None else 1.0 / rank

            output[idx, 0] = pair_strength
            output[idx, 1] = repeat_rate
            output[idx, 2] = pair_recency
            output[idx, 5] = recent_hit

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
