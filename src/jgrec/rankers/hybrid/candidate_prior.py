from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray

from .config import CandidatePriorConfig
from .stats import DENSE_NODE_LIMIT, STAT_FEATURE_NAMES

CANDIDATE_PRIOR_FEATURE_NAMES = (
    "candidate_train_seen",
    "candidate_test_freq",
    "candidate_unseen_test_freq",
    "candidate_dst_pop_row_rank",
    "candidate_dst_recency_row_rank",
    "candidate_test_freq_row_rank",
)
CANDIDATE_PRIOR_FEATURE_DIM = len(CANDIDATE_PRIOR_FEATURE_NAMES)
DST_POPULARITY_INDEX = STAT_FEATURE_NAMES.index("dst_popularity")
DST_RECENCY_INDEX = STAT_FEATURE_NAMES.index("dst_recency")


class CandidatePriorTower:
    def __init__(self, config: CandidatePriorConfig | None = None) -> None:
        self.config = config or CandidatePriorConfig()
        self.train_dst: set[int] = set()
        self.test_candidate_counts: Counter[int] = Counter()
        self.test_candidate_total = 0
        self.train_seen_dense: np.ndarray | None = None
        self.test_freq_dense: np.ndarray | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return CANDIDATE_PRIOR_FEATURE_NAMES

    def fit(
        self,
        interactions: InteractionTable,
        test_candidate_counts: Counter[int] | None = None,
    ) -> None:
        self.fit_from_counts(np.unique(interactions.dst).astype(int).tolist(), test_candidate_counts)

    def fit_from_counts(
        self,
        train_dst: set[int] | list[int] | tuple[int, ...],
        test_candidate_counts: Counter[int] | None = None,
    ) -> None:
        self.train_dst = {int(dst) for dst in train_dst}
        self.test_candidate_counts = Counter(test_candidate_counts or {})
        self.test_candidate_total = sum(self.test_candidate_counts.values())
        self._build_dense_features()

    def features_for_queries(self, queries: TestQueryArray | list[TestQuery], stat_features: np.ndarray) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, CANDIDATE_PRIOR_FEATURE_DIM), dtype=np.float32)
        if isinstance(queries, TestQueryArray):
            return self.features_for_query_array(queries, stat_features)
        return self.features_for_query_array(TestQueryArray.from_queries(queries), stat_features)

    def snapshot(self) -> dict[str, Any]:
        return {
            "train_dst": set(self.train_dst),
            "test_candidate_counts": Counter(self.test_candidate_counts),
            "test_candidate_total": self.test_candidate_total,
            "train_seen_dense": self.train_seen_dense,
            "test_freq_dense": self.test_freq_dense,
        }

    def hydrate(self, snapshot: dict[str, Any]) -> None:
        self.train_dst = set(snapshot["train_dst"])
        self.test_candidate_counts = Counter(snapshot["test_candidate_counts"])
        self.test_candidate_total = int(snapshot["test_candidate_total"])
        self.train_seen_dense = snapshot["train_seen_dense"]
        self.test_freq_dense = snapshot["test_freq_dense"]

    def features_for_query_array(self, queries: TestQueryArray, stat_features: np.ndarray) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, CANDIDATE_PRIOR_FEATURE_DIM), dtype=np.float32)
        features = np.zeros((len(queries), queries.candidate_count, CANDIDATE_PRIOR_FEATURE_DIM), dtype=np.float32)
        if not self.config.enabled:
            return features

        denominator = max(float(self.test_candidate_total), 1.0)
        candidate_ids = queries.candidates.astype(np.int64, copy=False)

        seen, test_freq = self._lookup_candidate_features(candidate_ids, denominator)
        features[:, :, 0] = seen.astype(np.float32, copy=False)
        features[:, :, 3] = _row_rank_features(stat_features[:, :, DST_POPULARITY_INDEX])
        features[:, :, 4] = _row_rank_features(stat_features[:, :, DST_RECENCY_INDEX])
        if self.config.include_test_frequency:
            features[:, :, 1] = test_freq
            features[:, :, 2] = np.where(seen, 0.0, test_freq).astype(np.float32, copy=False)
            features[:, :, 5] = _row_rank_features(test_freq)
        return features

    def _build_dense_features(self) -> None:
        self.train_seen_dense = None
        self.test_freq_dense = None
        max_id = -1
        values = list(self.train_dst)
        values.extend(int(value) for value in self.test_candidate_counts)
        if values:
            max_id = max(values)
        if min(values, default=0) < 0 or max_id < 0 or max_id > DENSE_NODE_LIMIT:
            return

        train_seen = np.zeros(max_id + 1, dtype=bool)
        if self.train_dst:
            train_seen[np.fromiter(self.train_dst, dtype=np.int64, count=len(self.train_dst))] = True
        test_freq = np.zeros(max_id + 1, dtype=np.float32)
        denominator = max(float(self.test_candidate_total), 1.0)
        for candidate, count in self.test_candidate_counts.items():
            test_freq[int(candidate)] = float(count) / denominator
        self.train_seen_dense = train_seen
        self.test_freq_dense = test_freq

    def _lookup_candidate_features(self, candidate_ids: np.ndarray, denominator: float) -> tuple[np.ndarray, np.ndarray]:
        if self.train_seen_dense is not None and self.test_freq_dense is not None:
            valid = (candidate_ids >= 0) & (candidate_ids < self.train_seen_dense.shape[0])
            seen = np.zeros(candidate_ids.shape, dtype=bool)
            test_freq = np.zeros(candidate_ids.shape, dtype=np.float32)
            if np.any(valid):
                seen[valid] = self.train_seen_dense[candidate_ids[valid]]
                test_freq[valid] = self.test_freq_dense[candidate_ids[valid]]
            return seen, test_freq

        seen = np.zeros(candidate_ids.shape, dtype=bool)
        test_freq = np.zeros(candidate_ids.shape, dtype=np.float32)
        for row_idx, row in enumerate(candidate_ids):
            seen[row_idx] = np.fromiter((int(candidate) in self.train_dst for candidate in row), dtype=bool)
            test_freq[row_idx] = np.fromiter(
                (self.test_candidate_counts.get(int(candidate), 0) / denominator for candidate in row),
                dtype=np.float32,
            )
        return seen, test_freq


def _row_rank_feature(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float32)
    i = 0
    n = values.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return (1.0 / ranks).astype(np.float32)


def _row_rank_features(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty(values.shape, dtype=np.float32)
    order = np.argsort(-values, axis=1, kind="mergesort")
    n_rows, n_cols = values.shape
    row_indices = np.arange(n_rows)[:, None]
    sorted_values = values[row_indices, order]
    ranks = np.empty(values.shape, dtype=np.float32)
    for row in range(n_rows):
        i = 0
        while i < n_cols:
            j = i
            while j + 1 < n_cols and sorted_values[row, j + 1] == sorted_values[row, i]:
                j += 1
            avg_rank = (i + j + 2) / 2.0
            ranks[row, order[row, i : j + 1]] = avg_rank
            i = j + 1
    return (1.0 / ranks).astype(np.float32, copy=False)
