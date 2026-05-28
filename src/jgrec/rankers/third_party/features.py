from __future__ import annotations

import math

import numpy as np

from jgrec.core.types import TestQuery

from .indexes import TemporalIndexes, decay, normalize_gap


THIRD_PARTY_FEATURE_NAMES = (
    "pair_log_count",
    "pair_repeat_rate",
    "pair_decay_short",
    "pair_decay_medium",
    "pair_decay_long",
    "pair_last_gap",
    "pair_seen",
    "recent_hit",
    "src_log_degree",
    "src_unique_dst_rate",
    "src_recency",
    "dst_log_count",
    "dst_unique_src",
    "dst_decay_short",
    "dst_decay_medium",
    "dst_decay_long",
    "dst_last_gap",
    "dst_pop_rank",
    "reverse_log_count",
    "reverse_recency",
    "common_neighbors",
    "jaccard",
    "cooccur_score",
    "transition_score",
)


class ThirdPartyFeatureExtractor:
    def __init__(self, indexes: TemporalIndexes) -> None:
        self.indexes = indexes
        self.feature_names = THIRD_PARTY_FEATURE_NAMES

    def features_for_queries(self, queries: list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, len(self.feature_names)), dtype=np.float32)
        candidate_count = len(queries[0].candidates)
        output = np.zeros((len(queries), candidate_count, len(self.feature_names)), dtype=np.float32)
        taus = self.indexes.taus()
        for row_idx, query in enumerate(queries):
            if len(query.candidates) != candidate_count:
                raise ValueError("all queries in a batch must have the same candidate count")
            self._fill_query(query, output[row_idx], taus)
        return output

    def _fill_query(self, query: TestQuery, output: np.ndarray, taus: tuple[float, float, float]) -> None:
        source = self.indexes.sources.get(query.src)
        src_total = source.total if source is not None else 0
        if source is not None:
            output[:, 8] = math.log1p(source.total) / self.indexes.log_total_edges
            output[:, 9] = len(source.dst_counts) / max(source.total, 1)
            output[:, 10] = decay(query.time - source.last_time, taus[2])

        recent_ranks: dict[int, int] = {}
        if source is not None:
            for rank, dst in enumerate(reversed(source.recent_dsts), start=1):
                recent_ranks.setdefault(dst, rank)

        for idx, dst in enumerate(query.candidates):
            dst_int = int(dst)
            destination = self.indexes.destinations.get(dst_int)
            if source is not None:
                pair_count = source.dst_counts.get(dst_int, 0)
                if pair_count:
                    last_time = source.dst_last_time[dst_int]
                    output[idx, 0] = math.log1p(pair_count)
                    output[idx, 1] = pair_count / max(src_total, 1)
                    output[idx, 2] = decay(query.time - last_time, taus[0])
                    output[idx, 3] = decay(query.time - last_time, taus[1])
                    output[idx, 4] = decay(query.time - last_time, taus[2])
                    output[idx, 5] = normalize_gap(query.time - last_time, self.indexes.graph_span)
                    output[idx, 6] = 1.0
                rank = recent_ranks.get(dst_int)
                if rank is not None:
                    output[idx, 7] = 1.0 / rank

            if destination is not None:
                output[idx, 11] = math.log1p(destination.total) / self.indexes.log_total_edges
                output[idx, 12] = math.log1p(len(destination.unique_sources)) / self.indexes.log_total_src
                output[idx, 13] = decay(query.time - destination.last_time, taus[0])
                output[idx, 14] = decay(query.time - destination.last_time, taus[1])
                output[idx, 15] = decay(query.time - destination.last_time, taus[2])
                output[idx, 16] = normalize_gap(query.time - destination.last_time, self.indexes.graph_span)
                rank = self.indexes.dst_rank.get(dst_int)
                if rank is not None:
                    output[idx, 17] = 1.0 / math.log2(rank + 1.0)

            reverse_key = (dst_int, query.src)
            reverse_count = self.indexes.reverse_pair_counts.get(reverse_key, 0)
            if reverse_count:
                output[idx, 18] = math.log1p(reverse_count)
                output[idx, 19] = decay(query.time - self.indexes.reverse_pair_last_time[reverse_key], taus[1])

            common_neighbors, jaccard = self.indexes.common_neighbor_features(query.src, dst_int)
            output[idx, 20] = common_neighbors
            output[idx, 21] = jaccard
            cooccur_score, transition_score = self.indexes.relation_scores_from_recent(source, dst_int)
            output[idx, 22] = cooccur_score
            output[idx, 23] = transition_score

