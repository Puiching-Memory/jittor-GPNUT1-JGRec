from __future__ import annotations

import math
from collections import Counter, OrderedDict
from typing import Any

import numpy as np

from jgrec.core.types import Interaction, InteractionTable, TestQuery, TestQueryArray
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex

from .config import StructureTowerConfig

STRUCTURE_FEATURE_NAMES = (
    "pair_decay_short",
    "pair_decay_medium",
    "pair_decay_long",
    "dst_unique_src",
    "dst_pop_rank",
    "reverse_log_count",
    "reverse_recency",
    "common_neighbors",
    "jaccard",
    "cooccur_score",
    "transition_score",
)
STRUCTURE_FEATURE_DIM = len(STRUCTURE_FEATURE_NAMES)
FULL_HISTORY_CACHE_LIMIT = 2048
FULL_COMMON_NEIGHBOR_CACHE_LIMIT = 2048
FULL_COOCCUR_CACHE_LIMIT = 4096
FULL_COOCCUR_PREAGGREGATE_NEIGHBOR_THRESHOLD = 256
DEFAULT_BRIDGE_OVERLAP_THRESHOLD = 0.50
DEFAULT_BRIDGE_MIN_ROLE_DEGREE = 2


class StructureFeatureTower:
    def __init__(self, config: StructureTowerConfig | None = None) -> None:
        self.config = config or StructureTowerConfig()
        self.index = TemporalInteractionIndex()
        self.min_time = 0
        self.max_time = 0
        self.graph_span = 1
        self.decay_windows: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._full_src_neighbor_cache: OrderedDict[int, set[int]] = OrderedDict()
        self._full_dst_source_cache: OrderedDict[int, set[int]] = OrderedDict()
        self._full_src_common_neighbor_cache: OrderedDict[int, dict[int, int]] = OrderedDict()
        self._full_src_cooccur_cache: OrderedDict[int, dict[int, int]] = OrderedDict()
        self._bridge_overlap_ratio = 0.0
        self._bridge_min_role_degree = DEFAULT_BRIDGE_MIN_ROLE_DEGREE
        self._allow_global_id_bridge = False

    @property
    def feature_names(self) -> tuple[str, ...]:
        return STRUCTURE_FEATURE_NAMES

    def fit(
        self,
        interactions: InteractionTable | list[Interaction],
        rng: np.random.Generator,
        verbose: bool = True,
    ) -> None:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")
        if not isinstance(interactions, InteractionTable):
            interactions = InteractionTable.from_events(interactions)
        interactions = interactions.sort_by_time()
        self.index.fit(
            interactions,
            build_transitions=self.config.transition_enabled,
            build_cooccurs=self.config.cooccur_enabled,
            cooccur_history_limit=self.config.cooccur_history_limit,
            future_only_transition_cooccur=self.config.future_only_transition_cooccur,
        )
        self.min_time = int(interactions.time[0])
        self.max_time = int(interactions.time[-1])
        self.graph_span = max(self.max_time - self.min_time, 1)
        self.decay_windows = (
            max(self.graph_span * 0.05, 1.0),
            max(self.graph_span * 0.20, 1.0),
            max(self.graph_span * 0.50, 1.0),
        )
        self.fit_bridge_policy(interactions)
        self._full_src_neighbor_cache.clear()
        self._full_dst_source_cache.clear()
        self._full_src_common_neighbor_cache.clear()
        self._full_src_cooccur_cache.clear()

    def compact_for_future_queries(self) -> None:
        self.index.compact_for_future_queries()
        self._full_src_neighbor_cache.clear()
        self._full_dst_source_cache.clear()
        self._full_src_common_neighbor_cache.clear()
        self._full_src_cooccur_cache.clear()

    def compact_transition_cooccur_for_future_queries(self) -> None:
        self.index.compact_transition_cooccur_for_future_queries()
        self._full_src_neighbor_cache.clear()
        self._full_dst_source_cache.clear()
        self._full_src_common_neighbor_cache.clear()
        self._full_src_cooccur_cache.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "index": self.index.shallow_copy(),
            "min_time": self.min_time,
            "max_time": self.max_time,
            "graph_span": self.graph_span,
            "decay_windows": self.decay_windows,
            "bridge_overlap_ratio": self._bridge_overlap_ratio,
            "bridge_min_role_degree": self._bridge_min_role_degree,
            "allow_global_id_bridge": self._allow_global_id_bridge,
        }

    def hydrate(self, snapshot: dict[str, Any]) -> None:
        self.index = snapshot["index"].shallow_copy()
        self.min_time = int(snapshot["min_time"])
        self.max_time = int(snapshot["max_time"])
        self.graph_span = int(snapshot["graph_span"])
        self.decay_windows = tuple(float(value) for value in snapshot["decay_windows"])
        self._bridge_overlap_ratio = float(snapshot.get("bridge_overlap_ratio", 0.0))
        self._bridge_min_role_degree = int(snapshot.get("bridge_min_role_degree", DEFAULT_BRIDGE_MIN_ROLE_DEGREE))
        self._allow_global_id_bridge = bool(snapshot.get("allow_global_id_bridge", False))
        self._full_src_neighbor_cache.clear()
        self._full_dst_source_cache.clear()
        self._full_src_common_neighbor_cache.clear()
        self._full_src_cooccur_cache.clear()

    def features_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, STRUCTURE_FEATURE_DIM), dtype=np.float32)
        return self.features_for_query_array(TestQueryArray.from_queries(queries))

    def features_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, STRUCTURE_FEATURE_DIM), dtype=np.float32)

        features = np.zeros((len(queries), queries.candidate_count, STRUCTURE_FEATURE_DIM), dtype=np.float32)
        full_history_src_counts = Counter(
            int(queries.src[row_idx])
            for row_idx in range(len(queries))
            if int(queries.time[row_idx]) > self.max_time
        )
        for row_idx, query in enumerate(queries):
            force_full_preaggregate = full_history_src_counts[int(query.src)] > 1
            self._fill_query_features(query, features[row_idx], force_full_preaggregate=force_full_preaggregate)
        return features

    def _fill_query_features(
        self,
        query: TestQuery,
        output: np.ndarray,
        *,
        force_full_preaggregate: bool = False,
    ) -> None:
        if query.time > self.max_time:
            self._fill_full_history_query_features(
                query,
                output,
                force_cooccur_preaggregate=force_full_preaggregate,
            )
            return

        src_view = self.index.source_view(query.src, query.time)
        src_neighbors = {int(dst) for dst in src_view.visible_dsts}
        src_neighbor_count = len(src_neighbors)
        last_visible_dst = int(src_view.visible_dsts[-1]) if src_view.cutoff > 0 else None

        for idx, dst in enumerate(query.candidates):
            dst_int = int(dst)
            pair_times = self.index.pair_times_before(query.src, dst_int, query.time)
            if pair_times.size:
                deltas = np.maximum(query.time - pair_times, 0)
                for feature_idx, window in enumerate(self.decay_windows):
                    output[idx, feature_idx] = float(np.exp(-deltas / window).sum())

            dst_view = self.index.destination_view(dst_int, query.time)
            dst_sources = {int(src) for src in dst_view.visible_srcs}
            dst_source_count = len(dst_sources)
            if dst_source_count:
                output[idx, 3] = math.log1p(dst_source_count)
                output[idx, 4] = 1.0 / math.log1p(dst_source_count + 1)

            reverse_times = self.index.reverse_pair_times_before(query.src, dst_int, query.time)
            if reverse_times.size:
                output[idx, 5] = math.log1p(reverse_times.size)
                reverse_last_time = int(reverse_times[-1])
                output[idx, 6] = math.exp(-max(query.time - reverse_last_time, 0) / self.graph_span)

            if src_neighbor_count and dst_source_count:
                bridge_nodes = self._valid_bridge_nodes(src_neighbors & dst_sources)
                common = len(bridge_nodes)
                union = src_neighbor_count + dst_source_count - common
                output[idx, 7] = math.log1p(common)
                output[idx, 8] = common / max(union, 1)

            if self.config.cooccur_enabled:
                cooccur = self.index.cooccur_count(query.src, dst_int, query.time)
                if cooccur:
                    output[idx, 9] = math.log1p(cooccur)

            if self.config.transition_enabled and last_visible_dst is not None:
                transition = self.index.transition_count(last_visible_dst, dst_int, query.time)
                if transition:
                    output[idx, 10] = math.log1p(transition)

    def _fill_full_history_query_features(
        self,
        query: TestQuery,
        output: np.ndarray,
        *,
        force_cooccur_preaggregate: bool = False,
    ) -> None:
        src_neighbors = self._full_src_neighbors(query.src)
        src_neighbor_count = len(src_neighbors)
        src_dsts = self.index.src_dsts.get(query.src)
        last_visible_dst = int(src_dsts[-1]) if src_dsts is not None and src_dsts.size else None
        candidate_ids = tuple(int(dst) for dst in query.candidates)
        common_counts = self._full_src_common_neighbors(query.src, src_neighbors) if src_neighbor_count else {}
        cooccur_counts = (
            self._full_cooccur_counts(
                query.src,
                src_neighbors,
                candidate_ids,
                force_preaggregate=force_cooccur_preaggregate,
            )
            if self.config.cooccur_enabled
            else np.zeros(len(candidate_ids), dtype=np.int32)
        )
        transition_counts = (
            self.index.future_transition_count_maps.get(last_visible_dst, {})
            if self.index.future_only and self.config.transition_enabled and last_visible_dst is not None
            else None
        )

        for idx, dst_int in enumerate(candidate_ids):
            pair_times = self.index.pair_times.get((query.src, dst_int))
            if pair_times is not None and pair_times.size:
                deltas = np.maximum(query.time - pair_times, 0)
                for feature_idx, window in enumerate(self.decay_windows):
                    output[idx, feature_idx] = float(np.exp(-deltas / window).sum())

            dst_source_count = self.index.dst_unique_src_counts.get(dst_int, 0)
            if dst_source_count:
                output[idx, 3] = math.log1p(dst_source_count)
                output[idx, 4] = 1.0 / math.log1p(dst_source_count + 1)

            reverse_times = self.index.pair_times.get((dst_int, query.src))
            if reverse_times is not None and reverse_times.size:
                output[idx, 5] = math.log1p(reverse_times.size)
                reverse_last_time = int(reverse_times[-1])
                output[idx, 6] = math.exp(-max(query.time - reverse_last_time, 0) / self.graph_span)

            if src_neighbor_count and dst_source_count:
                common = common_counts.get(dst_int, 0)
                union = src_neighbor_count + dst_source_count - common
                output[idx, 7] = math.log1p(common)
                output[idx, 8] = common / max(union, 1)

            if self.config.cooccur_enabled:
                cooccur = int(cooccur_counts[idx])
                if cooccur:
                    output[idx, 9] = math.log1p(cooccur)

            if self.config.transition_enabled and last_visible_dst is not None:
                transition = (
                    transition_counts.get(dst_int, 0)
                    if transition_counts is not None
                    else self.index.transition_count(last_visible_dst, dst_int, query.time)
                )
                if transition:
                    output[idx, 10] = math.log1p(transition)

    def _full_src_neighbors(self, src: int) -> set[int]:
        cached = self._full_src_neighbor_cache.get(src)
        if cached is not None:
            self._full_src_neighbor_cache.move_to_end(src)
            return cached

        dsts = self.index.src_dsts.get(src)
        neighbors = {int(dst) for dst in dsts} if dsts is not None else set()
        self._cache_put(self._full_src_neighbor_cache, src, neighbors, FULL_HISTORY_CACHE_LIMIT)
        return neighbors

    def _full_dst_sources(self, dst: int) -> set[int]:
        cached = self._full_dst_source_cache.get(dst)
        if cached is not None:
            self._full_dst_source_cache.move_to_end(dst)
            return cached

        srcs = self.index.dst_srcs.get(dst)
        sources = {int(src) for src in srcs} if srcs is not None else set()
        self._cache_put(self._full_dst_source_cache, dst, sources, FULL_HISTORY_CACHE_LIMIT)
        return sources

    def _full_src_common_neighbors(self, src: int, src_neighbors: set[int]) -> dict[int, int]:
        cached = self._full_src_common_neighbor_cache.get(src)
        if cached is not None:
            self._full_src_common_neighbor_cache.move_to_end(src)
            return cached

        counts: dict[int, int] = {}
        for neighbor_src in src_neighbors:
            if not self._is_valid_bridge_node(neighbor_src):
                continue
            neighbor_dsts = self.index.src_dsts.get(neighbor_src)
            if neighbor_dsts is None or not neighbor_dsts.size:
                continue
            for candidate in self._full_src_neighbors(neighbor_src):
                candidate_int = int(candidate)
                counts[candidate_int] = counts.get(candidate_int, 0) + 1
        self._cache_put(self._full_src_common_neighbor_cache, src, counts, FULL_COMMON_NEIGHBOR_CACHE_LIMIT)
        return counts

    def fit_bridge_policy(self, interactions: InteractionTable) -> None:
        self.fit_bridge_policy_from_values(
            src_values={int(value) for value in np.unique(interactions.src)},
            dst_values={int(value) for value in np.unique(interactions.dst)},
        )

    def fit_bridge_policy_from_values(self, src_values: set[int], dst_values: set[int]) -> None:
        if not src_values or not dst_values:
            self._bridge_overlap_ratio = 0.0
            self._allow_global_id_bridge = False
        else:
            overlap = len(src_values & dst_values)
            self._bridge_overlap_ratio = overlap / max(min(len(src_values), len(dst_values)), 1)
            threshold = float(getattr(self.config, "bridge_overlap_threshold", DEFAULT_BRIDGE_OVERLAP_THRESHOLD))
            self._allow_global_id_bridge = self._bridge_overlap_ratio >= threshold
        self._bridge_min_role_degree = max(
            int(getattr(self.config, "bridge_min_role_degree", DEFAULT_BRIDGE_MIN_ROLE_DEGREE)),
            1,
        )

    def _valid_bridge_nodes(self, nodes: set[int]) -> set[int]:
        if not nodes:
            return set()
        if self._allow_global_id_bridge:
            return nodes
        return {node for node in nodes if self._is_valid_bridge_node(node)}

    def _is_valid_bridge_node(self, node: int) -> bool:
        if self._allow_global_id_bridge:
            return True
        src_degree = len(self.index.src_dsts.get(int(node), ()))
        dst_degree = len(self.index.dst_srcs.get(int(node), ()))
        # In bipartite graphs (no src/dst overlap), bridge nodes cannot satisfy
        # both degree thresholds. Fall back to allowing any node that appears
        # in at least one role, to avoid zeroing out structure signals.
        if self._bridge_overlap_ratio == 0.0:
            return (src_degree + dst_degree) >= self._bridge_min_role_degree
        return src_degree >= self._bridge_min_role_degree and dst_degree >= self._bridge_min_role_degree

    def _full_cooccur_counts(
        self,
        src: int,
        src_neighbors: set[int],
        candidate_ids: tuple[int, ...],
        *,
        force_preaggregate: bool = False,
    ) -> np.ndarray:
        counts = np.zeros(len(candidate_ids), dtype=np.int32)
        if not src_neighbors or not candidate_ids:
            return counts

        has_grouped_cooccurs = (
            bool(self.index.future_cooccur_count_maps)
            if self.index.future_only
            else bool(self.index.cooccurs_by_left)
        )
        if force_preaggregate and has_grouped_cooccurs:
            candidate_counts = self._full_src_cooccurs(src, src_neighbors)
        elif len(src_neighbors) <= FULL_COOCCUR_PREAGGREGATE_NEIGHBOR_THRESHOLD or not has_grouped_cooccurs:
            unique_candidates = tuple(dict.fromkeys(candidate_ids))
            candidate_counts: dict[int, int] = {}
            for candidate in unique_candidates:
                total = 0
                for seen_dst in src_neighbors:
                    if seen_dst == candidate:
                        continue
                    if self.index.future_only:
                        total += self.index.future_cooccur_count_maps.get(seen_dst, {}).get(candidate, 0)
                    else:
                        times = self.index.cooccur_times.get((seen_dst, candidate))
                        if times is not None:
                            total += len(times)
                candidate_counts[candidate] = total
        else:
            candidate_counts = self._full_src_cooccurs(src, src_neighbors)

        for idx, candidate in enumerate(candidate_ids):
            counts[idx] = candidate_counts.get(candidate, 0)
        return counts

    def _full_src_cooccurs(self, src: int, src_neighbors: set[int]) -> dict[int, int]:
        cached = self._full_src_cooccur_cache.get(src)
        if cached is not None:
            self._full_src_cooccur_cache.move_to_end(src)
            return cached

        counts: dict[int, int] = {}
        for seen_dst in src_neighbors:
            if self.index.future_only:
                cooccur_items = self.index.future_cooccur_count_maps.get(seen_dst, {}).items()
            else:
                cooccur_items = self.index.cooccurs_by_left.get(seen_dst, ())
            for candidate, value in cooccur_items:
                candidate_int = int(candidate)
                count = int(value) if self.index.future_only else len(value)
                counts[candidate_int] = counts.get(candidate_int, 0) + count
        self._cache_put(self._full_src_cooccur_cache, src, counts, FULL_COOCCUR_CACHE_LIMIT)
        return counts

    @staticmethod
    def _cache_put(
        cache: OrderedDict[int, set[int]] | OrderedDict[int, dict[int, int]],
        key: int,
        value,
        limit: int,
    ) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
