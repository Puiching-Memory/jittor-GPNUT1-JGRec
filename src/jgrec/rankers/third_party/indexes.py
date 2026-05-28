from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from jgrec.core.types import Interaction


@dataclass
class SourceIndex:
    total: int = 0
    last_time: int = 0
    dst_counts: Counter[int] = field(default_factory=Counter)
    dst_last_time: dict[int, int] = field(default_factory=dict)
    dst_first_time: dict[int, int] = field(default_factory=dict)
    recent_dsts: deque[int] = field(default_factory=lambda: deque(maxlen=64))


@dataclass
class DestinationIndex:
    total: int = 0
    last_time: int = 0
    unique_sources: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class TemporalIndexConfig:
    recent_window: int = 64
    cooccur_recent_k: int = 16
    cooccur_max_sources: int = 500_000
    max_common_neighbors_degree: int = 512


class TemporalIndexes:
    def __init__(self, config: TemporalIndexConfig | None = None) -> None:
        self.config = config or TemporalIndexConfig()
        self.sources: dict[int, SourceIndex] = {}
        self.destinations: dict[int, DestinationIndex] = {}
        self.reverse_pair_counts: Counter[tuple[int, int]] = Counter()
        self.reverse_pair_last_time: dict[tuple[int, int], int] = {}
        self.out_neighbors: dict[int, set[int]] = {}
        self.in_neighbors: dict[int, set[int]] = {}
        self.cooccur: dict[int, Counter[int]] = {}
        self.transition: dict[int, Counter[int]] = {}
        self.dst_rank: dict[int, int] = {}
        self.min_time = 0
        self.max_time = 0
        self.graph_span = 1
        self.log_total_edges = 1.0
        self.log_total_src = 1.0

    def fit(self, interactions: list[Interaction]) -> None:
        if not interactions:
            raise ValueError("training interactions are empty")
        interactions = sorted(interactions, key=lambda item: item.time)
        self.min_time = interactions[0].time
        self.max_time = interactions[-1].time
        self.graph_span = max(self.max_time - self.min_time, 1)
        self.log_total_edges = math.log1p(len(interactions))
        self.sources = {}
        self.destinations = {}
        self.reverse_pair_counts = Counter()
        self.reverse_pair_last_time = {}
        out_neighbors: dict[int, set[int]] = defaultdict(set)
        in_neighbors: dict[int, set[int]] = defaultdict(set)

        sources: dict[int, SourceIndex] = defaultdict(
            lambda: SourceIndex(recent_dsts=deque(maxlen=self.config.recent_window))
        )
        destinations: dict[int, DestinationIndex] = defaultdict(DestinationIndex)
        for item in interactions:
            src_index = sources[item.src]
            src_index.total += 1
            src_index.last_time = item.time
            src_index.dst_counts[item.dst] += 1
            src_index.dst_last_time[item.dst] = item.time
            src_index.dst_first_time.setdefault(item.dst, item.time)
            src_index.recent_dsts.append(item.dst)

            dst_index = destinations[item.dst]
            dst_index.total += 1
            dst_index.last_time = item.time
            dst_index.unique_sources.add(item.src)

            out_neighbors[item.src].add(item.dst)
            in_neighbors[item.dst].add(item.src)
            self.reverse_pair_counts[(item.dst, item.src)] += 1
            self.reverse_pair_last_time[(item.dst, item.src)] = item.time

        self.sources = dict(sources)
        self.destinations = dict(destinations)
        self.out_neighbors = dict(out_neighbors)
        self.in_neighbors = dict(in_neighbors)
        self.log_total_src = math.log1p(max(len(self.sources), 1))
        self.dst_rank = {
            dst: rank
            for rank, (dst, _) in enumerate(
                sorted(self.destinations.items(), key=lambda pair: pair[1].total, reverse=True),
                start=1,
            )
        }
        self._build_cooccurrence(interactions)

    def _build_cooccurrence(self, interactions: list[Interaction]) -> None:
        by_src: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for item in interactions:
            by_src[item.src].append((item.dst, item.time))
        source_items = list(by_src.items())
        if len(source_items) > self.config.cooccur_max_sources:
            source_items = source_items[-self.config.cooccur_max_sources :]

        cooccur: dict[int, Counter[int]] = defaultdict(Counter)
        transition: dict[int, Counter[int]] = defaultdict(Counter)
        recent_k = max(self.config.cooccur_recent_k, 1)
        for _, history in source_items:
            if len(history) < 2:
                continue
            recent = history[-recent_k:]
            unique_recent = []
            seen: set[int] = set()
            for dst, _ in recent:
                if dst in seen:
                    continue
                seen.add(dst)
                unique_recent.append(dst)
            for i, dst_a in enumerate(unique_recent):
                for dst_b in unique_recent[i + 1 :]:
                    cooccur[dst_a][dst_b] += 1
                    cooccur[dst_b][dst_a] += 1
            for (dst_a, time_a), (dst_b, time_b) in zip(recent, recent[1:]):
                if dst_a == dst_b:
                    continue
                gap = max(time_b - time_a, 0)
                weight = max(1, int(10.0 * math.exp(-gap / self.graph_span)))
                transition[dst_a][dst_b] += weight

        self.cooccur = dict(cooccur)
        self.transition = dict(transition)

    def taus(self) -> tuple[float, float, float]:
        span = float(max(self.graph_span, 1))
        return (0.01 * span, 0.05 * span, 0.20 * span)

    def common_neighbor_features(self, src: int, dst: int) -> tuple[float, float]:
        src_neighbors = self.out_neighbors.get(src)
        dst_neighbors = self.out_neighbors.get(dst)
        if not src_neighbors or not dst_neighbors:
            return 0.0, 0.0
        if (
            len(src_neighbors) > self.config.max_common_neighbors_degree
            or len(dst_neighbors) > self.config.max_common_neighbors_degree
        ):
            return 0.0, 0.0
        common = len(src_neighbors & dst_neighbors)
        union = len(src_neighbors | dst_neighbors)
        jaccard = common / union if union else 0.0
        return math.log1p(common) / self.log_total_edges, jaccard

    def relation_scores_from_recent(self, source: SourceIndex | None, dst: int) -> tuple[float, float]:
        if source is None or not source.recent_dsts:
            return 0.0, 0.0
        cooccur_score = 0
        transition_score = 0
        for rank, recent_dst in enumerate(reversed(source.recent_dsts), start=1):
            weight = 1.0 / rank
            cooccur_score += weight * self.cooccur.get(recent_dst, {}).get(dst, 0)
            transition_score += weight * self.transition.get(recent_dst, {}).get(dst, 0)
        return math.log1p(cooccur_score) / self.log_total_edges, math.log1p(transition_score) / self.log_total_edges


def decay(gap: int, tau: float) -> float:
    return math.exp(-max(gap, 0) / max(tau, 1.0))


def normalize_gap(gap: int, span: int) -> float:
    return min(max(gap, 0) / max(span, 1), 1.0)


def as_candidate_matrix(candidates: list[tuple[int, ...]]) -> np.ndarray:
    if not candidates:
        return np.empty((0, 0), dtype=np.int64)
    return np.asarray(candidates, dtype=np.int64)

