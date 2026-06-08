from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np

from jgrec.core.types import InteractionTable
from jgrec.logging import log
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex

from .candidate_prior import CandidatePriorTower
from .config import CandidatePriorConfig, StructureTowerConfig
from .stats import SrcHistory, TemporalStats
from .structure import StructureFeatureTower


@dataclass(frozen=True)
class HybridDeterministicSnapshot:
    prefix_end: int
    stats: dict
    candidate_prior: dict
    structure: dict


class HybridPrefixStateCache:
    def __init__(
        self,
        interactions: InteractionTable,
        *,
        recent_window: int,
        candidate_prior_config: CandidatePriorConfig,
        structure_config: StructureTowerConfig,
        test_candidate_counts: Counter[int] | None,
        verbose: bool = True,
    ) -> None:
        if not interactions:
            raise ValueError("training interactions are empty")
        self.interactions = interactions
        self.recent_window = int(recent_window)
        self.candidate_prior_config = candidate_prior_config
        self.structure_config = structure_config
        self.test_candidate_counts = Counter(test_candidate_counts or {})
        self.verbose = verbose
        self._cursor = 0
        self._snapshots: dict[int, HybridDeterministicSnapshot] = {}

        self._src_times: dict[int, list[int]] = defaultdict(list)
        self._src_dsts: dict[int, list[int]] = defaultdict(list)
        self._dst_times: dict[int, list[int]] = defaultdict(list)
        self._dst_srcs: dict[int, list[int]] = defaultdict(list)
        self._pair_times: dict[tuple[int, int], list[int]] = defaultdict(list)
        self._event_times: list[int] = []
        self._histories: dict[int, SrcHistory] = defaultdict(
            lambda: SrcHistory(recent_dsts=deque(maxlen=self.recent_window))
        )
        self._dst_counts: Counter[int] = Counter()
        self._dst_recent_time: dict[int, int] = {}

    def snapshot_for_prefix(self, prefix_end: int) -> HybridDeterministicSnapshot:
        prefix_end = int(prefix_end)
        if prefix_end <= 0 or prefix_end > len(self.interactions):
            raise ValueError(f"invalid encoder cache prefix: {prefix_end}")
        if prefix_end < self._cursor:
            cached = self._snapshots.get(prefix_end)
            if cached is None:
                raise ValueError("encoder cache prefixes must be requested in ascending order")
            log(f"[encoder-cache] hit prefix={prefix_end}", enabled=self.verbose)
            return cached
        if prefix_end > self._cursor:
            self._advance(prefix_end)
        snapshot = self._build_snapshot(prefix_end)
        self._snapshots[prefix_end] = snapshot
        log(f"[encoder-cache] build prefix={prefix_end}", enabled=self.verbose)
        return snapshot

    def release_except(self, prefix_end: int | None = None) -> None:
        if prefix_end is None:
            self._snapshots.clear()
            return
        keep = self._snapshots.get(int(prefix_end))
        self._snapshots.clear()
        if keep is not None:
            self._snapshots[int(prefix_end)] = keep

    def release_builder_state(self) -> None:
        self.interactions = InteractionTable.empty()
        self._src_times.clear()
        self._src_dsts.clear()
        self._dst_times.clear()
        self._dst_srcs.clear()
        self._pair_times.clear()
        self._event_times.clear()
        self._histories.clear()
        self._dst_counts.clear()
        self._dst_recent_time.clear()

    def clear(self) -> None:
        self._snapshots.clear()
        self.release_builder_state()

    def _advance(self, prefix_end: int) -> None:
        window = self.interactions[self._cursor : prefix_end]
        for src, dst, time in zip(window.src, window.dst, window.time):
            src_int = int(src)
            dst_int = int(dst)
            time_int = int(time)
            self._event_times.append(time_int)
            self._src_times[src_int].append(time_int)
            self._src_dsts[src_int].append(dst_int)
            self._dst_times[dst_int].append(time_int)
            self._dst_srcs[dst_int].append(src_int)
            self._pair_times[(src_int, dst_int)].append(time_int)

            history = self._histories[src_int]
            history.total += 1
            history.last_time = time_int
            history.dst_counts[dst_int] += 1
            history.pair_recent_time[dst_int] = time_int
            history.recent_dsts.append(dst_int)

            self._dst_counts[dst_int] += 1
            self._dst_recent_time[dst_int] = time_int
        self._cursor = prefix_end

    def _build_snapshot(self, prefix_end: int) -> HybridDeterministicSnapshot:
        stats = self._build_stats(prefix_end)
        candidate_prior = CandidatePriorTower(self.candidate_prior_config)
        candidate_prior.fit_from_counts(set(self._dst_counts), self.test_candidate_counts)
        structure = StructureFeatureTower(self.structure_config)
        structure.index = self._build_structure_index(prefix_end)
        structure.min_time = int(self.interactions.time[0])
        structure.max_time = int(self.interactions.time[prefix_end - 1])
        structure.graph_span = max(structure.max_time - structure.min_time, 1)
        structure.decay_windows = (
            max(structure.graph_span * 0.05, 1.0),
            max(structure.graph_span * 0.20, 1.0),
            max(structure.graph_span * 0.50, 1.0),
        )
        return HybridDeterministicSnapshot(
            prefix_end=prefix_end,
            stats=stats.snapshot(),
            candidate_prior=candidate_prior.snapshot(),
            structure=structure.snapshot(),
        )

    def _build_stats(self, prefix_end: int) -> TemporalStats:
        stats = TemporalStats(recent_window=self.recent_window)
        stats.event_times = _compact_int_array(self._event_times)
        stats.total_edges = int(prefix_end)
        stats.min_time = int(self.interactions.time[0])
        stats.max_time = int(self.interactions.time[prefix_end - 1])
        stats.graph_span = max(stats.max_time - stats.min_time, 1)
        stats.log_total_edges = math.log1p(max(stats.total_edges, 1))
        stats.src_histories = {
            int(src): _finalize_history(history, stats.log_total_edges)
            for src, history in self._histories.items()
        }
        stats.src_event_times = {int(src): _compact_int_array(times) for src, times in self._src_times.items()}
        stats.src_event_dsts = {int(src): _compact_int_array(dsts) for src, dsts in self._src_dsts.items()}
        stats.dst_event_times = {int(dst): _compact_int_array(times) for dst, times in self._dst_times.items()}
        stats.pair_event_times = {pair: _compact_int_array(times) for pair, times in self._pair_times.items()}
        stats.dst_counts = Counter(self._dst_counts)
        stats.dst_recent_time = dict(self._dst_recent_time)
        stats.dst_popularity = {
            int(dst): math.log1p(count) / stats.log_total_edges
            for dst, count in stats.dst_counts.items()
        }
        stats._build_dense_dst_features()
        return stats

    def _build_structure_index(self, prefix_end: int) -> TemporalInteractionIndex:
        index = TemporalInteractionIndex()
        index.fit_grouped(
            src_times=self._src_times,
            src_dsts=self._src_dsts,
            dst_times=self._dst_times,
            dst_srcs=self._dst_srcs,
            pair_times=self._pair_times,
            max_time=int(self.interactions.time[prefix_end - 1]),
            total_edges=prefix_end,
            build_transitions=self.structure_config.transition_enabled,
            build_cooccurs=self.structure_config.cooccur_enabled,
            cooccur_history_limit=self.structure_config.cooccur_history_limit,
            future_only_transition_cooccur=self.structure_config.future_only_transition_cooccur,
        )
        return index


def hydrate_deterministic_state(
    *,
    snapshot: HybridDeterministicSnapshot,
    stats: TemporalStats,
    candidate_prior,
    structure,
) -> None:
    stats.hydrate(snapshot.stats)
    prior_hydrate = getattr(candidate_prior, "hydrate", None)
    if callable(prior_hydrate):
        prior_hydrate(snapshot.candidate_prior)
    structure_hydrate = getattr(structure, "hydrate", None)
    if callable(structure_hydrate):
        structure_hydrate(snapshot.structure)


def _finalize_history(source: SrcHistory, log_total_edges: float) -> SrcHistory:
    history = SrcHistory(recent_dsts=deque(source.recent_dsts, maxlen=source.recent_dsts.maxlen))
    history.total = int(source.total)
    history.last_time = int(source.last_time)
    history.activity = math.log1p(history.total) / log_total_edges
    history.dst_counts = Counter(source.dst_counts)
    history.pair_recent_time = dict(source.pair_recent_time)
    ranks: dict[int, int] = {}
    for rank, dst in enumerate(reversed(history.recent_dsts), start=1):
        ranks.setdefault(int(dst), rank)
    history.recent_ranks = ranks
    return history


def _compact_int_array(values: list[int]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.int32)
    min_value = min(values)
    max_value = max(values)
    int32 = np.iinfo(np.int32)
    dtype = np.int32 if int32.min <= min_value <= max_value <= int32.max else np.int64
    return np.asarray(values, dtype=dtype)
