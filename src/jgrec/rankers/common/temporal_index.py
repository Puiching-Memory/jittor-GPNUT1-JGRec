from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from jgrec.core.types import InteractionTable

DEFAULT_COOCCUR_HISTORY_LIMIT = 128


@dataclass(frozen=True)
class SourceHistoryView:
    dsts: np.ndarray
    times: np.ndarray
    cutoff: int

    @property
    def visible_dsts(self) -> np.ndarray:
        return self.dsts[: self.cutoff]

    @property
    def visible_times(self) -> np.ndarray:
        return self.times[: self.cutoff]


@dataclass(frozen=True)
class DestinationHistoryView:
    srcs: np.ndarray
    times: np.ndarray
    cutoff: int

    @property
    def visible_srcs(self) -> np.ndarray:
        return self.srcs[: self.cutoff]

    @property
    def visible_times(self) -> np.ndarray:
        return self.times[: self.cutoff]


class TemporalInteractionIndex:
    def __init__(self) -> None:
        self.src_times: dict[int, np.ndarray] = {}
        self.src_dsts: dict[int, np.ndarray] = {}
        self.dst_times: dict[int, np.ndarray] = {}
        self.dst_srcs: dict[int, np.ndarray] = {}
        self.dst_unique_src_counts: dict[int, int] = {}
        self.pair_times: dict[tuple[int, int], np.ndarray] = {}
        self.transition_times: dict[tuple[int, int], np.ndarray] = {}
        self.transitions_by_left: dict[int, tuple[tuple[int, np.ndarray], ...]] = {}
        self.cooccur_times: dict[tuple[int, int], np.ndarray] = {}
        self.cooccurs_by_left: dict[int, tuple[tuple[int, np.ndarray], ...]] = {}
        self.popular_dsts: tuple[int, ...] = ()
        self.max_time = 0
        self.total_edges = 0
        self.built_transitions = False
        self.built_cooccurs = False
        self.cooccur_history_limit = DEFAULT_COOCCUR_HISTORY_LIMIT
        self.future_only = False
        self.transition_counts_by_pair: dict[tuple[int, int], int] = {}
        self.cooccur_counts_by_pair: dict[tuple[int, int], int] = {}
        self.future_transition_count_maps: dict[int, dict[int, int]] = {}
        self.future_cooccur_count_maps: dict[int, dict[int, int]] = {}
        self.future_transitions_by_left: dict[int, tuple[tuple[int, int], ...]] = {}
        self.future_cooccurs_by_left: dict[int, tuple[tuple[int, int], ...]] = {}

    def fit(
        self,
        interactions: InteractionTable,
        *,
        build_transitions: bool = True,
        build_cooccurs: bool = True,
        cooccur_history_limit: int = DEFAULT_COOCCUR_HISTORY_LIMIT,
        future_only_transition_cooccur: bool = False,
    ) -> None:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")

        interactions = interactions.sort_by_time()
        self.max_time = int(interactions.time[-1])
        self.total_edges = len(interactions)
        self.built_transitions = bool(build_transitions)
        self.built_cooccurs = bool(build_cooccurs)
        self.cooccur_history_limit = int(cooccur_history_limit)
        src_times: dict[int, list[int]] = defaultdict(list)
        src_dsts: dict[int, list[int]] = defaultdict(list)
        dst_times: dict[int, list[int]] = defaultdict(list)
        dst_srcs: dict[int, list[int]] = defaultdict(list)
        pair_times: dict[tuple[int, int], list[int]] = defaultdict(list)

        for src, dst, time in zip(interactions.src, interactions.dst, interactions.time):
            src_int = int(src)
            dst_int = int(dst)
            time_int = int(time)
            src_times[src_int].append(time_int)
            src_dsts[src_int].append(dst_int)
            dst_times[dst_int].append(time_int)
            dst_srcs[dst_int].append(src_int)
            pair_times[(src_int, dst_int)].append(time_int)

        self.fit_grouped(
            src_times=src_times,
            src_dsts=src_dsts,
            dst_times=dst_times,
            dst_srcs=dst_srcs,
            pair_times=pair_times,
            max_time=int(interactions.time[-1]),
            total_edges=len(interactions),
            build_transitions=build_transitions,
            build_cooccurs=build_cooccurs,
            cooccur_history_limit=cooccur_history_limit,
            future_only_transition_cooccur=future_only_transition_cooccur,
        )

    def fit_grouped(
        self,
        *,
        src_times: Mapping[int, list[int]],
        src_dsts: Mapping[int, list[int]],
        dst_times: Mapping[int, list[int]],
        dst_srcs: Mapping[int, list[int]],
        pair_times: Mapping[tuple[int, int], list[int]],
        max_time: int,
        total_edges: int,
        build_transitions: bool = True,
        build_cooccurs: bool = True,
        cooccur_history_limit: int = DEFAULT_COOCCUR_HISTORY_LIMIT,
        future_only_transition_cooccur: bool = False,
    ) -> None:
        if total_edges <= 0:
            raise ValueError("training interactions are empty")
        self.max_time = int(max_time)
        self.total_edges = int(total_edges)
        self.built_transitions = bool(build_transitions)
        self.built_cooccurs = bool(build_cooccurs)
        self.cooccur_history_limit = int(cooccur_history_limit)
        self.src_times = {src: _compact_int_array(times) for src, times in src_times.items()}
        self.src_dsts = {src: _compact_int_array(dsts) for src, dsts in src_dsts.items()}
        self.dst_times = {dst: _compact_int_array(times) for dst, times in dst_times.items()}
        self.dst_srcs = {dst: _compact_int_array(srcs) for dst, srcs in dst_srcs.items()}
        self.dst_unique_src_counts = {int(dst): len(set(int(src) for src in srcs)) for dst, srcs in dst_srcs.items()}
        self.pair_times = {pair: _compact_int_array(times) for pair, times in pair_times.items()}
        if future_only_transition_cooccur:
            self.transition_times = {}
            self.transitions_by_left = {}
            self.transition_counts_by_pair = {}
            self.future_transition_count_maps = _transition_count_maps(src_dsts) if build_transitions else {}
            self.future_transitions_by_left = {}
            self.cooccur_times = {}
            self.cooccurs_by_left = {}
            self.cooccur_counts_by_pair = {}
            self.future_cooccur_count_maps = (
                _cooccur_count_maps(src_dsts, history_limit=cooccur_history_limit) if build_cooccurs else {}
            )
            self.future_cooccurs_by_left = {}
        else:
            self.transition_times = _transition_times(src_times, src_dsts) if build_transitions else {}
            self.transitions_by_left = _group_times_by_left(self.transition_times) if build_transitions else {}
            self.cooccur_times = (
                _cooccur_times(src_times, src_dsts, history_limit=cooccur_history_limit)
                if build_cooccurs
                else {}
            )
            self.cooccurs_by_left = _group_times_by_left(self.cooccur_times) if build_cooccurs else {}
            self.transition_counts_by_pair = {}
            self.cooccur_counts_by_pair = {}
            self.future_transition_count_maps = {}
            self.future_cooccur_count_maps = {}
            self.future_transitions_by_left = {}
            self.future_cooccurs_by_left = {}
        self.popular_dsts = tuple(
            dst
            for dst, _ in sorted(
                ((int(dst), len(times)) for dst, times in self.dst_times.items()),
                key=lambda item: (-item[1], item[0]),
            )
        )
        self.future_only = bool(future_only_transition_cooccur)

    def source_view(self, src: int, query_time: int | None = None) -> SourceHistoryView:
        times = self.src_times.get(src)
        dsts = self.src_dsts.get(src)
        if times is None or dsts is None:
            empty = np.empty(0, dtype=np.int32)
            return SourceHistoryView(dsts=empty, times=empty, cutoff=0)
        cutoff = len(times) if query_time is None or query_time > self.max_time else _cutoff(times, query_time)
        return SourceHistoryView(dsts=dsts, times=times, cutoff=cutoff)

    def destination_view(self, dst: int, query_time: int | None = None) -> DestinationHistoryView:
        times = self.dst_times.get(dst)
        srcs = self.dst_srcs.get(dst)
        if times is None or srcs is None:
            empty = np.empty(0, dtype=np.int32)
            return DestinationHistoryView(srcs=empty, times=empty, cutoff=0)
        cutoff = len(times) if query_time is None or query_time > self.max_time else _cutoff(times, query_time)
        return DestinationHistoryView(srcs=srcs, times=times, cutoff=cutoff)

    def pair_times_before(self, src: int, dst: int, query_time: int) -> np.ndarray:
        times = self.pair_times.get((src, dst))
        if times is None:
            return np.empty(0, dtype=np.int64)
        if query_time > self.max_time:
            return times
        return times[: _cutoff(times, query_time)]

    def reverse_pair_times_before(self, src: int, dst: int, query_time: int) -> np.ndarray:
        return self.pair_times_before(dst, src, query_time)

    def transition_count(self, previous_dst: int, candidate_dst: int, query_time: int) -> int:
        if self.future_only and query_time > self.max_time:
            return self.future_transition_count_maps.get(int(previous_dst), {}).get(int(candidate_dst), 0)
        times = self.transition_times.get((previous_dst, candidate_dst))
        if times is None:
            return 0
        if query_time > self.max_time:
            return len(times)
        return _cutoff(times, query_time)

    def cooccur_count(self, src: int, candidate_dst: int, query_time: int) -> int:
        total = 0
        src_dsts = set(int(dst) for dst in self.source_view(src, query_time).visible_dsts)
        src_dsts.discard(candidate_dst)
        for seen_dst in src_dsts:
            if self.future_only and query_time > self.max_time:
                total += self.future_cooccur_count_maps.get(int(seen_dst), {}).get(int(candidate_dst), 0)
                continue
            times = self.cooccur_times.get((seen_dst, candidate_dst))
            if times is None:
                continue
            total += len(times) if query_time > self.max_time else _cutoff(times, query_time)
        return total

    def transition_candidates(self, previous_dst: int, query_time: int, limit: int = 512) -> tuple[int, ...]:
        if self.future_only:
            return _top_count_candidates(self.future_transition_count_maps.get(int(previous_dst), {}), limit)
        candidates = self.transitions_by_left.get(int(previous_dst), ())
        return tuple(candidate for candidate, _ in candidates[:limit])

    def cooccur_candidates(self, src: int, query_time: int, limit: int = 512) -> tuple[int, ...]:
        source_view = self.source_view(src, query_time)
        candidates: list[int] = []
        seen_candidates: set[int] = set()
        for seen_dst in reversed(source_view.visible_dsts[-64:]):
            if self.future_only:
                cooccur_items = _top_count_items(self.future_cooccur_count_maps.get(int(seen_dst), {}), limit)
            else:
                cooccur_items = self.cooccurs_by_left.get(int(seen_dst), ())
            for candidate, _ in cooccur_items:
                candidate_int = int(candidate)
                if candidate_int in seen_candidates:
                    continue
                seen_candidates.add(candidate_int)
                candidates.append(candidate_int)
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break
        return tuple(candidates)

    def popular_destinations(self, query_time: int, limit: int = 2048) -> tuple[int, ...]:
        return self.popular_dsts[:limit]

    def compact_for_future_queries(self) -> None:
        self.src_times = {}
        self.dst_times = {}
        self.transitions_by_left = {}
        self.cooccurs_by_left = {}

    def compact_transition_cooccur_for_future_queries(self) -> None:
        if self.transition_times:
            self.future_transition_count_maps = _count_maps_from_time_pairs(self.transition_times)
        if self.cooccur_times:
            self.future_cooccur_count_maps = _count_maps_from_time_pairs(self.cooccur_times)
        self.transition_counts_by_pair = {}
        self.cooccur_counts_by_pair = {}
        self.future_transitions_by_left = {}
        self.future_cooccurs_by_left = {}
        self.transition_times = {}
        self.transitions_by_left = {}
        self.cooccur_times = {}
        self.cooccurs_by_left = {}
        self.future_only = True

    def shallow_copy(self) -> TemporalInteractionIndex:
        clone = TemporalInteractionIndex()
        clone.__dict__.update(self.__dict__)
        clone.src_times = dict(self.src_times)
        clone.src_dsts = dict(self.src_dsts)
        clone.dst_times = dict(self.dst_times)
        clone.dst_srcs = dict(self.dst_srcs)
        clone.dst_unique_src_counts = dict(self.dst_unique_src_counts)
        clone.pair_times = dict(self.pair_times)
        clone.transition_times = dict(self.transition_times)
        clone.transitions_by_left = dict(self.transitions_by_left)
        clone.cooccur_times = dict(self.cooccur_times)
        clone.cooccurs_by_left = dict(self.cooccurs_by_left)
        clone.transition_counts_by_pair = dict(self.transition_counts_by_pair)
        clone.cooccur_counts_by_pair = dict(self.cooccur_counts_by_pair)
        clone.future_transition_count_maps = _copy_nested_counts(self.future_transition_count_maps)
        clone.future_cooccur_count_maps = _copy_nested_counts(self.future_cooccur_count_maps)
        clone.future_transitions_by_left = dict(self.future_transitions_by_left)
        clone.future_cooccurs_by_left = dict(self.future_cooccurs_by_left)
        clone.popular_dsts = tuple(self.popular_dsts)
        return clone


def _cutoff(times: np.ndarray, query_time: int) -> int:
    return int(np.searchsorted(times, query_time, side="left"))


def _compact_int_array(values: list[int]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.int32)
    min_value = min(values)
    max_value = max(values)
    int32 = np.iinfo(np.int32)
    dtype = np.int32 if int32.min <= min_value <= max_value <= int32.max else np.int64
    return np.asarray(values, dtype=dtype)


def _transition_times(
    src_times: dict[int, list[int]],
    src_dsts: dict[int, list[int]],
) -> dict[tuple[int, int], np.ndarray]:
    times_by_transition: dict[tuple[int, int], list[int]] = defaultdict(list)
    for src, dsts in src_dsts.items():
        times = src_times[src]
        for previous, current, current_time in zip(dsts, dsts[1:], times[1:]):
            times_by_transition[(int(previous), int(current))].append(int(current_time))
    return {
        transition: _compact_int_array(times)
        for transition, times in times_by_transition.items()
    }


def _transition_count_maps(src_dsts: dict[int, list[int]]) -> dict[int, dict[int, int]]:
    counts_by_left: dict[int, dict[int, int]] = defaultdict(dict)
    for dsts in src_dsts.values():
        for previous, current in zip(dsts, dsts[1:]):
            left = int(previous)
            right = int(current)
            right_counts = counts_by_left[left]
            right_counts[right] = right_counts.get(right, 0) + 1
    return {left: dict(counts) for left, counts in counts_by_left.items()}


def _count_maps_from_time_pairs(times_by_pair: dict[tuple[int, int], np.ndarray]) -> dict[int, dict[int, int]]:
    counts_by_left: dict[int, dict[int, int]] = defaultdict(dict)
    for (left, right), times in times_by_pair.items():
        counts_by_left[int(left)][int(right)] = int(len(times))
    return {left: dict(counts) for left, counts in counts_by_left.items()}


def _copy_nested_counts(values: dict[int, dict[int, int]]) -> dict[int, dict[int, int]]:
    return {int(left): dict(counts) for left, counts in values.items()}


def _cooccur_times(
    src_times: dict[int, list[int]],
    src_dsts: dict[int, list[int]],
    history_limit: int = DEFAULT_COOCCUR_HISTORY_LIMIT,
) -> dict[tuple[int, int], np.ndarray]:
    history_limit = max(int(history_limit), 0)
    if history_limit <= 0:
        return {}

    times_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
    for src, dsts in src_dsts.items():
        seen: set[int] = set()
        recent_unique: list[int] = []
        for dst, event_time in zip(dsts, src_times[src]):
            dst_int = int(dst)
            if dst_int in seen:
                continue
            for other in recent_unique:
                times_by_pair[(other, dst_int)].append(int(event_time))
                times_by_pair[(dst_int, other)].append(int(event_time))
            seen.add(dst_int)
            recent_unique.append(dst_int)
            if len(seen) > history_limit:
                expired = recent_unique.pop(0)
                seen.remove(expired)
    return {
        pair: _compact_int_array(times)
        for pair, times in times_by_pair.items()
    }


def _cooccur_count_maps(
    src_dsts: dict[int, list[int]],
    history_limit: int = DEFAULT_COOCCUR_HISTORY_LIMIT,
) -> dict[int, dict[int, int]]:
    history_limit = max(int(history_limit), 0)
    if history_limit <= 0:
        return {}

    counts_by_left: dict[int, dict[int, int]] = defaultdict(dict)
    for dsts in src_dsts.values():
        seen: set[int] = set()
        recent_unique: list[int] = []
        for dst in dsts:
            dst_int = int(dst)
            if dst_int in seen:
                continue
            for other in recent_unique:
                other_counts = counts_by_left[other]
                other_counts[dst_int] = other_counts.get(dst_int, 0) + 1
                dst_counts = counts_by_left[dst_int]
                dst_counts[other] = dst_counts.get(other, 0) + 1
            seen.add(dst_int)
            recent_unique.append(dst_int)
            if len(seen) > history_limit:
                expired = recent_unique.pop(0)
                seen.remove(expired)
    return {left: dict(counts) for left, counts in counts_by_left.items()}


def _group_times_by_left(times_by_pair: dict[tuple[int, int], np.ndarray]) -> dict[int, tuple[tuple[int, np.ndarray], ...]]:
    grouped: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (left, right), times in times_by_pair.items():
        grouped[int(left)].append((int(right), times))
    return {
        left: tuple(sorted(values, key=lambda item: (-len(item[1]), item[0])))
        for left, values in grouped.items()
    }


def _top_count_candidates(counts: dict[int, int], limit: int) -> tuple[int, ...]:
    return tuple(candidate for candidate, _ in _top_count_items(counts, limit))


def _top_count_items(counts: dict[int, int], limit: int) -> tuple[tuple[int, int], ...]:
    if not counts or limit <= 0:
        return ()
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])
