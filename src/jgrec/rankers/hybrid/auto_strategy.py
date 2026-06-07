from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jgrec.core.io import read_interactions, read_test_queries
from jgrec.core.types import Interaction

REPEAT_MEMORY_MODE = "repeat_memory"
BALANCED_MODE = "balanced"
NEW_LINK_COLD_MODE = "new_link_cold"


@dataclass(frozen=True)
class DatasetProfile:
    holdout_pair_hit_rate: float
    holdout_new_pair_rate: float
    candidate_unseen_dst_rate: float
    candidate_seen_dst_rate: float
    src_history_p90: float
    test_candidate_top1pct_share: float
    test_candidate_total: int = 0
    test_candidate_counts: Counter[int] = field(default_factory=Counter)
    train_max_time: int = 0
    test_min_time: int = 0


@dataclass(frozen=True)
class AutoStrategy:
    mode: str
    test_candidate_negative_ratio: float


def profile_dataset(interactions: list[Interaction], test_path: Path, val_ratio: float = 0.15) -> DatasetProfile:
    if not interactions:
        return DatasetProfile(
            holdout_pair_hit_rate=0.0,
            holdout_new_pair_rate=1.0,
            candidate_unseen_dst_rate=0.0,
            candidate_seen_dst_rate=0.0,
            src_history_p90=0.0,
            test_candidate_top1pct_share=0.0,
        )

    ordered = sorted(interactions, key=lambda item: item.time)
    val_size = max(1, int(len(ordered) * val_ratio))
    train_end = max(1, len(ordered) - val_size)
    history = ordered[:train_end]
    holdout = ordered[train_end:]
    history_pairs = {(int(item.src), int(item.dst)) for item in history}
    pair_hits = sum(1 for item in holdout if (int(item.src), int(item.dst)) in history_pairs)
    holdout_count = max(len(holdout), 1)
    holdout_pair_hit_rate = pair_hits / holdout_count

    src_counts = Counter(int(item.src) for item in ordered)
    src_history_p90 = float(np.percentile(list(src_counts.values()), 90)) if src_counts else 0.0

    train_dst = {int(item.dst) for item in ordered}
    test_counts, total_candidates, unseen_candidates = _scan_test_candidates(test_path, train_dst)
    test_min_time = _scan_test_min_time(test_path)

    if total_candidates:
        candidate_unseen_dst_rate = unseen_candidates / total_candidates
        candidate_seen_dst_rate = 1.0 - candidate_unseen_dst_rate
    else:
        candidate_unseen_dst_rate = 0.0
        candidate_seen_dst_rate = 0.0

    test_candidate_top1pct_share = _top_share(test_counts, total_candidates, ratio=0.01)
    return DatasetProfile(
        holdout_pair_hit_rate=float(holdout_pair_hit_rate),
        holdout_new_pair_rate=float(1.0 - holdout_pair_hit_rate),
        candidate_unseen_dst_rate=float(candidate_unseen_dst_rate),
        candidate_seen_dst_rate=float(candidate_seen_dst_rate),
        src_history_p90=src_history_p90,
        test_candidate_top1pct_share=float(test_candidate_top1pct_share),
        test_candidate_total=int(total_candidates),
        test_candidate_counts=test_counts,
        train_max_time=int(ordered[-1].time),
        test_min_time=int(test_min_time),
    )


def profile_dataset_paths(train_path: Path, test_path: Path, val_ratio: float = 0.15) -> DatasetProfile:
    times: list[int] = []
    train_dst: set[int] = set()
    src_counts: Counter[int] = Counter()
    train_max_time = 0
    for item in read_interactions(train_path):
        item_time = int(item.time)
        times.append(item_time)
        train_max_time = max(train_max_time, item_time)
        train_dst.add(int(item.dst))
        src_counts[int(item.src)] += 1

    row_count = len(times)
    if row_count <= 0:
        return profile_dataset([], test_path, val_ratio=val_ratio)

    val_size = max(1, int(row_count * val_ratio))
    train_end = max(1, row_count - val_size)
    ordered_times = np.sort(np.asarray(times, dtype=np.int64))
    boundary_time = int(ordered_times[train_end - 1])
    history_before_boundary = int(np.searchsorted(ordered_times, boundary_time, side="left"))
    history_boundary_budget = train_end - history_before_boundary
    del ordered_times
    del times

    history_pairs: set[tuple[int, int]] = set()
    history_boundary_used = 0

    for item in read_interactions(train_path):
        src = int(item.src)
        dst = int(item.dst)
        time = int(item.time)
        in_history = time < boundary_time
        if time == boundary_time and history_boundary_used < history_boundary_budget:
            in_history = True
            history_boundary_used += 1

        if in_history:
            history_pairs.add((src, dst))

    holdout_count = 0
    pair_hits = 0
    history_boundary_used = 0
    for item in read_interactions(train_path):
        src = int(item.src)
        dst = int(item.dst)
        time = int(item.time)
        in_history = time < boundary_time
        if time == boundary_time and history_boundary_used < history_boundary_budget:
            in_history = True
            history_boundary_used += 1

        if not in_history:
            holdout_count += 1
            if (src, dst) in history_pairs:
                pair_hits += 1

    test_counts, total_candidates, unseen_candidates = _scan_test_candidates(test_path, train_dst)
    test_min_time = _scan_test_min_time(test_path)
    holdout_pair_hit_rate = pair_hits / max(holdout_count, 1)
    candidate_unseen_dst_rate = unseen_candidates / total_candidates if total_candidates else 0.0
    src_history_p90 = float(np.percentile(list(src_counts.values()), 90)) if src_counts else 0.0
    return DatasetProfile(
        holdout_pair_hit_rate=float(holdout_pair_hit_rate),
        holdout_new_pair_rate=float(1.0 - holdout_pair_hit_rate),
        candidate_unseen_dst_rate=float(candidate_unseen_dst_rate),
        candidate_seen_dst_rate=float(1.0 - candidate_unseen_dst_rate if total_candidates else 0.0),
        src_history_p90=src_history_p90,
        test_candidate_top1pct_share=float(_top_share(test_counts, total_candidates, ratio=0.01)),
        test_candidate_total=int(total_candidates),
        test_candidate_counts=test_counts,
        train_max_time=int(train_max_time),
        test_min_time=int(test_min_time),
    )


def choose_auto_strategy(profile: DatasetProfile) -> AutoStrategy:
    if profile.holdout_pair_hit_rate >= 0.25 and profile.candidate_unseen_dst_rate <= 0.20:
        return AutoStrategy(mode=REPEAT_MEMORY_MODE, test_candidate_negative_ratio=0.10)
    if profile.holdout_pair_hit_rate < 0.10 and profile.candidate_unseen_dst_rate >= 0.30:
        return AutoStrategy(mode=NEW_LINK_COLD_MODE, test_candidate_negative_ratio=0.60)
    return AutoStrategy(mode=BALANCED_MODE, test_candidate_negative_ratio=0.35)


def test_candidate_arrays(profile: DatasetProfile | None) -> tuple[np.ndarray, np.ndarray]:
    if profile is None or not profile.test_candidate_counts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    ordered = sorted(profile.test_candidate_counts.items(), key=lambda item: (-item[1], item[0]))
    values = np.asarray([item[0] for item in ordered], dtype=np.int64)
    weights = np.asarray([item[1] for item in ordered], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        weights = np.full(values.shape[0], 1.0 / max(values.shape[0], 1), dtype=np.float64)
    else:
        weights = weights / total
    return values, weights


def _top_share(counts: Counter[int], total: int, ratio: float) -> float:
    if total <= 0 or not counts:
        return 0.0
    top_count = max(1, int(math.ceil(len(counts) * ratio)))
    return sum(value for _, value in counts.most_common(top_count)) / total


def _scan_test_candidates(test_path: Path, train_dst: set[int]) -> tuple[Counter[int], int, int]:
    test_counts: Counter[int] = Counter()
    total_candidates = 0
    unseen_candidates = 0
    for query in read_test_queries(test_path):
        for candidate in query.candidates:
            candidate_int = int(candidate)
            test_counts[candidate_int] += 1
            total_candidates += 1
            if candidate_int not in train_dst:
                unseen_candidates += 1
    return test_counts, total_candidates, unseen_candidates


def _scan_test_min_time(test_path: Path) -> int:
    min_time: int | None = None
    for query in read_test_queries(test_path):
        query_time = int(query.time)
        if min_time is None or query_time < min_time:
            min_time = query_time
    return 0 if min_time is None else min_time
