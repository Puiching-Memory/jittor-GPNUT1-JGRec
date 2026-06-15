from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jgrec.core.io import read_interactions, read_test_queries
from jgrec.core.types import InteractionTable

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


def profile_dataset(interactions: InteractionTable, test_path: Path, val_ratio: float = 0.15) -> DatasetProfile:
    if len(interactions) == 0:
        return DatasetProfile(
            holdout_pair_hit_rate=0.0,
            holdout_new_pair_rate=1.0,
            candidate_unseen_dst_rate=0.0,
            candidate_seen_dst_rate=0.0,
            src_history_p90=0.0,
            test_candidate_top1pct_share=0.0,
        )

    ordered = interactions.sort_by_time()
    val_size = max(1, int(len(ordered) * val_ratio))
    train_end = max(1, len(ordered) - val_size)
    history = ordered[:train_end]
    holdout = ordered[train_end:]
    history_pairs = set(zip(history.src.astype(int, copy=False), history.dst.astype(int, copy=False), strict=True))
    pair_hits = sum(
        1
        for src, dst in zip(holdout.src.astype(int, copy=False), holdout.dst.astype(int, copy=False), strict=True)
        if (src, dst) in history_pairs
    )
    holdout_count = max(len(holdout), 1)
    holdout_pair_hit_rate = pair_hits / holdout_count

    _, src_count_values = np.unique(ordered.src, return_counts=True)
    src_history_p90 = float(np.percentile(src_count_values, 90)) if src_count_values.size else 0.0

    train_dst = set(np.unique(ordered.dst).astype(int).tolist())
    test_counts, total_candidates, unseen_candidates, test_min_time = _scan_test_profile(test_path, train_dst)

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
        train_max_time=int(ordered.time[-1]),
        test_min_time=int(test_min_time),
    )


def profile_dataset_paths(train_path: Path, test_path: Path, val_ratio: float = 0.15) -> DatasetProfile:
    interactions = read_interactions(train_path)
    row_count = len(interactions)
    if row_count <= 0:
        return profile_dataset(InteractionTable.empty(), test_path, val_ratio=val_ratio)

    src_values = interactions.src.astype(np.int64, copy=False)
    dst_values = interactions.dst.astype(np.int64, copy=False)
    times = interactions.time.astype(np.int64, copy=False)
    train_max_time = int(times.max(initial=0))
    train_dst = set(np.unique(dst_values).astype(int).tolist())
    unique_src, src_count_values = np.unique(src_values, return_counts=True)
    del unique_src

    val_size = max(1, int(row_count * val_ratio))
    train_end = max(1, row_count - val_size)
    ordered_times = np.sort(times)
    boundary_time = int(ordered_times[train_end - 1])
    history_before_boundary = int(np.searchsorted(ordered_times, boundary_time, side="left"))
    history_boundary_budget = train_end - history_before_boundary
    del ordered_times

    history_mask = times < boundary_time
    if history_boundary_budget > 0:
        boundary_indices = np.flatnonzero(times == boundary_time)[:history_boundary_budget]
        history_mask[boundary_indices] = True

    history_pairs = set(
        zip(
            src_values[history_mask].astype(int, copy=False),
            dst_values[history_mask].astype(int, copy=False),
            strict=True,
        )
    )
    holdout_mask = ~history_mask

    holdout_count = int(holdout_mask.sum())
    pair_hits = sum(
        1
        for src, dst in zip(
            src_values[holdout_mask].astype(int, copy=False),
            dst_values[holdout_mask].astype(int, copy=False),
            strict=True,
        )
        if (src, dst) in history_pairs
    )

    test_counts, total_candidates, unseen_candidates, test_min_time = _scan_test_profile(test_path, train_dst)
    holdout_pair_hit_rate = pair_hits / max(holdout_count, 1)
    candidate_unseen_dst_rate = unseen_candidates / total_candidates if total_candidates else 0.0
    src_history_p90 = float(np.percentile(src_count_values, 90)) if src_count_values.size else 0.0
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
        return AutoStrategy(mode=REPEAT_MEMORY_MODE, test_candidate_negative_ratio=0.0)
    if profile.holdout_pair_hit_rate < 0.10 and profile.candidate_unseen_dst_rate >= 0.30:
        return AutoStrategy(mode=NEW_LINK_COLD_MODE, test_candidate_negative_ratio=0.0)
    return AutoStrategy(mode=BALANCED_MODE, test_candidate_negative_ratio=0.0)


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
    top_count = max(1, math.ceil(len(counts) * ratio))
    return sum(value for _, value in counts.most_common(top_count)) / total


def _scan_test_profile(test_path: Path, train_dst: set[int]) -> tuple[Counter[int], int, int, int]:
    queries = read_test_queries(test_path)
    if len(queries) == 0:
        return Counter(), 0, 0, 0

    candidates = queries.candidates.astype(np.int64, copy=False).reshape(-1)
    values, counts = np.unique(candidates, return_counts=True)
    test_counts = Counter({int(value): int(count) for value, count in zip(values, counts, strict=True)})
    total_candidates = int(candidates.size)

    if train_dst:
        train_values = np.fromiter(train_dst, dtype=np.int64, count=len(train_dst))
        unseen_mask = ~np.isin(values, train_values)
        unseen_candidates = int(counts[unseen_mask].sum())
    else:
        unseen_candidates = total_candidates
    return test_counts, total_candidates, unseen_candidates, int(queries.time.min())
