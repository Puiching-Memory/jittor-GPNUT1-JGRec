from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence, Set

import numpy as np

ItemPair = tuple[int, int]


def canonical_item_pair(left: int, right: int) -> ItemPair:
    """Return an order-independent key for a pair of item identifiers."""
    left_int = int(left)
    right_int = int(right)
    return (left_int, right_int) if left_int <= right_int else (right_int, left_int)


def recent_unique_targets(dsts: np.ndarray, limit: int) -> np.ndarray:
    """Return the latest unique targets, preserving their chronological order."""
    if limit <= 0:
        return np.empty(0, dtype=np.asarray(dsts).dtype)

    selected: list[int] = []
    seen: set[int] = set()
    for dst in np.asarray(dsts)[::-1]:
        dst_int = int(dst)
        if dst_int in seen:
            continue
        seen.add(dst_int)
        selected.append(dst_int)
        if len(selected) >= limit:
            break
    selected.reverse()
    return np.asarray(selected, dtype=np.asarray(dsts).dtype)


def accumulate_required_cooccurrence_events(
    dsts: np.ndarray,
    times: np.ndarray,
    required_pairs: Set[ItemPair],
    output: MutableMapping[ItemPair, list[int]],
    *,
    history_limit: int = 128,
) -> None:
    """Collect timestamps for required pairs using the production cooccurrence rules."""
    if history_limit <= 0:
        return

    seen: set[int] = set()
    recent_unique: list[int] = []
    for dst, event_time in zip(np.asarray(dsts), np.asarray(times), strict=True):
        dst_int = int(dst)
        if dst_int in seen:
            continue
        for other in recent_unique:
            pair = canonical_item_pair(other, dst_int)
            if pair in required_pairs:
                output.setdefault(pair, []).append(int(event_time))
        seen.add(dst_int)
        recent_unique.append(dst_int)
        if len(seen) > history_limit:
            expired = recent_unique.pop(0)
            seen.remove(expired)


def two_hop_scores(
    *,
    query_time: int,
    source_history: np.ndarray,
    candidates: np.ndarray,
    pair_event_times: Mapping[ItemPair, Sequence[int]],
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score candidates by raw and exponentially decayed two-hop cooccurrences."""
    if tau <= 0:
        raise ValueError("tau must be positive")

    candidate_array = np.asarray(candidates)
    history_array = np.asarray(source_history)
    raw_scores = np.zeros(candidate_array.shape[0], dtype=np.float64)
    decayed_scores = np.zeros(candidate_array.shape[0], dtype=np.float64)

    for candidate_index, candidate in enumerate(candidate_array):
        candidate_int = int(candidate)
        for history_target in history_array:
            history_int = int(history_target)
            if history_int == candidate_int:
                continue
            event_times = pair_event_times.get(
                canonical_item_pair(history_int, candidate_int)
            )
            if event_times is None or len(event_times) == 0:
                continue
            eligible = np.asarray(event_times, dtype=np.int64)
            eligible = eligible[eligible < int(query_time)]
            raw_scores[candidate_index] += float(eligible.size)
            if eligible.size:
                ages = int(query_time) - eligible
                decayed_scores[candidate_index] += float(
                    np.exp(-ages.astype(np.float64) / float(tau)).sum()
                )
    return raw_scores, decayed_scores


def tie_neutral_mrr(scores: np.ndarray) -> float:
    """Compute MRR with the positive in column zero and average ranks for ties."""
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 2 or score_array.shape[1] == 0:
        raise ValueError("scores must be a non-empty 2D array")
    positive_scores = score_array[:, :1]
    greater = np.sum(score_array[:, 1:] > positive_scores, axis=1)
    equal_other = np.sum(score_array[:, 1:] == positive_scores, axis=1)
    average_ranks = 1.0 + greater + 0.5 * equal_other
    return float(np.mean(1.0 / average_ranks))


def passes_two_hop_proxy_gate(
    *,
    coverage: float,
    baseline_mrr: float,
    candidate_mrr: float,
    baseline_slice_mrrs: Sequence[float],
    candidate_slice_mrrs: Sequence[float],
    min_coverage: float = 0.20,
    min_full_gain: float = 0.01,
) -> bool:
    """Apply the frozen go/no-go criteria for the two-hop proxy experiment."""
    if len(baseline_slice_mrrs) != len(candidate_slice_mrrs):
        raise ValueError("baseline and candidate slices must have equal lengths")
    return bool(
        coverage >= min_coverage
        and candidate_mrr - baseline_mrr >= min_full_gain
        and all(
            candidate > baseline
            for baseline, candidate in zip(
                baseline_slice_mrrs, candidate_slice_mrrs, strict=True
            )
        )
    )
