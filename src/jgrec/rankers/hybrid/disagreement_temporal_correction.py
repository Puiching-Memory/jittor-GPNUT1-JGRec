from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

OOF_FEATURE_NAMES = (
    "expert_top1_vote_fraction",
    "consensus_top1_rank_std",
    "consensus_margin",
    "mean_candidate_rank_std",
    "maximum_candidate_rank_std",
    "expert_top1_diversity",
)

TEMPORAL_FEATURE_NAMES = (
    "temporal_top1_pair_seen",
    "temporal_top1_pair_count_log1p",
    "temporal_top1_recency",
    "temporal_top1_recent_global_log1p",
    "maximum_pair_count_log1p",
    "mean_pair_seen",
    "temporal_margin",
)

HYBRID_FEATURE_NAMES = (
    "signal_top1_agreement",
    "hybrid_margin",
    "mean_signal_rank_gap",
    "maximum_signal_rank_gap",
)

PROPOSAL_FEATURE_NAMES = (
    "base_top1_margin",
    "base_top1_probability",
    "base_entropy",
    "proposal_changes_top1",
    "proposed_vs_base_top1_signal_delta",
    "proposal_top1_signal_margin",
    "topk_score_reassignment_l1",
    "topk_score_reassignment_max",
)


@dataclass(frozen=True)
class CorrectionSignal:
    candidate_scores: np.ndarray
    row_features: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        scores = np.asarray(self.candidate_scores)
        features = np.asarray(self.row_features)
        if scores.ndim != 2 or features.ndim != 2:
            raise ValueError("correction signal arrays must be two dimensional")
        if features.shape[0] != scores.shape[0]:
            raise ValueError("correction signal row counts must align")
        if features.shape[1] != len(self.feature_names):
            raise ValueError("correction signal feature names must align")
        if not np.isfinite(scores).all() or not np.isfinite(features).all():
            raise FloatingPointError("correction signal contains NaN or Inf")


def row_percentile_scores(values: Any) -> np.ndarray:
    """Return tie-neutral within-row percentile ranks, with higher being better."""
    matrix = _matrix(values, label="row percentile values")
    rows, width = matrix.shape
    if width == 1:
        return np.ones((rows, 1), dtype=np.float32)
    order = np.argsort(matrix, axis=1, kind="stable")
    sorted_values = np.take_along_axis(matrix, order, axis=1)

    starts = np.ones((rows, width), dtype=bool)
    starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    columns = np.arange(width, dtype=np.int32)[None, :]
    start_indices = np.maximum.accumulate(
        np.where(starts, columns, 0),
        axis=1,
    )

    ends = np.ones((rows, width), dtype=bool)
    ends[:, :-1] = sorted_values[:, :-1] != sorted_values[:, 1:]
    end_indices = np.minimum.accumulate(
        np.where(ends, columns, width - 1)[:, ::-1],
        axis=1,
    )[:, ::-1]

    sorted_ranks = (
        0.5 * (start_indices.astype(np.float32) + end_indices)
        / float(width - 1)
    )
    output = np.empty((rows, width), dtype=np.float32)
    np.put_along_axis(output, order, sorted_ranks, axis=1)
    return output


def oof_disagreement_signal(
    expert_logits: Any,
    *,
    disagreement_penalty: float = 0.15,
) -> CorrectionSignal:
    experts = np.asarray(expert_logits)
    if experts.ndim != 3 or experts.shape[0] < 2:
        raise ValueError(
            "OOF disagreement needs [experts, rows, candidates] with 2+ experts"
        )
    if experts.shape[1] <= 0 or experts.shape[2] <= 1:
        raise ValueError("OOF disagreement dimensions are invalid")
    if not np.isfinite(experts).all():
        raise FloatingPointError("OOF expert logits contain NaN or Inf")
    penalty = float(disagreement_penalty)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("disagreement penalty must be finite and non-negative")

    percentiles = np.stack(
        [row_percentile_scores(experts[index]) for index in range(experts.shape[0])],
        axis=0,
    )
    consensus = np.mean(percentiles, axis=0, dtype=np.float32)
    rank_std = np.std(percentiles, axis=0, dtype=np.float32)
    candidate_scores = np.asarray(
        consensus - penalty * rank_std,
        dtype=np.float32,
    )
    top_indices = np.argmax(candidate_scores, axis=1)
    rows = np.arange(candidate_scores.shape[0])
    expert_top = np.argmax(percentiles, axis=2)
    vote_fraction = np.mean(
        expert_top == top_indices[None, :],
        axis=0,
        dtype=np.float32,
    )
    top_rank_std = rank_std[rows, top_indices]
    margin = _top_margin(candidate_scores)
    diversity = np.array(
        [
            np.unique(expert_top[:, row]).size / float(experts.shape[0])
            for row in range(expert_top.shape[1])
        ],
        dtype=np.float32,
    )
    row_features = np.column_stack(
        (
            vote_fraction,
            top_rank_std,
            margin,
            np.mean(rank_std, axis=1),
            np.max(rank_std, axis=1),
            diversity,
        )
    ).astype(np.float32, copy=False)
    return CorrectionSignal(
        candidate_scores=candidate_scores,
        row_features=row_features,
        feature_names=OOF_FEATURE_NAMES,
    )


def strict_temporal_support_signal(
    history_src: Any,
    history_dst: Any,
    history_time: Any,
    query_src: Any,
    candidate_ids: Any,
    *,
    origin_time: int | float,
    recent_rows: int = 20_000,
) -> CorrectionSignal:
    src = _vector(history_src, label="temporal history source")
    dst = _vector(history_dst, label="temporal history destination")
    times = _vector(history_time, label="temporal history time")
    if src.shape != dst.shape or src.shape != times.shape:
        raise ValueError("temporal history arrays must align")
    queries = _vector(query_src, label="temporal query source")
    candidates = _integer_matrix(candidate_ids, label="temporal candidates")
    if candidates.shape[0] != queries.shape[0]:
        raise ValueError("temporal query and candidate rows must align")
    if np.any(src < 0) or np.any(dst < 0):
        raise ValueError("temporal history IDs must be non-negative")
    if np.any(queries < 0) or np.any(candidates < 0):
        raise ValueError("temporal query IDs must be non-negative")
    origin = float(origin_time)
    if not math.isfinite(origin):
        raise ValueError("temporal origin must be finite")
    recent_count = int(recent_rows)
    if recent_count != recent_rows or recent_count <= 0:
        raise ValueError("recent_rows must be a positive integer")

    visible = np.asarray(times, dtype=np.float64) < origin
    visible_src = np.asarray(src[visible], dtype=np.int64)
    visible_dst = np.asarray(dst[visible], dtype=np.int64)
    visible_time = np.asarray(times[visible], dtype=np.float64)
    candidate_int = np.asarray(candidates, dtype=np.int64)
    query_int = np.asarray(queries, dtype=np.int64)
    shape = candidates.shape

    pair_count = np.zeros(shape, dtype=np.float32)
    pair_last = np.full(shape, -np.inf, dtype=np.float64)
    recent_global_count = np.zeros(shape, dtype=np.float32)
    if visible_src.size:
        maximum_item = int(
            max(
                int(np.max(visible_dst)),
                int(np.max(candidate_int)),
            )
        )
        stride = maximum_item + 1
        history_keys = visible_src * stride + visible_dst
        unique_keys, inverse, counts = np.unique(
            history_keys,
            return_inverse=True,
            return_counts=True,
        )
        last_times = np.full(unique_keys.shape, -np.inf, dtype=np.float64)
        np.maximum.at(last_times, inverse, visible_time)
        query_keys = (
            query_int[:, None] * stride + candidate_int
        ).reshape(-1)
        pair_count_flat, pair_last_flat = _lookup_sorted(
            unique_keys,
            np.asarray(counts, dtype=np.float32),
            last_times,
            query_keys,
        )
        pair_count = pair_count_flat.reshape(shape)
        pair_last = pair_last_flat.reshape(shape)

        recent_dst = visible_dst[-min(recent_count, visible_dst.size) :]
        recent_items, recent_counts = np.unique(
            recent_dst,
            return_counts=True,
        )
        recent_flat, _ = _lookup_sorted(
            recent_items,
            np.asarray(recent_counts, dtype=np.float32),
            np.zeros(recent_items.shape, dtype=np.float32),
            candidate_int.reshape(-1),
        )
        recent_global_count = recent_flat.reshape(shape)

    pair_seen = pair_count > 0
    pair_count_log = np.log1p(pair_count).astype(np.float32, copy=False)
    recent_global_log = np.log1p(recent_global_count).astype(
        np.float32,
        copy=False,
    )
    if visible_time.size:
        horizon = max(origin - float(np.min(visible_time)), 1.0)
        age_scale = max(horizon / 100.0, 1.0)
    else:
        age_scale = 1.0
    pair_recency = np.zeros(shape, dtype=np.float32)
    pair_recency[pair_seen] = np.exp(
        -np.maximum(origin - pair_last[pair_seen], 0.0) / age_scale
    ).astype(np.float32)
    raw_scores = (
        3.0 * pair_seen.astype(np.float32)
        + pair_count_log
        + pair_recency
        + 0.10 * recent_global_log
    )
    candidate_scores = row_percentile_scores(raw_scores)
    top_indices = np.argmax(candidate_scores, axis=1)
    rows = np.arange(candidate_scores.shape[0])
    row_features = np.column_stack(
        (
            pair_seen[rows, top_indices].astype(np.float32),
            pair_count_log[rows, top_indices],
            pair_recency[rows, top_indices],
            recent_global_log[rows, top_indices],
            np.max(pair_count_log, axis=1),
            np.mean(pair_seen, axis=1, dtype=np.float32),
            _top_margin(candidate_scores),
        )
    ).astype(np.float32, copy=False)
    return CorrectionSignal(
        candidate_scores=candidate_scores,
        row_features=row_features,
        feature_names=TEMPORAL_FEATURE_NAMES,
    )


def hybrid_consensus_signal(
    oof_signal: CorrectionSignal,
    temporal_signal: CorrectionSignal,
) -> CorrectionSignal:
    if oof_signal.candidate_scores.shape != temporal_signal.candidate_scores.shape:
        raise ValueError("hybrid correction signals must align")
    oof_ranks = row_percentile_scores(oof_signal.candidate_scores)
    temporal_ranks = row_percentile_scores(temporal_signal.candidate_scores)
    candidate_scores = np.asarray(
        0.5 * (oof_ranks + temporal_ranks),
        dtype=np.float32,
    )
    oof_top = np.argmax(oof_ranks, axis=1)
    temporal_top = np.argmax(temporal_ranks, axis=1)
    rank_gap = np.abs(oof_ranks - temporal_ranks)
    hybrid_features = np.column_stack(
        (
            (oof_top == temporal_top).astype(np.float32),
            _top_margin(candidate_scores),
            np.mean(rank_gap, axis=1),
            np.max(rank_gap, axis=1),
        )
    ).astype(np.float32, copy=False)
    return CorrectionSignal(
        candidate_scores=candidate_scores,
        row_features=np.concatenate(
            (
                oof_signal.row_features,
                temporal_signal.row_features,
                hybrid_features,
            ),
            axis=1,
        ),
        feature_names=(
            oof_signal.feature_names
            + temporal_signal.feature_names
            + HYBRID_FEATURE_NAMES
        ),
    )


def topk_score_multiset_proposal(
    base_scores: Any,
    signal_scores: Any,
    *,
    top_k: int,
) -> np.ndarray:
    base = _matrix(base_scores, label="top-k proposal base")
    signal = _matrix(signal_scores, label="top-k proposal signal")
    if base.shape != signal.shape:
        raise ValueError("top-k proposal base and signal must align")
    width = int(top_k)
    if width != top_k or not 1 <= width <= base.shape[1]:
        raise ValueError("top_k must be a valid integer")

    base_top = np.argsort(-base, axis=1, kind="stable")[:, :width]
    top_values = np.take_along_axis(base, base_top, axis=1)
    top_signal = np.take_along_axis(signal, base_top, axis=1)
    signal_order = np.argsort(-top_signal, axis=1, kind="stable")
    target_positions = np.take_along_axis(base_top, signal_order, axis=1)
    proposed = np.array(base, copy=True)
    np.put_along_axis(proposed, target_positions, top_values, axis=1)
    return proposed


def proposal_router_features(
    base_scores: Any,
    proposed_scores: Any,
    signal: CorrectionSignal,
    *,
    top_k: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    base = _matrix(base_scores, label="router base scores")
    proposed = _matrix(proposed_scores, label="router proposed scores")
    if base.shape != proposed.shape or base.shape != signal.candidate_scores.shape:
        raise ValueError("router score and signal arrays must align")
    width = int(top_k)
    if width != top_k or not 1 <= width <= base.shape[1]:
        raise ValueError("router top_k must be a valid integer")

    rows = np.arange(base.shape[0])
    base_top = np.argmax(base, axis=1)
    proposed_top = np.argmax(proposed, axis=1)
    signal_scores = signal.candidate_scores
    signal_top = np.argmax(signal_scores, axis=1)
    base_probability, entropy = _top_probability_and_entropy(base)
    top_indices = np.argsort(-base, axis=1, kind="stable")[:, :width]
    score_delta = np.take_along_axis(
        np.abs(proposed - base),
        top_indices,
        axis=1,
    )
    generic = np.column_stack(
        (
            _top_margin(base),
            base_probability,
            entropy,
            (proposed_top != base_top).astype(np.float32),
            (
                signal_scores[rows, proposed_top]
                - signal_scores[rows, base_top]
            ),
            (
                signal_scores[rows, signal_top]
                - np.partition(signal_scores, -2, axis=1)[:, -2]
            ),
            np.mean(score_delta, axis=1),
            np.max(score_delta, axis=1),
        )
    ).astype(np.float32, copy=False)
    features = np.concatenate((generic, signal.row_features), axis=1)
    if not np.isfinite(features).all():
        raise FloatingPointError("router features contain NaN or Inf")
    return features, PROPOSAL_FEATURE_NAMES + signal.feature_names


def score_multiset_correction_audit(
    base_scores: Any,
    proposed_scores: Any,
    routed_scores: Any,
    route_mask: Any,
    *,
    top_k: int,
    maximum_route_fraction: float,
) -> dict[str, float | int | bool]:
    base = _matrix(base_scores, label="multiset audit base")
    proposed = _matrix(proposed_scores, label="multiset audit proposal")
    routed = _matrix(routed_scores, label="multiset audit routed")
    rows = np.asarray(route_mask, dtype=bool)
    if (
        proposed.shape != base.shape
        or routed.shape != base.shape
        or rows.shape != (base.shape[0],)
    ):
        raise ValueError("multiset audit arrays must align")
    width = int(top_k)
    if width != top_k or not 1 <= width <= base.shape[1]:
        raise ValueError("multiset audit top_k must be a valid integer")
    fraction = float(maximum_route_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("maximum route fraction must be in [0, 1]")

    top_indices = np.argsort(-base, axis=1, kind="stable")[:, :width]
    top_mask = np.zeros(base.shape, dtype=bool)
    np.put_along_axis(top_mask, top_indices, True, axis=1)
    maximum_rows = math.floor(base.shape[0] * fraction)
    outside_exact = bool(
        np.array_equal(proposed[~top_mask], base[~top_mask])
    )
    proposal_multisets_exact = bool(
        np.array_equal(
            np.sort(proposed, axis=1),
            np.sort(base, axis=1),
        )
    )
    routed_multisets_exact = bool(
        np.array_equal(
            np.sort(routed, axis=1),
            np.sort(base, axis=1),
        )
    )
    unrouted_exact = bool(np.array_equal(routed[~rows], base[~rows]))
    routed_matches = bool(np.array_equal(routed[rows], proposed[rows]))
    routed_count = int(np.sum(rows))
    passed = bool(
        outside_exact
        and proposal_multisets_exact
        and routed_multisets_exact
        and unrouted_exact
        and routed_matches
        and routed_count <= maximum_rows
    )
    return {
        "passed": passed,
        "topk_outside_exact": outside_exact,
        "proposal_score_multisets_exact": proposal_multisets_exact,
        "routed_score_multisets_exact": routed_multisets_exact,
        "unrouted_rows_exact": unrouted_exact,
        "routed_rows_match_proposal": routed_matches,
        "routed_rows": routed_count,
        "maximum_routed_rows": maximum_rows,
        "route_fraction": float(np.mean(rows)),
        "top_k": width,
    }


def _lookup_sorted(
    keys: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(keys, query)
    clipped = np.minimum(indices, max(keys.size - 1, 0))
    valid = (indices < keys.size) & (keys[clipped] == query)
    primary_output = np.zeros(query.shape, dtype=primary.dtype)
    secondary_output = np.full(query.shape, -np.inf, dtype=secondary.dtype)
    primary_output[valid] = primary[clipped[valid]]
    secondary_output[valid] = secondary[clipped[valid]]
    return primary_output, secondary_output


def _top_margin(scores: np.ndarray) -> np.ndarray:
    top_two = np.partition(scores, -2, axis=1)[:, -2:]
    return np.asarray(
        np.max(top_two, axis=1) - np.min(top_two, axis=1),
        dtype=np.float32,
    )


def _top_probability_and_entropy(
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shifted = scores.astype(np.float64) - np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    top_probability = np.max(probabilities, axis=1).astype(np.float32)
    entropy = (
        -np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-12)),
            axis=1,
        )
        / math.log(scores.shape[1])
    ).astype(np.float32)
    return top_probability, entropy


def _matrix(values: Any, *, label: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError(f"{label} must be a non-empty matrix")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{label} must be numeric")
    if not np.isfinite(matrix).all():
        raise FloatingPointError(f"{label} contains NaN or Inf")
    return matrix


def _integer_matrix(values: Any, *, label: str) -> np.ndarray:
    matrix = _matrix(values, label=label)
    if not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError(f"{label} must contain integers")
    return matrix


def _vector(values: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1:
        raise ValueError(f"{label} must be one dimensional")
    if not np.issubdtype(vector.dtype, np.number):
        raise TypeError(f"{label} must be numeric")
    if not np.isfinite(vector).all():
        raise FloatingPointError(f"{label} contains NaN or Inf")
    return vector
