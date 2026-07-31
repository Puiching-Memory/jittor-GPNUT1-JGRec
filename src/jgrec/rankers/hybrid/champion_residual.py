from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChampionResidualRoutingResult:
    scores: np.ndarray
    use_residual: np.ndarray
    switch_gain: np.ndarray
    proposed_top1: np.ndarray


def champion_hard_negative_indices(
    champion_scores: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    """Select the champion's strongest negatives, excluding positive column 0."""

    scores = _score_matrix(champion_scores)
    width = _validated_top_k(
        top_k,
        candidate_count=scores.shape[1],
        maximum=scores.shape[1] - 1,
    )
    negative_order = np.argsort(
        -scores[:, 1:],
        axis=1,
        kind="stable",
    )
    return negative_order[:, :width].astype(np.int64, copy=False) + 1


def lambda_mrr_pair_weights(
    champion_scores: np.ndarray,
    hard_negative_indices: np.ndarray,
) -> np.ndarray:
    """Return static reciprocal-rank swap weights for positive/negative pairs."""

    scores = _score_matrix(champion_scores)
    negatives = np.asarray(hard_negative_indices)
    if (
        negatives.ndim != 2
        or negatives.shape[0] != scores.shape[0]
        or negatives.shape[1] < 1
        or negatives.dtype.kind not in "iu"
    ):
        raise ValueError(
            "hard negative indices must contain one integer matrix row "
            "per query"
        )
    if np.any(negatives <= 0) or np.any(negatives >= scores.shape[1]):
        raise ValueError("hard negative indices must exclude positive column 0")
    if np.any(
        np.sort(negatives, axis=1)[:, 1:]
        == np.sort(negatives, axis=1)[:, :-1]
    ):
        raise ValueError("hard negative indices must be unique within each row")

    positive_scores = scores[:, :1]
    positive_ranks = 1 + np.sum(
        scores > positive_scores,
        axis=1,
        dtype=np.int32,
    )
    negative_scores = np.take_along_axis(scores, negatives, axis=1)
    negative_ranks = 1 + np.sum(
        scores[:, :, None] > negative_scores[:, None, :],
        axis=1,
        dtype=np.int32,
    )
    weights = np.abs(
        1.0 / positive_ranks[:, None]
        - 1.0 / negative_ranks
    )
    return weights.astype(np.float32, copy=False)


def lambda_mrr_pairwise_loss(
    positive_logits: np.ndarray,
    negative_logits: np.ndarray,
    pair_weights: np.ndarray,
) -> float:
    """Evaluate the frozen weighted pairwise logistic objective in NumPy."""

    positive = np.asarray(positive_logits, dtype=np.float64)
    negatives = np.asarray(negative_logits, dtype=np.float64)
    weights = np.asarray(pair_weights, dtype=np.float64)
    if (
        positive.ndim != 1
        or negatives.ndim != 2
        or negatives.shape[0] != positive.shape[0]
        or weights.shape != negatives.shape
    ):
        raise ValueError(
            "pairwise loss requires [queries] positives and aligned "
            "[queries, negatives] logits and weights"
        )
    if (
        not np.all(np.isfinite(positive))
        or not np.all(np.isfinite(negatives))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or np.sum(weights) <= 0.0
    ):
        raise ValueError("pairwise logits and non-negative weights must be finite")
    margins = positive[:, None] - negatives
    losses = np.logaddexp(0.0, -margins)
    return float(np.sum(losses * weights) / np.sum(weights))


def route_champion_topk_residual(
    champion_scores: np.ndarray,
    residual_scores: np.ndarray,
    *,
    top_k: int,
    minimum_switch_gain: float,
) -> ChampionResidualRoutingResult:
    """Apply only confident top1 switches while preserving champion score mass."""

    champion = _score_matrix(champion_scores)
    residual = np.asarray(residual_scores, dtype=np.float32)
    if residual.shape != champion.shape or not np.all(np.isfinite(residual)):
        raise ValueError("residual scores must be finite and align with champion")
    width = _validated_top_k(
        top_k,
        candidate_count=champion.shape[1],
        maximum=champion.shape[1],
    )
    if (
        not np.isfinite(minimum_switch_gain)
        or minimum_switch_gain < 0.0
    ):
        raise ValueError("minimum switch gain must be finite and non-negative")

    champion_order = np.argsort(
        -champion,
        axis=1,
        kind="stable",
    )
    top_indices = champion_order[:, :width]
    top_values = np.take_along_axis(champion, top_indices, axis=1)
    top_residual = np.take_along_axis(residual, top_indices, axis=1)
    adjusted = np.log(
        np.maximum(top_values, np.finfo(np.float32).tiny)
    ) + top_residual
    residual_order = np.argsort(
        -adjusted,
        axis=1,
        kind="stable",
    )
    proposed_order = np.take_along_axis(
        top_indices,
        residual_order,
        axis=1,
    )
    proposed_top1 = proposed_order[:, 0]
    row_indices = np.arange(champion.shape[0])
    switch_gain = (
        np.take_along_axis(adjusted, residual_order[:, :1], axis=1)[:, 0]
        - adjusted[:, 0]
    )

    proposed = champion.copy()
    proposed[row_indices[:, None], proposed_order] = top_values
    changed = np.any(proposed != champion, axis=1)
    use_residual = (
        (proposed_top1 != top_indices[:, 0])
        & (switch_gain >= minimum_switch_gain)
        & changed
    )
    routed = champion.copy()
    routed[use_residual] = proposed[use_residual]
    return ChampionResidualRoutingResult(
        scores=routed,
        use_residual=use_residual,
        switch_gain=switch_gain.astype(np.float32, copy=False),
        proposed_top1=proposed_top1,
    )


def _score_matrix(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("champion scores must be query-by-candidate")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("champion scores must be finite and non-negative")
    if np.any(scores.sum(axis=1) <= 0.0):
        raise ValueError("champion scores must have positive row mass")
    return scores


def _validated_top_k(
    top_k: int,
    *,
    candidate_count: int,
    maximum: int,
) -> int:
    width = int(top_k)
    if width != top_k or not 1 <= width <= maximum:
        raise ValueError(
            f"top_k must be between 1 and {maximum} "
            f"for {candidate_count} candidates"
        )
    return width
