from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_METRIC_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ConservativeWindowCandidate:
    alpha: float
    prefix_mrr: float
    slice_mrrs: tuple[float, float]
    prefix_delta: float
    slice_deltas: tuple[float, float]
    eligible: bool


@dataclass(frozen=True)
class ConservativeWindowSelection:
    selected_alpha: float | None
    baseline_prefix_mrr: float
    baseline_slice_mrrs: tuple[float, float]
    minimum_prefix_delta: float
    forward_metrics_read: bool
    candidates: tuple[ConservativeWindowCandidate, ...]


@dataclass(frozen=True)
class ConservativeWindowGate:
    passed: bool
    selected_alpha: float
    baseline_full_mrr: float
    candidate_full_mrr: float
    full_delta: float
    baseline_slice_mrrs: tuple[float, float, float]
    candidate_slice_mrrs: tuple[float, float, float]
    slice_deltas: tuple[float, float, float]
    minimum_full_delta: float


def conservative_window_scores(
    champion_scores: np.ndarray,
    window_scores: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Shrink a fixed window candidate residual toward champion scores."""

    champion, window = _validated_score_pair(
        champion_scores,
        window_scores,
        require_finite=True,
    )
    weight = _validated_alpha(alpha, allow_zero=True)
    if weight == 0.0:
        return champion.copy()
    if weight == 1.0:
        return window.copy()
    return champion + weight * (window - champion)


def select_conservative_window_blend_on_prefix(
    champion_scores: np.ndarray,
    window_scores: np.ndarray,
    *,
    alphas: tuple[float, ...],
    first_slice_stop: int,
    selection_stop: int,
    minimum_prefix_delta: float,
) -> ConservativeWindowSelection:
    """Select a fixed residual weight without reading forward score rows."""

    champion, window = _validated_score_pair(
        champion_scores,
        window_scores,
        require_finite=False,
    )
    if not 1 <= first_slice_stop < selection_stop < champion.shape[0]:
        raise ValueError(
            "slice boundaries must leave two visible slices and forward rows"
        )
    threshold = float(minimum_prefix_delta)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum prefix delta must be finite and non-negative")
    weights = _validated_alpha_grid(alphas)
    champion_prefix = champion[:selection_stop]
    window_prefix = window[:selection_stop]
    if not (
        np.all(np.isfinite(champion_prefix))
        and np.all(np.isfinite(window_prefix))
    ):
        raise ValueError("selection prefix scores must be finite")

    baseline_slices = (
        _ranking_mrr_tie_neutral(champion_prefix[:first_slice_stop]),
        _ranking_mrr_tie_neutral(champion_prefix[first_slice_stop:]),
    )
    baseline_prefix = _ranking_mrr_tie_neutral(champion_prefix)
    reports: list[ConservativeWindowCandidate] = []
    for alpha in weights:
        candidate = conservative_window_scores(
            champion_prefix,
            window_prefix,
            alpha=alpha,
        )
        slice_mrrs = (
            _ranking_mrr_tie_neutral(candidate[:first_slice_stop]),
            _ranking_mrr_tie_neutral(candidate[first_slice_stop:]),
        )
        prefix_mrr = _ranking_mrr_tie_neutral(candidate)
        slice_deltas = tuple(
            float(actual - baseline)
            for actual, baseline in zip(
                slice_mrrs,
                baseline_slices,
                strict=True,
            )
        )
        prefix_delta = float(prefix_mrr - baseline_prefix)
        reports.append(
            ConservativeWindowCandidate(
                alpha=alpha,
                prefix_mrr=prefix_mrr,
                slice_mrrs=slice_mrrs,
                prefix_delta=prefix_delta,
                slice_deltas=slice_deltas,
                eligible=bool(
                    all(
                        delta + _METRIC_TOLERANCE >= 0.0
                        for delta in slice_deltas
                    )
                    and prefix_delta + _METRIC_TOLERANCE >= threshold
                ),
            )
        )
    eligible = [report for report in reports if report.eligible]
    selected = (
        max(
            eligible,
            key=lambda report: (
                report.prefix_mrr,
                -report.alpha,
            ),
        )
        if eligible
        else None
    )
    return ConservativeWindowSelection(
        selected_alpha=None if selected is None else selected.alpha,
        baseline_prefix_mrr=baseline_prefix,
        baseline_slice_mrrs=baseline_slices,
        minimum_prefix_delta=threshold,
        forward_metrics_read=False,
        candidates=tuple(reports),
    )


def evaluate_conservative_window_gate(
    champion_scores: np.ndarray,
    window_scores: np.ndarray,
    *,
    selected_alpha: float,
    first_slice_stop: int,
    selection_stop: int,
    minimum_full_delta: float,
) -> ConservativeWindowGate:
    """Evaluate one locked non-zero weight on all three time slices."""

    champion, window = _validated_score_pair(
        champion_scores,
        window_scores,
        require_finite=True,
    )
    if not 1 <= first_slice_stop < selection_stop < champion.shape[0]:
        raise ValueError(
            "slice boundaries must leave two visible slices and forward rows"
        )
    alpha = _validated_alpha(selected_alpha, allow_zero=False)
    threshold = float(minimum_full_delta)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("minimum full delta must be finite and non-negative")
    candidate = conservative_window_scores(
        champion,
        window,
        alpha=alpha,
    )
    slices = (
        slice(0, first_slice_stop),
        slice(first_slice_stop, selection_stop),
        slice(selection_stop, champion.shape[0]),
    )
    baseline_slice_mrrs = tuple(
        _ranking_mrr_tie_neutral(champion[part])
        for part in slices
    )
    candidate_slice_mrrs = tuple(
        _ranking_mrr_tie_neutral(candidate[part])
        for part in slices
    )
    slice_deltas = tuple(
        float(actual - baseline)
        for actual, baseline in zip(
            candidate_slice_mrrs,
            baseline_slice_mrrs,
            strict=True,
        )
    )
    baseline_full = _ranking_mrr_tie_neutral(champion)
    candidate_full = _ranking_mrr_tie_neutral(candidate)
    full_delta = float(candidate_full - baseline_full)
    return ConservativeWindowGate(
        passed=bool(
            all(
                delta + _METRIC_TOLERANCE >= 0.0
                for delta in slice_deltas
            )
            and full_delta + _METRIC_TOLERANCE >= threshold
        ),
        selected_alpha=alpha,
        baseline_full_mrr=baseline_full,
        candidate_full_mrr=candidate_full,
        full_delta=full_delta,
        baseline_slice_mrrs=baseline_slice_mrrs,
        candidate_slice_mrrs=candidate_slice_mrrs,
        slice_deltas=slice_deltas,
        minimum_full_delta=threshold,
    )


def _validated_score_pair(
    champion_scores: np.ndarray,
    window_scores: np.ndarray,
    *,
    require_finite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    champion = np.asarray(champion_scores, dtype=np.float64)
    window = np.asarray(window_scores, dtype=np.float64)
    if champion.shape != window.shape:
        raise ValueError("champion and window scores must have the same shape")
    if champion.ndim != 2 or champion.shape[0] < 1 or champion.shape[1] < 2:
        raise ValueError(
            "scores must contain at least one query and two candidates"
        )
    if require_finite and not (
        np.all(np.isfinite(champion))
        and np.all(np.isfinite(window))
    ):
        raise ValueError("scores must be finite")
    return champion, window


def _validated_alpha(alpha: float, *, allow_zero: bool) -> float:
    weight = float(alpha)
    lower_bound_satisfied = weight >= 0.0 if allow_zero else weight > 0.0
    if (
        not np.isfinite(weight)
        or not lower_bound_satisfied
        or weight > 1.0
    ):
        qualifier = "zero and one" if allow_zero else "zero-exclusive and one"
        raise ValueError(f"alpha must be between {qualifier}")
    return weight


def _validated_alpha_grid(alphas: tuple[float, ...]) -> tuple[float, ...]:
    if not alphas:
        raise ValueError("alpha grid must not be empty")
    weights = tuple(
        _validated_alpha(alpha, allow_zero=False)
        for alpha in alphas
    )
    if len(set(weights)) != len(weights):
        raise ValueError("alpha grid must contain unique values")
    return weights


def _ranking_mrr_tie_neutral(scores: np.ndarray) -> float:
    positive = scores[:, 0:1]
    negatives = scores[:, 1:]
    greater = np.sum(negatives > positive, axis=1)
    equal = np.sum(negatives == positive, axis=1)
    average_ranks = 1.0 + greater + 0.5 * equal
    return float(np.mean(1.0 / average_ranks))
