from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class TimeRampTrial:
    power: float
    prefix_mrr: float
    prefix_delta: float
    slice_mrrs: tuple[float, float]
    slice_deltas: tuple[float, float]
    mean_weight: float
    eligible: bool


@dataclass(frozen=True)
class TimeRampSelection:
    selected_power: float | None
    selection_rows: tuple[int, int]
    forward_rows: tuple[int, int]
    forward_metrics_read: bool
    trials: tuple[TimeRampTrial, ...]


@dataclass(frozen=True)
class TimeRampGateResult:
    passed: bool
    baseline_full_mrr: float
    candidate_full_mrr: float
    full_delta: float
    baseline_slice_mrrs: tuple[float, ...]
    candidate_slice_mrrs: tuple[float, ...]
    slice_deltas: tuple[float, ...]


def time_progress(
    query_times: np.ndarray,
    *,
    minimum_time: float | None = None,
    maximum_time: float | None = None,
) -> np.ndarray:
    """Normalize a complete query horizon to a finite [0, 1] time axis."""

    times = np.asarray(query_times, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("query times must be one-dimensional")
    if not np.all(np.isfinite(times)):
        raise ValueError("query times must be finite")
    if (minimum_time is None) != (maximum_time is None):
        raise ValueError("minimum and maximum time must be provided together")
    if minimum_time is not None and maximum_time is not None:
        minimum = float(minimum_time)
        maximum = float(maximum_time)
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError("time bounds must be finite")
        if maximum < minimum:
            raise ValueError("maximum time cannot precede minimum time")
    elif times.size:
        minimum = float(times.min())
        maximum = float(times.max())
    else:
        minimum = 0.0
        maximum = 0.0
    if times.size == 0:
        return np.zeros(0, dtype=np.float64)
    span = maximum - minimum
    if span <= 0.0:
        return np.zeros(times.shape, dtype=np.float64)
    return np.clip((times - minimum) / span, 0.0, 1.0)


def time_ramp_weights(
    query_times: np.ndarray,
    *,
    power: float,
    minimum_time: float | None = None,
    maximum_time: float | None = None,
) -> np.ndarray:
    """Return the frozen monotonic power-ramp expert weights."""

    exponent = float(power)
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("time-ramp power must be finite and positive")
    return np.power(
        time_progress(
            query_times,
            minimum_time=minimum_time,
            maximum_time=maximum_time,
        ),
        exponent,
    )


def blend_query_scores(
    champion_scores: np.ndarray,
    expert_scores: np.ndarray,
    query_weights: np.ndarray,
) -> np.ndarray:
    """Blend aligned experts with one bounded weight per query."""

    champion, expert = _aligned_scores(champion_scores, expert_scores)
    weights = np.asarray(query_weights, dtype=np.float64)
    if weights.shape != (champion.shape[0],):
        raise ValueError("query weights must contain one value per query")
    if not np.all(np.isfinite(weights)):
        raise ValueError("query weights must be finite")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("query weights must lie in [0, 1]")
    output = (
        (1.0 - weights[:, None]) * champion
        + weights[:, None] * expert
    )
    output[weights == 0.0] = champion[weights == 0.0]
    output[weights == 1.0] = expert[weights == 1.0]
    return output


def apply_time_ramp(
    champion_scores: np.ndarray,
    expert_scores: np.ndarray,
    query_times: np.ndarray,
    *,
    power: float | None,
    minimum_time: float | None = None,
    maximum_time: float | None = None,
) -> np.ndarray:
    """Apply a selected ramp or return the champion exact fallback."""

    champion, expert = _aligned_scores(champion_scores, expert_scores)
    if power is None:
        return champion.copy()
    return blend_query_scores(
        champion,
        expert,
        time_ramp_weights(
            query_times,
            power=power,
            minimum_time=minimum_time,
            maximum_time=maximum_time,
        ),
    )


def select_time_ramp_on_prefix(
    champion_scores: np.ndarray,
    expert_scores: np.ndarray,
    query_times: np.ndarray,
    *,
    powers: tuple[float, ...],
    first_slice_stop: int,
    selection_stop: int,
    minimum_prefix_delta: float,
) -> TimeRampSelection:
    """Select a frozen ramp without computing any forward-row metric."""

    champion, expert = _aligned_scores(champion_scores, expert_scores)
    row_count = champion.shape[0]
    if not 0 < first_slice_stop < selection_stop <= row_count:
        raise ValueError("invalid time-ramp selection boundaries")
    if not powers:
        raise ValueError("at least one time-ramp power is required")
    if not np.isfinite(minimum_prefix_delta):
        raise ValueError("minimum prefix delta must be finite")

    baseline_first = _mrr(champion[:first_slice_stop])
    baseline_second = _mrr(
        champion[first_slice_stop:selection_stop]
    )
    baseline_prefix = _mrr(champion[:selection_stop])
    trials: list[TimeRampTrial] = []
    for power in powers:
        weights = time_ramp_weights(query_times, power=power)
        candidate_prefix = blend_query_scores(
            champion[:selection_stop],
            expert[:selection_stop],
            weights[:selection_stop],
        )
        first_mrr = _mrr(candidate_prefix[:first_slice_stop])
        second_mrr = _mrr(candidate_prefix[first_slice_stop:])
        prefix_mrr = _mrr(candidate_prefix)
        slice_deltas = (
            first_mrr - baseline_first,
            second_mrr - baseline_second,
        )
        prefix_delta = prefix_mrr - baseline_prefix
        trials.append(
            TimeRampTrial(
                power=float(power),
                prefix_mrr=prefix_mrr,
                prefix_delta=prefix_delta,
                slice_mrrs=(first_mrr, second_mrr),
                slice_deltas=slice_deltas,
                mean_weight=float(weights[:selection_stop].mean()),
                eligible=bool(
                    slice_deltas[0] >= 0.0
                    and slice_deltas[1] >= 0.0
                    and prefix_delta >= minimum_prefix_delta
                ),
            )
        )
    eligible = [trial for trial in trials if trial.eligible]
    selected = (
        max(eligible, key=lambda trial: (trial.prefix_mrr, trial.power))
        if eligible
        else None
    )
    return TimeRampSelection(
        selected_power=None if selected is None else selected.power,
        selection_rows=(0, selection_stop),
        forward_rows=(selection_stop, row_count),
        forward_metrics_read=False,
        trials=tuple(trials),
    )


def passes_time_ramp_gate(
    champion_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    slice_stops: tuple[int, ...],
    minimum_full_delta: float,
) -> TimeRampGateResult:
    """Require minimum full gain and non-decreasing chronological slices."""

    champion, candidate = _aligned_scores(champion_scores, candidate_scores)
    if not np.isfinite(minimum_full_delta):
        raise ValueError("minimum full delta must be finite")
    boundaries = (0, *slice_stops, champion.shape[0])
    if any(
        start >= stop
        for start, stop in pairwise(boundaries)
    ):
        raise ValueError("time-ramp gate slices must be strictly increasing")
    baseline_slices = tuple(
        _mrr(champion[start:stop])
        for start, stop in pairwise(boundaries)
    )
    candidate_slices = tuple(
        _mrr(candidate[start:stop])
        for start, stop in pairwise(boundaries)
    )
    slice_deltas = tuple(
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(
            baseline_slices,
            candidate_slices,
            strict=True,
        )
    )
    baseline_full = _mrr(champion)
    candidate_full = _mrr(candidate)
    full_delta = candidate_full - baseline_full
    return TimeRampGateResult(
        passed=bool(
            full_delta >= minimum_full_delta
            and all(delta >= 0.0 for delta in slice_deltas)
        ),
        baseline_full_mrr=baseline_full,
        candidate_full_mrr=candidate_full,
        full_delta=full_delta,
        baseline_slice_mrrs=baseline_slices,
        candidate_slice_mrrs=candidate_slices,
        slice_deltas=slice_deltas,
    )


def _aligned_scores(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] < 2:
        raise ValueError(
            "expert scores must share a query-by-candidate shape"
        )
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("expert scores must be finite")
    return left, right


def _mrr(scores: np.ndarray) -> float:
    if scores.shape[0] == 0:
        return 0.0
    ranks = 1 + np.sum(scores[:, 1:] > scores[:, :1], axis=1)
    return float(np.mean(1.0 / ranks))
