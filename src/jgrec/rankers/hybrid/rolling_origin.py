from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RollingOriginFold:
    index: int
    train_rows: tuple[int, int]
    score_rows: tuple[int, int]
    role: Literal["selection", "gate"]


@dataclass(frozen=True)
class RollingOriginTimeBoundary:
    fold_index: int
    train_time_min: int
    train_time_max: int
    score_time_min: int
    score_time_max: int
    equal_origin_timestamp: bool


@dataclass(frozen=True)
class RollingOriginTrial:
    name: str
    fold_deltas: tuple[float, ...]
    mean_delta: float
    worst_fold_delta: float
    eligible: bool


@dataclass(frozen=True)
class RollingOriginSelection:
    selected_name: str | None
    selection_fold_count: int
    forward_metrics_read: bool
    trials: tuple[RollingOriginTrial, ...]


@dataclass(frozen=True)
class RollingOriginGateResult:
    passed: bool
    selection_fold_deltas: tuple[float, ...]
    forward_delta: float
    all_fold_deltas: tuple[float, ...]
    overall_mean_delta: float


def sliding_rolling_origin_folds(
    *,
    row_count: int,
    train_window_rows: int,
    score_rows: int,
    fold_count: int,
    step_rows: int,
    selection_fold_count: int,
) -> tuple[RollingOriginFold, ...]:
    """Build fixed-width sliding histories followed by disjoint score ranges."""

    total = int(row_count)
    train_width = int(train_window_rows)
    score_width = int(score_rows)
    count = int(fold_count)
    step = int(step_rows)
    selection_count = int(selection_fold_count)
    if total <= 0 or train_width <= 0 or score_width <= 0:
        raise ValueError("rolling-origin row counts must be positive")
    if count < 2:
        raise ValueError("rolling-origin requires at least two folds")
    if not 0 < selection_count < count:
        raise ValueError(
            "selection folds must leave at least one independent gate fold"
        )
    if step < score_width:
        raise ValueError("rolling-origin score ranges cannot overlap")

    first_score_start = total - score_width - (count - 1) * step
    if first_score_start - train_width < 0:
        raise ValueError(
            "rolling-origin history is too short for the requested folds"
        )

    folds: list[RollingOriginFold] = []
    for index in range(count):
        score_start = first_score_start + index * step
        score_stop = score_start + score_width
        train_stop = score_start
        train_start = train_stop - train_width
        folds.append(
            RollingOriginFold(
                index=index,
                train_rows=(train_start, train_stop),
                score_rows=(score_start, score_stop),
                role=(
                    "selection"
                    if index < selection_count
                    else "gate"
                ),
            )
        )
    if folds[-1].score_rows[1] != total:
        raise RuntimeError("rolling-origin final fold must end at row count")
    return tuple(folds)


def validate_rolling_origin_times(
    query_times: np.ndarray,
    folds: tuple[RollingOriginFold, ...],
) -> tuple[RollingOriginTimeBoundary, ...]:
    """Validate chronological sidecars and describe every origin boundary."""

    times = np.asarray(query_times)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("rolling-origin times must be a non-empty vector")
    if not np.issubdtype(times.dtype, np.integer):
        raise ValueError("rolling-origin times must use an integer dtype")
    if np.any(np.diff(times.astype(np.int64, copy=False)) < 0):
        raise ValueError("rolling-origin times must be non-decreasing")
    boundaries: list[RollingOriginTimeBoundary] = []
    for fold in folds:
        train_start, train_stop = fold.train_rows
        score_start, score_stop = fold.score_rows
        if (
            train_start < 0
            or train_start >= train_stop
            or train_stop > score_start
            or score_start >= score_stop
            or score_stop > times.size
        ):
            raise ValueError(
                f"rolling-origin fold {fold.index} has invalid row bounds"
            )
        train_time_max = int(times[train_stop - 1])
        score_time_min = int(times[score_start])
        if train_time_max > score_time_min:
            raise ValueError(
                f"rolling-origin fold {fold.index} crosses future time"
            )
        boundaries.append(
            RollingOriginTimeBoundary(
                fold_index=fold.index,
                train_time_min=int(times[train_start]),
                train_time_max=train_time_max,
                score_time_min=score_time_min,
                score_time_max=int(times[score_stop - 1]),
                equal_origin_timestamp=(
                    train_time_max == score_time_min
                ),
            )
        )
    return tuple(boundaries)


def select_candidate_on_rolling_origins(
    *,
    baseline_mrrs: tuple[float, ...],
    candidate_mrrs: Mapping[str, tuple[float, ...]],
    minimum_mean_delta: float,
    tie_break_order: tuple[str, ...],
) -> RollingOriginSelection:
    """Select only from supplied selection folds; no forward input exists."""

    baseline = _finite_vector(baseline_mrrs, label="baseline MRR")
    threshold = float(minimum_mean_delta)
    if not np.isfinite(threshold):
        raise ValueError("minimum rolling-origin mean delta must be finite")
    if not candidate_mrrs:
        raise ValueError("rolling-origin selection requires candidates")
    names = tuple(str(name) for name in candidate_mrrs)
    if len(set(tie_break_order)) != len(tie_break_order) or set(
        tie_break_order
    ) != set(names):
        raise ValueError(
            "rolling-origin tie-break order must contain every candidate once"
        )
    tie_rank = {
        name: index for index, name in enumerate(tie_break_order)
    }
    trials: list[RollingOriginTrial] = []
    for name, values in candidate_mrrs.items():
        candidate = _finite_vector(
            values,
            label=f"{name} MRR",
            expected_size=baseline.size,
        )
        deltas = candidate - baseline
        mean_delta = float(deltas.mean())
        worst_delta = float(deltas.min())
        trials.append(
            RollingOriginTrial(
                name=str(name),
                fold_deltas=tuple(float(value) for value in deltas),
                mean_delta=mean_delta,
                worst_fold_delta=worst_delta,
                eligible=bool(
                    worst_delta >= 0.0
                    and mean_delta >= threshold
                ),
            )
        )
    eligible = [trial for trial in trials if trial.eligible]
    selected = (
        max(
            eligible,
            key=lambda trial: (
                trial.worst_fold_delta,
                trial.mean_delta,
                tie_rank[trial.name],
            ),
        )
        if eligible
        else None
    )
    return RollingOriginSelection(
        selected_name=None if selected is None else selected.name,
        selection_fold_count=int(baseline.size),
        forward_metrics_read=False,
        trials=tuple(trials),
    )


def passes_rolling_origin_gate(
    *,
    selection_fold_deltas: tuple[float, ...],
    baseline_forward_mrr: float,
    candidate_forward_mrr: float,
    minimum_overall_mean_delta: float,
) -> RollingOriginGateResult:
    """Gate one locked candidate on a previously unseen forward fold."""

    selection = _finite_vector(
        selection_fold_deltas,
        label="selection fold delta",
    )
    baseline = float(baseline_forward_mrr)
    candidate = float(candidate_forward_mrr)
    threshold = float(minimum_overall_mean_delta)
    if not all(np.isfinite(value) for value in (
        baseline,
        candidate,
        threshold,
    )):
        raise ValueError("rolling-origin gate values must be finite")
    forward_delta = candidate - baseline
    all_deltas = np.append(selection, forward_delta)
    overall_mean = float(all_deltas.mean())
    return RollingOriginGateResult(
        passed=bool(
            forward_delta >= 0.0
            and overall_mean >= threshold
        ),
        selection_fold_deltas=tuple(
            float(value) for value in selection
        ),
        forward_delta=float(forward_delta),
        all_fold_deltas=tuple(float(value) for value in all_deltas),
        overall_mean_delta=overall_mean,
    )


def _finite_vector(
    values: tuple[float, ...],
    *,
    label: str,
    expected_size: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty vector")
    if expected_size is not None and array.size != expected_size:
        raise ValueError(f"{label} fold count differs from baseline")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array
