from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BlendScanResult:
    reference_weight: float
    mrr: dict[str, float]
    weights_tested: int


@dataclass(frozen=True)
class RankBlendScanResult:
    reference_weight: float
    method: str
    selection_mrr: float
    mrr: dict[str, float]
    weights_tested: int


@dataclass(frozen=True)
class HighWeightBlendScanResult:
    primary_weight: float
    selection_mrr: float
    mrr: dict[str, float]
    weights_tested: int


@dataclass(frozen=True)
class SetwiseModelBlendSelection:
    model_name: str
    primary_weight: float
    selection_mrr: float
    models_tested: int
    weights_tested: int


def inclusive_weight_grid(
    start: float,
    stop: float,
    step: float,
) -> tuple[float, ...]:
    values = np.asarray([start, stop, step], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("weight grid values must be finite")
    if not 0.0 <= start <= stop <= 1.0:
        raise ValueError("weight grid must stay between zero and one")
    if step <= 0.0:
        raise ValueError("weight grid step must be positive")
    interval_count = (stop - start) / step
    rounded_count = round(interval_count)
    if not np.isclose(interval_count, rounded_count, atol=1e-10, rtol=0.0):
        raise ValueError("weight grid step must divide the interval exactly")
    return tuple(
        round(start + index * step, 12)
        for index in range(rounded_count + 1)
    )


def authorized_setwise_weight(evaluation: Mapping[str, Any]) -> float:
    if (
        evaluation.get("status") != "passed"
        or not evaluation.get("gate_passed")
        or not evaluation.get("package_authorized")
        or evaluation.get("winner") != "setwise"
    ):
        raise ValueError("evaluation did not authorize a Setwise package")
    setwise = evaluation.get("setwise")
    if not isinstance(setwise, Mapping) or not setwise.get("gate_passed"):
        raise ValueError("selected Setwise candidate did not pass its gate")
    weight = float(setwise.get("selected_weight", np.nan))
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("selected Setwise weight must be between zero and one")
    return weight


def ranking_mrr_slices(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("ranking scores must contain at least two queries and two candidates")
    midpoint = values.shape[0] // 2
    return {
        "full": _ranking_mrr(values),
        "early": _ranking_mrr(values[:midpoint]),
        "late": _ranking_mrr(values[midpoint:]),
    }


def ranking_mrr_three_slices(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 2:
        raise ValueError(
            "ranking scores must contain at least three queries and two candidates"
        )
    base_size, remainder = divmod(values.shape[0], 3)
    sizes = tuple(base_size + (1 if index < remainder else 0) for index in range(3))
    first_stop = sizes[0]
    second_stop = first_stop + sizes[1]
    parts = (
        values[:first_stop],
        values[first_stop:second_stop],
        values[second_stop:],
    )
    return {
        "full": _ranking_mrr_tie_neutral(values),
        **{
            f"slice_{index}": _ranking_mrr_tie_neutral(part)
            for index, part in enumerate(parts)
        },
    }


def uniform_rank_average(score_models: tuple[np.ndarray, ...]) -> np.ndarray:
    """Average query-local percentile ranks with equal model weights."""
    if len(score_models) < 2:
        raise ValueError("rank averaging requires at least two score models")
    values = tuple(np.asarray(scores, dtype=np.float64) for scores in score_models)
    expected_shape = values[0].shape
    if (
        len(expected_shape) != 2
        or expected_shape[0] < 1
        or expected_shape[1] < 2
    ):
        raise ValueError(
            "score models must contain queries with at least two candidates"
        )
    if any(scores.shape != expected_shape for scores in values[1:]):
        raise ValueError("rank-average score models must have the same shape")
    if any(not np.all(np.isfinite(scores)) for scores in values):
        raise ValueError("rank-average score models must contain only finite scores")

    rows = np.arange(expected_shape[0])[:, None]
    candidate_ranks = np.arange(expected_shape[1], dtype=np.float64)[None, :]
    averaged = np.zeros(expected_shape, dtype=np.float64)
    for scores in values:
        order = np.argsort(-scores, axis=1, kind="stable")
        percentiles = np.empty(expected_shape, dtype=np.float64)
        percentiles[rows, order] = (
            1.0 - candidate_ranks / float(expected_shape[1] - 1)
        )
        averaged += percentiles
    return averaged / float(len(values))


def scan_probability_blend(reference: np.ndarray, alternate: np.ndarray) -> BlendScanResult:
    reference_values = np.asarray(reference, dtype=np.float64)
    alternate_values = np.asarray(alternate, dtype=np.float64)
    if reference_values.shape != alternate_values.shape:
        raise ValueError("blend inputs must have the same shape")

    best_weight = 0.0
    best_mrr: dict[str, float] | None = None
    for weight_int in range(101):
        weight = weight_int / 100.0
        blended = weight * reference_values + (1.0 - weight) * alternate_values
        mrr = ranking_mrr_slices(blended)
        if best_mrr is None or mrr["full"] > best_mrr["full"] or (
            mrr["full"] == best_mrr["full"] and weight > best_weight
        ):
            best_weight = weight
            best_mrr = mrr

    if best_mrr is None:
        raise RuntimeError("blend scan produced no result")
    return BlendScanResult(reference_weight=best_weight, mrr=best_mrr, weights_tested=101)


def scan_rank_blend_on_prefix(
    reference: np.ndarray,
    alternate: np.ndarray,
    *,
    selection_stop: int,
    method: str = "probability",
) -> RankBlendScanResult:
    reference_values = np.asarray(reference, dtype=np.float64)
    alternate_values = np.asarray(alternate, dtype=np.float64)
    if reference_values.shape != alternate_values.shape:
        raise ValueError("blend inputs must have the same shape")
    if (
        reference_values.ndim != 2
        or reference_values.shape[0] < 3
        or reference_values.shape[1] < 2
    ):
        raise ValueError(
            "blend inputs must contain at least three queries and two candidates"
        )
    if not np.all(np.isfinite(reference_values)) or not np.all(
        np.isfinite(alternate_values)
    ):
        raise ValueError("blend inputs must contain only finite scores")
    if selection_stop < 2 or selection_stop >= reference_values.shape[0]:
        raise ValueError("selection_stop must leave non-empty selection and final rows")

    normalized_reference = _normalize_blend_scores(reference_values, method)
    normalized_alternate = _normalize_blend_scores(alternate_values, method)
    best_weight = 0.0
    best_selection_mrr = -np.inf
    for weight_int in range(101):
        weight = weight_int / 100.0
        blended_selection = (
            weight * normalized_reference[:selection_stop]
            + (1.0 - weight) * normalized_alternate[:selection_stop]
        )
        selection_mrr = _ranking_mrr_tie_neutral(blended_selection)
        if selection_mrr > best_selection_mrr or (
            selection_mrr == best_selection_mrr and weight > best_weight
        ):
            best_weight = weight
            best_selection_mrr = selection_mrr

    blended = (
        best_weight * normalized_reference
        + (1.0 - best_weight) * normalized_alternate
    )
    return RankBlendScanResult(
        reference_weight=best_weight,
        method=method,
        selection_mrr=float(best_selection_mrr),
        mrr=ranking_mrr_three_slices(blended),
        weights_tested=101,
    )


def scan_high_weight_blend_on_prefix(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    selection_stop: int,
    primary_weights: tuple[float, ...] | None = None,
) -> HighWeightBlendScanResult:
    primary_values = np.asarray(primary, dtype=np.float64)
    secondary_values = np.asarray(secondary, dtype=np.float64)
    if primary_values.shape != secondary_values.shape:
        raise ValueError("blend inputs must have the same shape")
    if (
        primary_values.ndim != 2
        or primary_values.shape[0] < 3
        or primary_values.shape[1] < 2
    ):
        raise ValueError(
            "blend inputs must contain at least three queries and two candidates"
        )
    if not np.all(np.isfinite(primary_values)) or not np.all(
        np.isfinite(secondary_values)
    ):
        raise ValueError("blend inputs must contain only finite scores")
    if selection_stop < 2 or selection_stop >= primary_values.shape[0]:
        raise ValueError("selection_stop must leave non-empty forward rows")
    weights = primary_weights or inclusive_weight_grid(0.8, 1.0, 0.01)
    if not weights:
        raise ValueError("primary weight grid must not be empty")
    if not all(np.isfinite(weight) and 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("primary weights must be finite and between zero and one")

    best_weight = weights[0]
    best_selection_mrr = -np.inf
    for primary_weight in weights:
        blended_selection = (
            primary_weight * primary_values[:selection_stop]
            + (1.0 - primary_weight) * secondary_values[:selection_stop]
        )
        selection_mrr = _ranking_mrr_tie_neutral(blended_selection)
        if selection_mrr > best_selection_mrr or (
            selection_mrr == best_selection_mrr
            and primary_weight > best_weight
        ):
            best_weight = primary_weight
            best_selection_mrr = selection_mrr

    blended = (
        best_weight * primary_values
        + (1.0 - best_weight) * secondary_values
    )
    return HighWeightBlendScanResult(
        primary_weight=best_weight,
        selection_mrr=float(best_selection_mrr),
        mrr=ranking_mrr_three_slices(blended),
        weights_tested=len(weights),
    )


def select_setwise_model_blend_on_prefix(
    primary_models: Mapping[str, np.ndarray],
    secondary: np.ndarray,
    *,
    selection_stop: int,
    primary_weights: tuple[float, ...],
    model_tie_break_order: tuple[str, ...],
) -> SetwiseModelBlendSelection:
    """Select one Setwise model and blend weight without reading forward rows."""
    model_names = tuple(primary_models)
    if not model_names:
        raise ValueError("Setwise model candidates must not be empty")
    secondary_values = np.asarray(secondary, dtype=np.float64)
    if (
        secondary_values.ndim != 2
        or secondary_values.shape[0] < 3
        or secondary_values.shape[1] < 2
    ):
        raise ValueError(
            "blend inputs must contain at least three queries and two candidates"
        )
    if selection_stop < 2 or selection_stop >= secondary_values.shape[0]:
        raise ValueError("selection_stop must leave non-empty forward rows")
    if not primary_weights:
        raise ValueError("primary weight grid must not be empty")
    if not all(
        np.isfinite(weight) and 0.0 <= weight <= 1.0
        for weight in primary_weights
    ):
        raise ValueError(
            "primary weights must be finite and between zero and one"
        )
    if (
        len(model_tie_break_order) != len(set(model_tie_break_order))
        or set(model_tie_break_order) != set(model_names)
    ):
        raise ValueError(
            "model tie-break order must contain every candidate exactly once"
        )
    priority = {
        model_name: index
        for index, model_name in enumerate(model_tie_break_order)
    }

    secondary_selection = secondary_values[:selection_stop]
    if not np.all(np.isfinite(secondary_selection)):
        raise ValueError("selection inputs must contain only finite scores")
    selected_name = ""
    selected_weight = primary_weights[0]
    selected_mrr = -np.inf
    for model_name in model_names:
        primary_values = np.asarray(
            primary_models[model_name],
            dtype=np.float64,
        )
        if primary_values.shape != secondary_values.shape:
            raise ValueError("blend inputs must have the same shape")
        primary_selection = primary_values[:selection_stop]
        if not np.all(np.isfinite(primary_selection)):
            raise ValueError("selection inputs must contain only finite scores")
        model_weight = primary_weights[0]
        model_mrr = -np.inf
        for primary_weight in primary_weights:
            blended_selection = (
                primary_weight * primary_selection
                + (1.0 - primary_weight) * secondary_selection
            )
            selection_mrr = _ranking_mrr_tie_neutral(blended_selection)
            if selection_mrr > model_mrr or (
                selection_mrr == model_mrr
                and primary_weight > model_weight
            ):
                model_weight = primary_weight
                model_mrr = selection_mrr
        if (
            model_mrr > selected_mrr
            or (
                model_mrr == selected_mrr
                and model_weight > selected_weight
            )
            or (
                model_mrr == selected_mrr
                and model_weight == selected_weight
                and priority[model_name] > priority[selected_name]
            )
        ):
            selected_name = model_name
            selected_weight = model_weight
            selected_mrr = model_mrr

    return SetwiseModelBlendSelection(
        model_name=selected_name,
        primary_weight=selected_weight,
        selection_mrr=float(selected_mrr),
        models_tested=len(model_names),
        weights_tested=len(primary_weights),
    )


def _normalize_blend_scores(scores: np.ndarray, method: str) -> np.ndarray:
    if method == "probability":
        return scores
    if method == "row_zscore":
        centered = scores - np.mean(scores, axis=1, keepdims=True)
        scales = np.std(centered, axis=1, keepdims=True)
        return np.divide(
            centered,
            scales,
            out=np.zeros_like(centered),
            where=scales > 0.0,
        )
    if method == "rank_percentile":
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.empty_like(order)
        rows = np.arange(scores.shape[0])[:, None]
        ranks[rows, order] = np.arange(scores.shape[1])[None, :]
        return 1.0 - ranks / float(scores.shape[1] - 1)
    raise ValueError(f"unsupported rank blend method: {method}")


def _ranking_mrr(scores: np.ndarray) -> float:
    ranks = 1 + (scores[:, 1:] > scores[:, 0:1]).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _ranking_mrr_tie_neutral(scores: np.ndarray) -> float:
    positive = scores[:, 0:1]
    negatives = scores[:, 1:]
    greater = np.sum(negatives > positive, axis=1)
    equal = np.sum(negatives == positive, axis=1)
    average_ranks = 1.0 + greater + 0.5 * equal
    return float(np.mean(1.0 / average_ranks))
