from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class SubsetPrefixCandidate:
    experts: tuple[str, ...]
    selection_mrr: float


@dataclass(frozen=True)
class UniformSubsetSelection:
    selected_experts: tuple[str, ...]
    selection_mrr: float
    candidates: tuple[SubsetPrefixCandidate, ...]


@dataclass(frozen=True)
class TemporalCandidatePrefixReport:
    name: str
    selection_mrr: float
    slice_0_mrr: float
    slice_1_mrr: float
    eligible: bool
    complexity: int


@dataclass(frozen=True)
class TemporalRobustSelection:
    selected_name: str
    selection_mrr: float
    candidates: tuple[TemporalCandidatePrefixReport, ...]


def recent_window_view(features: np.ndarray, requested_rows: int) -> np.ndarray:
    """Return an exact chronological tail without copying the source array."""
    if requested_rows <= 0:
        raise ValueError("requested window rows must be positive")
    available_rows = int(features.shape[0])
    if requested_rows > available_rows:
        raise ValueError(
            f"requested {requested_rows} window rows but cache has only "
            f"{available_rows}"
        )
    return features[available_rows - requested_rows :]


def normalized_exponential_recency_weights(
    *,
    row_count: int,
    half_life_rows: int,
) -> np.ndarray:
    """Return oldest-to-newest exponential weights normalized to mean one."""
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if half_life_rows <= 0:
        raise ValueError("half_life_rows must be positive")
    age = np.arange(row_count - 1, -1, -1, dtype=np.float64)
    raw = np.exp2(-age / float(half_life_rows))
    normalized = raw / np.mean(raw, dtype=np.float64)
    weights = normalized.astype(np.float32)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise FloatingPointError("recency weights must be finite and positive")
    return weights


def select_uniform_subset_on_prefix(
    expert_probabilities: Mapping[str, np.ndarray],
    secondary_probabilities: np.ndarray,
    *,
    selection_stop: int,
    expert_weight: float,
    expert_order: tuple[str, ...],
) -> UniformSubsetSelection:
    """Select a uniform expert subset using only the chronological prefix."""
    secondary = np.asarray(secondary_probabilities, dtype=np.float64)
    _validate_score_shape(secondary)
    if selection_stop < 2 or selection_stop >= secondary.shape[0]:
        raise ValueError("selection_stop must leave non-empty forward rows")
    if not np.isfinite(expert_weight) or not 0.0 <= expert_weight <= 1.0:
        raise ValueError("expert_weight must be finite and between zero and one")
    if (
        not expert_order
        or len(expert_order) != len(set(expert_order))
        or set(expert_order) != set(expert_probabilities)
    ):
        raise ValueError(
            "expert_order must contain every expert exactly once"
        )

    secondary_prefix = secondary[:selection_stop]
    if not np.all(np.isfinite(secondary_prefix)):
        raise ValueError("selection inputs must contain only finite scores")
    prefix_by_expert: dict[str, np.ndarray] = {}
    for name in expert_order:
        values = np.asarray(expert_probabilities[name], dtype=np.float64)
        if values.shape != secondary.shape:
            raise ValueError("expert and secondary scores must have the same shape")
        prefix = values[:selection_stop]
        if not np.all(np.isfinite(prefix)):
            raise ValueError("selection inputs must contain only finite scores")
        prefix_by_expert[name] = prefix

    candidates: list[SubsetPrefixCandidate] = []
    selected_experts: tuple[str, ...] | None = None
    selected_mrr = -np.inf
    for subset_size in range(1, len(expert_order) + 1):
        for subset in combinations(expert_order, subset_size):
            expert_mean = np.mean(
                np.stack([prefix_by_expert[name] for name in subset], axis=0),
                axis=0,
            )
            blended = (
                expert_weight * expert_mean
                + (1.0 - expert_weight) * secondary_prefix
            )
            selection_mrr = _ranking_mrr_tie_neutral(blended)
            candidates.append(
                SubsetPrefixCandidate(
                    experts=subset,
                    selection_mrr=selection_mrr,
                )
            )
            if selection_mrr > selected_mrr:
                selected_experts = subset
                selected_mrr = selection_mrr

    if selected_experts is None:
        raise RuntimeError("uniform subset selection produced no candidate")
    return UniformSubsetSelection(
        selected_experts=selected_experts,
        selection_mrr=float(selected_mrr),
        candidates=tuple(candidates),
    )


def blend_expert_subset(
    expert_probabilities: Mapping[str, np.ndarray],
    secondary_probabilities: np.ndarray,
    *,
    selected_experts: tuple[str, ...],
    expert_weight: float,
) -> np.ndarray:
    """Build the fixed outer blend for an already selected expert subset."""
    if not selected_experts or len(selected_experts) != len(set(selected_experts)):
        raise ValueError("selected_experts must contain unique expert names")
    if not np.isfinite(expert_weight) or not 0.0 <= expert_weight <= 1.0:
        raise ValueError("expert_weight must be finite and between zero and one")
    secondary = np.asarray(secondary_probabilities, dtype=np.float64)
    _validate_score_shape(secondary)
    if not np.all(np.isfinite(secondary)):
        raise ValueError("blend inputs must contain only finite scores")
    selected: list[np.ndarray] = []
    for name in selected_experts:
        if name not in expert_probabilities:
            raise ValueError(f"selected expert is missing: {name}")
        values = np.asarray(expert_probabilities[name], dtype=np.float64)
        if values.shape != secondary.shape:
            raise ValueError("expert and secondary scores must have the same shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("blend inputs must contain only finite scores")
        selected.append(values)
    expert_mean = np.mean(np.stack(selected, axis=0), axis=0)
    return expert_weight * expert_mean + (1.0 - expert_weight) * secondary


def select_temporally_robust_candidate_on_prefix(
    candidates: Mapping[str, np.ndarray],
    champion_scores: np.ndarray,
    *,
    first_slice_stop: int,
    selection_stop: int,
    candidate_complexity: Mapping[str, int],
    candidate_order: tuple[str, ...],
) -> TemporalRobustSelection:
    """Select on two visible slices while requiring both to match champion."""
    names = tuple(candidates)
    if (
        not names
        or len(candidate_order) != len(set(candidate_order))
        or set(candidate_order) != set(names)
    ):
        raise ValueError(
            "candidate_order must contain every candidate exactly once"
        )
    if set(candidate_complexity) != set(names) or any(
        int(candidate_complexity[name]) <= 0
        for name in names
    ):
        raise ValueError(
            "candidate_complexity must contain one positive value per candidate"
        )
    champion = np.asarray(champion_scores, dtype=np.float64)
    _validate_score_shape(champion)
    if not 1 <= first_slice_stop < selection_stop < champion.shape[0]:
        raise ValueError(
            "slice boundaries must leave two visible slices and forward rows"
        )
    champion_prefix = champion[:selection_stop]
    if not np.all(np.isfinite(champion_prefix)):
        raise ValueError("selection inputs must contain only finite scores")
    champion_slice_0 = _ranking_mrr_tie_neutral(
        champion_prefix[:first_slice_stop]
    )
    champion_slice_1 = _ranking_mrr_tie_neutral(
        champion_prefix[first_slice_stop:]
    )

    reports: list[TemporalCandidatePrefixReport] = []
    selected_name = ""
    selected_mrr = -np.inf
    selected_complexity = 0
    for name in candidate_order:
        scores = np.asarray(candidates[name], dtype=np.float64)
        if scores.shape != champion.shape:
            raise ValueError("candidate and champion scores must have same shape")
        prefix = scores[:selection_stop]
        if not np.all(np.isfinite(prefix)):
            raise ValueError("selection inputs must contain only finite scores")
        slice_0_mrr = _ranking_mrr_tie_neutral(
            prefix[:first_slice_stop]
        )
        slice_1_mrr = _ranking_mrr_tie_neutral(
            prefix[first_slice_stop:]
        )
        selection_mrr = _ranking_mrr_tie_neutral(prefix)
        eligible = bool(
            slice_0_mrr + 1e-12 >= champion_slice_0
            and slice_1_mrr + 1e-12 >= champion_slice_1
        )
        complexity = int(candidate_complexity[name])
        reports.append(
            TemporalCandidatePrefixReport(
                name=name,
                selection_mrr=selection_mrr,
                slice_0_mrr=slice_0_mrr,
                slice_1_mrr=slice_1_mrr,
                eligible=eligible,
                complexity=complexity,
            )
        )
        if eligible and (
            selection_mrr > selected_mrr
            or (
                selection_mrr == selected_mrr
                and complexity < selected_complexity
            )
        ):
            selected_name = name
            selected_mrr = selection_mrr
            selected_complexity = complexity

    if not selected_name:
        raise RuntimeError("temporal robust selection found no eligible candidate")
    return TemporalRobustSelection(
        selected_name=selected_name,
        selection_mrr=float(selected_mrr),
        candidates=tuple(reports),
    )


def _validate_score_shape(scores: np.ndarray) -> None:
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 2:
        raise ValueError(
            "scores must contain at least one query and two candidates"
        )


def _ranking_mrr_tie_neutral(scores: np.ndarray) -> float:
    positive = scores[:, 0:1]
    negatives = scores[:, 1:]
    greater = np.sum(negatives > positive, axis=1)
    equal = np.sum(negatives == positive, axis=1)
    average_ranks = 1.0 + greater + 0.5 * equal
    return float(np.mean(1.0 / average_ranks))
