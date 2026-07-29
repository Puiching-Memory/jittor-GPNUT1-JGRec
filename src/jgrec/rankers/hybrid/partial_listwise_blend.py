from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

ALLOWED_AUXILIARY_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
_METRIC_TOLERANCE = 1e-15


def blend_partial_listwise(
    champion_scores: np.ndarray,
    expert_scores: np.ndarray,
    *,
    auxiliary_weight: float,
) -> np.ndarray:
    champion, expert = _validated_score_pair(
        champion_scores,
        expert_scores,
    )
    weight = _validated_weight(auxiliary_weight)
    return (1.0 - weight) * champion + weight * expert


def descending_midrank_probabilities(scores: np.ndarray) -> np.ndarray:
    values = _validated_scores(scores, label="scores")
    row_count, candidate_count = values.shape
    strengths = np.empty((row_count, candidate_count), dtype=np.float64)
    for row_index, row in enumerate(values):
        order = np.argsort(-row, kind="stable")
        sorted_values = row[order]
        start = 0
        while start < candidate_count:
            stop = start + 1
            while (
                stop < candidate_count
                and sorted_values[stop] == sorted_values[start]
            ):
                stop += 1
            first_rank = start + 1
            last_rank = stop
            midrank = 0.5 * (first_rank + last_rank)
            strengths[row_index, order[start:stop]] = (
                candidate_count + 1.0 - midrank
            )
            start = stop
    return strengths / strengths.sum(axis=1, keepdims=True)


def select_auxiliary_weight(
    *,
    expert_name: str,
    champion_slice0: np.ndarray,
    expert_slice0: np.ndarray,
    candidate_manifest_sha256: str,
    weights: Iterable[float] = ALLOWED_AUXILIARY_WEIGHTS,
) -> dict[str, Any]:
    champion, expert = _validated_score_pair(
        champion_slice0,
        expert_slice0,
    )
    if not expert_name:
        raise ValueError("expert_name must not be empty")
    if not candidate_manifest_sha256:
        raise ValueError("candidate_manifest_sha256 must not be empty")
    scan = scan_auxiliary_weights(
        champion_slice0=champion,
        expert_slice0=expert,
        weights=weights,
    )
    baseline_mrr = float(scan["baseline_mrr"])
    trials = list(scan["trials"])
    eligible = [trial for trial in trials if trial["eligible"]]
    if not eligible:
        raise ValueError("no auxiliary weight is non-decreasing on slice_0")
    winner = min(
        eligible,
        key=lambda trial: (
            -float(trial["candidate_mrr"]),
            float(trial["weight"]),
        ),
    )
    selection: dict[str, Any] = {
        "schema_version": 1,
        "expert_name": expert_name,
        "selection_slice": "slice_0",
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "selected_weight": float(winner["weight"]),
        "baseline_mrr": baseline_mrr,
        "candidate_mrr": float(winner["candidate_mrr"]),
        "delta": float(winner["delta"]),
        "tie_break": "smaller auxiliary weight",
        "trials": trials,
    }
    selection["lock_sha256"] = selection_lock_sha256(selection)
    return selection


def scan_auxiliary_weights(
    *,
    champion_slice0: np.ndarray,
    expert_slice0: np.ndarray,
    weights: Iterable[float] = ALLOWED_AUXILIARY_WEIGHTS,
) -> dict[str, Any]:
    champion, expert = _validated_score_pair(
        champion_slice0,
        expert_slice0,
    )
    frozen_weights = tuple(_validated_weight(weight) for weight in weights)
    if not frozen_weights:
        raise ValueError("at least one auxiliary weight is required")
    if len(set(frozen_weights)) != len(frozen_weights):
        raise ValueError("auxiliary weights must be unique")
    baseline_mrr = ranking_mrr(champion)
    trials = []
    for weight in frozen_weights:
        candidate_mrr = ranking_mrr(
            blend_partial_listwise(
                champion,
                expert,
                auxiliary_weight=weight,
            )
        )
        delta = candidate_mrr - baseline_mrr
        trials.append(
            {
                "weight": weight,
                "baseline_mrr": baseline_mrr,
                "candidate_mrr": candidate_mrr,
                "delta": delta,
                "eligible": bool(delta >= -_METRIC_TOLERANCE),
            }
        )
    return {
        "selection_slice": "slice_0",
        "baseline_mrr": baseline_mrr,
        "trials": trials,
        "eligible_weight_count": sum(
            bool(trial["eligible"]) for trial in trials
        ),
    }


def selection_lock_sha256(selection: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in selection.items()
        if key != "lock_sha256"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_forward_gate(
    *,
    selection: dict[str, Any],
    champion_slice0: np.ndarray,
    expert_slice0: np.ndarray,
    champion_slice1: np.ndarray,
    expert_slice1: np.ndarray,
    candidate_manifest_sha256: str,
    minimum_prefix_delta: float,
) -> dict[str, Any]:
    _validate_selection_lock(
        selection,
        candidate_manifest_sha256=candidate_manifest_sha256,
    )
    champion0, expert0 = _validated_score_pair(
        champion_slice0,
        expert_slice0,
    )
    champion1, expert1 = _validated_score_pair(
        champion_slice1,
        expert_slice1,
    )
    if champion0.shape[1] != champion1.shape[1]:
        raise ValueError("slice candidate counts differ")
    weight = _validated_weight(selection["selected_weight"])
    candidate0 = blend_partial_listwise(
        champion0,
        expert0,
        auxiliary_weight=weight,
    )
    candidate1 = blend_partial_listwise(
        champion1,
        expert1,
        auxiliary_weight=weight,
    )
    slice1_baseline_mrr = ranking_mrr(champion1)
    slice1_candidate_mrr = ranking_mrr(candidate1)
    slice1_delta = slice1_candidate_mrr - slice1_baseline_mrr
    champion_prefix = np.concatenate((champion0, champion1), axis=0)
    candidate_prefix = np.concatenate((candidate0, candidate1), axis=0)
    prefix_baseline_mrr = ranking_mrr(champion_prefix)
    prefix_candidate_mrr = ranking_mrr(candidate_prefix)
    prefix_delta = prefix_candidate_mrr - prefix_baseline_mrr
    passed = bool(
        slice1_delta >= -_METRIC_TOLERANCE
        and prefix_delta + _METRIC_TOLERANCE >= minimum_prefix_delta
    )
    return {
        "schema_version": 1,
        "expert_name": selection["expert_name"],
        "selected_weight": weight,
        "selection_lock_sha256": selection["lock_sha256"],
        "slice_1_baseline_mrr": slice1_baseline_mrr,
        "slice_1_candidate_mrr": slice1_candidate_mrr,
        "slice_1_delta": slice1_delta,
        "prefix_baseline_mrr": prefix_baseline_mrr,
        "prefix_candidate_mrr": prefix_candidate_mrr,
        "prefix_delta": prefix_delta,
        "minimum_prefix_delta": float(minimum_prefix_delta),
        "passed": passed,
    }


def choose_forward_winner(
    forward_reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [report for report in forward_reports if report.get("passed")]
    if not eligible:
        raise ValueError("no expert passed the forward gate")
    winner = min(
        eligible,
        key=lambda report: (
            -float(report["prefix_candidate_mrr"]),
            float(report["selected_weight"]),
            str(report["expert_name"]),
        ),
    )
    return {
        "expert_name": str(winner["expert_name"]),
        "selected_weight": float(winner["selected_weight"]),
        "prefix_candidate_mrr": float(winner["prefix_candidate_mrr"]),
        "selection_lock_sha256": str(
            winner["selection_lock_sha256"]
        ),
    }


def evaluate_final_gate(
    *,
    selection: dict[str, Any],
    forward_report: dict[str, Any],
    champion_scores: np.ndarray,
    expert_scores: np.ndarray,
    slices: Sequence[tuple[int, int]],
    expected_selection_lock_sha256: str,
    minimum_full_delta: float,
) -> dict[str, Any]:
    actual_lock_sha256 = selection.get("lock_sha256")
    if (
        actual_lock_sha256 != expected_selection_lock_sha256
        or selection_lock_sha256(selection) != expected_selection_lock_sha256
        or forward_report.get("selection_lock_sha256")
        != expected_selection_lock_sha256
    ):
        raise ValueError("selection-lock hash mismatch")
    if not forward_report.get("passed"):
        raise ValueError("final gate requires a passing forward report")
    champion, expert = _validated_score_pair(
        champion_scores,
        expert_scores,
    )
    normalized_slices = _validated_slices(slices, row_count=champion.shape[0])
    weight = _validated_weight(selection["selected_weight"])
    candidate = blend_partial_listwise(
        champion,
        expert,
        auxiliary_weight=weight,
    )
    baseline_full = ranking_mrr(champion)
    candidate_full = ranking_mrr(candidate)
    full_delta = candidate_full - baseline_full
    slice_metrics = []
    for index, (start, stop) in enumerate(normalized_slices):
        baseline_mrr = ranking_mrr(champion[start:stop])
        candidate_mrr = ranking_mrr(candidate[start:stop])
        slice_metrics.append(
            {
                "index": index,
                "rows": [start, stop],
                "baseline_mrr": baseline_mrr,
                "candidate_mrr": candidate_mrr,
                "delta": candidate_mrr - baseline_mrr,
            }
        )
    all_slices_non_decreasing = all(
        metric["delta"] >= -_METRIC_TOLERANCE
        for metric in slice_metrics
    )
    passed = bool(
        full_delta + _METRIC_TOLERANCE >= minimum_full_delta
        and all_slices_non_decreasing
    )
    return {
        "schema_version": 1,
        "expert_name": selection["expert_name"],
        "selected_weight": weight,
        "selection_lock_sha256": expected_selection_lock_sha256,
        "baseline_full_mrr": baseline_full,
        "candidate_full_mrr": candidate_full,
        "full_delta": full_delta,
        "minimum_full_delta": float(minimum_full_delta),
        "slices": slice_metrics,
        "all_slices_non_decreasing": all_slices_non_decreasing,
        "passed": passed,
    }


def ranking_mrr(scores: np.ndarray) -> float:
    values = _validated_scores(scores, label="scores")
    ranks = 1 + np.sum(values[:, 1:] > values[:, :1], axis=1)
    return float(np.mean(1.0 / ranks))


def _validated_score_pair(
    champion_scores: np.ndarray,
    expert_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    champion = _validated_scores(champion_scores, label="champion scores")
    expert = _validated_scores(expert_scores, label="expert scores")
    if champion.shape != expert.shape:
        raise ValueError("champion and expert scores must have the same shape")
    return champion, expert


def _validated_scores(scores: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(f"{label} must be a non-empty 2D candidate matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values")
    return values


def _validated_weight(weight: float) -> float:
    value = float(weight)
    for allowed in ALLOWED_AUXILIARY_WEIGHTS:
        if abs(value - allowed) <= 1e-12:
            return float(allowed)
    raise ValueError(
        "weight is not in the frozen auxiliary-weight grid "
        f"{ALLOWED_AUXILIARY_WEIGHTS}"
    )


def _validate_selection_lock(
    selection: dict[str, Any],
    *,
    candidate_manifest_sha256: str,
) -> None:
    stored_hash = selection.get("lock_sha256")
    if (
        not isinstance(stored_hash, str)
        or stored_hash != selection_lock_sha256(selection)
    ):
        raise ValueError("selection-lock hash mismatch")
    if (
        selection.get("candidate_manifest_sha256")
        != candidate_manifest_sha256
    ):
        raise ValueError("candidate manifest hash mismatch")


def _validated_slices(
    slices: Sequence[tuple[int, int]],
    *,
    row_count: int,
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(start), int(stop)) for start, stop in slices)
    if len(normalized) != 3:
        raise ValueError("final gate requires exactly three slices")
    expected_start = 0
    for start, stop in normalized:
        if start != expected_start or stop <= start or stop > row_count:
            raise ValueError(
                "slices must be contiguous, non-empty, and in bounds"
            )
        expected_start = stop
    if expected_start != row_count:
        raise ValueError("slices must cover every score row")
    return normalized
