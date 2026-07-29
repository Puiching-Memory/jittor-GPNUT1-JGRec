from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

EXPECTED_STATUS = "frozen_before_successor_v2_metrics"
EXPECTED_EXPERIMENT_ID = "dataset2_cooccur_lift_successor_v2_duel_20260729"
EXPECTED_CANDIDATE_IDS = (
    "cooccur_lift_full_only_v2",
    "cooccur_lift_gap_aware_v2",
)
EXPECTED_TRAINING_DEVICE = "cpu"
EXPECTED_INTERNAL_SCORING_DEVICE = "cpu"
EXPECTED_REPLAY_RUNS = 2
EXPECTED_RTOL = 2e-5
EXPECTED_ATOL = 2e-6
EXPECTED_LEGACY_ROLE = "diagnostic_only"
EXPECTED_WEIGHT = 0.5


def validate_successor_execution_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject execution paths that reintroduce the diagnosed V1 drift."""
    if contract.get("schema_version") != 1:
        raise ValueError("successor execution schema differs")
    if contract.get("status") != EXPECTED_STATUS:
        raise ValueError("successor execution contract is not frozen")
    if contract.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError("successor execution experiment differs")
    if tuple(contract.get("candidate_ids", ())) != EXPECTED_CANDIDATE_IDS:
        raise ValueError("successor execution candidate space differs")
    if contract.get("training_device") != EXPECTED_TRAINING_DEVICE:
        raise ValueError("successor training device must be cpu")
    if (
        contract.get("internal_scoring_device")
        != EXPECTED_INTERNAL_SCORING_DEVICE
    ):
        raise ValueError("successor internal scoring device must be cpu")
    if (
        contract.get("historical_near_v1_manifest_role")
        != EXPECTED_LEGACY_ROLE
    ):
        raise ValueError("historical V1 scores must remain diagnostic only")
    replay = contract.get("deterministic_replay_gate")
    if not isinstance(replay, Mapping):
        raise ValueError("successor execution lacks deterministic replay gate")
    if replay.get("runs") != EXPECTED_REPLAY_RUNS:
        raise ValueError("successor deterministic replay requires two runs")
    if (
        float(replay.get("rtol", -1.0)) != EXPECTED_RTOL
        or float(replay.get("atol", -1.0)) != EXPECTED_ATOL
        or replay.get("tolerance_relaxation_authorized") is not False
    ):
        raise ValueError("successor deterministic replay tolerance differs")
    if contract.get("external_authorized") is not False:
        raise ValueError("successor execution must keep external closed")
    return {
        "training_device": EXPECTED_TRAINING_DEVICE,
        "internal_scoring_device": EXPECTED_INTERNAL_SCORING_DEVICE,
        "runs": EXPECTED_REPLAY_RUNS,
        "rtol": EXPECTED_RTOL,
        "atol": EXPECTED_ATOL,
        "legacy_role": EXPECTED_LEGACY_ROLE,
    }


def build_deterministic_replay_report(
    *,
    first_state: Mapping[str, np.ndarray],
    second_state: Mapping[str, np.ndarray],
    first_losses: Sequence[float],
    second_losses: Sequence[float],
    first_predictions: Mapping[str, np.ndarray],
    second_predictions: Mapping[str, np.ndarray],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare two independently trained heads under one frozen contract."""
    state_keys_match = first_state.keys() == second_state.keys()
    state_errors = [
        _maximum_absolute_error(first_state[key], second_state[key])
        for key in first_state
        if key in second_state
    ]
    state_matched = state_keys_match and all(
        np.allclose(
            np.asarray(first_state[key]),
            np.asarray(second_state[key]),
            rtol=rtol,
            atol=atol,
        )
        for key in first_state
    )
    first_loss_array = np.asarray(first_losses, dtype=np.float64)
    second_loss_array = np.asarray(second_losses, dtype=np.float64)
    loss_shape_matched = first_loss_array.shape == second_loss_array.shape
    loss_matched = loss_shape_matched and bool(
        np.allclose(
            first_loss_array,
            second_loss_array,
            rtol=rtol,
            atol=atol,
        )
    )
    prediction_keys_match = (
        first_predictions.keys() == second_predictions.keys()
    )
    prediction_errors: list[np.ndarray] = []
    probability_matched = prediction_keys_match
    for key, first in first_predictions.items():
        if key not in second_predictions:
            probability_matched = False
            continue
        first_array = np.asarray(first)
        second_array = np.asarray(second_predictions[key])
        if first_array.shape != second_array.shape:
            probability_matched = False
            continue
        error = np.abs(
            first_array.astype(np.float64)
            - second_array.astype(np.float64)
        )
        prediction_errors.append(error)
        probability_matched = probability_matched and bool(
            np.allclose(
                first_array,
                second_array,
                rtol=rtol,
                atol=atol,
            )
        )
    maximum_probability_error = max(
        (float(np.max(error, initial=0.0)) for error in prediction_errors),
        default=float("inf") if not prediction_keys_match else 0.0,
    )
    probability_error_count = sum(error.size for error in prediction_errors)
    mean_probability_error = (
        sum(float(np.sum(error)) for error in prediction_errors)
        / probability_error_count
        if probability_error_count
        else (float("inf") if not prediction_keys_match else 0.0)
    )
    return {
        "matched": bool(
            state_matched and loss_matched and probability_matched
        ),
        "runs": 2,
        "rtol": float(rtol),
        "atol": float(atol),
        "tolerance_relaxed": False,
        "state_matched": bool(state_matched),
        "loss_matched": bool(loss_matched),
        "probability_matched": bool(probability_matched),
        "state_max_abs_error": max(state_errors, default=0.0),
        "loss_max_abs_error": (
            _maximum_absolute_error(first_loss_array, second_loss_array)
            if loss_shape_matched
            else float("inf")
        ),
        "max_abs_error": maximum_probability_error,
        "mean_abs_error": mean_probability_error,
        "prediction_arms": sorted(first_predictions),
    }


def resolve_bugfixed_v1_fold_baseline(
    *,
    prior_baseline: np.ndarray,
    cpu_auxiliary: np.ndarray,
    legacy_cuda_v1: np.ndarray,
    replay: Mapping[str, Any],
    weight: float,
) -> dict[str, Any]:
    """Use the reproducible CPU V1 and retain historical CUDA only as context."""
    if replay.get("matched") is not True:
        raise ValueError("bugfixed V1 deterministic replay did not pass")
    if replay.get("tolerance_relaxed") is not False:
        raise ValueError("bugfixed V1 deterministic replay relaxed tolerance")
    if float(weight) != EXPECTED_WEIGHT:
        raise ValueError("bugfixed V1 baseline weight differs from 0.50")
    prior = np.asarray(prior_baseline, dtype=np.float64)
    auxiliary = np.asarray(cpu_auxiliary, dtype=np.float64)
    legacy = np.asarray(legacy_cuda_v1, dtype=np.float64)
    if prior.shape != auxiliary.shape or prior.shape != legacy.shape:
        raise ValueError("bugfixed V1 fold score shapes differ")
    baseline = (1.0 - weight) * prior + weight * auxiliary
    return {
        "baseline": baseline,
        "deterministic_replay": dict(replay),
        "legacy_cuda_max_abs_error": _maximum_absolute_error(
            baseline,
            legacy,
        ),
        "legacy_cuda_role": EXPECTED_LEGACY_ROLE,
    }


def _maximum_absolute_error(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape:
        return float("inf")
    return float(
        np.max(np.abs(first_array - second_array), initial=0.0)
    )
