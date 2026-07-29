from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .expert_fusion import (
    ExpertBlendCalibration,
    blend_expert_logits,
)
from .time_ramp import apply_time_ramp

BASE_CONTEXT_INTEGRATION_ID = (
    "dataset1_base_fusion_setwise_context_v1_time_ramp_g050"
)
_CONTEXT_VERSION_KEY = "context_transform_version"


@dataclass(frozen=True)
class BaseContextBlendProtocol:
    """Frozen Dataset1 integration shared by rolling and external gates."""

    mlp_weight: float
    expert_calibration: ExpertBlendCalibration
    time_ramp_power: float = 0.5


@dataclass(frozen=True)
class BaseContextScoreComparison:
    integration_id: str
    control: np.ndarray
    candidate: np.ndarray


@dataclass(frozen=True)
class BaseContextPackageAuthorization:
    integration_id: str
    selection_lock_sha256: str
    candidate_head_sha256: str
    source_checkpoint_sha256: str
    external_evaluation_sha256: str


def compose_dataset1_final_scores(
    *,
    control_mlp_logits: np.ndarray,
    candidate_mlp_logits: np.ndarray,
    shared_lgbm_logits: np.ndarray,
    shared_setwise_logits: np.ndarray,
    query_times: np.ndarray,
    protocol: BaseContextBlendProtocol,
    minimum_time: float | None = None,
    maximum_time: float | None = None,
) -> BaseContextScoreComparison:
    """Compare v0/v1 after the exact shared LGBM and Setwise branches."""

    control_backbone = blend_expert_logits(
        control_mlp_logits,
        shared_lgbm_logits,
        protocol.mlp_weight,
        calibration=protocol.expert_calibration,
    )
    candidate_backbone = blend_expert_logits(
        candidate_mlp_logits,
        shared_lgbm_logits,
        protocol.mlp_weight,
        calibration=protocol.expert_calibration,
    )
    setwise_probabilities = _softmax(shared_setwise_logits)
    control = apply_time_ramp(
        control_backbone,
        setwise_probabilities,
        query_times,
        power=protocol.time_ramp_power,
        minimum_time=minimum_time,
        maximum_time=maximum_time,
    )
    candidate = apply_time_ramp(
        candidate_backbone,
        setwise_probabilities,
        query_times,
        power=protocol.time_ramp_power,
        minimum_time=minimum_time,
        maximum_time=maximum_time,
    )
    return BaseContextScoreComparison(
        integration_id=BASE_CONTEXT_INTEGRATION_ID,
        control=control,
        candidate=candidate,
    )


def validate_context_only_difference(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    """Require an explicit v0-vs-v1 comparison with no other drift."""

    control_version = control.get(_CONTEXT_VERSION_KEY)
    candidate_version = candidate.get(_CONTEXT_VERSION_KEY)
    if control_version != 0:
        raise ValueError("control context transform must be v0")
    if candidate_version != 1:
        raise ValueError("candidate context transform must be v1")

    keys = set(control) | set(candidate)
    differences = [
        key
        for key in sorted(keys)
        if key != _CONTEXT_VERSION_KEY
        and not _same_value(control.get(key), candidate.get(key))
    ]
    if differences:
        raise ValueError(
            "control and candidate differ outside context transform: "
            + ", ".join(differences)
        )


def authorize_base_context_package(
    *,
    external_result: Mapping[str, Any],
    external_evaluation: Mapping[str, Any],
    actual_external_evaluation_sha256: str,
    actual_candidate_head_sha256: str,
    actual_source_checkpoint_sha256: str,
) -> BaseContextPackageAuthorization:
    """Bind packaging to one accepted external result and exact artifacts."""

    if (
        external_result.get("status") != "accepted"
        or external_result.get("external_pass") is not True
        or external_result.get("package_authorized") is not True
        or external_evaluation.get("status") != "accepted"
        or external_evaluation.get("failed_gates") != []
    ):
        raise ValueError("packaging requires an accepted external gate")
    if (
        external_evaluation.get("integration_id")
        != BASE_CONTEXT_INTEGRATION_ID
        or float(external_evaluation.get("selected_weight", -1.0)) != 1.0
    ):
        raise ValueError("external gate does not bind the context v1 candidate")
    if (
        external_result.get("weight_rescan_authorized") is not False
        or external_evaluation.get("weight_rescan_authorized") is not False
        or external_evaluation.get("leaderboard_tuning_authorized") is not False
    ):
        raise ValueError("external gate permits forbidden post-holdout tuning")

    _require_bound_hash(
        external_result,
        "external_evaluation_sha256",
        actual_external_evaluation_sha256,
        "external evaluation",
    )
    _require_bound_hash(
        external_result,
        "candidate_head_sha256",
        actual_candidate_head_sha256,
        "candidate head",
    )
    _require_bound_hash(
        external_result,
        "source_checkpoint_sha256",
        actual_source_checkpoint_sha256,
        "source checkpoint",
    )
    result_lock = str(external_result.get("selection_lock_sha256", ""))
    evaluation_lock = str(
        external_evaluation.get("selection_lock_sha256", "")
    )
    if not result_lock or result_lock != evaluation_lock:
        raise ValueError("selection lock hash is not consistently bound")
    return BaseContextPackageAuthorization(
        integration_id=BASE_CONTEXT_INTEGRATION_ID,
        selection_lock_sha256=result_lock,
        candidate_head_sha256=actual_candidate_head_sha256,
        source_checkpoint_sha256=actual_source_checkpoint_sha256,
        external_evaluation_sha256=actual_external_evaluation_sha256,
    )


def _require_bound_hash(
    payload: Mapping[str, Any],
    key: str,
    actual: str,
    label: str,
) -> None:
    expected = str(payload.get(key, ""))
    if not expected or expected != actual:
        raise ValueError(f"{label} hash does not match its external result")


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    return left == right


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] < 2
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "Setwise logits must be a finite query-by-candidate matrix"
        )
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
