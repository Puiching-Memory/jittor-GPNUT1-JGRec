from __future__ import annotations

import numpy as np
import pytest

from jgrec.rankers.hybrid.base_context_gate import (
    BASE_CONTEXT_INTEGRATION_ID,
    BaseContextBlendProtocol,
    authorize_base_context_package,
    compose_dataset1_final_scores,
    validate_context_only_difference,
)
from jgrec.rankers.hybrid.expert_fusion import ExpertBlendCalibration


def test_context_comparison_changes_only_the_base_mlp_branch() -> None:
    control_mlp = np.asarray(
        [[2.0, 0.0, -1.0], [0.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    candidate_mlp = np.asarray(
        [[8.0, 0.0, -1.0], [2.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    lgbm = np.asarray(
        [[0.0, 2.0, -1.0], [0.0, 2.0, -1.0]],
        dtype=np.float64,
    )
    setwise = np.asarray(
        [[0.0, 0.0, 3.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    protocol = BaseContextBlendProtocol(
        mlp_weight=0.45,
        expert_calibration=ExpertBlendCalibration(mode="probability"),
        time_ramp_power=0.5,
    )

    comparison = compose_dataset1_final_scores(
        control_mlp_logits=control_mlp,
        candidate_mlp_logits=candidate_mlp,
        shared_lgbm_logits=lgbm,
        shared_setwise_logits=setwise,
        query_times=np.asarray([10, 20], dtype=np.int64),
        protocol=protocol,
        minimum_time=10,
        maximum_time=20,
    )

    assert comparison.integration_id == BASE_CONTEXT_INTEGRATION_ID
    # At the start of the horizon the result is exactly the MLP+LGBM backbone.
    assert comparison.control[0].argmax() == 1
    assert comparison.candidate[0].argmax() == 0
    # At the end of the horizon both paths are exactly the same Setwise expert.
    np.testing.assert_array_equal(comparison.control[1], comparison.candidate[1])
    assert comparison.control[1].argmax() == 0


def test_context_only_difference_rejects_any_other_training_change() -> None:
    control = {
        "context_transform_version": 0,
        "seed": 60,
        "epochs": 15,
        "hidden_dim": 64,
        "feature_indices": list(range(63)),
    }
    candidate = {
        **control,
        "context_transform_version": 1,
    }

    validate_context_only_difference(control, candidate)

    candidate["hidden_dim"] = 32
    with pytest.raises(ValueError, match="hidden_dim"):
        validate_context_only_difference(control, candidate)


def test_context_only_difference_requires_v0_and_v1() -> None:
    shared = {
        "seed": 60,
        "epochs": 15,
        "hidden_dim": 64,
    }

    with pytest.raises(ValueError, match=r"control.*v0"):
        validate_context_only_difference(
            {**shared, "context_transform_version": 1},
            {**shared, "context_transform_version": 1},
        )


def test_package_requires_accepted_external_and_matching_artifacts() -> None:
    evaluation_sha = "a" * 64
    head_sha = "b" * 64
    source_sha = "c" * 64
    lock_sha = "d" * 64
    evaluation = {
        "status": "accepted",
        "integration_id": BASE_CONTEXT_INTEGRATION_ID,
        "selected_weight": 1.0,
        "selection_lock_sha256": lock_sha,
        "failed_gates": [],
        "weight_rescan_authorized": False,
        "leaderboard_tuning_authorized": False,
    }
    result = {
        "status": "accepted",
        "external_pass": True,
        "package_authorized": True,
        "weight_rescan_authorized": False,
        "candidate_head_sha256": head_sha,
        "source_checkpoint_sha256": source_sha,
        "selection_lock_sha256": lock_sha,
        "external_evaluation_sha256": evaluation_sha,
    }

    authorization = authorize_base_context_package(
        external_result=result,
        external_evaluation=evaluation,
        actual_external_evaluation_sha256=evaluation_sha,
        actual_candidate_head_sha256=head_sha,
        actual_source_checkpoint_sha256=source_sha,
    )

    assert authorization.integration_id == BASE_CONTEXT_INTEGRATION_ID
    assert authorization.selection_lock_sha256 == lock_sha
    assert authorization.candidate_head_sha256 == head_sha


def test_package_rejects_failed_or_unbound_external_gate() -> None:
    evaluation = {
        "status": "rejected",
        "integration_id": BASE_CONTEXT_INTEGRATION_ID,
        "selected_weight": 1.0,
        "selection_lock_sha256": "d" * 64,
        "failed_gates": ["mrr_strictly_increasing"],
        "weight_rescan_authorized": False,
        "leaderboard_tuning_authorized": False,
    }
    result = {
        "status": "rejected",
        "external_pass": False,
        "package_authorized": False,
        "weight_rescan_authorized": False,
        "candidate_head_sha256": "b" * 64,
        "source_checkpoint_sha256": "c" * 64,
        "selection_lock_sha256": "d" * 64,
        "external_evaluation_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="accepted external"):
        authorize_base_context_package(
            external_result=result,
            external_evaluation=evaluation,
            actual_external_evaluation_sha256="a" * 64,
            actual_candidate_head_sha256="b" * 64,
            actual_source_checkpoint_sha256="c" * 64,
        )

    evaluation["status"] = "accepted"
    evaluation["failed_gates"] = []
    result["status"] = "accepted"
    result["external_pass"] = True
    result["package_authorized"] = True
    with pytest.raises(ValueError, match="candidate head hash"):
        authorize_base_context_package(
            external_result=result,
            external_evaluation=evaluation,
            actual_external_evaluation_sha256="a" * 64,
            actual_candidate_head_sha256="e" * 64,
            actual_source_checkpoint_sha256="c" * 64,
        )
