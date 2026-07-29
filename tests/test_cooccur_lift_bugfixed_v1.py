from __future__ import annotations

import copy

import pytest

from jgrec.cooccur_lift_bugfixed_v1 import (
    validate_bugfixed_v1_materialization_inputs,
    validate_bugfixed_v1_training_report,
)


def _contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "frozen_before_bugfixed_refit",
        "candidate_id": "cooccur_lift_aux_expert_v1_bugfixed_refit_20260729",
        "integration_id": "cooccur_lift_aux_expert_v1",
        "selected_weight": 0.5,
        "full_origin_seed": 33100,
        "deterministic_replay_gate": {
            "rtol": 2e-5,
            "atol": 2e-6,
        },
        "training_assets": {
            "frozen_config_sha256": "config-sha",
            "selection_lock_sha256": "lock-sha",
            "source_checkpoint_sha256": "checkpoint-sha",
        },
        "implementation_contract": {
            "training_device": "cpu",
            "test_scoring_device": "cuda",
        },
    }


def _training_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete_deterministic_bugfixed_v1_refit",
        "candidate_id": "cooccur_lift_aux_expert_v1_bugfixed_refit_20260729",
        "integration_id": "cooccur_lift_aux_expert_v1",
        "selected_weight": 0.5,
        "full_origin_seed": 33100,
        "candidate_contract_sha256": "contract-sha",
        "model_sha256": "model-sha",
        "source_checkpoint_sha256": "checkpoint-sha",
        "frozen_config_sha256": "config-sha",
        "selection_lock_sha256": "lock-sha",
        "training_device": "cpu",
        "deterministic_replay": {
            "matched": True,
            "rtol": 2e-5,
            "atol": 2e-6,
            "max_abs_error": 0.0,
        },
    }


def test_training_report_must_prove_the_frozen_replay_gate() -> None:
    contract = _contract()
    report = _training_report()

    evidence = validate_bugfixed_v1_training_report(
        contract=contract,
        contract_sha256="contract-sha",
        report=report,
        actual_model_sha256="model-sha",
    )

    assert evidence == {
        "candidate_id": "cooccur_lift_aux_expert_v1_bugfixed_refit_20260729",
        "model_sha256": "model-sha",
        "source_checkpoint_sha256": "checkpoint-sha",
        "selected_weight": 0.5,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report.update(model_sha256="other-model"),
            "model hash",
        ),
        (
            lambda report: report["deterministic_replay"].update(
                matched=False,
            ),
            "deterministic replay",
        ),
        (
            lambda report: report["deterministic_replay"].update(
                rtol=3e-5,
            ),
            "replay tolerance",
        ),
        (
            lambda report: report["deterministic_replay"].update(
                max_abs_error=0.10732766809698313,
            ),
            "deterministic replay",
        ),
        (
            lambda report: report.update(training_device="cuda"),
            "training device",
        ),
    ],
)
def test_training_report_rejects_unbound_or_drifted_evidence(
    mutation: object,
    message: str,
) -> None:
    report = copy.deepcopy(_training_report())
    mutation(report)

    with pytest.raises(ValueError, match=message):
        validate_bugfixed_v1_training_report(
            contract=_contract(),
            contract_sha256="contract-sha",
            report=report,
            actual_model_sha256="model-sha",
        )


def test_materialization_rejects_a_model_swap_before_scoring() -> None:
    training = _training_report()

    with pytest.raises(ValueError, match="model hash"):
        validate_bugfixed_v1_materialization_inputs(
            contract=_contract(),
            contract_sha256="contract-sha",
            training_report=training,
            actual_model_sha256="substituted-model-sha",
            actual_source_checkpoint_sha256="checkpoint-sha",
        )


def test_materialization_binds_model_checkpoint_and_candidate() -> None:
    evidence = validate_bugfixed_v1_materialization_inputs(
        contract=_contract(),
        contract_sha256="contract-sha",
        training_report=_training_report(),
        actual_model_sha256="model-sha",
        actual_source_checkpoint_sha256="checkpoint-sha",
    )

    assert evidence["candidate_id"].endswith("bugfixed_refit_20260729")
    assert evidence["model_sha256"] == "model-sha"
    assert evidence["selected_weight"] == 0.5
