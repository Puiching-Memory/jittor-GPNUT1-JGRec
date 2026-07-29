from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EXPECTED_CONTRACT_STATUS = "frozen_before_bugfixed_refit"
EXPECTED_TRAINING_STATUS = "complete_deterministic_bugfixed_v1_refit"
EXPECTED_INTEGRATION_ID = "cooccur_lift_aux_expert_v1"
EXPECTED_WEIGHT = 0.5
EXPECTED_FULL_ORIGIN_SEED = 33100


def validate_bugfixed_v1_training_report(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    report: Mapping[str, Any],
    actual_model_sha256: str,
) -> dict[str, Any]:
    """Bind one corrected v1 model to its frozen contract and replay proof."""
    candidate_id = _validate_contract(contract)
    if report.get("status") != EXPECTED_TRAINING_STATUS:
        raise ValueError("bugfixed v1 training report is incomplete")
    _require_equal(report, "candidate_id", candidate_id)
    _require_equal(report, "integration_id", EXPECTED_INTEGRATION_ID)
    _require_equal(report, "selected_weight", EXPECTED_WEIGHT)
    _require_equal(report, "full_origin_seed", EXPECTED_FULL_ORIGIN_SEED)
    _require_equal(report, "candidate_contract_sha256", contract_sha256)

    model_sha256 = str(report.get("model_sha256", ""))
    if not model_sha256 or model_sha256 != actual_model_sha256:
        raise ValueError("bugfixed v1 model hash differs from training evidence")
    source_checkpoint_sha256 = str(
        report.get("source_checkpoint_sha256", "")
    )
    if not source_checkpoint_sha256:
        raise ValueError("bugfixed v1 training report lacks checkpoint hash")
    training_assets = contract.get("training_assets")
    if not isinstance(training_assets, Mapping):
        raise ValueError("bugfixed v1 contract lacks training assets")
    for key in (
        "source_checkpoint_sha256",
        "frozen_config_sha256",
        "selection_lock_sha256",
    ):
        expected = str(training_assets.get(key, ""))
        if not expected or report.get(key) != expected:
            raise ValueError(
                f"bugfixed v1 training {key} differs from frozen assets"
            )
    implementation = contract.get("implementation_contract")
    if not isinstance(implementation, Mapping):
        raise ValueError("bugfixed v1 contract lacks implementation evidence")
    if (
        implementation.get("training_device") != "cpu"
        or implementation.get("test_scoring_device") != "cuda"
        or report.get("training_device")
        != implementation.get("training_device")
    ):
        raise ValueError("bugfixed v1 training device differs from frozen contract")

    replay = report.get("deterministic_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("bugfixed v1 training report lacks deterministic replay")
    frozen_gate = contract.get("deterministic_replay_gate")
    if not isinstance(frozen_gate, Mapping):
        raise ValueError("bugfixed v1 contract lacks deterministic replay gate")
    rtol = float(frozen_gate.get("rtol", -1.0))
    atol = float(frozen_gate.get("atol", -1.0))
    if (
        float(replay.get("rtol", -2.0)) != rtol
        or float(replay.get("atol", -2.0)) != atol
    ):
        raise ValueError("bugfixed v1 replay tolerance differs from frozen gate")
    maximum_error = float(replay.get("max_abs_error", float("inf")))
    if (
        replay.get("matched") is not True
        or maximum_error < 0.0
        or maximum_error > rtol + atol
    ):
        raise ValueError("bugfixed v1 deterministic replay did not pass")

    return {
        "candidate_id": candidate_id,
        "model_sha256": model_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "selected_weight": EXPECTED_WEIGHT,
    }


def validate_bugfixed_v1_materialization_inputs(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    training_report: Mapping[str, Any],
    actual_model_sha256: str,
    actual_source_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Reject model/checkpoint substitution before online test scoring."""
    evidence = validate_bugfixed_v1_training_report(
        contract=contract,
        contract_sha256=contract_sha256,
        report=training_report,
        actual_model_sha256=actual_model_sha256,
    )
    if (
        evidence["source_checkpoint_sha256"]
        != actual_source_checkpoint_sha256
    ):
        raise ValueError(
            "bugfixed v1 source checkpoint hash differs from training evidence"
        )
    return evidence


def _validate_contract(contract: Mapping[str, Any]) -> str:
    if contract.get("status") != EXPECTED_CONTRACT_STATUS:
        raise ValueError("bugfixed v1 candidate contract is not frozen")
    _require_equal(contract, "schema_version", 1)
    _require_equal(contract, "integration_id", EXPECTED_INTEGRATION_ID)
    _require_equal(contract, "selected_weight", EXPECTED_WEIGHT)
    _require_equal(contract, "full_origin_seed", EXPECTED_FULL_ORIGIN_SEED)
    candidate_id = str(contract.get("candidate_id", ""))
    if not candidate_id:
        raise ValueError("bugfixed v1 candidate contract lacks candidate_id")
    return candidate_id


def _require_equal(
    payload: Mapping[str, Any],
    key: str,
    expected: Any,
) -> None:
    if payload.get(key) != expected:
        raise ValueError(f"bugfixed v1 {key} differs from frozen contract")
