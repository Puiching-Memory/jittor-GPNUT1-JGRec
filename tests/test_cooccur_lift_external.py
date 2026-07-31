from __future__ import annotations

import json
from pathlib import Path

import pytest

from jgrec.cooccur_lift_external import (
    build_external_manifest,
    validate_locked_external_setup,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "docs" / "experiments" / "cooccur-lift-aux-expert-v1.frozen.json"


def _lock_payload() -> dict:
    return {
        "schema_version": 1,
        "protocol": "exact_integrated_weight_selection_lock_v1",
        "integration_id": "cooccur_lift_aux_expert_v1",
        "selection_manifest_sha256": "1" * 64,
        "selected_weight": 0.5,
        "selected_candidate_scores": [],
        "selection_rule": [],
        "external_holdout_read": False,
    }


def test_external_setup_requires_exact_lock_and_uses_next_fold_seed(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "selection-lock.json"
    lock_path.write_text(json.dumps(_lock_payload()), encoding="utf-8")

    contract = validate_locked_external_setup(
        frozen_config_path=FROZEN,
        selection_lock_path=lock_path,
    )

    assert contract.integration_id == "cooccur_lift_aux_expert_v1"
    assert contract.selected_weight == 0.5
    assert contract.full_origin_seed == 33100
    assert len(contract.selection_lock_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("protocol", "wrong", "protocol"),
        ("integration_id", "wrong", "integration_id"),
        ("selected_weight", 0.15, "selected_weight"),
        ("external_holdout_read", True, "external_holdout_read"),
    ],
)
def test_external_setup_rejects_lock_drift(
    tmp_path: Path,
    field: str,
    value,
    match: str,
) -> None:
    payload = _lock_payload()
    payload[field] = value
    lock_path = tmp_path / "selection-lock.json"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        validate_locked_external_setup(
            frozen_config_path=FROZEN,
            selection_lock_path=lock_path,
        )


def test_external_manifest_is_bound_to_lock_and_candidate_order(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "selection-lock.json"
    lock_path.write_text(json.dumps(_lock_payload()), encoding="utf-8")
    contract = validate_locked_external_setup(
        frozen_config_path=FROZEN,
        selection_lock_path=lock_path,
    )

    manifest = build_external_manifest(
        contract=contract,
        candidate_fingerprint="2" * 64,
        training_time_max=100,
        score_time_min=101,
        score_time_max=200,
        baseline_path=tmp_path / "baseline.npy",
        baseline_sha256="3" * 64,
        candidate_path=tmp_path / "candidate.npy",
        candidate_sha256="4" * 64,
    )

    assert manifest["protocol"] == "exact_integrated_external_holdout_v1"
    assert manifest["selection_lock_sha256"] == contract.selection_lock_sha256
    assert manifest["minimum_train_to_score_gap"] == 1
    assert manifest["candidate"]["candidate_fingerprint"] == "2" * 64
    assert manifest["candidate"]["weight"] == 0.5
