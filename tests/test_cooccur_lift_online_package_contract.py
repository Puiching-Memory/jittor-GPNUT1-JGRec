from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from jgrec.cooccur_lift_online_package_contract import (
    validate_k512_online_package_preflight,
)

GATES = [
    "mrr_meets_minimum",
    "hit_at_1_meets_minimum",
    "hit_at_3_meets_minimum",
    "hit_at_10_meets_minimum",
    "ndcg_at_10_meets_minimum",
    "mean_rank_meets_maximum",
    "improved_minus_worsened_meets_minimum",
]


def test_preflight_accepts_exact_current_run_lineage(tmp_path: Path) -> None:
    contract_path = _write_fixture(tmp_path)

    report = validate_k512_online_package_preflight(
        root=tmp_path,
        contract_path=contract_path,
    )

    assert report["status"] == "passed"
    assert report["candidate_id"] == "cooccur_lift_gap_aware_v2"
    assert report["selected_weight"] == 0.5
    assert report["all_seven_gates_passed"] is True
    assert report["outputs_absent"] is True


def test_preflight_rejects_bugfixed_v1_model_substitution(
    tmp_path: Path,
) -> None:
    contract_path = _write_fixture(tmp_path)
    (tmp_path / "inputs/v1-model.npz").write_bytes(b"historical-v1")

    with pytest.raises(
        ValueError,
        match="input hash differs: bugfixed_v1_model",
    ):
        validate_k512_online_package_preflight(
            root=tmp_path,
            contract_path=contract_path,
        )


def _write_fixture(root: Path) -> Path:
    inputs = root / "inputs"
    implementations = root / "implementations"
    inputs.mkdir()
    implementations.mkdir()

    paths = {
        "candidate_config": "inputs/candidate.json",
        "selection_lock": "inputs/selection.json",
        "external_report": "inputs/external-report.json",
        "external_open_receipt": "inputs/external-receipt.json",
        "external_execution_contract": "inputs/external-contract.json",
        "external_materialization_report": "inputs/external-materialization.json",
        "external_manifest": "inputs/external-manifest.json",
        "gap_aware_model": "inputs/gap-aware.npz",
        "bugfixed_v1_contract": "inputs/v1-contract.json",
        "bugfixed_v1_training_report": "inputs/v1-training.json",
        "bugfixed_v1_model": "inputs/v1-model.npz",
        "bugfixed_v1_frozen_config": "inputs/v1-config.json",
        "bugfixed_v1_selection_lock": "inputs/v1-selection.json",
        "source_checkpoint": "inputs/source.pkl",
        "train_csv": "inputs/train.csv",
        "test_csv": "inputs/test.csv",
        "source_champion_zip": "inputs/source.zip",
    }
    implementation_paths = {
        "bugfixed_v1_materializer": "implementations/v1-materialize.py",
        "bugfixed_v1_packager": "implementations/v1-package.py",
        "successor_materializer": "implementations/v2-materialize.py",
        "successor_packager": "implementations/v2-package.py",
        "bugfixed_v1_contract_module": "implementations/v1-contract.py",
        "successor_external_module": "implementations/v2-external.py",
        "submission_module": "implementations/submission.py",
        "native_materializer_module": "implementations/native.py",
        "gap_aware_view_module": "implementations/gap-aware.py",
    }
    for name, relative in implementation_paths.items():
        (root / relative).write_text(name, encoding="utf-8")

    (root / paths["candidate_config"]).write_text(
        "{}\n",
        encoding="utf-8",
    )
    (root / paths["gap_aware_model"]).write_bytes(b"gap-aware")
    (root / paths["bugfixed_v1_model"]).write_bytes(b"current-k512-v1")
    (root / paths["bugfixed_v1_frozen_config"]).write_text(
        "{}\n",
        encoding="utf-8",
    )
    (root / paths["bugfixed_v1_selection_lock"]).write_text(
        "{}\n",
        encoding="utf-8",
    )
    (root / paths["source_checkpoint"]).write_bytes(b"checkpoint")
    (root / paths["train_csv"]).write_text("train\n", encoding="utf-8")
    (root / paths["test_csv"]).write_text("test\n", encoding="utf-8")

    source_zip = root / paths["source_champion_zip"]
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("dataset1.csv", "0.5,0.5\n")
        archive.writestr("dataset2.csv", "0.4,0.6\n")
    with zipfile.ZipFile(source_zip) as archive:
        member_sha256 = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
        }

    external_contract = {"status": "frozen_before_external_open"}
    _write_json(root / paths["external_execution_contract"], external_contract)
    external_manifest = {"status": "metrics_unread"}
    _write_json(root / paths["external_manifest"], external_manifest)
    external_contract_sha256 = _sha256(
        root / paths["external_execution_contract"]
    )
    external_manifest_sha256 = _sha256(
        root / paths["external_manifest"]
    )

    selection = {
        "selected_candidate": {
            "candidate_id": "cooccur_lift_gap_aware_v2",
            "config_sha256": _sha256(root / paths["candidate_config"]),
        },
        "weight_rescan_authorized": False,
        "feature_rescan_authorized": False,
    }
    _write_json(root / paths["selection_lock"], selection)
    selection_sha256 = _sha256(root / paths["selection_lock"])
    receipt = {
        "selected_candidate_id": "cooccur_lift_gap_aware_v2",
        "selection_lock_sha256": selection_sha256,
        "external_manifest_sha256": external_manifest_sha256,
    }
    _write_json(root / paths["external_open_receipt"], receipt)
    external = {
        "status": "accepted",
        "package_authorized": True,
        "failed_gates": [],
        "gates": dict.fromkeys(GATES, True),
        "selection_lock_sha256": selection_sha256,
        "selected_candidate": {
            "candidate_id": "cooccur_lift_gap_aware_v2",
            "config_sha256": _sha256(root / paths["candidate_config"]),
        },
        "external_manifest_sha256": external_manifest_sha256,
        "weight_rescan_authorized": False,
        "feature_rescan_authorized": False,
        "leaderboard_tuning_authorized": False,
    }
    _write_json(root / paths["external_report"], external)

    v1_contract = {
        "status": "frozen_before_bugfixed_refit",
        "selected_weight": 0.5,
        "package_contract": {
            "champion_zip_sha256": _sha256(source_zip),
            "dataset1_member_sha256": member_sha256["dataset1.csv"],
            "dataset2_member_sha256": member_sha256["dataset2.csv"],
        },
    }
    _write_json(root / paths["bugfixed_v1_contract"], v1_contract)
    v1_training = {
        "status": "complete_deterministic_bugfixed_v1_refit",
        "candidate_contract_sha256": _sha256(
            root / paths["bugfixed_v1_contract"]
        ),
        "model_sha256": _sha256(root / paths["bugfixed_v1_model"]),
        "selected_weight": 0.5,
        "deterministic_replay": {
            "matched": True,
            "tolerance_relaxed": False,
        },
    }
    _write_json(root / paths["bugfixed_v1_training_report"], v1_training)

    external_materialization = {
        "status": "external_candidate_materialized_metrics_unread",
        "candidate_id": "cooccur_lift_gap_aware_v2",
        "candidate_config_sha256": _sha256(
            root / paths["candidate_config"]
        ),
        "selection_lock_sha256": selection_sha256,
        "execution_contract_sha256": external_contract_sha256,
        "external_manifest_sha256": external_manifest_sha256,
        "model_sha256": _sha256(root / paths["gap_aware_model"]),
        "selected_weight": 0.5,
        "effect_size_estimation_authorized": False,
        "deterministic_replay": {
            "matched": True,
            "tolerance_relaxed": False,
        },
    }
    _write_json(
        root / paths["external_materialization_report"],
        external_materialization,
    )

    contract = {
        "schema_version": 1,
        "protocol": "cooccur_lift_k512_successor_v2_online_package_v1",
        "status": "frozen_after_external_acceptance_before_online_scoring",
        "candidate_id": "cooccur_lift_gap_aware_v2",
        "selected_weight": 0.5,
        "baseline_role": "current_k512_bugfixed_v1_new_champion",
        "external_decision_role": "safety_gate_only",
        "external_effect_size_used": False,
        "input_paths": paths,
        "input_sha256": {
            name: _sha256(root / relative)
            for name, relative in paths.items()
        },
        "implementation_paths": implementation_paths,
        "implementation_sha256": {
            name: _sha256(root / relative)
            for name, relative in implementation_paths.items()
        },
        "source_package_contract": {
            "zip": paths["source_champion_zip"],
            "dataset1_member_sha256": member_sha256["dataset1.csv"],
            "dataset2_member_sha256": member_sha256["dataset2.csv"],
            "dataset1_rows": 1,
            "dataset2_rows": 1,
            "candidate_count": 2,
        },
        "stage_contract": {
            "v1_materialization_dir": "outputs/v1-materialization",
            "v1_output_dir": "outputs/v1-submission",
            "v2_materialization_dir": "outputs/v2-materialization",
            "v2_output_dir": "outputs/v2-submission",
        },
        "authorization": {
            "external_gates_required": GATES,
            "package_authorized": True,
            "weight_rescan_authorized": False,
            "feature_rescan_authorized": False,
            "external_reopen_authorized": False,
        },
    }
    contract_path = root / "contract.json"
    _write_json(contract_path, contract)
    return contract_path


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
