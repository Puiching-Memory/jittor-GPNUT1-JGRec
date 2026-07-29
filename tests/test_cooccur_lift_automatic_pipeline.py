from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from jgrec.cooccur_lift_automatic_pipeline import (
    automatic_stage_order,
    build_duel_manifest_bindings,
    build_frozen_validation_recovery_marker,
    validate_external_transition,
    validate_full_predict_prerequisite,
    validate_stage_order,
)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_frozen_validation_recovery_marker_requires_real_exact_artifacts(
    tmp_path: Path,
) -> None:
    train_feature = tmp_path / "train.npy"
    train_feature.write_bytes(b"train-feature")
    train_report_path = tmp_path / "train-report.json"
    validation_report_path = tmp_path / "validation-report.json"
    output_paths = {
        name: tmp_path / f"output-{name}.npy"
        for name in (
            "features",
            "candidates",
            "src",
            "dst",
            "time",
            "row_indices",
        )
    }
    reference_paths = {
        name: tmp_path / f"reference-{name}.npy"
        for name in (
            "candidates",
            "src",
            "dst",
            "time",
            "row_indices",
        )
    }
    for name, path in output_paths.items():
        path.write_bytes(f"frozen-{name}".encode())
    for name, path in reference_paths.items():
        path.write_bytes(output_paths[name].read_bytes())

    train_report = {
        "status": "complete",
        "dataset_name": "dataset2",
        "joint_build": {
            "id": "joint-build-123",
            "pid": 4321,
            "role": "train",
        },
        "checkpoint_sha256": "checkpoint-sha",
        "artifacts": {
            "features": _descriptor(train_feature),
        },
    }
    train_report_path.write_text(
        json.dumps(train_report),
        encoding="utf-8",
    )
    validation_report = {
        "status": "complete",
        "dataset_name": "dataset2",
        "joint_build": {
            "id": "joint-build-123",
            "pid": 9876,
            "role": "validation",
        },
        "train_cache_report": str(train_report_path.resolve()),
        "train_cache_report_sha256": _sha256(train_report_path),
        "train_feature_sha256": _sha256(train_feature),
        "checkpoint_sha256": "checkpoint-sha",
        "validation_shape": [20_000, 100, 63],
        "candidate_shape": [20_000, 100],
        "artifacts": {
            name: _descriptor(path)
            for name, path in output_paths.items()
        },
        "recovery": {
            "protocol": "frozen_candidate_validation_recovery_v1",
            "recovery_process_id": 9876,
            "train_joint_build_id": "joint-build-123",
            "train_joint_process_id": 4321,
            "train_feature_sha256": _sha256(train_feature),
            "candidate_sampling_performed": False,
            "frozen_query_alignment_exact": True,
            "reference_sidecars_exact": dict.fromkeys(
                reference_paths,
                True,
            ),
            "reference_artifacts": {
                name: _descriptor(path)
                for name, path in reference_paths.items()
            },
        },
    }
    validation_report_path.write_text(
        json.dumps(validation_report),
        encoding="utf-8",
    )
    old_validation_report = tmp_path / "old-validation-report.json"
    old_validation_report.write_text('{"status":"old"}', encoding="utf-8")
    marker = {
        "schema_version": 1,
        "status": "complete",
        "stage": "build_joint_cache",
        "evidence": {"return_code": 0},
        "outputs": [
            _descriptor(train_report_path),
            {
                **_descriptor(old_validation_report),
                "path": str(validation_report_path.resolve()),
            },
        ],
    }

    rebound = build_frozen_validation_recovery_marker(
        marker=marker,
        train_report_path=train_report_path,
        validation_report_path=validation_report_path,
        rebound_at_utc="2026-07-30T01:02:03Z",
    )

    assert rebound["schema_version"] == 2
    assert rebound["outputs"][1] == _descriptor(validation_report_path)
    assert rebound["evidence"]["validation_recovery_rebind"][
        "validation_process_id"
    ] == 9876
    assert rebound["evidence"]["validation_recovery_rebind"][
        "previous_validation_report_sha256"
    ] == _sha256(old_validation_report)

    output_paths["candidates"].write_bytes(b"different")
    with pytest.raises(ValueError, match="candidates artifact differs"):
        build_frozen_validation_recovery_marker(
            marker=marker,
            train_report_path=train_report_path,
            validation_report_path=validation_report_path,
            rebound_at_utc="2026-07-30T01:02:03Z",
        )


def test_full_predict_prerequisite_requires_both_zero_exit_codes(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "final-exit-code.txt", "0\n")
    _write(tmp_path / "validation-exit-code.txt", "7\n")
    _write(tmp_path / "dataset2.csv", "0.1,0.9\n")
    _write(tmp_path / "SHA256SUMS", "proof\n")

    with pytest.raises(
        ValueError,
        match="validation exit code must be zero",
    ):
        validate_full_predict_prerequisite(tmp_path)

    _write(tmp_path / "validation-exit-code.txt", "0\n")
    evidence = validate_full_predict_prerequisite(tmp_path)

    assert evidence["status"] == "complete"
    assert evidence["final_exit_code"] == 0
    assert evidence["validation_exit_code"] == 0
    assert evidence["dataset2_csv_bytes"] > 0


def test_duel_manifest_binding_carries_frozen_baseline_hash() -> None:
    plan_lock = {
        "baseline_sha256": "a" * 64,
        "plan_lock_sha256": "b" * 64,
    }

    assert build_duel_manifest_bindings(plan_lock) == {
        "baseline_sha256": "a" * 64,
    }


def test_external_transition_requires_new_selected_gap_aware_lock(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "selection-report.json"
    lock_path = tmp_path / "selection-lock.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "selected",
                "selected_candidate_id": "cooccur_lift_gap_aware_v2",
                "external_holdout_read": False,
            }
        ),
        encoding="utf-8",
    )
    lock_path.write_text(
        json.dumps(
            {
                "protocol": "standard_validation_selection_lock_v1",
                "external_holdout_read": False,
                "selected_candidate": {
                    "candidate_id": "cooccur_lift_full_only_v2",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selected candidate differs"):
        validate_external_transition(
            selection_report_path=report_path,
            selection_lock_path=lock_path,
            expected_candidate_id="cooccur_lift_gap_aware_v2",
        )

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["selected_candidate"]["candidate_id"] = (
        "cooccur_lift_gap_aware_v2"
    )
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    evidence = validate_external_transition(
        selection_report_path=report_path,
        selection_lock_path=lock_path,
        expected_candidate_id="cooccur_lift_gap_aware_v2",
    )

    assert evidence["status"] == "authorized"
    assert evidence["candidate_id"] == "cooccur_lift_gap_aware_v2"
    assert evidence["selection_lock_sha256"]


def test_automatic_stage_order_keeps_external_after_dual_horizon_gate() -> None:
    stages = automatic_stage_order()

    assert stages.index("select_dual_horizon") < stages.index(
        "open_external_gate"
    )
    assert stages[-1] == "open_external_gate"
    validate_stage_order(stages)

    unsafe = list(stages)
    unsafe.remove("open_external_gate")
    unsafe.insert(unsafe.index("select_dual_horizon"), "open_external_gate")
    with pytest.raises(
        ValueError,
        match="external gate must follow dual-horizon selection",
    ):
        validate_stage_order(tuple(unsafe))
