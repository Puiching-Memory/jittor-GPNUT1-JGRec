from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from jgrec.rankers.hybrid.full100_training import (
    validate_joint_cache_reports,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_AUTOMATIC_STAGE_ORDER = (
    "wait_full_predict",
    "build_joint_cache",
    "materialize_near_lift",
    "freeze_v1_contract",
    "freeze_duel_contract",
    "train_duel",
    "select_dual_horizon",
    "train_v1_full_origin",
    "freeze_external_contract",
    "materialize_external",
    "preflight_external",
    "open_external_gate",
)


def automatic_stage_order() -> tuple[str, ...]:
    """Return the frozen automatic stage order."""
    return _AUTOMATIC_STAGE_ORDER


def validate_stage_order(stages: tuple[str, ...]) -> None:
    """Reject orchestration that can read external before selection."""
    if len(stages) != len(set(stages)):
        raise ValueError("automatic pipeline stages must be unique")
    required = set(_AUTOMATIC_STAGE_ORDER)
    if set(stages) != required:
        raise ValueError("automatic pipeline stage set differs")
    if stages.index("open_external_gate") < stages.index(
        "select_dual_horizon"
    ):
        raise ValueError(
            "external gate must follow dual-horizon selection"
        )
    if stages[-1] != "open_external_gate":
        raise ValueError("external gate must be the terminal stage")


def validate_full_predict_prerequisite(
    result_dir: Path | str,
) -> dict[str, Any]:
    """Require the complete, structurally validated K512 prediction."""
    root = Path(result_dir)
    final_code = _read_exit_code(
        root / "final-exit-code.txt",
        label="full predict final",
    )
    if final_code != 0:
        raise ValueError("full predict final exit code must be zero")
    validation_code = _read_exit_code(
        root / "validation-exit-code.txt",
        label="full predict validation",
    )
    if validation_code != 0:
        raise ValueError("full predict validation exit code must be zero")
    csv_path = root / "dataset2.csv"
    checksum_path = root / "SHA256SUMS"
    if not csv_path.is_file() or csv_path.stat().st_size <= 0:
        raise ValueError("full predict dataset2.csv is missing or empty")
    if not checksum_path.is_file() or checksum_path.stat().st_size <= 0:
        raise ValueError("full predict SHA256SUMS is missing or empty")
    return {
        "status": "complete",
        "final_exit_code": final_code,
        "validation_exit_code": validation_code,
        "dataset2_csv": str(csv_path.resolve()),
        "dataset2_csv_bytes": csv_path.stat().st_size,
        "dataset2_csv_sha256": _sha256(csv_path),
        "checksums_sha256": _sha256(checksum_path),
    }


def build_duel_manifest_bindings(
    plan_lock: dict[str, Any],
) -> dict[str, str]:
    """Propagate the immutable baseline binding into the score manifest."""
    baseline_sha256 = plan_lock.get("baseline_sha256")
    if (
        not isinstance(baseline_sha256, str)
        or _SHA256_PATTERN.fullmatch(baseline_sha256) is None
    ):
        raise ValueError("plan lock baseline_sha256 is invalid")
    return {"baseline_sha256": baseline_sha256}


def build_frozen_validation_recovery_marker(
    *,
    marker: dict[str, Any],
    train_report_path: Path | str,
    validation_report_path: Path | str,
    rebound_at_utc: str,
) -> dict[str, Any]:
    """Rebind a completed cache stage only to a proven frozen-query recovery."""
    if (
        marker.get("status") != "complete"
        or marker.get("stage") != "build_joint_cache"
    ):
        raise ValueError("joint cache stage marker differs")
    descriptors = marker.get("outputs")
    if not isinstance(descriptors, list):
        raise ValueError("joint cache stage outputs are malformed")
    train_path = Path(train_report_path)
    validation_path = Path(validation_report_path)
    frozen_train = _find_descriptor(descriptors, train_path)
    frozen_validation = _find_descriptor(descriptors, validation_path)
    current_train = _artifact_descriptor(train_path)
    current_validation = _artifact_descriptor(validation_path)
    if frozen_train != current_train:
        raise ValueError(
            "training cache report changed during validation recovery"
        )
    if frozen_validation == current_validation:
        raise ValueError("validation recovery marker rebind is not required")

    train_report = _read_json(train_path)
    validation_report = _read_json(validation_path)
    lineage = validate_joint_cache_reports(
        train_report,
        validation_report,
    )
    if lineage.get("validation_recovery") is not True:
        raise ValueError("validation report is not a recovery")
    if (
        train_report.get("checkpoint_sha256")
        != validation_report.get("checkpoint_sha256")
    ):
        raise ValueError("validation recovery checkpoint differs")
    if (
        validation_report.get("train_cache_report")
        != str(train_path.resolve())
        or validation_report.get("train_cache_report_sha256")
        != current_train["sha256"]
    ):
        raise ValueError("validation recovery training report binding differs")
    if tuple(validation_report.get("validation_shape", ())) != (
        20_000,
        100,
        63,
    ):
        raise ValueError("validation recovery feature shape differs")
    if tuple(validation_report.get("candidate_shape", ())) != (
        20_000,
        100,
    ):
        raise ValueError("validation recovery candidate shape differs")

    train_artifacts = train_report.get("artifacts")
    if not isinstance(train_artifacts, dict):
        raise ValueError("training cache artifacts are missing")
    _validate_artifact_descriptor(
        train_artifacts.get("features"),
        label="training features",
    )

    artifacts = validation_report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("validation recovery artifacts are missing")
    required_artifacts = (
        "features",
        "candidates",
        "src",
        "dst",
        "time",
        "row_indices",
    )
    for name in required_artifacts:
        _validate_artifact_descriptor(
            artifacts.get(name),
            label=name,
        )

    recovery = validation_report.get("recovery")
    if not isinstance(recovery, dict):
        raise ValueError("validation recovery evidence is missing")
    references = recovery.get("reference_artifacts")
    if not isinstance(references, dict):
        raise ValueError("validation recovery references are missing")
    for name in required_artifacts[1:]:
        output = _validate_artifact_descriptor(
            artifacts.get(name),
            label=name,
        )
        reference = _validate_artifact_descriptor(
            references.get(name),
            label=f"reference {name}",
        )
        if (
            output["bytes"] != reference["bytes"]
            or output["sha256"] != reference["sha256"]
        ):
            raise ValueError(
                f"validation recovery {name} artifact differs "
                "from its frozen reference"
            )

    evidence = marker.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("joint cache stage evidence is malformed")
    rebound = dict(marker)
    rebound["schema_version"] = 2
    rebound["evidence"] = {
        **evidence,
        "validation_recovery_rebind": {
            **lineage,
            "protocol": "frozen_candidate_validation_recovery_v1",
            "rebound_at_utc": rebound_at_utc,
            "previous_validation_report_sha256": frozen_validation[
                "sha256"
            ],
            "validation_report_sha256": current_validation["sha256"],
        },
    }
    rebound["outputs"] = [
        (
            current_validation
            if Path(str(item.get("path", ""))).resolve()
            == validation_path.resolve()
            else item
        )
        for item in descriptors
        if isinstance(item, dict)
    ]
    if len(rebound["outputs"]) != len(descriptors):
        raise ValueError("joint cache stage output descriptors differ")
    return rebound


def validate_external_transition(
    *,
    selection_report_path: Path | str,
    selection_lock_path: Path | str,
    expected_candidate_id: str,
) -> dict[str, str]:
    """Authorize one external stage only from a matching selected lock."""
    report_path = Path(selection_report_path)
    lock_path = Path(selection_lock_path)
    report = _read_json(report_path)
    lock = _read_json(lock_path)
    if report.get("status") != "selected":
        raise ValueError("selection report did not select a candidate")
    if report.get("external_holdout_read") is not False:
        raise ValueError("selection report already consumed external")
    report_candidate = report.get("selected_candidate_id")
    selected = lock.get("selected_candidate")
    if not isinstance(selected, dict):
        raise ValueError("selection lock has no selected candidate")
    lock_candidate = selected.get("candidate_id")
    if report_candidate != lock_candidate:
        raise ValueError("selection report and lock selected candidate differs")
    if report_candidate != expected_candidate_id:
        raise ValueError("selected candidate differs from external implementation")
    if (
        lock.get("protocol")
        != "standard_validation_selection_lock_v1"
    ):
        raise ValueError("selection lock protocol differs")
    if lock.get("external_holdout_read") is not False:
        raise ValueError("selection lock already consumed external")
    return {
        "status": "authorized",
        "candidate_id": expected_candidate_id,
        "selection_report_sha256": _sha256(report_path),
        "selection_lock_sha256": _sha256(lock_path),
    }


def _read_exit_code(path: Path, *, label: str) -> int:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return int(raw)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} exit code is missing or invalid") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _find_descriptor(
    descriptors: list[Any],
    path: Path,
) -> dict[str, Any]:
    matched = [
        item
        for item in descriptors
        if isinstance(item, dict)
        and Path(str(item.get("path", ""))).resolve() == path.resolve()
    ]
    if len(matched) != 1:
        raise ValueError(f"stage output descriptor differs: {path}")
    return matched[0]


def _validate_artifact_descriptor(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} artifact descriptor is missing")
    path = Path(str(value.get("path", "")))
    actual = _artifact_descriptor(path)
    if value != actual:
        raise ValueError(f"{label} artifact differs")
    return actual


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"artifact is missing: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
