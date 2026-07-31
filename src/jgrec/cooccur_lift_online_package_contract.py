from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPECTED_PROTOCOL = "cooccur_lift_k512_successor_v2_online_package_v1"
EXPECTED_STATUS = "frozen_after_external_acceptance_before_online_scoring"
EXPECTED_CANDIDATE_ID = "cooccur_lift_gap_aware_v2"
EXPECTED_BASELINE_ROLE = "current_k512_bugfixed_v1_new_champion"
EXPECTED_GATE_NAMES = (
    "mrr_meets_minimum",
    "hit_at_1_meets_minimum",
    "hit_at_3_meets_minimum",
    "hit_at_10_meets_minimum",
    "ndcg_at_10_meets_minimum",
    "mean_rank_meets_maximum",
    "improved_minus_worsened_meets_minimum",
)
EXPECTED_WEIGHT = 0.5


def validate_k512_online_package_preflight(
    *,
    root: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Validate the complete current-run lineage before online scoring."""
    root = Path(root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _read_json(contract_path)
    _validate_contract_header(contract)

    input_paths = _path_map(contract, "input_paths")
    input_hashes = _string_map(contract, "input_sha256")
    implementation_paths = _path_map(contract, "implementation_paths")
    implementation_hashes = _string_map(
        contract,
        "implementation_sha256",
    )
    resolved_inputs = _verify_path_hashes(
        root=root,
        paths=input_paths,
        expected_hashes=input_hashes,
        role="input",
    )
    _verify_path_hashes(
        root=root,
        paths=implementation_paths,
        expected_hashes=implementation_hashes,
        role="implementation",
    )

    selection = _read_json(resolved_inputs["selection_lock"])
    external = _read_json(resolved_inputs["external_report"])
    receipt = _read_json(resolved_inputs["external_open_receipt"])
    materialization = _read_json(
        resolved_inputs["external_materialization_report"]
    )
    v1_contract = _read_json(resolved_inputs["bugfixed_v1_contract"])
    v1_training = _read_json(
        resolved_inputs["bugfixed_v1_training_report"]
    )
    _validate_selection_and_external(
        contract=contract,
        selection=selection,
        external=external,
        receipt=receipt,
        materialization=materialization,
        input_hashes=input_hashes,
    )
    _validate_current_v1(
        contract=v1_contract,
        training=v1_training,
        input_hashes=input_hashes,
    )
    member_hashes = _validate_source_package(
        root=root,
        contract=contract,
        source_zip=resolved_inputs["source_champion_zip"],
        v1_contract=v1_contract,
    )
    _validate_outputs_absent(root=root, contract=contract)

    return {
        "schema_version": 1,
        "protocol": EXPECTED_PROTOCOL,
        "status": "passed",
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "selected_weight": EXPECTED_WEIGHT,
        "baseline_role": EXPECTED_BASELINE_ROLE,
        "input_count": len(input_paths),
        "implementation_count": len(implementation_paths),
        "all_seven_gates_passed": True,
        "outputs_absent": True,
        "source_member_sha256": member_hashes,
        "external_effect_size_used": False,
    }


def _validate_contract_header(contract: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "protocol": EXPECTED_PROTOCOL,
        "status": EXPECTED_STATUS,
        "candidate_id": EXPECTED_CANDIDATE_ID,
        "selected_weight": EXPECTED_WEIGHT,
        "baseline_role": EXPECTED_BASELINE_ROLE,
        "external_decision_role": "safety_gate_only",
        "external_effect_size_used": False,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"online package contract {key} differs")
    authorization = _mapping(contract, "authorization")
    if (
        tuple(authorization.get("external_gates_required", ()))
        != EXPECTED_GATE_NAMES
        or authorization.get("package_authorized") is not True
        or authorization.get("weight_rescan_authorized") is not False
        or authorization.get("feature_rescan_authorized") is not False
        or authorization.get("external_reopen_authorized") is not False
    ):
        raise ValueError("online package authorization differs")


def _validate_selection_and_external(
    *,
    contract: Mapping[str, Any],
    selection: Mapping[str, Any],
    external: Mapping[str, Any],
    receipt: Mapping[str, Any],
    materialization: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> None:
    selected = _mapping(selection, "selected_candidate")
    if (
        selected.get("candidate_id") != EXPECTED_CANDIDATE_ID
        or selected.get("config_sha256")
        != input_hashes["candidate_config"]
        or selection.get("weight_rescan_authorized") is not False
        or selection.get("feature_rescan_authorized") is not False
    ):
        raise ValueError("selected successor candidate differs")

    gates = _mapping(external, "gates")
    if (
        external.get("status") != "accepted"
        or external.get("package_authorized") is not True
        or external.get("failed_gates") != []
        or set(gates) != set(EXPECTED_GATE_NAMES)
        or not all(gates[name] is True for name in EXPECTED_GATE_NAMES)
        or external.get("selection_lock_sha256")
        != input_hashes["selection_lock"]
        or external.get("weight_rescan_authorized") is not False
        or external.get("feature_rescan_authorized") is not False
        or external.get("leaderboard_tuning_authorized") is not False
    ):
        raise ValueError("external package authorization differs")
    external_selected = _mapping(external, "selected_candidate")
    if (
        external_selected.get("candidate_id") != EXPECTED_CANDIDATE_ID
        or external_selected.get("config_sha256")
        != input_hashes["candidate_config"]
    ):
        raise ValueError("external selected candidate differs")
    if (
        receipt.get("selected_candidate_id") != EXPECTED_CANDIDATE_ID
        or receipt.get("selection_lock_sha256")
        != input_hashes["selection_lock"]
        or receipt.get("external_manifest_sha256")
        != input_hashes["external_manifest"]
        or external.get("external_manifest_sha256")
        != input_hashes["external_manifest"]
    ):
        raise ValueError("external receipt binding differs")

    replay = _mapping(materialization, "deterministic_replay")
    if (
        materialization.get("status")
        != "external_candidate_materialized_metrics_unread"
        or materialization.get("candidate_id") != EXPECTED_CANDIDATE_ID
        or materialization.get("candidate_config_sha256")
        != input_hashes["candidate_config"]
        or materialization.get("selection_lock_sha256")
        != input_hashes["selection_lock"]
        or materialization.get("execution_contract_sha256")
        != input_hashes["external_execution_contract"]
        or materialization.get("external_manifest_sha256")
        != input_hashes["external_manifest"]
        or materialization.get("model_sha256")
        != input_hashes["gap_aware_model"]
        or float(materialization.get("selected_weight", -1.0))
        != EXPECTED_WEIGHT
        or materialization.get("effect_size_estimation_authorized")
        is not False
        or replay.get("matched") is not True
        or replay.get("tolerance_relaxed") is not False
    ):
        raise ValueError("external materialization binding differs")
    if contract.get("external_effect_size_used") is not False:
        raise ValueError("external effect size use is forbidden")


def _validate_current_v1(
    *,
    contract: Mapping[str, Any],
    training: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> None:
    replay = _mapping(training, "deterministic_replay")
    if (
        contract.get("status") != "frozen_before_bugfixed_refit"
        or float(contract.get("selected_weight", -1.0))
        != EXPECTED_WEIGHT
        or training.get("status")
        != "complete_deterministic_bugfixed_v1_refit"
        or training.get("candidate_contract_sha256")
        != input_hashes["bugfixed_v1_contract"]
        or training.get("model_sha256")
        != input_hashes["bugfixed_v1_model"]
        or float(training.get("selected_weight", -1.0))
        != EXPECTED_WEIGHT
        or replay.get("matched") is not True
        or replay.get("tolerance_relaxed") is not False
    ):
        raise ValueError("current K512 bugfixed V1 binding differs")


def _validate_source_package(
    *,
    root: Path,
    contract: Mapping[str, Any],
    source_zip: Path,
    v1_contract: Mapping[str, Any],
) -> dict[str, str]:
    package = _mapping(contract, "source_package_contract")
    if _resolve_under_root(root, str(package.get("zip", ""))) != source_zip:
        raise ValueError("source package path differs")
    with zipfile.ZipFile(source_zip) as archive:
        names = archive.namelist()
        if sorted(names) != ["dataset1.csv", "dataset2.csv"]:
            raise ValueError("source package members differ")
        member_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in names
        }
    if (
        member_hashes["dataset1.csv"]
        != package.get("dataset1_member_sha256")
        or member_hashes["dataset2.csv"]
        != package.get("dataset2_member_sha256")
    ):
        raise ValueError("source package member hash differs")
    v1_package = _mapping(v1_contract, "package_contract")
    if (
        v1_package.get("champion_zip_sha256") != _sha256(source_zip)
        or v1_package.get("dataset1_member_sha256")
        != member_hashes["dataset1.csv"]
        or v1_package.get("dataset2_member_sha256")
        != member_hashes["dataset2.csv"]
    ):
        raise ValueError("current V1 source package contract differs")
    return member_hashes


def _validate_outputs_absent(
    *,
    root: Path,
    contract: Mapping[str, Any],
) -> None:
    stages = _mapping(contract, "stage_contract")
    output_keys = (
        "v1_materialization_dir",
        "v1_output_dir",
        "v2_materialization_dir",
        "v2_output_dir",
    )
    for key in output_keys:
        path = _resolve_under_root(root, str(stages.get(key, "")))
        if path.exists():
            raise ValueError(f"refusing to overwrite package output: {key}")


def _verify_path_hashes(
    *,
    root: Path,
    paths: Mapping[str, str],
    expected_hashes: Mapping[str, str],
    role: str,
) -> dict[str, Path]:
    if paths.keys() != expected_hashes.keys():
        raise ValueError(f"{role} path/hash key set differs")
    resolved: dict[str, Path] = {}
    for name, relative in paths.items():
        path = _resolve_under_root(root, relative)
        if not path.is_file():
            raise ValueError(f"{role} file is missing: {name}")
        actual = _sha256(path)
        if actual != expected_hashes[name]:
            raise ValueError(f"{role} hash differs: {name}")
        resolved[name] = path
    return resolved


def _resolve_under_root(root: Path, value: str) -> Path:
    if not value:
        raise ValueError("package contract path is empty")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("package contract paths must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("package contract path escapes root") from error
    return resolved


def _path_map(value: Mapping[str, Any], name: str) -> dict[str, str]:
    result = _string_map(value, name)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _string_map(value: Mapping[str, Any], name: str) -> dict[str, str]:
    result = _mapping(value, name)
    if not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in result.items()
    ):
        raise ValueError(f"{name} must map strings to strings")
    return dict(result)


def _mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise ValueError(f"{name} must be an object")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
