"""Frozen external-gate helpers for the cooccur-lift successor.

The external holdout is a one-shot near-horizon safety gate.  Far-horizon
evidence belongs to the already-completed gapped-fold validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXTERNAL_GATE_NAMES = (
    "mrr_meets_minimum",
    "hit_at_1_meets_minimum",
    "hit_at_3_meets_minimum",
    "hit_at_10_meets_minimum",
    "ndcg_at_10_meets_minimum",
    "mean_rank_meets_maximum",
    "improved_minus_worsened_meets_minimum",
)

_EXPECTED_MINIMUM_DELTAS = {
    "mrr": 0.0,
    "hit_at_1": 0.0,
    "hit_at_3": 0.0,
    "hit_at_10": 0.0,
    "ndcg_at_10": 0.0,
}
_EXPECTED_MAXIMUM_DELTAS = {"mean_rank": 0.0}


@dataclass(frozen=True)
class SuccessorExternalSetup:
    """Validated immutable inputs for the one-shot successor external gate."""

    experiment_id: str
    candidate_id: str
    config_sha256: str
    selection_lock_sha256: str
    baseline_sha256: str
    selected_weight: float
    full_origin_seed: int
    short_window_seconds: int
    collapsed_fraction: float
    gap_seconds: tuple[int, ...]
    holdout_id: str
    lineage_sha256: str
    deployment_horizon_seconds: int
    minimum_horizon_seconds: int
    minimum_start_gap_seconds: int
    external_gate_names: tuple[str, ...] = EXTERNAL_GATE_NAMES


def validate_successor_external_setup(
    *,
    candidate_config_path: Path,
    selection_lock_path: Path,
) -> SuccessorExternalSetup:
    """Validate the chosen candidate and exact seven-gate external contract."""

    candidate_config_path = Path(candidate_config_path)
    selection_lock_path = Path(selection_lock_path)
    config = _read_json(candidate_config_path)
    lock = _read_json(selection_lock_path)

    if config.get("candidate_id") != "cooccur_lift_gap_aware_v2":
        raise ValueError("external candidate must be gap-aware v2")
    if lock.get("protocol") != "standard_validation_selection_lock_v1":
        raise ValueError("unexpected standard selection-lock protocol")
    if lock.get("external_holdout_read") is not False:
        raise ValueError("external holdout was already read")
    for field in (
        "weight_rescan_authorized",
        "feature_rescan_authorized",
        "leaderboard_tuning_authorized",
    ):
        if lock.get(field) is not False:
            raise ValueError(f"{field} must remain false")

    config_sha256 = _sha256(candidate_config_path)
    selected = lock.get("selected_candidate")
    if not isinstance(selected, dict):
        raise ValueError("selection lock has no selected candidate")
    if selected.get("candidate_id") != config["candidate_id"]:
        raise ValueError("selected candidate differs from candidate config")
    if selected.get("config_sha256") != config_sha256:
        raise ValueError("selected candidate config hash differs")

    selected_weight = float(config["integration"]["selected_weight"])
    if selected_weight != 0.5:
        raise ValueError("successor integration weight must remain 0.5")
    if config["integration"].get("weight_rescan_authorized", False):
        raise ValueError("successor weight rescan is forbidden")

    horizon = config["horizon_training"]
    support_contract = horizon["support_state_training_contract"]
    collapsed_fraction = float(
        support_contract["collapsed_copy_weight"]
    )
    near_fraction = float(support_contract["near_copy_weight"])
    if not np.isclose(
        near_fraction + collapsed_fraction,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("support-state training weights must sum to one")
    gaps = tuple(
        int(spec["gap_seconds"])
        for spec in horizon["gapped_fold_specs"]
    )
    if len(gaps) != 3:
        raise ValueError("gap-aware v2 requires exactly three gapped states")

    short_window_seconds = int(
        config["short_window"]["short_window_seconds"]
    )
    if any(gap < short_window_seconds for gap in gaps):
        raise ValueError("every gapped state must cover the short window")

    randomness = config["randomness"]
    full_origin_seed = (
        int(randomness["base_seed"])
        + len(gaps) * 1009
        + int(randomness["seed_salt"])
    )

    external = lock.get("external_gate")
    if not isinstance(external, dict):
        raise ValueError("selection lock has no external gate")
    _validate_exact_external_gate(external)

    baseline_sha256 = lock.get("baseline_sha256")
    if not _is_sha256(baseline_sha256):
        raise ValueError("selection lock baseline hash is invalid")

    return SuccessorExternalSetup(
        experiment_id=str(lock["experiment_id"]),
        candidate_id=str(config["candidate_id"]),
        config_sha256=config_sha256,
        selection_lock_sha256=_sha256(selection_lock_path),
        baseline_sha256=baseline_sha256,
        selected_weight=selected_weight,
        full_origin_seed=full_origin_seed,
        short_window_seconds=short_window_seconds,
        collapsed_fraction=collapsed_fraction,
        gap_seconds=gaps,
        holdout_id=str(external["holdout_id"]),
        lineage_sha256=str(external["lineage_sha256"]),
        deployment_horizon_seconds=int(
            external["deployment_horizon_seconds"]
        ),
        minimum_horizon_seconds=int(external["minimum_horizon_seconds"]),
        minimum_start_gap_seconds=int(
            external["minimum_start_gap_seconds"]
        ),
    )


def full_origin_copy_weights(
    *,
    collapsed_fraction: float,
    gapped_copy_count: int,
) -> tuple[float, ...]:
    """Return near-copy weight followed by equal per-gap collapsed weights."""

    collapsed_fraction = float(collapsed_fraction)
    if not 0.0 < collapsed_fraction < 1.0:
        raise ValueError("collapsed_fraction must be strictly within (0, 1)")
    if gapped_copy_count <= 0:
        raise ValueError("gapped_copy_count must be positive")
    collapsed_copy = collapsed_fraction / gapped_copy_count
    return (1.0 - collapsed_fraction,) + (
        collapsed_copy,
    ) * gapped_copy_count


def short_window_support_from_availability(
    query_time: np.ndarray,
    availability_time: np.ndarray,
    *,
    short_window_seconds: int,
) -> np.ndarray:
    """Compute the frozen strict-boundary support indicator."""

    query = np.asarray(query_time, dtype=np.int64)
    availability = np.asarray(availability_time, dtype=np.int64)
    if query.shape != availability.shape:
        raise ValueError("query and availability times must have equal shape")
    if short_window_seconds <= 0:
        raise ValueError("short_window_seconds must be positive")
    age = query - availability
    if np.any(age < 0):
        raise ValueError("availability time cannot follow query time")
    return (age < short_window_seconds).astype(np.float32)


def build_standard_external_manifest(
    *,
    setup: SuccessorExternalSetup,
    candidate_fingerprint: str,
    training_time_max: int,
    score_time_min: int,
    score_time_max: int,
    baseline_path: Path,
    baseline_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
    scored_rows: int,
    supported_rows: int,
) -> dict[str, Any]:
    """Build a standard manifest without opening or scoring the holdout."""

    if not _is_sha256(candidate_fingerprint):
        raise ValueError("candidate fingerprint is invalid")
    for label, value in (
        ("baseline artifact", baseline_sha256),
        ("candidate artifact", candidate_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{label} hash is invalid")
    if scored_rows <= 0 or not 0 <= supported_rows <= scored_rows:
        raise ValueError("short-window support row counts are invalid")
    if score_time_min < training_time_max:
        raise ValueError("external score interval precedes training")
    if (
        score_time_max - training_time_max
        < setup.minimum_horizon_seconds
    ):
        raise ValueError("external score interval is shorter than required")

    supported_values = [1] if supported_rows == scored_rows else [0, 1]
    return {
        "schema_version": 1,
        "protocol": "standard_external_scores_v1",
        "selection_lock_sha256": setup.selection_lock_sha256,
        "experiment_id": setup.experiment_id,
        "holdout_id": setup.holdout_id,
        "lineage_sha256": setup.lineage_sha256,
        "baseline_sha256": setup.baseline_sha256,
        "selected_candidate_id": setup.candidate_id,
        "selected_candidate_config_sha256": setup.config_sha256,
        "positive_candidate_column": 0,
        "candidate_fingerprint": candidate_fingerprint,
        "training_time_max": int(training_time_max),
        "score_time_min": int(score_time_min),
        "score_time_max": int(score_time_max),
        "baseline": {
            "path": str(Path(baseline_path)),
            "sha256": baseline_sha256,
        },
        "candidate": {
            "path": str(Path(candidate_path)),
            "sha256": candidate_sha256,
            "candidate_id": setup.candidate_id,
            "config_sha256": setup.config_sha256,
            "candidate_fingerprint": candidate_fingerprint,
        },
        "short_window_support": {
            "collapsed_fraction": float(
                (scored_rows - supported_rows) / scored_rows
            ),
            "supported_rows": int(supported_rows),
            "total_rows": int(scored_rows),
            "unique_values": supported_values,
        },
    }


def authorize_successor_package(
    *,
    external_report: dict[str, Any],
    external_report_sha256: str,
    expected_selection_lock_sha256: str,
    expected_candidate_id: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Authorize packaging from bindings and the seven booleans only."""

    if external_report.get("status") != "accepted":
        raise ValueError("external seven-gate status is not accepted")
    if external_report.get("package_authorized") is not True:
        raise ValueError("external seven-gate package authorization is false")
    if (
        external_report.get("selection_lock_sha256")
        != expected_selection_lock_sha256
    ):
        raise ValueError("external report selection-lock binding differs")

    selected = external_report.get("selected_candidate")
    if not isinstance(selected, dict):
        raise ValueError("external report selected candidate is missing")
    if selected.get("candidate_id") != expected_candidate_id:
        raise ValueError("external report candidate binding differs")
    if selected.get("config_sha256") != expected_config_sha256:
        raise ValueError("external report config binding differs")

    gates = external_report.get("gates")
    if (
        not isinstance(gates, dict)
        or len(gates) != len(EXTERNAL_GATE_NAMES)
        or set(gates) != set(EXTERNAL_GATE_NAMES)
    ):
        raise ValueError("external report does not contain the exact seven-gate")
    if any(gates[name] is not True for name in EXTERNAL_GATE_NAMES):
        raise ValueError("external seven-gate did not pass")
    if not _is_sha256(external_report_sha256):
        raise ValueError("external report hash is invalid")

    return {
        "decision_role": "safety_gate_only",
        "effect_size_estimation_authorized": False,
        "external_report_sha256": external_report_sha256,
        "selection_lock_sha256": expected_selection_lock_sha256,
        "candidate_id": expected_candidate_id,
        "config_sha256": expected_config_sha256,
        "gate_count": len(EXTERNAL_GATE_NAMES),
        "gates": dict.fromkeys(EXTERNAL_GATE_NAMES, True),
    }


def _validate_exact_external_gate(external: dict[str, Any]) -> None:
    if external.get("strictly_increasing_metrics") != ["mrr"]:
        raise ValueError("external gate must require strict MRR improvement")
    if external.get("minimum_deltas") != _EXPECTED_MINIMUM_DELTAS:
        raise ValueError("external gate minimum deltas changed")
    if external.get("maximum_deltas") != _EXPECTED_MAXIMUM_DELTAS:
        raise ValueError("external gate maximum deltas changed")
    if external.get("minimum_improved_minus_worsened") != 1:
        raise ValueError("external movement gate changed")
    policy = external.get("interpretation_policy")
    if not isinstance(policy, dict):
        raise ValueError("time-local external interpretation is missing")
    if policy.get("decision_role") != "safety_gate_only":
        raise ValueError("external must remain a safety gate only")
    if policy.get("effect_size_estimation_authorized") is not False:
        raise ValueError("external effect-size estimation is forbidden")
    if float(policy.get("calibration_discount_factor", 0.0)) != 19.5:
        raise ValueError("external calibration discount changed")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
