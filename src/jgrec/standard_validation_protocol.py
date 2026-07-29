from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .robust_weight_selection import ranking_metrics

STANDARD_METRIC_NAMES = (
    "mrr",
    "hit_at_1",
    "hit_at_3",
    "hit_at_10",
    "ndcg_at_10",
    "mean_rank",
)
_SELECTION_ORDER = [
    "maximum_mean_fold_mrr_delta",
    "maximum_worst_fold_mrr_delta",
    "maximum_mean_fold_ndcg_at_10_delta",
    "minimum_tie_break_priority",
]
_FAR_HORIZON_SELECTION_ORDER = [
    "maximum_deployment_mixture_mrr_delta",
    "maximum_worst_gapped_mrr_delta",
    "maximum_deployment_mixture_ndcg_at_10_delta",
    "minimum_tie_break_priority",
]
_DUAL_HORIZON_SELECTION_ORDER = [
    "maximum_mean_gapped_mrr_delta",
    "maximum_worst_gapped_mrr_delta",
    "maximum_mean_near_mrr_delta",
    "minimum_tie_break_priority",
]
_DEPLOYMENT_MIXTURE_ELIGIBILITY = "deployment_mixture"
_DUAL_HORIZON_ELIGIBILITY = (
    "near_non_decreasing_and_gapped_strict_improvement"
)
_TOLERANCE = 1e-15


def freeze_standard_validation_plan(
    *,
    plan_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Freeze all candidate, selection, and external decisions before scoring."""

    plan_path = Path(plan_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    plan = _read_json(plan_path)
    (
        experiment_id,
        candidate_space,
        selection_policy,
        external_policy,
        temporal_scope,
        far_horizon_policy,
        baseline,
    ) = _validate_plan(plan)
    candidate_space_sha256 = _json_sha256(candidate_space)
    selection_policy_sha256 = _json_sha256(selection_policy)
    external_policy_sha256 = _json_sha256(external_policy)
    temporal_scope_sha256 = (
        None if temporal_scope is None else _json_sha256(temporal_scope)
    )
    far_horizon_policy_sha256 = (
        None
        if far_horizon_policy is None
        else _json_sha256(far_horizon_policy)
    )
    baseline_sha256 = (
        None if baseline is None else _json_sha256(baseline)
    )
    lock = {
        "schema_version": 1,
        "protocol": "standard_validation_plan_lock_v1",
        "experiment_id": experiment_id,
        "candidate_family": plan["candidate_family"],
        "source_plan": str(plan_path.resolve()),
        "source_plan_sha256": _sha256(plan_path),
        "candidate_ids": [
            candidate["candidate_id"] for candidate in candidate_space
        ],
        "candidate_space": candidate_space,
        "candidate_space_sha256": candidate_space_sha256,
        "rolling_selection": selection_policy,
        "selection_policy_sha256": selection_policy_sha256,
        "external_gate": external_policy,
        "external_policy_sha256": external_policy_sha256,
        "selection_metrics_read": False,
        "reserved_fold_metrics_read": False,
        "external_holdout_read": False,
    }
    if baseline is not None:
        lock["baseline"] = baseline
        lock["baseline_sha256"] = baseline_sha256
    if temporal_scope is not None:
        lock["temporal_scope"] = temporal_scope
        lock["temporal_scope_sha256"] = temporal_scope_sha256
        lock["far_horizon_validation"] = far_horizon_policy
        lock["far_horizon_policy_sha256"] = far_horizon_policy_sha256
    report = {
        "schema_version": 1,
        "protocol": "standard_validation_plan_preflight_v1",
        "status": "ready_for_rolling_selection",
        "experiment_id": experiment_id,
        "source_plan_sha256": lock["source_plan_sha256"],
        "candidate_space_sha256": candidate_space_sha256,
        "selection_policy_sha256": selection_policy_sha256,
        "external_policy_sha256": external_policy_sha256,
        "candidate_count": len(candidate_space),
        "minimum_selection_folds": selection_policy["minimum_folds"],
        "reserved_gate_folds": selection_policy[
            "reserved_gate_folds"
        ],
        "deployment_horizon_seconds": external_policy[
            "deployment_horizon_seconds"
        ],
        "minimum_horizon_seconds": external_policy[
            "minimum_horizon_seconds"
        ],
        "selection_metrics_read": False,
        "reserved_fold_metrics_read": False,
        "external_holdout_read": False,
        "package_authorized": False,
    }
    if baseline is not None:
        report["baseline_id"] = baseline["baseline_id"]
        report["baseline_sha256"] = baseline_sha256
    if temporal_scope is not None:
        report.update(
            {
                "temporal_scope": temporal_scope,
                "temporal_scope_sha256": temporal_scope_sha256,
                "far_horizon_policy_sha256": far_horizon_policy_sha256,
                "minimum_gapped_folds": far_horizon_policy[
                    "minimum_gapped_folds"
                ],
                "deployment_collapsed_fraction": far_horizon_policy[
                    "deployment_collapsed_fraction"
                ],
                "external_decision_role": external_policy[
                    "interpretation_policy"
                ]["decision_role"],
                "eligibility_rule": far_horizon_policy.get(
                    "eligibility_rule",
                    _DEPLOYMENT_MIXTURE_ELIGIBILITY,
                ),
            }
        )
    output_dir.mkdir(parents=True)
    _write_json_exclusive(
        output_dir / "validation-plan-lock.json",
        lock,
    )
    _write_json_exclusive(
        output_dir / "preflight-report.json",
        report,
    )
    return report


def select_standard_rolling_candidate(
    *,
    manifest_path: Path,
    plan_lock_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Select one preregistered candidate from equal-weight rolling folds."""

    manifest_path = Path(manifest_path)
    plan_lock_path = Path(plan_lock_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    plan_lock = _read_json(plan_lock_path)
    manifest = _read_json(manifest_path)
    (
        candidate_space,
        selection_policy,
        folds,
        far_horizon_policy,
        gapped_folds,
    ) = _validate_rolling_contract(
        manifest=manifest,
        plan_lock=plan_lock,
        plan_lock_sha256=_sha256(plan_lock_path),
    )
    positive_column = int(manifest.get("positive_candidate_column", 0))
    candidate_reports: dict[str, dict[str, Any]] = {}
    candidate_score_hashes: dict[str, Any] = {}
    near_baselines, near_baseline_metrics = _load_fold_group_baselines(
        folds=folds,
        manifest_dir=manifest_path.parent,
        positive_column=positive_column,
    )
    gapped_baselines: list[np.ndarray] = []
    gapped_baseline_metrics: list[dict[str, Any]] = []
    counterfactual_baselines: dict[
        str,
        tuple[list[np.ndarray], list[dict[str, Any]]],
    ] = {}
    if far_horizon_policy is not None:
        gapped_baselines, gapped_baseline_metrics = (
            _load_fold_group_baselines(
                folds=gapped_folds,
                manifest_dir=manifest_path.parent,
                positive_column=positive_column,
            )
        )
        counterfactual_policy = far_horizon_policy[
            "zero_short_counterfactual"
        ]
        if counterfactual_policy["enabled"]:
            arm_id = counterfactual_policy["arm_id"]
            counterfactual_baselines[arm_id] = (
                _load_fold_group_baselines(
                    folds=folds,
                    manifest_dir=manifest_path.parent,
                    positive_column=positive_column,
                    counterfactual_arm_id=arm_id,
                )
            )

    for candidate in candidate_space:
        candidate_id = candidate["candidate_id"]
        near = _score_candidate_fold_group(
            candidate_id=candidate_id,
            folds=folds,
            baselines=near_baselines,
            baseline_metrics=near_baseline_metrics,
            manifest_dir=manifest_path.parent,
            positive_column=positive_column,
        )
        report_fields: dict[str, Any] = {
            "candidate_id": candidate_id,
            "config_sha256": candidate["config_sha256"],
            "folds": near["folds"],
            "mean_fold_baseline": near["mean_baseline"],
            "mean_fold_candidate": near["mean_candidate"],
            "mean_fold_delta": near["mean_delta"],
            "query_movements": near["query_movements"],
            "stability": {
                "fold_mrr_deltas": [
                    delta["mrr"] for delta in near["fold_deltas"]
                ],
                "fold_ndcg_at_10_deltas": [
                    delta["ndcg_at_10"]
                    for delta in near["fold_deltas"]
                ],
                "worst_fold_mrr_delta": float(
                    min(delta["mrr"] for delta in near["fold_deltas"])
                ),
            },
            "tie_break_priority": candidate["tie_break_priority"],
        }
        if far_horizon_policy is None:
            gates = _rolling_gates(
                fold_deltas=near["fold_deltas"],
                mean_delta=near["mean_delta"],
                improved=near["query_movements"]["improved"],
                worsened=near["query_movements"]["worsened"],
                policy=selection_policy,
            )
            candidate_score_hashes[candidate_id] = near["score_hashes"]
        else:
            gapped = _score_candidate_fold_group(
                candidate_id=candidate_id,
                folds=gapped_folds,
                baselines=gapped_baselines,
                baseline_metrics=gapped_baseline_metrics,
                manifest_dir=manifest_path.parent,
                positive_column=positive_column,
            )
            collapsed_fraction = float(
                far_horizon_policy["deployment_collapsed_fraction"]
            )
            deployment_delta = _weighted_metric_panels(
                near["mean_delta"],
                gapped["mean_delta"],
                right_weight=collapsed_fraction,
            )
            deployment_movement_rate = (
                (1.0 - collapsed_fraction) * near["movement_balance_rate"]
                + collapsed_fraction * gapped["movement_balance_rate"]
            )
            gates = _far_horizon_gates(
                near_fold_deltas=near["fold_deltas"],
                gapped_fold_deltas=gapped["fold_deltas"],
                deployment_delta=deployment_delta,
                deployment_movement_rate=deployment_movement_rate,
                rolling_policy=selection_policy,
                far_horizon_policy=far_horizon_policy,
            )
            counterfactual_reports: dict[str, Any] = {}
            counterfactual_hashes: dict[str, Any] = {}
            counterfactual_policy = far_horizon_policy[
                "zero_short_counterfactual"
            ]
            if counterfactual_policy["enabled"]:
                arm_id = counterfactual_policy["arm_id"]
                arm_baselines, arm_baseline_metrics = (
                    counterfactual_baselines[arm_id]
                )
                counterfactual = _score_candidate_fold_group(
                    candidate_id=candidate_id,
                    folds=folds,
                    baselines=arm_baselines,
                    baseline_metrics=arm_baseline_metrics,
                    manifest_dir=manifest_path.parent,
                    positive_column=positive_column,
                    counterfactual_arm_id=arm_id,
                )
                counterfactual_reports[arm_id] = {
                    "participates_in_selection": False,
                    "folds": counterfactual["folds"],
                    "mean_fold_baseline": counterfactual["mean_baseline"],
                    "mean_fold_candidate": counterfactual["mean_candidate"],
                    "mean_fold_delta": counterfactual["mean_delta"],
                    "query_movements": counterfactual["query_movements"],
                }
                counterfactual_hashes[arm_id] = counterfactual[
                    "score_hashes"
                ]
            report_fields.update(
                {
                    "gapped_folds": gapped["folds"],
                    "mean_gapped_fold_baseline": gapped["mean_baseline"],
                    "mean_gapped_fold_candidate": gapped["mean_candidate"],
                    "mean_gapped_fold_delta": gapped["mean_delta"],
                    "gapped_query_movements": gapped["query_movements"],
                    "deployment_collapsed_fraction": collapsed_fraction,
                    "deployment_mixture_delta": deployment_delta,
                    "deployment_movement_balance_rate": (
                        deployment_movement_rate
                    ),
                    "counterfactual_arms": counterfactual_reports,
                }
            )
            report_fields["stability"].update(
                {
                    "gapped_fold_mrr_deltas": [
                        delta["mrr"] for delta in gapped["fold_deltas"]
                    ],
                    "worst_gapped_mrr_delta": float(
                        min(
                            delta["mrr"]
                            for delta in gapped["fold_deltas"]
                        )
                    ),
                }
            )
            candidate_score_hashes[candidate_id] = {
                "near": near["score_hashes"],
                "gapped": gapped["score_hashes"],
                "counterfactual_arms": counterfactual_hashes,
            }
        failed_gates = [
            name for name, passed in gates.items() if not passed
        ]
        candidate_reports[candidate_id] = {
            **report_fields,
            "eligible": not failed_gates,
            "failed_gates": failed_gates,
            "gates": gates,
        }

    eligible = [
        candidate
        for candidate in candidate_space
        if candidate_reports[candidate["candidate_id"]]["eligible"]
    ]
    selected = (
        max(
            eligible,
            key=lambda candidate: _selection_key(
                candidate_reports[candidate["candidate_id"]],
                far_horizon_policy=far_horizon_policy,
            ),
        )
        if eligible
        else None
    )
    selected_id = (
        None if selected is None else selected["candidate_id"]
    )
    report = {
        "schema_version": 1,
        "protocol": "standard_rolling_origin_selection_v1",
        "status": "selected" if selected is not None else "rejected",
        "experiment_id": plan_lock["experiment_id"],
        "plan_lock_sha256": _sha256(plan_lock_path),
        "rolling_manifest_sha256": _sha256(manifest_path),
        "candidate_space_sha256": plan_lock[
            "candidate_space_sha256"
        ],
        "selection_policy_sha256": plan_lock[
            "selection_policy_sha256"
        ],
        "external_policy_sha256": plan_lock[
            "external_policy_sha256"
        ],
        "aggregation": (
            selection_policy["aggregation"]
            if far_horizon_policy is None
            else (
                "deployment_horizon_mixture"
                if far_horizon_policy.get(
                    "eligibility_rule",
                    _DEPLOYMENT_MIXTURE_ELIGIBILITY,
                )
                == _DEPLOYMENT_MIXTURE_ELIGIBILITY
                else "dual_horizon_gate"
            )
        ),
        "fold_count": len(folds),
        "gapped_fold_count": len(gapped_folds),
        "reserved_fold_count": len(manifest["reserved_folds"]),
        "reserved_fold_metrics_read": False,
        "external_holdout_read": False,
        "candidates": candidate_reports,
        "selected_candidate_id": selected_id,
    }
    if far_horizon_policy is not None:
        report.update(
            {
                "far_horizon_policy_sha256": plan_lock[
                    "far_horizon_policy_sha256"
                ],
                "deployment_collapsed_fraction": far_horizon_policy[
                    "deployment_collapsed_fraction"
                ],
                "counterfactual_arms_participate_in_selection": False,
                "eligibility_rule": far_horizon_policy.get(
                    "eligibility_rule",
                    _DEPLOYMENT_MIXTURE_ELIGIBILITY,
                ),
            }
        )
    output_dir.mkdir(parents=True)
    _write_json_exclusive(
        output_dir / "selection-report.json",
        report,
    )
    if selected is not None:
        selection_lock = {
            "schema_version": 1,
            "protocol": "standard_validation_selection_lock_v1",
            "experiment_id": plan_lock["experiment_id"],
            "plan_lock_sha256": report["plan_lock_sha256"],
            "rolling_manifest_sha256": report[
                "rolling_manifest_sha256"
            ],
            "candidate_space_sha256": report[
                "candidate_space_sha256"
            ],
            "selection_policy_sha256": report[
                "selection_policy_sha256"
            ],
            "external_policy_sha256": report[
                "external_policy_sha256"
            ],
            "selected_candidate": selected,
            "selected_candidate_scores": candidate_score_hashes[
                selected_id
            ],
            "selection_order": (
                selection_policy["selection_order"]
                if far_horizon_policy is None
                else far_horizon_policy["selection_order"]
            ),
            "external_gate": plan_lock["external_gate"],
            "external_holdout_read": False,
            "weight_rescan_authorized": False,
            "feature_rescan_authorized": False,
            "leaderboard_tuning_authorized": False,
        }
        if "baseline" in plan_lock:
            selection_lock["baseline"] = plan_lock["baseline"]
            selection_lock["baseline_sha256"] = plan_lock[
                "baseline_sha256"
            ]
        if far_horizon_policy is not None:
            selection_lock.update(
                {
                    "temporal_scope": plan_lock["temporal_scope"],
                    "temporal_scope_sha256": plan_lock[
                        "temporal_scope_sha256"
                    ],
                    "far_horizon_validation": far_horizon_policy,
                    "far_horizon_policy_sha256": plan_lock[
                        "far_horizon_policy_sha256"
                    ],
                }
            )
        _write_json_exclusive(
            output_dir / "selection-lock.json",
            selection_lock,
        )
    return report


def evaluate_standard_external_gate(
    *,
    manifest_path: Path,
    selection_lock_path: Path,
    state_dir: Path,
) -> dict[str, Any]:
    """Open one preregistered long-horizon external holdout exactly once."""

    manifest_path = Path(manifest_path)
    selection_lock_path = Path(selection_lock_path)
    state_dir = Path(state_dir)
    manifest = _read_json(manifest_path)
    selection_lock = _read_json(selection_lock_path)
    _validate_external_contract(
        manifest=manifest,
        selection_lock=selection_lock,
        selection_lock_sha256=_sha256(selection_lock_path),
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = state_dir / "external-open-receipt.json"
    if receipt_path.exists():
        raise FileExistsError(
            f"external holdout already opened: {receipt_path}"
        )
    external_policy = selection_lock["external_gate"]
    receipt = {
        "schema_version": 1,
        "protocol": "standard_external_open_receipt_v1",
        "opened_at_utc": datetime.now(UTC).isoformat(),
        "external_manifest_sha256": _sha256(manifest_path),
        "selection_lock_sha256": _sha256(selection_lock_path),
        "experiment_id": selection_lock["experiment_id"],
        "selected_candidate_id": selection_lock[
            "selected_candidate"
        ]["candidate_id"],
        "holdout_id": external_policy["holdout_id"],
        "lineage_sha256": external_policy["lineage_sha256"],
    }
    _write_json_exclusive(receipt_path, receipt)

    baseline = _load_score_artifact(
        manifest["baseline"],
        manifest_dir=manifest_path.parent,
        label="external baseline",
    )
    candidate = _load_score_artifact(
        manifest["candidate"],
        manifest_dir=manifest_path.parent,
        label="external candidate",
    )
    if candidate.shape != baseline.shape:
        raise ValueError(
            "external candidate score shape differs from baseline"
        )
    positive_column = int(manifest.get("positive_candidate_column", 0))
    baseline_metrics = ranking_metrics(
        baseline,
        positive_candidate_column=positive_column,
    )
    candidate_metrics = ranking_metrics(
        candidate,
        baseline_scores=baseline,
        positive_candidate_column=positive_column,
    )
    deltas = _metric_deltas(candidate_metrics, baseline_metrics)
    gates = _external_gates(
        deltas=deltas,
        movements=candidate_metrics["query_movements"],
        policy=external_policy,
    )
    failed_gates = [
        name for name, passed in gates.items() if not passed
    ]
    accepted = not failed_gates
    actual_start_gap = (
        manifest["score_time_min"] - manifest["training_time_max"]
    )
    actual_horizon = (
        manifest["score_time_max"] - manifest["training_time_max"]
    )
    report = {
        "schema_version": 1,
        "protocol": "standard_external_evaluation_v1",
        "status": "accepted" if accepted else "rejected",
        "experiment_id": selection_lock["experiment_id"],
        "selected_candidate": selection_lock["selected_candidate"],
        "selection_lock_sha256": receipt["selection_lock_sha256"],
        "external_manifest_sha256": receipt[
            "external_manifest_sha256"
        ],
        "external_open_receipt": str(receipt_path.resolve()),
        "holdout_id": external_policy["holdout_id"],
        "lineage_sha256": external_policy["lineage_sha256"],
        "actual_start_gap_seconds": actual_start_gap,
        "actual_horizon_seconds": actual_horizon,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_candidate_minus_baseline": deltas,
        "gates": gates,
        "failed_gates": failed_gates,
        "weight_rescan_authorized": False,
        "feature_rescan_authorized": False,
        "leaderboard_tuning_authorized": False,
        "package_authorized": accepted,
    }
    interpretation_policy = external_policy.get("interpretation_policy")
    if interpretation_policy is not None:
        discount = float(
            interpretation_policy["calibration_discount_factor"]
        )
        report["effect_size_interpretation"] = {
            **interpretation_policy,
            "raw_deltas_participate_in_safety_gate": True,
            "calibrated_effect_size_proxy": {
                metric: float(delta / discount)
                for metric, delta in deltas.items()
            },
        }
    _write_json_exclusive(
        state_dir / "external-evaluation-report.json",
        report,
    )
    return report


def _validate_plan(
    plan: dict[str, Any],
) -> tuple[
    str,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    if plan.get("protocol") != "standard_validation_plan_v1":
        raise ValueError("unexpected standard validation plan protocol")
    experiment_id = plan.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be a non-empty string")
    family = plan.get("candidate_family")
    if not isinstance(family, str) or not family:
        raise ValueError("candidate_family must be a non-empty string")
    candidate_space = plan.get("candidate_space")
    if not isinstance(candidate_space, list) or not candidate_space:
        raise ValueError("candidate_space must be a non-empty list")
    candidate_ids: set[str] = set()
    priorities: set[int] = set()
    for candidate in candidate_space:
        if not isinstance(candidate, dict):
            raise ValueError("every candidate must be an object")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        _validate_sha256(
            candidate.get("config_sha256"),
            f"{candidate_id} config_sha256",
        )
        priority = candidate.get("tie_break_priority")
        if (
            not isinstance(priority, int)
            or priority < 0
            or priority in priorities
        ):
            raise ValueError(
                "tie_break_priority must be unique non-negative integers"
            )
        priorities.add(priority)

    baseline = plan.get("baseline")
    if baseline is not None:
        _validate_baseline(baseline)

    selection = plan.get("rolling_selection")
    if not isinstance(selection, dict):
        raise ValueError("rolling_selection must be an object")
    if int(selection.get("minimum_folds", 0)) < 3:
        raise ValueError(
            "standard rolling selection requires at least three folds"
        )
    if int(selection.get("reserved_gate_folds", 0)) < 1:
        raise ValueError(
            "standard validation requires a reserved rolling gate fold"
        )
    if selection.get("aggregation") != "equal_weight_fold_mean":
        raise ValueError(
            "standard rolling aggregation must be equal_weight_fold_mean"
        )
    _validate_thresholds(
        selection.get("per_fold_minimum_deltas"),
        allowed={"mrr", "ndcg_at_10"},
        required={"mrr", "ndcg_at_10"},
        label="per_fold_minimum_deltas",
    )
    _validate_thresholds(
        selection.get("mean_minimum_deltas"),
        allowed=set(STANDARD_METRIC_NAMES) - {"mean_rank"},
        required={
            "mrr",
            "hit_at_1",
            "hit_at_3",
            "hit_at_10",
            "ndcg_at_10",
        },
        label="mean_minimum_deltas",
    )
    _validate_thresholds(
        selection.get("mean_maximum_deltas"),
        allowed={"mean_rank"},
        required={"mean_rank"},
        label="mean_maximum_deltas",
    )
    if int(selection.get("minimum_improved_minus_worsened", 0)) < 1:
        raise ValueError(
            "rolling selection must require more improved than worsened queries"
        )
    if selection.get("selection_order") != _SELECTION_ORDER:
        raise ValueError("unexpected standard selection_order")

    temporal_scope = plan.get("temporal_scope")
    far_horizon_policy = plan.get("far_horizon_validation")
    if temporal_scope is None:
        if far_horizon_policy is not None:
            raise ValueError(
                "far_horizon_validation requires temporal_scope=time_local"
            )
    else:
        if not isinstance(temporal_scope, dict):
            raise ValueError("temporal_scope must be an object")
        if temporal_scope.get("kind") != "time_local":
            raise ValueError("temporal_scope.kind must be time_local")
        short_window = _finite_positive_number(
            temporal_scope.get("short_window_seconds"),
            label="temporal_scope.short_window_seconds",
        )
        if far_horizon_policy is None:
            raise ValueError(
                "time_local temporal_scope requires far_horizon_validation"
            )
        _validate_far_horizon_policy(
            far_horizon_policy,
            short_window_seconds=short_window,
        )

    external = plan.get("external_gate")
    if not isinstance(external, dict):
        raise ValueError("external_gate must be an object")
    holdout_id = external.get("holdout_id")
    if not isinstance(holdout_id, str) or not holdout_id:
        raise ValueError("external holdout_id must be a non-empty string")
    _validate_sha256(
        external.get("lineage_sha256"),
        "external lineage_sha256",
    )
    deployment_horizon = _finite_nonnegative_number(
        external.get("deployment_horizon_seconds"),
        label="deployment_horizon_seconds",
    )
    minimum_horizon = _finite_nonnegative_number(
        external.get("minimum_horizon_seconds"),
        label="minimum_horizon_seconds",
    )
    if deployment_horizon <= 0 or minimum_horizon < deployment_horizon:
        raise ValueError(
            "external minimum horizon must cover the deployment horizon"
        )
    _finite_nonnegative_number(
        external.get("minimum_start_gap_seconds"),
        label="minimum_start_gap_seconds",
    )
    strict_metrics = external.get("strictly_increasing_metrics")
    if (
        not isinstance(strict_metrics, list)
        or strict_metrics != ["mrr"]
    ):
        raise ValueError(
            "standard external gate requires strict MRR improvement"
        )
    _validate_thresholds(
        external.get("minimum_deltas"),
        allowed=set(STANDARD_METRIC_NAMES) - {"mean_rank"},
        required={
            "mrr",
            "hit_at_1",
            "hit_at_3",
            "hit_at_10",
            "ndcg_at_10",
        },
        label="external minimum_deltas",
    )
    _validate_thresholds(
        external.get("maximum_deltas"),
        allowed={"mean_rank"},
        required={"mean_rank"},
        label="external maximum_deltas",
    )
    if int(external.get("minimum_improved_minus_worsened", 0)) < 1:
        raise ValueError(
            "external gate must require more improved than worsened queries"
        )
    interpretation_policy = external.get("interpretation_policy")
    if temporal_scope is not None:
        _validate_time_local_external_interpretation(interpretation_policy)
    elif interpretation_policy is not None:
        raise ValueError(
            "external interpretation_policy is reserved for time_local plans"
        )
    return (
        experiment_id,
        candidate_space,
        selection,
        external,
        temporal_scope,
        far_horizon_policy,
        baseline,
    )


def _validate_baseline(baseline: Any) -> None:
    if not isinstance(baseline, dict):
        raise ValueError("baseline must be an object")
    for field in ("baseline_id", "integration_id"):
        value = baseline.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"baseline.{field} must be a non-empty string")
    _validate_sha256(
        baseline.get("checkpoint_sha256"),
        "baseline checkpoint_sha256",
    )
    promoted_manifest_sha256 = baseline.get("promoted_manifest_sha256")
    if promoted_manifest_sha256 is not None:
        _validate_sha256(
            promoted_manifest_sha256,
            "baseline promoted_manifest_sha256",
        )
    selected_weight = _finite_nonnegative_number(
        baseline.get("selected_weight"),
        label="baseline.selected_weight",
    )
    if selected_weight > 1:
        raise ValueError("baseline.selected_weight must be within [0, 1]")


def _validate_far_horizon_policy(
    policy: Any,
    *,
    short_window_seconds: float,
) -> None:
    if not isinstance(policy, dict):
        raise ValueError("far_horizon_validation must be an object")
    if policy.get("protocol") != "standard_far_horizon_validation_v1":
        raise ValueError("unexpected far_horizon_validation protocol")
    minimum_gapped_folds = policy.get("minimum_gapped_folds")
    if (
        not isinstance(minimum_gapped_folds, int)
        or minimum_gapped_folds < 2
    ):
        raise ValueError(
            "far_horizon_validation requires at least two gapped folds"
        )
    if policy.get("gapped_fold_aggregation") != "equal_weight_fold_mean":
        raise ValueError(
            "gapped fold aggregation must be equal_weight_fold_mean"
        )
    if policy.get("deployment_aggregation") != "near_collapsed_mixture":
        raise ValueError(
            "deployment aggregation must be near_collapsed_mixture"
        )
    collapsed_fraction = _finite_nonnegative_number(
        policy.get("deployment_collapsed_fraction"),
        label="deployment_collapsed_fraction",
    )
    if not 0 < collapsed_fraction < 1:
        raise ValueError(
            "deployment_collapsed_fraction must be strictly between zero and one"
        )
    specs = policy.get("gapped_fold_specs")
    if (
        not isinstance(specs, list)
        or len(specs) < minimum_gapped_folds
        or len(specs) != minimum_gapped_folds
    ):
        raise ValueError(
            "gapped_fold_specs must exactly match minimum_gapped_folds"
        )
    seen_ids: set[str] = set()
    previous_quantile = 0.0
    previous_gap = 0.0
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("every gapped fold spec must be an object")
        fold_id = spec.get("fold_id")
        if (
            not isinstance(fold_id, str)
            or not fold_id
            or fold_id in seen_ids
        ):
            raise ValueError("gapped fold specs require unique fold_id values")
        seen_ids.add(fold_id)
        quantile = _finite_positive_number(
            spec.get("deployment_horizon_quantile"),
            label=f"{fold_id}.deployment_horizon_quantile",
        )
        if quantile > 1 or quantile <= previous_quantile:
            raise ValueError(
                "gapped deployment horizon quantiles must strictly increase "
                "within (0, 1]"
            )
        gap = _finite_positive_number(
            spec.get("minimum_gap_seconds"),
            label=f"{fold_id}.minimum_gap_seconds",
        )
        if gap < short_window_seconds:
            raise ValueError(
                f"{fold_id} minimum gap is shorter than the short window"
            )
        if gap <= previous_gap:
            raise ValueError("gapped minimum gaps must strictly increase")
        previous_quantile = quantile
        previous_gap = gap
    _validate_thresholds(
        policy.get("gapped_per_fold_minimum_deltas"),
        allowed={"mrr", "ndcg_at_10"},
        required={"mrr", "ndcg_at_10"},
        label="gapped_per_fold_minimum_deltas",
    )
    eligibility_rule = policy.get(
        "eligibility_rule",
        _DEPLOYMENT_MIXTURE_ELIGIBILITY,
    )
    if eligibility_rule not in {
        _DEPLOYMENT_MIXTURE_ELIGIBILITY,
        _DUAL_HORIZON_ELIGIBILITY,
    }:
        raise ValueError("unexpected far-horizon eligibility_rule")
    strict_metrics = policy.get("gapped_strictly_increasing_metrics")
    if eligibility_rule == _DUAL_HORIZON_ELIGIBILITY:
        if strict_metrics != ["mrr"]:
            raise ValueError(
                "dual-horizon gate requires strict gapped MRR improvement"
            )
        expected_selection_order = _DUAL_HORIZON_SELECTION_ORDER
    else:
        if strict_metrics not in (None, []):
            raise ValueError(
                "gapped strict metrics require the dual-horizon gate"
            )
        expected_selection_order = _FAR_HORIZON_SELECTION_ORDER
    movement_threshold = policy.get(
        "minimum_deployment_improved_minus_worsened_rate"
    )
    if (
        not isinstance(movement_threshold, (int, float))
        or not np.isfinite(float(movement_threshold))
        or not -1 <= float(movement_threshold) <= 1
    ):
        raise ValueError(
            "minimum deployment movement rate must be finite within [-1, 1]"
        )
    if policy.get("selection_order") != expected_selection_order:
        raise ValueError("unexpected far-horizon selection_order")
    counterfactual = policy.get("zero_short_counterfactual")
    if not isinstance(counterfactual, dict):
        raise ValueError("zero_short_counterfactual must be an object")
    if not isinstance(counterfactual.get("enabled"), bool):
        raise ValueError("zero_short_counterfactual.enabled must be boolean")
    arm_id = counterfactual.get("arm_id")
    if not isinstance(arm_id, str) or not arm_id:
        raise ValueError(
            "zero_short_counterfactual.arm_id must be non-empty"
        )
    if counterfactual.get("participates_in_selection") is not False:
        raise ValueError(
            "zero-short counterfactual cannot participate in selection"
        )


def _validate_time_local_external_interpretation(policy: Any) -> None:
    if not isinstance(policy, dict):
        raise ValueError(
            "time_local external gate requires interpretation_policy"
        )
    if policy.get("decision_role") != "safety_gate_only":
        raise ValueError(
            "time-local external decision role must be safety_gate_only"
        )
    if policy.get("effect_size_estimation_authorized") is not False:
        raise ValueError(
            "time-local external effect-size estimation must be disabled"
        )
    discount = _finite_positive_number(
        policy.get("calibration_discount_factor"),
        label="calibration_discount_factor",
    )
    if abs(discount - 19.5) > _TOLERANCE:
        raise ValueError(
            "time-local external calibration discount must be 19.5x"
        )


def _validate_rolling_contract(
    *,
    manifest: dict[str, Any],
    plan_lock: dict[str, Any],
    plan_lock_sha256: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    if (
        plan_lock.get("protocol")
        != "standard_validation_plan_lock_v1"
    ):
        raise ValueError("unexpected validation plan lock protocol")
    if (
        manifest.get("protocol")
        != "standard_rolling_origin_scores_v1"
    ):
        raise ValueError("unexpected rolling score protocol")
    if manifest.get("plan_lock_sha256") != plan_lock_sha256:
        raise ValueError("rolling manifest plan lock hash differs")
    _validate_manifest_baseline_binding(
        manifest=manifest,
        lock=plan_lock,
        label="rolling",
    )
    candidate_space = plan_lock["candidate_space"]
    if (
        _json_sha256(candidate_space)
        != plan_lock.get("candidate_space_sha256")
    ):
        raise ValueError("validation plan candidate space hash differs")
    selection = plan_lock["rolling_selection"]
    if (
        _json_sha256(selection)
        != plan_lock.get("selection_policy_sha256")
    ):
        raise ValueError("validation plan selection policy hash differs")
    temporal_scope = plan_lock.get("temporal_scope")
    far_horizon_policy = plan_lock.get("far_horizon_validation")
    if temporal_scope is None:
        if far_horizon_policy is not None:
            raise ValueError("validation plan temporal policy is incomplete")
    else:
        if (
            _json_sha256(temporal_scope)
            != plan_lock.get("temporal_scope_sha256")
        ):
            raise ValueError("validation plan temporal scope hash differs")
        if (
            not isinstance(far_horizon_policy, dict)
            or _json_sha256(far_horizon_policy)
            != plan_lock.get("far_horizon_policy_sha256")
        ):
            raise ValueError(
                "validation plan far-horizon policy hash differs"
            )
    candidate_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in candidate_space
    }
    folds = manifest.get("folds")
    if (
        not isinstance(folds, list)
        or len(folds) < int(selection["minimum_folds"])
    ):
        raise ValueError(
            "rolling manifest has too few selection folds"
        )
    _validate_fold_chronology(folds, role="selection")
    for fold in folds:
        _validate_scored_fold_artifacts(
            fold=fold,
            candidate_by_id=candidate_by_id,
            label="selection",
        )

    reserved = manifest.get("reserved_folds")
    if (
        not isinstance(reserved, list)
        or len(reserved) < int(selection["reserved_gate_folds"])
    ):
        raise ValueError("rolling manifest has too few reserved gate folds")
    _validate_fold_chronology(reserved, role="gate")
    if any(
        "baseline" in fold or "candidates" in fold for fold in reserved
    ):
        raise ValueError(
            "reserved gate fold metrics cannot enter rolling selection"
        )
    last_selection_score = folds[-1]["score_time_max"]
    if reserved[0]["score_time_min"] <= last_selection_score:
        raise ValueError(
            "reserved gate folds must follow all selection folds"
        )

    gapped_folds: list[dict[str, Any]] = []
    if far_horizon_policy is None:
        if "gapped_folds" in manifest:
            raise ValueError(
                "gapped_folds require a preregistered far-horizon policy"
            )
    else:
        gapped = manifest.get("gapped_folds")
        if (
            not isinstance(gapped, list)
            or len(gapped)
            < int(far_horizon_policy["minimum_gapped_folds"])
        ):
            raise ValueError("rolling manifest has too few gapped folds")
        specs = far_horizon_policy["gapped_fold_specs"]
        if [fold.get("fold_id") for fold in gapped] != [
            spec["fold_id"] for spec in specs
        ]:
            raise ValueError(
                "gapped folds differ from preregistered horizon quantiles"
            )
        _validate_fold_chronology(gapped, role="gapped")
        short_window = float(temporal_scope["short_window_seconds"])
        for fold, spec in zip(gapped, specs, strict=True):
            gap = float(fold["score_time_min"] - fold["train_time_max"])
            if gap < short_window:
                raise ValueError(
                    f"{fold['fold_id']} gap is shorter than the short window"
                )
            if gap < float(spec["minimum_gap_seconds"]):
                raise ValueError(
                    f"{fold['fold_id']} gap is shorter than its "
                    "preregistered deployment horizon quantile"
                )
            if (
                float(fold.get("deployment_horizon_quantile", -1))
                != float(spec["deployment_horizon_quantile"])
            ):
                raise ValueError(
                    f"{fold['fold_id']} deployment horizon quantile differs"
                )
            _validate_scored_fold_artifacts(
                fold=fold,
                candidate_by_id=candidate_by_id,
                label="gapped",
            )
        gapped_folds = gapped
        counterfactual = far_horizon_policy[
            "zero_short_counterfactual"
        ]
        if counterfactual["enabled"]:
            arm_id = counterfactual["arm_id"]
            for fold in folds:
                arms = fold.get("counterfactual_arms")
                if not isinstance(arms, dict) or set(arms) != {arm_id}:
                    raise ValueError(
                        f"{fold['fold_id']} must contain only the "
                        f"{arm_id} counterfactual arm"
                    )
                _validate_scored_fold_artifacts(
                    fold=fold,
                    candidate_by_id=candidate_by_id,
                    label=f"counterfactual {arm_id}",
                    counterfactual_arm_id=arm_id,
                )
    return (
        candidate_space,
        selection,
        folds,
        far_horizon_policy,
        gapped_folds,
    )


def _validate_external_contract(
    *,
    manifest: dict[str, Any],
    selection_lock: dict[str, Any],
    selection_lock_sha256: str,
) -> None:
    if (
        selection_lock.get("protocol")
        != "standard_validation_selection_lock_v1"
    ):
        raise ValueError("unexpected standard selection lock protocol")
    if manifest.get("protocol") != "standard_external_scores_v1":
        raise ValueError("unexpected standard external score protocol")
    if manifest.get("selection_lock_sha256") != selection_lock_sha256:
        raise ValueError("external selection lock hash differs")
    _validate_manifest_baseline_binding(
        manifest=manifest,
        lock=selection_lock,
        label="external",
    )
    if manifest.get("experiment_id") != selection_lock.get(
        "experiment_id"
    ):
        raise ValueError("external experiment_id differs from lock")
    selected = selection_lock["selected_candidate"]
    if manifest.get("selected_candidate_id") != selected[
        "candidate_id"
    ]:
        raise ValueError("external selected candidate differs from lock")
    if (
        manifest.get("selected_candidate_config_sha256")
        != selected["config_sha256"]
    ):
        raise ValueError(
            "external selected candidate config differs from lock"
        )
    policy = selection_lock["external_gate"]
    if manifest.get("holdout_id") != policy["holdout_id"]:
        raise ValueError("external holdout_id differs from preregistration")
    if manifest.get("lineage_sha256") != policy["lineage_sha256"]:
        raise ValueError(
            "external lineage differs from preregistration"
        )
    candidate = manifest.get("candidate")
    _validate_artifact_descriptor(candidate, "external candidate")
    _validate_artifact_descriptor(
        manifest.get("baseline"),
        "external baseline",
    )
    if candidate.get("candidate_id") != selected["candidate_id"]:
        raise ValueError("external candidate_id differs from lock")
    if candidate.get("config_sha256") != selected["config_sha256"]:
        raise ValueError(
            "external candidate config_sha256 differs from lock"
        )
    if (
        candidate.get("candidate_fingerprint")
        != manifest.get("candidate_fingerprint")
    ):
        raise ValueError(
            "external candidate fingerprint differs from baseline order"
        )
    train_max = manifest.get("training_time_max")
    score_min = manifest.get("score_time_min")
    score_max = manifest.get("score_time_max")
    if not all(
        isinstance(value, (int, float))
        and np.isfinite(float(value))
        for value in (train_max, score_min, score_max)
    ):
        raise ValueError("external time boundaries must be finite")
    if not train_max <= score_min <= score_max:
        raise ValueError(
            "external score interval must not precede training"
        )
    start_gap = score_min - train_max
    if start_gap < policy["minimum_start_gap_seconds"]:
        raise ValueError(
            "external start gap is shorter than preregistered"
        )
    horizon = score_max - train_max
    if horizon < policy["minimum_horizon_seconds"]:
        raise ValueError(
            "external score interval does not cover deployment horizon"
        )


def _validate_manifest_baseline_binding(
    *,
    manifest: dict[str, Any],
    lock: dict[str, Any],
    label: str,
) -> None:
    baseline = lock.get("baseline")
    baseline_sha256 = lock.get("baseline_sha256")
    if baseline is None:
        if baseline_sha256 is not None:
            raise ValueError("validation lock baseline contract is incomplete")
        return
    if (
        not isinstance(baseline, dict)
        or _json_sha256(baseline) != baseline_sha256
    ):
        raise ValueError("validation lock baseline hash differs")
    if manifest.get("baseline_sha256") != baseline_sha256:
        raise ValueError(
            f"{label} manifest baseline differs from the frozen plan"
        )


def _validate_scored_fold_artifacts(
    *,
    fold: dict[str, Any],
    candidate_by_id: dict[str, dict[str, Any]],
    label: str,
    counterfactual_arm_id: str | None = None,
) -> None:
    fold_id = fold["fold_id"]
    fingerprint = fold.get("candidate_fingerprint")
    _validate_sha256(
        fingerprint,
        f"{fold_id} candidate_fingerprint",
    )
    container = _fold_score_container(
        fold,
        counterfactual_arm_id=counterfactual_arm_id,
    )
    _validate_artifact_descriptor(
        container.get("baseline"),
        f"{fold_id} {label} baseline",
    )
    candidates = container.get("candidates")
    if (
        not isinstance(candidates, dict)
        or set(candidates) != set(candidate_by_id)
    ):
        raise ValueError(
            f"every {label} fold must contain the frozen candidate space"
        )
    for candidate_id, expected in candidate_by_id.items():
        descriptor = candidates[candidate_id]
        _validate_artifact_descriptor(
            descriptor,
            f"{fold_id} {label} candidate {candidate_id}",
        )
        if descriptor.get("candidate_id") != candidate_id:
            raise ValueError(f"{fold_id} candidate_id differs from plan")
        if descriptor.get("config_sha256") != expected["config_sha256"]:
            raise ValueError(
                f"{fold_id} candidate {candidate_id} "
                "config_sha256 differs from the frozen plan"
            )
        if descriptor.get("candidate_fingerprint") != fingerprint:
            raise ValueError(f"{fold_id} candidate fingerprint differs")


def _fold_score_container(
    fold: dict[str, Any],
    *,
    counterfactual_arm_id: str | None,
) -> dict[str, Any]:
    if counterfactual_arm_id is None:
        return fold
    arms = fold["counterfactual_arms"]
    return arms[counterfactual_arm_id]


def _load_fold_group_baselines(
    *,
    folds: list[dict[str, Any]],
    manifest_dir: Path,
    positive_column: int,
    counterfactual_arm_id: str | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    baselines: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    for fold in folds:
        container = _fold_score_container(
            fold,
            counterfactual_arm_id=counterfactual_arm_id,
        )
        arm_label = (
            ""
            if counterfactual_arm_id is None
            else f" {counterfactual_arm_id}"
        )
        baseline = _load_score_artifact(
            container["baseline"],
            manifest_dir=manifest_dir,
            label=f"{fold['fold_id']}{arm_label} baseline",
        )
        _validate_positive_column(
            positive_column,
            candidate_count=baseline.shape[1],
        )
        baselines.append(baseline)
        metrics.append(
            ranking_metrics(
                baseline,
                positive_candidate_column=positive_column,
            )
        )
    return baselines, metrics


def _score_candidate_fold_group(
    *,
    candidate_id: str,
    folds: list[dict[str, Any]],
    baselines: list[np.ndarray],
    baseline_metrics: list[dict[str, Any]],
    manifest_dir: Path,
    positive_column: int,
    counterfactual_arm_id: str | None = None,
) -> dict[str, Any]:
    fold_reports: list[dict[str, Any]] = []
    fold_deltas: list[dict[str, float]] = []
    score_hashes: list[dict[str, str]] = []
    movements = {"improved": 0, "unchanged": 0, "worsened": 0}
    for fold, baseline, baseline_panel in zip(
        folds,
        baselines,
        baseline_metrics,
        strict=True,
    ):
        container = _fold_score_container(
            fold,
            counterfactual_arm_id=counterfactual_arm_id,
        )
        descriptor = container["candidates"][candidate_id]
        arm_label = (
            ""
            if counterfactual_arm_id is None
            else f" {counterfactual_arm_id}"
        )
        scores = _load_score_artifact(
            descriptor,
            manifest_dir=manifest_dir,
            label=(
                f"{fold['fold_id']}{arm_label} candidate {candidate_id}"
            ),
        )
        if scores.shape != baseline.shape:
            raise ValueError(
                f"{fold['fold_id']} candidate {candidate_id} score "
                "shape differs from baseline"
            )
        candidate_metrics = ranking_metrics(
            scores,
            baseline_scores=baseline,
            positive_candidate_column=positive_column,
        )
        deltas = _metric_deltas(candidate_metrics, baseline_panel)
        query_movements = candidate_metrics["query_movements"]
        for key in movements:
            movements[key] += int(query_movements[key])
        fold_deltas.append(deltas)
        fold_reports.append(
            {
                "fold_id": fold["fold_id"],
                "baseline": baseline_panel,
                "candidate": candidate_metrics,
                "delta_candidate_minus_baseline": deltas,
            }
        )
        score_hashes.append(
            {
                "fold_id": fold["fold_id"],
                "score_sha256": descriptor["sha256"],
            }
        )
    total_movements = sum(movements.values())
    return {
        "folds": fold_reports,
        "fold_deltas": fold_deltas,
        "mean_baseline": _mean_metric_panels(baseline_metrics),
        "mean_candidate": _mean_metric_panels(
            [fold["candidate"] for fold in fold_reports]
        ),
        "mean_delta": _mean_metric_panels(fold_deltas),
        "query_movements": movements,
        "movement_balance_rate": float(
            (movements["improved"] - movements["worsened"])
            / total_movements
        ),
        "score_hashes": score_hashes,
    }


def _rolling_gates(
    *,
    fold_deltas: list[dict[str, float]],
    mean_delta: dict[str, float],
    improved: int,
    worsened: int,
    policy: dict[str, Any],
) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    for metric, threshold in policy[
        "per_fold_minimum_deltas"
    ].items():
        gates[f"all_folds_{metric}_meet_minimum"] = all(
            delta[metric] >= float(threshold) - _TOLERANCE
            for delta in fold_deltas
        )
    for metric, threshold in policy["mean_minimum_deltas"].items():
        gates[f"mean_fold_{metric}_meets_minimum"] = (
            mean_delta[metric] >= float(threshold) - _TOLERANCE
        )
    for metric, threshold in policy["mean_maximum_deltas"].items():
        gates[f"mean_fold_{metric}_meets_maximum"] = (
            mean_delta[metric] <= float(threshold) + _TOLERANCE
        )
    gates["improved_minus_worsened_meets_minimum"] = (
        improved - worsened
        >= int(policy["minimum_improved_minus_worsened"])
    )
    return gates


def _far_horizon_gates(
    *,
    near_fold_deltas: list[dict[str, float]],
    gapped_fold_deltas: list[dict[str, float]],
    deployment_delta: dict[str, float],
    deployment_movement_rate: float,
    rolling_policy: dict[str, Any],
    far_horizon_policy: dict[str, Any],
) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    eligibility_rule = far_horizon_policy.get(
        "eligibility_rule",
        _DEPLOYMENT_MIXTURE_ELIGIBILITY,
    )
    if eligibility_rule == _DUAL_HORIZON_ELIGIBILITY:
        for metric, threshold in rolling_policy[
            "per_fold_minimum_deltas"
        ].items():
            gates[f"all_near_folds_{metric}_meet_minimum"] = all(
                delta[metric] >= float(threshold) - _TOLERANCE
                for delta in near_fold_deltas
            )
        strict_metrics = set(
            far_horizon_policy["gapped_strictly_increasing_metrics"]
        )
        for metric, threshold in far_horizon_policy[
            "gapped_per_fold_minimum_deltas"
        ].items():
            if metric in strict_metrics:
                name = f"all_gapped_folds_{metric}_strictly_improve"
                gates[name] = all(
                    delta[metric] > float(threshold) + _TOLERANCE
                    for delta in gapped_fold_deltas
                )
            else:
                name = f"all_gapped_folds_{metric}_meet_minimum"
                gates[name] = all(
                    delta[metric] >= float(threshold) - _TOLERANCE
                    for delta in gapped_fold_deltas
                )
        return gates

    for metric, threshold in far_horizon_policy[
        "gapped_per_fold_minimum_deltas"
    ].items():
        gates[f"all_gapped_folds_{metric}_meet_minimum"] = all(
            delta[metric] >= float(threshold) - _TOLERANCE
            for delta in gapped_fold_deltas
        )
    for metric, threshold in rolling_policy[
        "mean_minimum_deltas"
    ].items():
        gates[f"deployment_mixture_{metric}_meets_minimum"] = (
            deployment_delta[metric]
            >= float(threshold) - _TOLERANCE
        )
    for metric, threshold in rolling_policy[
        "mean_maximum_deltas"
    ].items():
        gates[f"deployment_mixture_{metric}_meets_maximum"] = (
            deployment_delta[metric]
            <= float(threshold) + _TOLERANCE
        )
    gates["deployment_movement_rate_meets_minimum"] = (
        deployment_movement_rate
        >= float(
            far_horizon_policy[
                "minimum_deployment_improved_minus_worsened_rate"
            ]
        )
        - _TOLERANCE
    )
    return gates


def _external_gates(
    *,
    deltas: dict[str, float],
    movements: dict[str, int],
    policy: dict[str, Any],
) -> dict[str, bool]:
    strict = set(policy["strictly_increasing_metrics"])
    gates: dict[str, bool] = {}
    for metric, threshold in policy["minimum_deltas"].items():
        if metric in strict:
            passed = deltas[metric] > float(threshold) + _TOLERANCE
        else:
            passed = deltas[metric] >= float(threshold) - _TOLERANCE
        gates[f"{metric}_meets_minimum"] = passed
    for metric, threshold in policy["maximum_deltas"].items():
        gates[f"{metric}_meets_maximum"] = (
            deltas[metric] <= float(threshold) + _TOLERANCE
        )
    gates["improved_minus_worsened_meets_minimum"] = (
        int(movements["improved"]) - int(movements["worsened"])
        >= int(policy["minimum_improved_minus_worsened"])
    )
    return gates


def _selection_key(
    report: dict[str, Any],
    *,
    far_horizon_policy: dict[str, Any] | None,
) -> tuple[float, ...]:
    if far_horizon_policy is not None:
        eligibility_rule = far_horizon_policy.get(
            "eligibility_rule",
            _DEPLOYMENT_MIXTURE_ELIGIBILITY,
        )
        if eligibility_rule == _DUAL_HORIZON_ELIGIBILITY:
            return (
                report["mean_gapped_fold_delta"]["mrr"],
                report["stability"]["worst_gapped_mrr_delta"],
                report["mean_fold_delta"]["mrr"],
                -float(report["tie_break_priority"]),
            )
        return (
            report["deployment_mixture_delta"]["mrr"],
            report["stability"]["worst_gapped_mrr_delta"],
            report["deployment_mixture_delta"]["ndcg_at_10"],
            -float(report["tie_break_priority"]),
        )
    return (
        report["mean_fold_delta"]["mrr"],
        report["stability"]["worst_fold_mrr_delta"],
        report["mean_fold_delta"]["ndcg_at_10"],
        -float(report["tie_break_priority"]),
    )


def _mean_metric_panels(
    panels: list[dict[str, Any]],
) -> dict[str, float]:
    return {
        metric: float(np.mean([panel[metric] for panel in panels]))
        for metric in STANDARD_METRIC_NAMES
    }


def _metric_deltas(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    return {
        metric: float(candidate[metric] - baseline[metric])
        for metric in STANDARD_METRIC_NAMES
    }


def _weighted_metric_panels(
    left: dict[str, float],
    right: dict[str, float],
    *,
    right_weight: float,
) -> dict[str, float]:
    left_weight = 1.0 - right_weight
    return {
        metric: float(
            left_weight * float(left[metric])
            + right_weight * float(right[metric])
        )
        for metric in STANDARD_METRIC_NAMES
    }


def _validate_fold_chronology(
    folds: list[dict[str, Any]],
    *,
    role: str,
) -> None:
    seen: set[str] = set()
    previous_train_max: float | int | None = None
    previous_score_max: float | int | None = None
    for fold in folds:
        fold_id = fold.get("fold_id")
        if (
            not isinstance(fold_id, str)
            or not fold_id
            or fold_id in seen
        ):
            raise ValueError(f"{role} folds require unique fold_id values")
        seen.add(fold_id)
        if fold.get("role") != role:
            raise ValueError(f"{fold_id} must have role={role}")
        train_max = fold.get("train_time_max")
        score_min = fold.get("score_time_min")
        score_max = fold.get("score_time_max")
        if not all(
            isinstance(value, (int, float))
            and np.isfinite(float(value))
            for value in (train_max, score_min, score_max)
        ):
            raise ValueError(f"{fold_id} time boundaries must be finite")
        if not train_max < score_min <= score_max:
            raise ValueError(
                f"{fold_id} score interval must follow training"
            )
        if (
            previous_train_max is not None
            and train_max <= previous_train_max
        ):
            raise ValueError(
                f"{fold_id} training origin must increase"
            )
        if (
            previous_score_max is not None
            and score_min <= previous_score_max
        ):
            raise ValueError(
                f"{fold_id} score interval must follow the previous fold"
            )
        previous_train_max = train_max
        previous_score_max = score_max


def _validate_thresholds(
    thresholds: Any,
    *,
    allowed: set[str],
    required: set[str],
    label: str,
) -> None:
    if not isinstance(thresholds, dict) or set(thresholds) != required:
        raise ValueError(
            f"{label} must contain exactly {sorted(required)}"
        )
    if not set(thresholds) <= allowed:
        raise ValueError(f"{label} contains unsupported metrics")
    for metric, value in thresholds.items():
        if (
            not isinstance(value, (int, float))
            or not np.isfinite(float(value))
        ):
            raise ValueError(f"{label}.{metric} must be finite")


def _finite_nonnegative_number(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _finite_positive_number(value: Any, *, label: str) -> float:
    number = _finite_nonnegative_number(value, label=label)
    if number <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return number


def _load_score_artifact(
    descriptor: dict[str, Any],
    *,
    manifest_dir: Path,
    label: str,
) -> np.ndarray:
    _validate_artifact_descriptor(descriptor, label)
    path = Path(descriptor["path"])
    if not path.is_absolute():
        path = manifest_dir / path
    actual_sha256 = _sha256(path)
    if actual_sha256 != descriptor["sha256"]:
        raise ValueError(
            f"{label} hash mismatch: actual={actual_sha256} "
            f"expected={descriptor['sha256']}"
        )
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    values = np.asarray(values)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 2
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            f"{label} must be a finite non-empty 2D score matrix"
        )
    return values


def _validate_artifact_descriptor(
    descriptor: Any,
    label: str,
) -> None:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{label} artifact descriptor must be an object")
    if not isinstance(descriptor.get("path"), str):
        raise ValueError(f"{label} artifact requires a path")
    _validate_sha256(
        descriptor.get("sha256"),
        f"{label} artifact SHA-256",
    )


def _validate_positive_column(
    positive_column: int,
    *,
    candidate_count: int,
) -> None:
    if not 0 <= positive_column < candidate_count:
        raise ValueError(
            "positive_candidate_column is outside score matrix"
        )


def _validate_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_exclusive(
    path: Path,
    payload: dict[str, Any],
) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
