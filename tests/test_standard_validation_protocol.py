from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jgrec.standard_validation_protocol import (
    evaluate_standard_external_gate,
    freeze_standard_validation_plan,
    select_standard_rolling_candidate,
)

DEPLOYMENT_HORIZON_SECONDS = 468 * 24 * 60 * 60
DEPLOYMENT_COLLAPSED_FRACTION = 0.39971972363446745
SHORT_WINDOW_SECONDS = 100
STABLE_ID = "feature:setwise_context_v1"
UNSTABLE_ID = "ensemble:weight_0.20"
STABLE_CONFIG_SHA256 = "a" * 64
UNSTABLE_CONFIG_SHA256 = "b" * 64


def test_plan_freeze_preregisters_candidate_space_policies_and_long_horizon(
    tmp_path: Path,
) -> None:
    plan_path = _write_plan(tmp_path)

    report = freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )

    assert report["status"] == "ready_for_rolling_selection"
    assert report["selection_metrics_read"] is False
    assert report["reserved_fold_metrics_read"] is False
    assert report["external_holdout_read"] is False
    assert report["package_authorized"] is False
    assert report["deployment_horizon_seconds"] == DEPLOYMENT_HORIZON_SECONDS
    lock = json.loads(
        (tmp_path / "frozen" / "validation-plan-lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert lock["candidate_space_sha256"]
    assert lock["selection_policy_sha256"]
    assert lock["external_policy_sha256"]
    assert lock["candidate_ids"] == [STABLE_ID, UNSTABLE_ID]
    assert lock["rolling_selection"]["selection_order"][0] == (
        "maximum_mean_fold_mrr_delta"
    )


def test_plan_freeze_binds_the_declared_baseline(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["baseline"] = {
        "baseline_id": "cooccur_lift_aux_expert_v1_promoted_champion",
        "integration_id": "cooccur_lift_aux_expert_v1",
        "checkpoint_sha256": "c" * 64,
        "promoted_manifest_sha256": "d" * 64,
        "selected_weight": 0.5,
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    report = freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )

    lock = json.loads(
        (tmp_path / "frozen" / "validation-plan-lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert lock["baseline"] == plan["baseline"]
    assert lock["baseline_sha256"]
    assert report["baseline_id"] == plan["baseline"]["baseline_id"]
    assert report["baseline_sha256"] == lock["baseline_sha256"]


def test_time_local_example_plan_freezes_far_horizon_contract(
    tmp_path: Path,
) -> None:
    plan_path = (
        Path(__file__).parents[1]
        / "docs"
        / "experiments"
        / "standard-validation-time-local-plan.example.json"
    )

    report = freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )

    assert report["minimum_gapped_folds"] == 3
    assert report["deployment_collapsed_fraction"] == pytest.approx(
        0.39971972363446745
    )
    assert report["external_decision_role"] == "safety_gate_only"


def test_rolling_selection_uses_equal_fold_means_and_rejects_single_fold_peak(
    tmp_path: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )
    plan_lock = tmp_path / "frozen" / "validation-plan-lock.json"
    manifest_path = _write_rolling_manifest(
        tmp_path,
        plan_lock=plan_lock,
    )

    report = select_standard_rolling_candidate(
        manifest_path=manifest_path,
        plan_lock_path=plan_lock,
        output_dir=tmp_path / "selection",
    )

    assert report["status"] == "selected"
    assert report["selected_candidate_id"] == STABLE_ID
    assert report["aggregation"] == "equal_weight_fold_mean"
    assert report["reserved_fold_metrics_read"] is False
    stable = report["candidates"][STABLE_ID]
    unstable = report["candidates"][UNSTABLE_ID]
    expected_mean_mrr_delta = np.mean([0.25, 0.125, 0.0625])
    assert stable["mean_fold_delta"]["mrr"] == pytest.approx(
        expected_mean_mrr_delta
    )
    assert unstable["eligible"] is False
    assert "all_folds_mrr_meet_minimum" in unstable["failed_gates"]
    assert set(stable["mean_fold_candidate"]) == {
        "mrr",
        "hit_at_1",
        "hit_at_3",
        "hit_at_10",
        "ndcg_at_10",
        "mean_rank",
    }
    lock = json.loads(
        (tmp_path / "selection" / "selection-lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert lock["selected_candidate"]["candidate_id"] == STABLE_ID
    assert (
        lock["selected_candidate"]["config_sha256"]
        == STABLE_CONFIG_SHA256
    )
    assert lock["candidate_space_sha256"] == report[
        "candidate_space_sha256"
    ]


def test_rolling_selection_rejects_candidate_space_mutation(
    tmp_path: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )
    plan_lock = tmp_path / "frozen" / "validation-plan-lock.json"
    manifest_path = _write_rolling_manifest(
        tmp_path,
        plan_lock=plan_lock,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["folds"][1]["candidates"][STABLE_ID]["config_sha256"] = (
        "c" * 64
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config_sha256"):
        select_standard_rolling_candidate(
            manifest_path=manifest_path,
            plan_lock_path=plan_lock,
            output_dir=tmp_path / "selection",
        )


def test_rolling_selection_rejects_declared_baseline_mutation(
    tmp_path: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["baseline"] = {
        "baseline_id": "cooccur_lift_aux_expert_v1_promoted_champion",
        "integration_id": "cooccur_lift_aux_expert_v1",
        "checkpoint_sha256": "c" * 64,
        "selected_weight": 0.5,
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )
    plan_lock = tmp_path / "frozen" / "validation-plan-lock.json"
    manifest_path = _write_rolling_manifest(
        tmp_path,
        plan_lock=plan_lock,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["baseline_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline"):
        select_standard_rolling_candidate(
            manifest_path=manifest_path,
            plan_lock_path=plan_lock,
            output_dir=tmp_path / "selection",
        )


def test_time_local_plan_requires_preregistered_far_horizon_policy(
    tmp_path: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["temporal_scope"] = {
        "kind": "time_local",
        "short_window_seconds": SHORT_WINDOW_SECONDS,
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="far_horizon_validation"):
        freeze_standard_validation_plan(
            plan_path=plan_path,
            output_dir=tmp_path / "frozen",
        )


def test_far_horizon_deployment_mixture_can_accept_near_horizon_regression(
    tmp_path: Path,
) -> None:
    plan_path = _write_time_local_plan(tmp_path)
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )
    plan_lock = tmp_path / "frozen" / "validation-plan-lock.json"
    manifest_path = _write_time_local_rolling_manifest(
        tmp_path,
        plan_lock=plan_lock,
    )

    report = select_standard_rolling_candidate(
        manifest_path=manifest_path,
        plan_lock_path=plan_lock,
        output_dir=tmp_path / "selection",
    )

    assert report["status"] == "selected"
    assert report["selected_candidate_id"] == STABLE_ID
    assert report["aggregation"] == "deployment_horizon_mixture"
    assert report["gapped_fold_count"] == 3
    candidate = report["candidates"][STABLE_ID]
    assert candidate["mean_fold_delta"]["mrr"] < 0
    assert candidate["mean_gapped_fold_delta"]["mrr"] > 0
    assert candidate["deployment_mixture_delta"]["mrr"] > 0
    assert candidate["eligible"] is True
    assert candidate["counterfactual_arms"]["zero_short"][
        "participates_in_selection"
    ] is False
    assert candidate["counterfactual_arms"]["zero_short"][
        "mean_fold_delta"
    ]["mrr"] < 0
    lock = json.loads(
        (tmp_path / "selection" / "selection-lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert lock["far_horizon_policy_sha256"]
    assert lock["selected_candidate_scores"]["gapped"]
    assert lock["selected_candidate_scores"]["counterfactual_arms"][
        "zero_short"
    ]


def test_dual_horizon_gate_rejects_near_horizon_regression(
    tmp_path: Path,
) -> None:
    plan_path = _write_dual_horizon_plan(tmp_path)
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )
    plan_lock = tmp_path / "frozen" / "validation-plan-lock.json"
    manifest_path = _write_time_local_rolling_manifest(
        tmp_path,
        plan_lock=plan_lock,
    )

    report = select_standard_rolling_candidate(
        manifest_path=manifest_path,
        plan_lock_path=plan_lock,
        output_dir=tmp_path / "selection",
    )

    candidate = report["candidates"][STABLE_ID]
    assert candidate["mean_gapped_fold_delta"]["mrr"] > 0
    assert candidate["deployment_mixture_delta"]["mrr"] > 0
    assert candidate["eligible"] is False
    assert "all_near_folds_mrr_meet_minimum" in candidate["failed_gates"]
    assert report["status"] == "rejected"


def test_dual_horizon_gate_requires_strict_mrr_gain_in_every_gapped_fold(
    tmp_path: Path,
) -> None:
    plan_path = _write_dual_horizon_plan(tmp_path)
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )
    plan_lock = tmp_path / "frozen" / "validation-plan-lock.json"
    manifest_path = _write_time_local_rolling_manifest(
        tmp_path,
        plan_lock=plan_lock,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _replace_candidate_fold_ranks(
        tmp_path,
        manifest["folds"],
        candidate_id=STABLE_ID,
        ranks_by_fold=[[2, 2, 2, 2]] * 3,
    )
    _replace_candidate_fold_ranks(
        tmp_path,
        manifest["gapped_folds"],
        candidate_id=STABLE_ID,
        ranks_by_fold=[
            [2, 2, 2, 2],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = select_standard_rolling_candidate(
        manifest_path=manifest_path,
        plan_lock_path=plan_lock,
        output_dir=tmp_path / "selection",
    )

    candidate = report["candidates"][STABLE_ID]
    assert candidate["eligible"] is False
    assert (
        "all_gapped_folds_mrr_strictly_improve"
        in candidate["failed_gates"]
    )
    assert report["status"] == "rejected"


def test_dual_horizon_gate_selects_by_far_then_near_after_both_gates_pass(
    tmp_path: Path,
) -> None:
    plan_path = _write_dual_horizon_plan(tmp_path)
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )
    plan_lock = tmp_path / "frozen" / "validation-plan-lock.json"
    manifest_path = _write_time_local_rolling_manifest(
        tmp_path,
        plan_lock=plan_lock,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _replace_candidate_fold_ranks(
        tmp_path,
        manifest["folds"],
        candidate_id=STABLE_ID,
        ranks_by_fold=[[2, 2, 2, 2]] * 3,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = select_standard_rolling_candidate(
        manifest_path=manifest_path,
        plan_lock_path=plan_lock,
        output_dir=tmp_path / "selection",
    )

    assert report["status"] == "selected"
    assert report["selected_candidate_id"] == STABLE_ID
    assert report["eligibility_rule"] == (
        "near_non_decreasing_and_gapped_strict_improvement"
    )
    assert report["candidates"][STABLE_ID]["eligible"] is True
    lock = json.loads(
        (tmp_path / "selection" / "selection-lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert lock["selection_order"] == [
        "maximum_mean_gapped_mrr_delta",
        "maximum_worst_gapped_mrr_delta",
        "maximum_mean_near_mrr_delta",
        "minimum_tie_break_priority",
    ]


def test_far_horizon_fold_gap_must_cover_short_window(
    tmp_path: Path,
) -> None:
    plan_path = _write_time_local_plan(tmp_path)
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )
    plan_lock = tmp_path / "frozen" / "validation-plan-lock.json"
    manifest_path = _write_time_local_rolling_manifest(
        tmp_path,
        plan_lock=plan_lock,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["gapped_folds"][0]
    first["score_time_min"] = (
        first["train_time_max"] + SHORT_WINDOW_SECONDS - 1
    )
    first["score_time_max"] = first["score_time_min"] + 9
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="short window"):
        select_standard_rolling_candidate(
            manifest_path=manifest_path,
            plan_lock_path=plan_lock,
            output_dir=tmp_path / "selection",
        )


def test_time_local_external_is_safety_gate_not_effect_size_estimate(
    tmp_path: Path,
) -> None:
    selection_lock = _select_time_local_candidate(tmp_path)
    manifest_path = _write_external_manifest(
        tmp_path,
        selection_lock=selection_lock,
        horizon_seconds=DEPLOYMENT_HORIZON_SECONDS,
    )

    report = evaluate_standard_external_gate(
        manifest_path=manifest_path,
        selection_lock_path=selection_lock,
        state_dir=tmp_path / "external-state",
    )

    interpretation = report["effect_size_interpretation"]
    assert interpretation["decision_role"] == "safety_gate_only"
    assert interpretation["effect_size_estimation_authorized"] is False
    assert interpretation["calibration_discount_factor"] == 19.5
    assert interpretation["raw_deltas_participate_in_safety_gate"] is True
    assert interpretation["calibrated_effect_size_proxy"]["mrr"] == (
        pytest.approx(report["delta_candidate_minus_baseline"]["mrr"] / 19.5)
    )


def test_external_gate_rejects_declared_baseline_mutation(
    tmp_path: Path,
) -> None:
    plan_path = _write_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["baseline"] = {
        "baseline_id": "cooccur_lift_aux_expert_v1_promoted_champion",
        "integration_id": "cooccur_lift_aux_expert_v1",
        "checkpoint_sha256": "c" * 64,
        "selected_weight": 0.5,
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=tmp_path / "frozen",
    )
    plan_lock = tmp_path / "frozen" / "validation-plan-lock.json"
    lock = json.loads(plan_lock.read_text(encoding="utf-8"))
    rolling_manifest_path = _write_rolling_manifest(
        tmp_path,
        plan_lock=plan_lock,
    )
    rolling_manifest = json.loads(
        rolling_manifest_path.read_text(encoding="utf-8")
    )
    rolling_manifest["baseline_sha256"] = lock["baseline_sha256"]
    rolling_manifest_path.write_text(
        json.dumps(rolling_manifest, indent=2),
        encoding="utf-8",
    )
    selection_report = select_standard_rolling_candidate(
        manifest_path=rolling_manifest_path,
        plan_lock_path=plan_lock,
        output_dir=tmp_path / "selection",
    )
    assert selection_report["status"] == "selected"
    selection_lock = tmp_path / "selection" / "selection-lock.json"
    external_manifest_path = _write_external_manifest(
        tmp_path,
        selection_lock=selection_lock,
        horizon_seconds=DEPLOYMENT_HORIZON_SECONDS,
    )
    external_manifest = json.loads(
        external_manifest_path.read_text(encoding="utf-8")
    )
    external_manifest["baseline_sha256"] = "f" * 64
    external_manifest_path.write_text(
        json.dumps(external_manifest, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline"):
        evaluate_standard_external_gate(
            manifest_path=external_manifest_path,
            selection_lock_path=selection_lock,
            state_dir=tmp_path / "external-state",
        )


def test_external_gate_enforces_preregistered_horizon_before_opening(
    tmp_path: Path,
) -> None:
    selection_lock = _select_stable_candidate(tmp_path)
    manifest_path = _write_external_manifest(
        tmp_path,
        selection_lock=selection_lock,
        horizon_seconds=DEPLOYMENT_HORIZON_SECONDS - 1,
    )
    state_dir = tmp_path / "external-state"

    with pytest.raises(ValueError, match="deployment horizon"):
        evaluate_standard_external_gate(
            manifest_path=manifest_path,
            selection_lock_path=selection_lock,
            state_dir=state_dir,
        )

    assert not (state_dir / "external-open-receipt.json").exists()


def test_external_gate_writes_one_shot_receipt_before_loading_scores(
    tmp_path: Path,
) -> None:
    selection_lock = _select_stable_candidate(tmp_path)
    manifest_path = _write_external_manifest(
        tmp_path,
        selection_lock=selection_lock,
        horizon_seconds=DEPLOYMENT_HORIZON_SECONDS,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate"]["path"] = "missing-candidate.npy"
    manifest["candidate"]["sha256"] = "f" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    state_dir = tmp_path / "external-state"

    with pytest.raises(FileNotFoundError):
        evaluate_standard_external_gate(
            manifest_path=manifest_path,
            selection_lock_path=selection_lock,
            state_dir=state_dir,
        )

    assert (state_dir / "external-open-receipt.json").is_file()
    with pytest.raises(FileExistsError, match="already opened"):
        evaluate_standard_external_gate(
            manifest_path=manifest_path,
            selection_lock_path=selection_lock,
            state_dir=state_dir,
        )


def test_external_gate_reports_full_metric_panel_and_forbids_rescan(
    tmp_path: Path,
) -> None:
    selection_lock = _select_stable_candidate(tmp_path)
    manifest_path = _write_external_manifest(
        tmp_path,
        selection_lock=selection_lock,
        horizon_seconds=DEPLOYMENT_HORIZON_SECONDS,
    )

    report = evaluate_standard_external_gate(
        manifest_path=manifest_path,
        selection_lock_path=selection_lock,
        state_dir=tmp_path / "external-state",
    )

    assert report["status"] == "accepted"
    assert report["actual_horizon_seconds"] == DEPLOYMENT_HORIZON_SECONDS
    assert report["candidate"]["query_movements"] == {
        "improved": 2,
        "unchanged": 2,
        "worsened": 0,
    }
    assert report["weight_rescan_authorized"] is False
    assert report["feature_rescan_authorized"] is False
    assert report["leaderboard_tuning_authorized"] is False
    assert report["package_authorized"] is True


def _select_stable_candidate(root: Path) -> Path:
    plan_path = _write_plan(root)
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=root / "frozen",
    )
    plan_lock = root / "frozen" / "validation-plan-lock.json"
    manifest_path = _write_rolling_manifest(
        root,
        plan_lock=plan_lock,
    )
    select_standard_rolling_candidate(
        manifest_path=manifest_path,
        plan_lock_path=plan_lock,
        output_dir=root / "selection",
    )
    return root / "selection" / "selection-lock.json"


def _select_time_local_candidate(root: Path) -> Path:
    plan_path = _write_time_local_plan(root)
    freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=root / "frozen",
    )
    plan_lock = root / "frozen" / "validation-plan-lock.json"
    manifest_path = _write_time_local_rolling_manifest(
        root,
        plan_lock=plan_lock,
    )
    select_standard_rolling_candidate(
        manifest_path=manifest_path,
        plan_lock_path=plan_lock,
        output_dir=root / "selection",
    )
    return root / "selection" / "selection-lock.json"


def _write_plan(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "protocol": "standard_validation_plan_v1",
        "experiment_id": "dataset2_standard_protocol_fixture",
        "candidate_family": "final_integrated_fusion_variants",
        "candidate_space": [
            {
                "candidate_id": STABLE_ID,
                "config_sha256": STABLE_CONFIG_SHA256,
                "tie_break_priority": 0,
            },
            {
                "candidate_id": UNSTABLE_ID,
                "config_sha256": UNSTABLE_CONFIG_SHA256,
                "tie_break_priority": 1,
            },
        ],
        "rolling_selection": {
            "minimum_folds": 3,
            "reserved_gate_folds": 1,
            "aggregation": "equal_weight_fold_mean",
            "per_fold_minimum_deltas": {
                "mrr": 0.0,
                "ndcg_at_10": 0.0,
            },
            "mean_minimum_deltas": {
                "mrr": 0.0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_10": 0.0,
                "ndcg_at_10": 0.0,
            },
            "mean_maximum_deltas": {"mean_rank": 0.0},
            "minimum_improved_minus_worsened": 1,
            "selection_order": [
                "maximum_mean_fold_mrr_delta",
                "maximum_worst_fold_mrr_delta",
                "maximum_mean_fold_ndcg_at_10_delta",
                "minimum_tie_break_priority",
            ],
        },
        "external_gate": {
            "holdout_id": "dataset2_external_20k_v1",
            "lineage_sha256": "e" * 64,
            "deployment_horizon_seconds": DEPLOYMENT_HORIZON_SECONDS,
            "minimum_horizon_seconds": DEPLOYMENT_HORIZON_SECONDS,
            "minimum_start_gap_seconds": 0,
            "strictly_increasing_metrics": ["mrr"],
            "minimum_deltas": {
                "mrr": 0.0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_10": 0.0,
                "ndcg_at_10": 0.0,
            },
            "maximum_deltas": {"mean_rank": 0.0},
            "minimum_improved_minus_worsened": 1,
        },
    }
    path = root / "validation-plan.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def _write_time_local_plan(root: Path) -> Path:
    path = _write_plan(root)
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["temporal_scope"] = {
        "kind": "time_local",
        "short_window_seconds": SHORT_WINDOW_SECONDS,
    }
    plan["far_horizon_validation"] = {
        "protocol": "standard_far_horizon_validation_v1",
        "eligibility_rule": "deployment_mixture",
        "minimum_gapped_folds": 3,
        "gapped_fold_aggregation": "equal_weight_fold_mean",
        "deployment_aggregation": "near_collapsed_mixture",
        "deployment_collapsed_fraction": DEPLOYMENT_COLLAPSED_FRACTION,
        "gapped_fold_specs": [
            {
                "fold_id": "gapped-p75",
                "deployment_horizon_quantile": 0.75,
                "minimum_gap_seconds": 200,
            },
            {
                "fold_id": "gapped-p90",
                "deployment_horizon_quantile": 0.90,
                "minimum_gap_seconds": 300,
            },
            {
                "fold_id": "gapped-p100",
                "deployment_horizon_quantile": 1.0,
                "minimum_gap_seconds": 400,
            },
        ],
        "gapped_per_fold_minimum_deltas": {
            "mrr": 0.0,
            "ndcg_at_10": 0.0,
        },
        "minimum_deployment_improved_minus_worsened_rate": 0.0,
        "selection_order": [
            "maximum_deployment_mixture_mrr_delta",
            "maximum_worst_gapped_mrr_delta",
            "maximum_deployment_mixture_ndcg_at_10_delta",
            "minimum_tie_break_priority",
        ],
        "zero_short_counterfactual": {
            "enabled": True,
            "arm_id": "zero_short",
            "participates_in_selection": False,
        },
    }
    plan["external_gate"]["interpretation_policy"] = {
        "decision_role": "safety_gate_only",
        "effect_size_estimation_authorized": False,
        "calibration_discount_factor": 19.5,
    }
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def _write_dual_horizon_plan(root: Path) -> Path:
    path = _write_time_local_plan(root)
    plan = json.loads(path.read_text(encoding="utf-8"))
    far_horizon = plan["far_horizon_validation"]
    far_horizon["eligibility_rule"] = (
        "near_non_decreasing_and_gapped_strict_improvement"
    )
    far_horizon["gapped_strictly_increasing_metrics"] = ["mrr"]
    far_horizon["selection_order"] = [
        "maximum_mean_gapped_mrr_delta",
        "maximum_worst_gapped_mrr_delta",
        "maximum_mean_near_mrr_delta",
        "minimum_tie_break_priority",
    ]
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def _write_rolling_manifest(
    root: Path,
    *,
    plan_lock: Path,
) -> Path:
    candidate_fingerprint = "d" * 64
    fold_sizes = [2, 4, 8]
    folds = []
    for index, row_count in enumerate(fold_sizes):
        baseline_ranks = [2] * row_count
        stable_ranks = [1] + [2] * (row_count - 1)
        unstable_ranks = (
            [1] * row_count
            if index == 0
            else ([3] * row_count if index == 1 else baseline_ranks)
        )
        baseline_path = root / f"fold-{index}-baseline.npy"
        stable_path = root / f"fold-{index}-stable.npy"
        unstable_path = root / f"fold-{index}-unstable.npy"
        np.save(baseline_path, _scores_for_ranks(baseline_ranks))
        np.save(stable_path, _scores_for_ranks(stable_ranks))
        np.save(unstable_path, _scores_for_ranks(unstable_ranks))
        folds.append(
            {
                "fold_id": f"fold-{index}",
                "role": "selection",
                "train_time_max": index * 10 + 9,
                "score_time_min": index * 10 + 10,
                "score_time_max": index * 10 + 19,
                "candidate_fingerprint": candidate_fingerprint,
                "baseline": _artifact(root, baseline_path),
                "candidates": {
                    STABLE_ID: {
                        **_artifact(root, stable_path),
                        "candidate_id": STABLE_ID,
                        "config_sha256": STABLE_CONFIG_SHA256,
                        "candidate_fingerprint": candidate_fingerprint,
                    },
                    UNSTABLE_ID: {
                        **_artifact(root, unstable_path),
                        "candidate_id": UNSTABLE_ID,
                        "config_sha256": UNSTABLE_CONFIG_SHA256,
                        "candidate_fingerprint": candidate_fingerprint,
                    },
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "protocol": "standard_rolling_origin_scores_v1",
        "plan_lock_sha256": _sha256(plan_lock),
        "positive_candidate_column": 0,
        "folds": folds,
        "reserved_folds": [
            {
                "fold_id": "fold-3",
                "role": "gate",
                "train_time_max": 39,
                "score_time_min": 40,
                "score_time_max": 49,
            }
        ],
    }
    path = root / "rolling-manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _write_time_local_rolling_manifest(
    root: Path,
    *,
    plan_lock: Path,
) -> Path:
    path = _write_rolling_manifest(root, plan_lock=plan_lock)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    candidate_fingerprint = "d" * 64

    for index, fold in enumerate(manifest["folds"]):
        baseline_path = root / f"time-local-near-{index}-baseline.npy"
        stable_path = root / f"time-local-near-{index}-stable.npy"
        unstable_path = root / f"time-local-near-{index}-unstable.npy"
        zero_short_baseline_path = (
            root / f"time-local-near-{index}-zero-short-baseline.npy"
        )
        zero_short_stable_path = (
            root / f"time-local-near-{index}-zero-short-stable.npy"
        )
        zero_short_unstable_path = (
            root / f"time-local-near-{index}-zero-short-unstable.npy"
        )
        np.save(baseline_path, _scores_for_ranks([2, 2, 2, 2]))
        np.save(stable_path, _scores_for_ranks([3, 2, 2, 2]))
        np.save(unstable_path, _scores_for_ranks([2, 2, 2, 2]))
        np.save(zero_short_baseline_path, _scores_for_ranks([2, 2, 2, 2]))
        np.save(zero_short_stable_path, _scores_for_ranks([12, 12, 12, 12]))
        np.save(
            zero_short_unstable_path,
            _scores_for_ranks([2, 2, 2, 2]),
        )
        fold["baseline"] = _artifact(root, baseline_path)
        fold["candidates"] = {
            STABLE_ID: _candidate_artifact(
                root,
                stable_path,
                candidate_id=STABLE_ID,
                config_sha256=STABLE_CONFIG_SHA256,
                candidate_fingerprint=candidate_fingerprint,
            ),
            UNSTABLE_ID: _candidate_artifact(
                root,
                unstable_path,
                candidate_id=UNSTABLE_ID,
                config_sha256=UNSTABLE_CONFIG_SHA256,
                candidate_fingerprint=candidate_fingerprint,
            ),
        }
        fold["counterfactual_arms"] = {
            "zero_short": {
                "baseline": _artifact(root, zero_short_baseline_path),
                "candidates": {
                    STABLE_ID: _candidate_artifact(
                        root,
                        zero_short_stable_path,
                        candidate_id=STABLE_ID,
                        config_sha256=STABLE_CONFIG_SHA256,
                        candidate_fingerprint=candidate_fingerprint,
                    ),
                    UNSTABLE_ID: _candidate_artifact(
                        root,
                        zero_short_unstable_path,
                        candidate_id=UNSTABLE_ID,
                        config_sha256=UNSTABLE_CONFIG_SHA256,
                        candidate_fingerprint=candidate_fingerprint,
                    ),
                },
            }
        }

    gapped_folds = []
    for index, (fold_id, quantile, gap) in enumerate(
        (
            ("gapped-p75", 0.75, 200),
            ("gapped-p90", 0.90, 300),
            ("gapped-p100", 1.0, 400),
        )
    ):
        baseline_path = root / f"{fold_id}-baseline.npy"
        stable_path = root / f"{fold_id}-stable.npy"
        unstable_path = root / f"{fold_id}-unstable.npy"
        np.save(baseline_path, _scores_for_ranks([2, 2, 2, 2]))
        np.save(stable_path, _scores_for_ranks([1, 1, 1, 1]))
        np.save(unstable_path, _scores_for_ranks([2, 2, 2, 2]))
        train_time_max = index * 1_000
        gapped_folds.append(
            {
                "fold_id": fold_id,
                "role": "gapped",
                "deployment_horizon_quantile": quantile,
                "train_time_max": train_time_max,
                "score_time_min": train_time_max + gap,
                "score_time_max": train_time_max + gap + 9,
                "candidate_fingerprint": candidate_fingerprint,
                "baseline": _artifact(root, baseline_path),
                "candidates": {
                    STABLE_ID: _candidate_artifact(
                        root,
                        stable_path,
                        candidate_id=STABLE_ID,
                        config_sha256=STABLE_CONFIG_SHA256,
                        candidate_fingerprint=candidate_fingerprint,
                    ),
                    UNSTABLE_ID: _candidate_artifact(
                        root,
                        unstable_path,
                        candidate_id=UNSTABLE_ID,
                        config_sha256=UNSTABLE_CONFIG_SHA256,
                        candidate_fingerprint=candidate_fingerprint,
                    ),
                },
            }
        )
    manifest["gapped_folds"] = gapped_folds
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _replace_candidate_fold_ranks(
    root: Path,
    folds: list[dict[str, object]],
    *,
    candidate_id: str,
    ranks_by_fold: list[list[int]],
) -> None:
    for index, (fold, ranks) in enumerate(
        zip(folds, ranks_by_fold, strict=True)
    ):
        path = root / f"replacement-{fold['fold_id']}-{candidate_id}-{index}.npy"
        np.save(path, _scores_for_ranks(ranks))
        descriptor = fold["candidates"][candidate_id]
        descriptor.update(_artifact(root, path))


def _write_external_manifest(
    root: Path,
    *,
    selection_lock: Path,
    horizon_seconds: int,
) -> Path:
    baseline_path = root / "external-baseline.npy"
    candidate_path = root / "external-candidate.npy"
    np.save(baseline_path, _scores_for_ranks([2, 2, 3, 3]))
    np.save(candidate_path, _scores_for_ranks([1, 2, 2, 3]))
    train_time_max = 100
    candidate_fingerprint = "c" * 64
    manifest = {
        "schema_version": 1,
        "protocol": "standard_external_scores_v1",
        "selection_lock_sha256": _sha256(selection_lock),
        "experiment_id": "dataset2_standard_protocol_fixture",
        "holdout_id": "dataset2_external_20k_v1",
        "lineage_sha256": "e" * 64,
        "selected_candidate_id": STABLE_ID,
        "selected_candidate_config_sha256": STABLE_CONFIG_SHA256,
        "positive_candidate_column": 0,
        "candidate_fingerprint": candidate_fingerprint,
        "training_time_max": train_time_max,
        "score_time_min": train_time_max,
        "score_time_max": train_time_max + horizon_seconds,
        "baseline": _artifact(root, baseline_path),
        "candidate": {
            **_artifact(root, candidate_path),
            "candidate_id": STABLE_ID,
            "config_sha256": STABLE_CONFIG_SHA256,
            "candidate_fingerprint": candidate_fingerprint,
        },
    }
    path = root / "external-manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _artifact(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
    }


def _candidate_artifact(
    root: Path,
    path: Path,
    *,
    candidate_id: str,
    config_sha256: str,
    candidate_fingerprint: str,
) -> dict[str, str]:
    return {
        **_artifact(root, path),
        "candidate_id": candidate_id,
        "config_sha256": config_sha256,
        "candidate_fingerprint": candidate_fingerprint,
    }


def _scores_for_ranks(
    ranks: list[int],
    *,
    candidate_count: int = 12,
) -> np.ndarray:
    scores = np.full((len(ranks), candidate_count), 0.1, dtype=np.float64)
    scores[:, 0] = 0.5
    for row, rank in enumerate(ranks):
        if rank < 1 or rank > candidate_count:
            raise ValueError("rank outside candidate count")
        scores[row, 1:rank] = 0.6
    return scores


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
