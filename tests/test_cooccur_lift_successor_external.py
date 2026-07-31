from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jgrec.cooccur_lift_successor_external import (
    authorize_successor_package,
    build_standard_external_manifest,
    full_origin_copy_weights,
    short_window_support_from_availability,
    validate_successor_external_setup,
)

CANDIDATE_ID = "cooccur_lift_gap_aware_v2"
CANDIDATE_SHA256 = "2" * 64
BASELINE_SHA256 = "b" * 64
COLLAPSED_FRACTION = 0.39971972363446745
SHORT_WINDOW_SECONDS = 17_038_080
GAPS = (21_686_400, 26_611_200, 30_153_600)


def test_successor_external_setup_binds_gap_aware_and_exact_seven_gates(
    tmp_path: Path,
) -> None:
    config_path, lock_path = _write_contracts(tmp_path)

    setup = validate_successor_external_setup(
        candidate_config_path=config_path,
        selection_lock_path=lock_path,
    )

    assert setup.candidate_id == CANDIDATE_ID
    assert setup.selected_weight == 0.5
    assert setup.full_origin_seed == 33_100
    assert setup.short_window_seconds == SHORT_WINDOW_SECONDS
    assert setup.collapsed_fraction == COLLAPSED_FRACTION
    assert setup.gap_seconds == GAPS
    assert setup.baseline_sha256 == BASELINE_SHA256
    assert setup.external_gate_names == (
        "mrr_meets_minimum",
        "hit_at_1_meets_minimum",
        "hit_at_3_meets_minimum",
        "hit_at_10_meets_minimum",
        "ndcg_at_10_meets_minimum",
        "mean_rank_meets_maximum",
        "improved_minus_worsened_meets_minimum",
    )


def test_full_origin_copy_weights_equalize_the_three_gapped_states() -> None:
    weights = full_origin_copy_weights(
        collapsed_fraction=COLLAPSED_FRACTION,
        gapped_copy_count=3,
    )

    assert weights[0] == pytest.approx(1.0 - COLLAPSED_FRACTION)
    assert weights[1:] == pytest.approx(
        (COLLAPSED_FRACTION / 3,) * 3
    )
    assert sum(weights) == pytest.approx(1.0)


def test_short_window_support_uses_feature_availability_not_score_horizon() -> None:
    query_time = np.asarray([100, 111, 200], dtype=np.int64)
    near_availability = query_time.copy()
    deployed_history_end = np.asarray([90, 100, 100], dtype=np.int64)

    near = short_window_support_from_availability(
        query_time,
        near_availability,
        short_window_seconds=11,
    )
    deployed = short_window_support_from_availability(
        query_time,
        deployed_history_end,
        short_window_seconds=11,
    )

    np.testing.assert_array_equal(near, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(deployed, [1.0, 0.0, 0.0])


def test_deployed_support_is_not_forced_to_the_training_mixture_fraction() -> None:
    query_time = np.asarray([100, 110, 120, 130], dtype=np.int64)
    history_end = np.full(4, 100, dtype=np.int64)

    support = short_window_support_from_availability(
        query_time,
        history_end,
        short_window_seconds=20,
    )

    np.testing.assert_array_equal(support, [1.0, 1.0, 0.0, 0.0])
    assert float(np.mean(1.0 - support)) != pytest.approx(
        COLLAPSED_FRACTION
    )


def test_external_manifest_binds_frozen_baseline_and_zero_collapse(
    tmp_path: Path,
) -> None:
    config_path, lock_path = _write_contracts(tmp_path)
    setup = validate_successor_external_setup(
        candidate_config_path=config_path,
        selection_lock_path=lock_path,
    )
    baseline_path = tmp_path / "baseline.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(baseline_path, np.ones((2, 3), dtype=np.float32))
    np.save(candidate_path, np.ones((2, 3), dtype=np.float32))

    manifest = build_standard_external_manifest(
        setup=setup,
        candidate_fingerprint="c" * 64,
        training_time_max=100,
        score_time_min=100,
        score_time_max=100 + 40_435_200,
        baseline_path=baseline_path,
        baseline_sha256=_sha256(baseline_path),
        candidate_path=candidate_path,
        candidate_sha256=_sha256(candidate_path),
        scored_rows=2,
        supported_rows=2,
    )

    assert manifest["protocol"] == "standard_external_scores_v1"
    assert manifest["baseline_sha256"] == BASELINE_SHA256
    assert manifest["selected_candidate_id"] == CANDIDATE_ID
    assert manifest["candidate"]["config_sha256"] == setup.config_sha256
    assert manifest["short_window_support"] == {
        "collapsed_fraction": 0.0,
        "supported_rows": 2,
        "total_rows": 2,
        "unique_values": [1],
    }


def test_package_authorization_reads_only_accepted_seven_gate_state() -> None:
    report = {
        "status": "accepted",
        "selected_candidate": {
            "candidate_id": CANDIDATE_ID,
            "config_sha256": CANDIDATE_SHA256,
        },
        "selection_lock_sha256": "a" * 64,
        "gates": {
            "mrr_meets_minimum": True,
            "hit_at_1_meets_minimum": True,
            "hit_at_3_meets_minimum": True,
            "hit_at_10_meets_minimum": True,
            "ndcg_at_10_meets_minimum": True,
            "mean_rank_meets_maximum": True,
            "improved_minus_worsened_meets_minimum": True,
        },
        "package_authorized": True,
    }

    evidence = authorize_successor_package(
        external_report=report,
        external_report_sha256="e" * 64,
        expected_selection_lock_sha256="a" * 64,
        expected_candidate_id=CANDIDATE_ID,
        expected_config_sha256=CANDIDATE_SHA256,
    )

    assert evidence["external_report_sha256"] == "e" * 64
    assert evidence["gate_count"] == 7
    rejected = json.loads(json.dumps(report))
    rejected["gates"]["hit_at_10_meets_minimum"] = False
    with pytest.raises(ValueError, match="seven-gate"):
        authorize_successor_package(
            external_report=rejected,
            external_report_sha256="e" * 64,
            expected_selection_lock_sha256="a" * 64,
            expected_candidate_id=CANDIDATE_ID,
            expected_config_sha256=CANDIDATE_SHA256,
        )


def _write_contracts(tmp_path: Path) -> tuple[Path, Path]:
    config = {
        "candidate_id": CANDIDATE_ID,
        "short_window": {
            "short_window_seconds": SHORT_WINDOW_SECONDS,
        },
        "horizon_training": {
            "gapped_fold_specs": [
                {"gap_seconds": gap} for gap in GAPS
            ],
            "support_state_training_contract": {
                "near_copy_weight": 1.0 - COLLAPSED_FRACTION,
                "collapsed_copy_weight": COLLAPSED_FRACTION,
            },
        },
        "randomness": {
            "base_seed": 60,
            "seed_salt": 30_013,
        },
        "integration": {
            "selected_weight": 0.5,
        },
    }
    config_path = tmp_path / "candidate.json"
    config_path.write_text(
        json.dumps(config, sort_keys=True),
        encoding="utf-8",
    )
    lock = {
        "protocol": "standard_validation_selection_lock_v1",
        "experiment_id": "dataset2_cooccur-lift-successor-v2",
        "selected_candidate": {
            "candidate_id": CANDIDATE_ID,
            "config_sha256": _sha256(config_path),
        },
        "baseline": {
            "baseline_id": "cooccur_lift_aux_expert_v1_promoted_champion",
            "checkpoint_sha256": "d" * 64,
            "integration_id": "cooccur_lift_aux_expert_v1",
            "selected_weight": 0.5,
        },
        "baseline_sha256": BASELINE_SHA256,
        "external_holdout_read": False,
        "external_gate": {
            "holdout_id": "dataset2_external_20k_v1",
            "lineage_sha256": "f" * 64,
            "deployment_horizon_seconds": 40_435_200,
            "minimum_horizon_seconds": 40_435_200,
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
            "interpretation_policy": {
                "decision_role": "safety_gate_only",
                "effect_size_estimation_authorized": False,
                "calibration_discount_factor": 19.5,
            },
        },
        "weight_rescan_authorized": False,
        "feature_rescan_authorized": False,
        "leaderboard_tuning_authorized": False,
    }
    lock_path = tmp_path / "selection-lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    return config_path, lock_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
