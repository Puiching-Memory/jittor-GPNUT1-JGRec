from __future__ import annotations

import numpy as np
import pytest

from jgrec.rankers.hybrid.config import TwoTowerConfig
from jgrec.rankers.hybrid.tower_optimization_experiment import (
    paired_rank_movements,
    positive_ranks,
    ranking_metrics,
    two_tower_screen_config,
    two_tower_screen_gate,
)


@pytest.mark.parametrize(
    ("arm", "schedule", "weight_decay", "in_batch"),
    [
        ("control", "constant", 0.0, False),
        ("optimizer_only", "cosine", 1e-4, False),
        ("inbatch_only", "constant", 0.0, True),
        ("combined", "cosine", 1e-4, True),
    ],
)
def test_two_tower_screen_config_changes_only_frozen_factors(
    arm: str,
    schedule: str,
    weight_decay: float,
    in_batch: bool,
) -> None:
    base = TwoTowerConfig(
        epochs=50,
        max_samples=200_000,
        num_negatives=99,
        objective="listwise",
        early_stop_metric="mrr",
        in_batch_negative_weight=1.0,
        in_batch_temperature=1.0,
    )

    actual = two_tower_screen_config(base, arm)

    assert actual.lr_schedule == schedule
    assert actual.min_lr_ratio == pytest.approx(
        0.1 if schedule == "cosine" else 0.0
    )
    assert actual.weight_decay == pytest.approx(weight_decay)
    assert actual.in_batch_negatives is in_batch
    assert actual.epochs == base.epochs
    assert actual.max_samples == base.max_samples
    assert actual.num_negatives == base.num_negatives
    assert actual.objective == base.objective
    assert actual.early_stop_metric == base.early_stop_metric
    assert actual.in_batch_negative_weight == pytest.approx(1.0)
    assert actual.in_batch_temperature == pytest.approx(1.0)


def test_ranking_metrics_report_the_frozen_non_mrr_contract() -> None:
    scores = np.asarray(
        [
            [3.0, 2.0, 1.0, 0.0],
            [2.0, 3.0, 1.0, 0.0],
            [1.0, 4.0, 3.0, 2.0],
        ],
        dtype=np.float32,
    )

    metrics = ranking_metrics(scores)

    assert metrics["mrr"] == pytest.approx((1.0 + 0.5 + 0.25) / 3.0)
    assert metrics["hit_at_1"] == pytest.approx(1.0 / 3.0)
    assert metrics["hit_at_3"] == pytest.approx(2.0 / 3.0)
    assert metrics["hit_at_10"] == pytest.approx(1.0)
    assert metrics["ndcg_at_10"] == pytest.approx(
        (
            1.0
            + 1.0 / np.log2(3.0)
            + 1.0 / np.log2(5.0)
        )
        / 3.0
    )
    assert metrics["mean_rank"] == pytest.approx(7.0 / 3.0)


def test_positive_ranks_are_neutral_to_exact_score_ties() -> None:
    scores = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 2.0, 1.0],
            [2.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )

    ranks = positive_ranks(scores)

    np.testing.assert_array_equal(
        ranks,
        np.asarray([2.5, 1.5, 2.5], dtype=np.float64),
    )
    metrics = ranking_metrics(scores[:1])
    assert metrics["mrr"] == pytest.approx(0.4)
    assert metrics["hit_at_1"] == pytest.approx(0.0)
    assert metrics["mean_rank"] == pytest.approx(2.5)


def test_paired_rank_movements_count_improved_and_worsened_queries() -> None:
    control = np.asarray([3, 2, 4, 1, 2], dtype=np.int32)
    candidate = np.asarray([2, 2, 3, 2, 1], dtype=np.int32)

    movements = paired_rank_movements(control, candidate)

    assert movements == {
        "improved_queries": 3,
        "worsened_queries": 1,
        "unchanged_queries": 1,
        "net_improved_queries": 2,
        "mean_rank_delta": pytest.approx(-0.4),
    }


def test_two_tower_screen_gate_requires_all_metrics_and_slices() -> None:
    control = {
        "mrr": 0.40,
        "hit_at_1": 0.30,
        "hit_at_3": 0.50,
        "hit_at_10": 0.75,
        "ndcg_at_10": 0.48,
        "mean_rank": 8.0,
    }
    candidate = {
        "mrr": 0.41,
        "hit_at_1": 0.31,
        "hit_at_3": 0.51,
        "hit_at_10": 0.76,
        "ndcg_at_10": 0.49,
        "mean_rank": 7.9,
    }
    slices = [
        {"mrr": 0.40, "ndcg_at_10": 0.48},
        {"mrr": 0.39, "ndcg_at_10": 0.47},
        {"mrr": 0.38, "ndcg_at_10": 0.46},
    ]
    candidate_slices = [
        {"mrr": 0.41, "ndcg_at_10": 0.49},
        {"mrr": 0.40, "ndcg_at_10": 0.48},
        {"mrr": 0.39, "ndcg_at_10": 0.47},
    ]
    movements = {"improved_queries": 10, "worsened_queries": 7}

    passed = two_tower_screen_gate(
        control,
        candidate,
        slices,
        candidate_slices,
        movements,
    )
    rank_regression = two_tower_screen_gate(
        control,
        {**candidate, "mean_rank": 8.1},
        slices,
        candidate_slices,
        movements,
    )
    slice_regression = two_tower_screen_gate(
        control,
        candidate,
        slices,
        [
            candidate_slices[0],
            {**candidate_slices[1], "ndcg_at_10": 0.46},
            candidate_slices[2],
        ],
        movements,
    )
    movement_regression = two_tower_screen_gate(
        control,
        candidate,
        slices,
        candidate_slices,
        {"improved_queries": 7, "worsened_queries": 10},
    )

    assert passed["passed"] is True
    assert rank_regression["passed"] is False
    assert "full_mean_rank" in rank_regression["failed_checks"]
    assert slice_regression["passed"] is False
    assert "slice_1_ndcg_at_10" in slice_regression["failed_checks"]
    assert movement_regression["passed"] is False
    assert "query_movements" in movement_regression["failed_checks"]
