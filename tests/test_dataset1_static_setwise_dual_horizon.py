from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jgrec.rankers.hybrid.static_setwise import (
    apply_prediction_history_limit,
    blend_static_setwise,
    evaluate_external_safety_deltas,
    select_dual_horizon_static_weight,
    static_setwise_weight_grid,
)


@dataclass(frozen=True)
class _PredictionConfig:
    structure_predict_neighbor_limit: int = 512
    source_profile_predict_history_limit: int = 512
    untouched: int = 7


def test_dataset1_static_setwise_grid_is_exactly_preregistered() -> None:
    assert static_setwise_weight_grid() == tuple(
        index / 100.0 for index in range(5, 81, 5)
    )


def test_static_setwise_blend_uses_one_weight_for_every_query() -> None:
    backbone = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    setwise = np.asarray([[0.6, 0.4], [0.7, 0.3]])

    actual = blend_static_setwise(backbone, setwise, weight=0.75)

    np.testing.assert_allclose(
        actual,
        0.25 * backbone + 0.75 * setwise,
    )


def test_k256_overrides_both_prediction_limits_only() -> None:
    original = _PredictionConfig()

    actual = apply_prediction_history_limit(original, limit=256)

    assert actual.structure_predict_neighbor_limit == 256
    assert actual.source_profile_predict_history_limit == 256
    assert actual.untouched == 7
    assert original.structure_predict_neighbor_limit == 512
    assert original.source_profile_predict_history_limit == 512


def test_dual_horizon_selector_requires_every_near_and_gapped_gate() -> None:
    trials = {
        0.70: {
            "near_mrr": (0.001, 0.002, 0.001),
            "near_ndcg_at_10": (0.001, 0.001, 0.001),
            "gapped_mrr": (0.003, 0.004, 0.002),
            "gapped_ndcg_at_10": (0.002, 0.002, 0.001),
        },
        0.75: {
            "near_mrr": (0.002, -0.000001, 0.002),
            "near_ndcg_at_10": (0.002, 0.001, 0.002),
            "gapped_mrr": (0.005, 0.005, 0.005),
            "gapped_ndcg_at_10": (0.003, 0.003, 0.003),
        },
        0.80: {
            "near_mrr": (0.001, 0.002, 0.001),
            "near_ndcg_at_10": (0.001, 0.001, 0.001),
            "gapped_mrr": (0.004, 0.005, 0.003),
            "gapped_ndcg_at_10": (0.002, 0.002, 0.001),
        },
    }

    selection = select_dual_horizon_static_weight(trials)

    assert selection["selected_weight"] == 0.80
    assert selection["eligible_weights"] == [0.70, 0.80]
    assert selection["trials"]["0.75"]["eligible"] is False


def test_external_safety_gate_is_seven_directional_gates_only() -> None:
    accepted = evaluate_external_safety_deltas(
        {
            "mrr": 0.0001,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0002,
            "hit_at_10": 0.0,
            "ndcg_at_10": 0.0001,
            "mean_rank": -0.01,
        },
        improved=11,
        worsened=10,
    )
    rejected = evaluate_external_safety_deltas(
        {
            "mrr": 0.0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "mean_rank": 0.0,
        },
        improved=11,
        worsened=10,
    )

    assert accepted["accepted"] is True
    assert len(accepted["gates"]) == 7
    assert rejected["accepted"] is False
    assert rejected["gates"]["mrr_strictly_increases"] is False
