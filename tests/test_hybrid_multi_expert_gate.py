from __future__ import annotations

import numpy as np
import pytest

from jgrec.rankers.hybrid.multi_expert_gate import (
    MultiExpertGateConfig,
    expert_top1_feature_deltas,
    fit_multi_expert_gate,
    multi_expert_score_descriptors,
    predict_multi_expert_gate,
    route_multi_expert,
    select_multi_expert_config_on_forward_slice,
)

EXPERT_ORDER = (
    "current_gate",
    "v1_champion",
    "multi_interest",
)


def test_expert_top1_feature_deltas_are_tie_neutral_and_permutation_invariant(
) -> None:
    scores = {
        "current_gate": np.asarray(
            [
                [0.9, 0.9, 0.1, 0.0],
                [0.2, 0.3, 0.8, 0.1],
            ],
            dtype=np.float32,
        ),
        "multi_interest": np.asarray(
            [
                [0.1, 0.2, 0.8, 0.8],
                [0.9, 0.1, 0.2, 0.3],
            ],
            dtype=np.float32,
        ),
        "window_ensemble": np.asarray(
            [
                [0.1, 0.9, 0.2, 0.3],
                [0.1, 0.8, 0.2, 0.8],
            ],
            dtype=np.float32,
        ),
    }
    candidate_features = np.asarray(
        [
            [
                [1.0, 10.0, 100.0],
                [3.0, 30.0, 300.0],
                [5.0, 50.0, 500.0],
                [9.0, 90.0, 900.0],
            ],
            [
                [2.0, 20.0, 200.0],
                [4.0, 40.0, 400.0],
                [8.0, 80.0, 800.0],
                [10.0, 100.0, 1000.0],
            ],
        ],
        dtype=np.float32,
    )

    descriptors, names = expert_top1_feature_deltas(
        scores,
        candidate_features,
        candidate_feature_names=("raw_a", "raw_b", "unused"),
        selected_feature_names=("raw_a", "raw_b"),
        fallback_expert="current_gate",
        alternative_order=("multi_interest", "window_ensemble"),
    )

    assert names == (
        "multi_interest__vs__current_gate_top1_delta__raw_a",
        "multi_interest__vs__current_gate_top1_delta__raw_b",
        "window_ensemble__vs__current_gate_top1_delta__raw_a",
        "window_ensemble__vs__current_gate_top1_delta__raw_b",
    )
    np.testing.assert_allclose(
        descriptors,
        np.asarray(
            [
                [5.0, 50.0, 1.0, 10.0],
                [-6.0, -60.0, -1.0, -10.0],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=0.0,
    )

    permutation = np.asarray([2, 0, 3, 1])
    permuted, permuted_names = expert_top1_feature_deltas(
        {
            name: values[:, permutation]
            for name, values in scores.items()
        },
        candidate_features[:, permutation, :],
        candidate_feature_names=("raw_a", "raw_b", "unused"),
        selected_feature_names=("raw_a", "raw_b"),
        fallback_expert="current_gate",
        alternative_order=("multi_interest", "window_ensemble"),
    )
    assert permuted_names == names
    np.testing.assert_array_equal(permuted, descriptors)


def test_expert_top1_feature_deltas_reject_invalid_feature_schema() -> None:
    scores = {
        "current_gate": np.asarray([[0.7, 0.3]], dtype=np.float32),
        "multi_interest": np.asarray([[0.2, 0.8]], dtype=np.float32),
    }
    candidate_features = np.ones((1, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="unique"):
        expert_top1_feature_deltas(
            scores,
            candidate_features,
            candidate_feature_names=("support", "support"),
            selected_feature_names=("support",),
            fallback_expert="current_gate",
            alternative_order=("multi_interest",),
        )
    with pytest.raises(ValueError, match="missing"):
        expert_top1_feature_deltas(
            scores,
            candidate_features,
            candidate_feature_names=("support", "other"),
            selected_feature_names=("unknown",),
            fallback_expert="current_gate",
            alternative_order=("multi_interest",),
        )


def test_multi_expert_descriptors_are_candidate_permutation_invariant() -> None:
    scores = {
        "current_gate": np.asarray(
            [
                [0.55, 0.25, 0.15, 0.05],
                [0.10, 0.40, 0.40, 0.10],
            ],
            dtype=np.float32,
        ),
        "v1_champion": np.asarray(
            [
                [0.45, 0.35, 0.15, 0.05],
                [0.20, 0.40, 0.40, 0.00],
            ],
            dtype=np.float32,
        ),
        "multi_interest": np.asarray(
            [
                [0.20, 0.60, 0.15, 0.05],
                [0.40, 0.10, 0.10, 0.40],
            ],
            dtype=np.float32,
        ),
    }
    descriptors, names = multi_expert_score_descriptors(
        scores,
        expert_order=EXPERT_ORDER,
    )

    permutation = np.asarray([2, 0, 3, 1])
    permuted, permuted_names = multi_expert_score_descriptors(
        {
            name: values[:, permutation]
            for name, values in scores.items()
        },
        expert_order=EXPERT_ORDER,
    )

    assert names == permuted_names
    assert descriptors.shape == (2, 42)
    np.testing.assert_allclose(permuted, descriptors, rtol=0.0, atol=1e-7)

    index = {name: position for position, name in enumerate(names)}
    np.testing.assert_allclose(
        descriptors[:, index["current_gate_top_margin"]],
        np.asarray([0.30, 0.00]),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        descriptors[
            :,
            index[
                "current_gate__multi_interest_"
                "current_gate_own_top_preference"
            ],
        ],
        np.asarray([0.30, 0.30]),
        rtol=0.0,
        atol=1e-7,
    )
    assert (
        descriptors[
            1,
            index[
                "current_gate__v1_champion_top1_jaccard"
            ],
        ]
        == 1.0
    )


def test_multi_expert_router_uses_best_lift_and_exact_fallback() -> None:
    fallback = np.asarray(
        [[0.6, 0.3, 0.1], [0.5, 0.3, 0.2]],
        dtype=np.float32,
    )
    alternatives = {
        "v1_champion": np.asarray(
            [[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]],
            dtype=np.float32,
        ),
        "multi_interest": np.asarray(
            [[0.2, 0.7, 0.1], [0.3, 0.5, 0.2]],
            dtype=np.float32,
        ),
        "window_ensemble": np.asarray(
            [[0.3, 0.2, 0.5], [0.2, 0.3, 0.5]],
            dtype=np.float32,
        ),
    }
    predicted_lifts = np.asarray(
        [[0.012, 0.025, 0.020], [0.009, -0.010, 0.005]],
        dtype=np.float32,
    )

    result = route_multi_expert(
        fallback,
        alternatives,
        predicted_lifts,
        expert_order=(
            "v1_champion",
            "multi_interest",
            "window_ensemble",
        ),
        minimum_predicted_lift=0.01,
    )

    assert result.selected_experts.tolist() == [
        "multi_interest",
        "current_gate",
    ]
    np.testing.assert_array_equal(
        result.scores[0],
        alternatives["multi_interest"][0],
    )
    np.testing.assert_array_equal(result.scores[1], fallback[1])
    assert np.array_equal(
        result.scores[~result.use_alternative],
        fallback[~result.use_alternative],
    )


def test_multi_expert_router_ties_follow_frozen_expert_order() -> None:
    fallback = np.asarray([[0.6, 0.4]], dtype=np.float32)
    alternatives = {
        "v1_champion": np.asarray([[0.7, 0.3]], dtype=np.float32),
        "multi_interest": np.asarray([[0.2, 0.8]], dtype=np.float32),
        "window_ensemble": np.asarray([[0.1, 0.9]], dtype=np.float32),
    }

    result = route_multi_expert(
        fallback,
        alternatives,
        np.asarray([[0.02, 0.02, 0.01]], dtype=np.float32),
        expert_order=(
            "v1_champion",
            "multi_interest",
            "window_ensemble",
        ),
        minimum_predicted_lift=0.01,
    )

    assert result.selected_experts.tolist() == ["v1_champion"]
    np.testing.assert_array_equal(
        result.scores,
        alternatives["v1_champion"],
    )


def test_multi_expert_gate_learns_one_reward_model_per_expert() -> None:
    descriptors = np.asarray(
        [[0.0], [0.1], [0.9], [1.0]],
        dtype=np.float32,
    )
    rewards = np.asarray(
        [
            [0.5, -0.1],
            [0.5, -0.1],
            [-0.1, 0.5],
            [-0.1, 0.5],
        ],
        dtype=np.float32,
    )
    config = MultiExpertGateConfig(
        max_depth=1,
        min_samples_leaf=1,
        minimum_predicted_lift=0.1,
    )

    model = fit_multi_expert_gate(
        descriptors,
        rewards,
        config,
        descriptor_names=("signal",),
        expert_order=("v1_champion", "multi_interest"),
        seed=60,
    )
    predicted = predict_multi_expert_gate(
        model,
        np.asarray([[0.05], [0.95]], dtype=np.float32),
        descriptor_names=("signal",),
    )

    assert model.expert_order == ("v1_champion", "multi_interest")
    assert predicted.shape == (2, 2)
    assert predicted[0, 0] > predicted[0, 1]
    assert predicted[1, 1] > predicted[1, 0]


def test_forward_selector_does_not_read_gate_rows() -> None:
    descriptors = np.asarray(
        [[0.0], [0.1], [0.9], [1.0], [0.05], [0.95], [np.nan], [np.nan]],
        dtype=np.float32,
    )
    fallback = np.tile(
        np.asarray([[0.4, 0.6]], dtype=np.float32),
        (8, 1),
    )
    v1 = fallback.copy()
    multi_interest = fallback.copy()
    low_rows = np.asarray([0, 1, 4])
    high_rows = np.asarray([2, 3, 5])
    v1[low_rows] = np.asarray([0.7, 0.3], dtype=np.float32)
    multi_interest[high_rows] = np.asarray(
        [0.7, 0.3],
        dtype=np.float32,
    )
    v1[6:] = np.nan
    multi_interest[6:] = np.nan
    fallback[6:] = np.nan

    selection = select_multi_expert_config_on_forward_slice(
        descriptors,
        fallback,
        {
            "v1_champion": v1,
            "multi_interest": multi_interest,
        },
        configs=(
            MultiExpertGateConfig(
                max_depth=1,
                min_samples_leaf=1,
                minimum_predicted_lift=0.1,
            ),
        ),
        descriptor_names=("signal",),
        expert_order=("v1_champion", "multi_interest"),
        train_rows=(0, 4),
        selection_rows=(4, 6),
        minimum_selection_delta=0.4,
        maximum_coverage=1.0,
        seed=60,
    )

    assert selection is not None
    assert selection.selection_mrr == 1.0
    assert selection.fallback_mrr == 0.5
    assert selection.delta == 0.5
    assert selection.coverage == 1.0
