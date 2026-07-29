from __future__ import annotations

import numpy as np

from jgrec.rankers.hybrid.champion_residual import (
    champion_hard_negative_indices,
    lambda_mrr_pair_weights,
    lambda_mrr_pairwise_loss,
    route_champion_topk_residual,
)


def test_champion_hard_negatives_exclude_positive_and_follow_stable_rank() -> None:
    champion_scores = np.asarray(
        [
            [0.40, 0.90, 0.10, 0.80, 0.20],
            [0.95, 0.20, 0.70, 0.70, 0.10],
        ],
        dtype=np.float32,
    )

    hard_negatives = champion_hard_negative_indices(
        champion_scores,
        top_k=2,
    )
    weights = lambda_mrr_pair_weights(
        champion_scores,
        hard_negatives,
    )

    np.testing.assert_array_equal(
        hard_negatives,
        np.asarray([[1, 3], [2, 3]], dtype=np.int64),
    )
    assert not np.any(hard_negatives == 0)
    np.testing.assert_allclose(
        weights,
        np.asarray(
            [
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 2.0, 1.0 / 2.0],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-7,
    )


def test_lambda_mrr_pairwise_loss_rewards_larger_positive_margin() -> None:
    positive = np.asarray([-1.0, 0.5], dtype=np.float32)
    negatives = np.asarray(
        [[0.5, -0.5], [0.8, -0.2]],
        dtype=np.float32,
    )
    weights = np.asarray(
        [[2.0 / 3.0, 1.0 / 6.0], [0.5, 0.5]],
        dtype=np.float32,
    )

    baseline_loss = lambda_mrr_pairwise_loss(
        positive,
        negatives,
        weights,
    )
    improved_loss = lambda_mrr_pairwise_loss(
        positive + 1.0,
        negatives - 1.0,
        weights,
    )

    assert improved_loss < baseline_loss


def test_topk_residual_routes_only_high_confidence_switches() -> None:
    champion = np.asarray(
        [
            [0.60, 0.25, 0.10, 0.05],
            [0.60, 0.25, 0.10, 0.05],
            [0.60, 0.25, 0.10, 0.05],
        ],
        dtype=np.float32,
    )
    residual = np.asarray(
        [
            [0.0, 0.60, 20.0, 0.0],
            [0.0, 0.95, 20.0, 0.0],
            [0.0, 1.50, 20.0, 0.0],
        ],
        dtype=np.float32,
    )

    routed = route_champion_topk_residual(
        champion,
        residual,
        top_k=2,
        minimum_switch_gain=0.20,
    )

    assert routed.use_residual.tolist() == [False, False, True]
    np.testing.assert_array_equal(routed.scores[:2], champion[:2])
    np.testing.assert_array_equal(
        routed.scores[2],
        np.asarray([0.25, 0.60, 0.10, 0.05], dtype=np.float32),
    )
    np.testing.assert_array_equal(routed.scores[:, 2:], champion[:, 2:])
    np.testing.assert_array_equal(
        np.sort(routed.scores, axis=1),
        np.sort(champion, axis=1),
    )
    assert routed.switch_gain[1] < 0.20
    assert routed.switch_gain[2] > 0.20


def test_topk_residual_route_follows_candidate_permutation() -> None:
    champion = np.asarray(
        [[0.60, 0.25, 0.10, 0.05]],
        dtype=np.float32,
    )
    residual = np.asarray(
        [[0.0, 1.50, 20.0, 0.0]],
        dtype=np.float32,
    )
    permutation = np.asarray([2, 0, 3, 1])

    expected = route_champion_topk_residual(
        champion,
        residual,
        top_k=2,
        minimum_switch_gain=0.20,
    )
    permuted = route_champion_topk_residual(
        champion[:, permutation],
        residual[:, permutation],
        top_k=2,
        minimum_switch_gain=0.20,
    )

    np.testing.assert_array_equal(
        permuted.scores,
        expected.scores[:, permutation],
    )
    np.testing.assert_array_equal(
        permuted.use_residual,
        expected.use_residual,
    )
    np.testing.assert_allclose(
        permuted.switch_gain,
        expected.switch_gain,
        rtol=0.0,
        atol=1e-7,
    )
