from __future__ import annotations

import numpy as np
import pytest

jt = pytest.importorskip("jittor")

from jgrec.rankers.hybrid.oof_utility_router import (  # noqa: E402
    OOFUtilityRouter,
    OOFUtilityRouterConfig,
    OOFUtilityRouterTrainingConfig,
    action_utility_features,
    build_utility_targets,
    fit_oof_utility_router,
    load_oof_utility_router_checkpoint,
    predict_oof_utility,
    route_by_expected_utility,
    save_oof_utility_router_checkpoint,
    utility_hurdle_loss,
)


def test_utility_targets_separate_gain_no_change_loss_and_unavailable():
    rewards = np.array([0.25, 0.0, -0.5, 1e-9], dtype=np.float32)
    available = np.array([True, True, True, False])

    targets = build_utility_targets(
        rewards,
        available=available,
        zero_tolerance=1e-8,
    )

    assert targets.labels.tolist() == [1, 0, 2, -1]
    assert targets.changed.tolist() == [True, False, True, False]
    assert targets.gain.tolist() == [True, False, False, False]
    np.testing.assert_allclose(
        targets.magnitude,
        np.array([0.25, 0.0, 0.5, 0.0], dtype=np.float32),
    )


def test_action_features_are_label_free_and_candidate_permutation_invariant():
    rng = np.random.default_rng(60)
    rows, candidates, raw_width = 4, 7, 3
    default = rng.normal(size=(rows, candidates)).astype(np.float32)
    short_residual = rng.normal(
        scale=0.03,
        size=(rows, candidates),
    ).astype(np.float32)
    action_residual = rng.normal(
        scale=0.03,
        size=(rows, candidates),
    ).astype(np.float32)
    action = default + action_residual - short_residual
    raw = rng.normal(size=(rows, candidates, raw_width)).astype(np.float32)
    permutation = np.array([3, 0, 6, 2, 5, 1, 4])

    before, names = action_utility_features(
        default,
        short_residual,
        action_residual,
        np.full(rows, 7.0, dtype=np.float32),
        np.full(rows, 42.0, dtype=np.float32),
        action,
        raw,
        ("support", "recent", "multi_interest"),
        action_index=1,
    )
    after, after_names = action_utility_features(
        default[:, permutation],
        short_residual[:, permutation],
        action_residual[:, permutation],
        np.full(rows, 7.0, dtype=np.float32),
        np.full(rows, 42.0, dtype=np.float32),
        action[:, permutation],
        raw[:, permutation],
        ("support", "recent", "multi_interest"),
        action_index=1,
    )

    assert names == after_names
    assert all("positive" not in name for name in names)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)


def test_utility_route_abstains_masks_unavailable_and_honors_hard_quota():
    default = np.zeros((5, 3), dtype=np.float32)
    medium = np.full((5, 3), 1.0, dtype=np.float32)
    long = np.full((5, 3), 2.0, dtype=np.float32)
    utility = np.array(
        [
            [0.90, 0.10],
            [0.70, 0.80],
            [0.20, 0.95],
            [-0.10, 0.60],
            [0.50, 0.40],
        ],
        dtype=np.float32,
    )
    change_probability = np.array(
        [
            [0.90, 0.90],
            [0.90, 0.90],
            [0.90, 0.90],
            [0.90, 0.20],
            [0.30, 0.90],
        ],
        dtype=np.float32,
    )
    available = np.array(
        [
            [True, True],
            [True, True],
            [True, False],
            [True, True],
            [True, True],
        ]
    )

    routed = route_by_expected_utility(
        default,
        (medium, long),
        utility,
        change_probability,
        available=available,
        minimum_utility=0.0,
        minimum_change_probability=0.5,
        maximum_route_fraction=0.4,
    )

    assert routed.quota == 2
    assert routed.route_index.tolist() == [1, 2, 0, 0, 0]
    assert np.array_equal(routed.scores[0], medium[0])
    assert np.array_equal(routed.scores[1], long[1])
    assert np.array_equal(routed.scores[2:], default[2:])


def test_hurdle_loss_backpropagates_through_all_utility_heads():
    jt.flags.use_cuda = 0
    model = OOFUtilityRouter(
        OOFUtilityRouterConfig(input_dim=3, hidden_dim=8, dropout=0.0)
    )
    features = jt.array(
        np.array(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, -0.5],
                [0.5, 0.5, 0.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    rewards = jt.array(
        np.array([0.2, -0.1, 0.0, 0.3], dtype=np.float32)
    )
    raw = model(features)

    total, parts = utility_hurdle_loss(
        raw,
        rewards,
        change_positive_weight=2.0,
        loss_direction_weight=2.0,
        magnitude_weight=1.0,
    )
    gradients = jt.grad(total, model.parameters())
    jt.sync_all()

    assert float(total.item()) > 0.0
    assert set(parts) == {"change", "direction", "magnitude"}
    assert all(float(value.item()) >= 0.0 for value in parts.values())
    assert all(
        np.any(np.abs(gradient.numpy()) > 0.0)
        for gradient in gradients
    )


def test_utility_router_checkpoint_replays_predictions(tmp_path):
    jt.flags.use_cuda = 0
    rng = np.random.default_rng(61)
    features = rng.normal(size=(48, 5)).astype(np.float32)
    rewards = np.zeros(48, dtype=np.float32)
    rewards[features[:, 0] > 0.8] = 0.2
    rewards[features[:, 1] > 1.0] = -0.1
    model_config = OOFUtilityRouterConfig(
        input_dim=5,
        hidden_dim=8,
        dropout=0.0,
    )
    training_config = OOFUtilityRouterTrainingConfig(
        epochs=2,
        batch_size=16,
        learning_rate=0.005,
        weight_decay=0.0,
        reward_scale=10.0,
        change_positive_weight=2.0,
        loss_direction_weight=2.0,
        magnitude_weight=0.5,
        hard_negative_weight=2.0,
        regret_weight=2.0,
        seed=61,
    )

    model, result = fit_oof_utility_router(
        features,
        rewards,
        model_config=model_config,
        training_config=training_config,
        feature_names=("f0", "f1", "f2", "f3", "f4"),
        verbose=False,
    )
    before = predict_oof_utility(
        model,
        features,
        result=result,
        batch_size=17,
    )
    checkpoint = tmp_path / "utility-router.npz"
    save_oof_utility_router_checkpoint(checkpoint, model, result)
    loaded, loaded_result = load_oof_utility_router_checkpoint(checkpoint)
    after = predict_oof_utility(
        loaded,
        features,
        result=loaded_result,
        batch_size=13,
    )

    assert result.trainable_frameworks == ("jittor",)
    assert result.non_jittor_trainable_models == ()
    assert result.feature_names == ("f0", "f1", "f2", "f3", "f4")
    np.testing.assert_allclose(
        before.change_probability,
        after.change_probability,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        before.expected_utility,
        after.expected_utility,
        rtol=0.0,
        atol=1e-7,
    )
