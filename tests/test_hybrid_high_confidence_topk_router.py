import numpy as np

from jgrec.rankers.hybrid.high_confidence_topk_router import (
    ResidualAdvantageRouterConfig,
    ResidualAdvantageRouterTrainingConfig,
    audit_bounded_topk_route,
    bounded_topk_alternative,
    fit_residual_advantage_router,
    hard_high_confidence_route,
    load_residual_advantage_router_checkpoint,
    predict_residual_advantages,
    route_reward_targets,
    router_candidate_support_features,
    router_summary_features,
    save_residual_advantage_router_checkpoint,
    timestamp_router_split,
)


def _scores() -> np.ndarray:
    return np.array(
        [
            [0.8, 0.6, 0.5, 0.2, 0.1, -0.2],
            [0.7, 0.4, 0.3, 0.1, -0.1, -0.3],
            [0.9, 0.5, 0.2, 0.0, -0.2, -0.4],
            [0.6, 0.55, 0.4, 0.2, 0.0, -0.5],
        ],
        dtype=np.float32,
    )


def _residuals() -> np.ndarray:
    short = np.array(
        [
            [0.02, -0.01, 0.01, -0.01, 0.00, -0.01],
            [0.01, 0.02, -0.01, -0.02, 0.00, 0.00],
            [0.03, -0.01, -0.01, 0.00, 0.00, -0.01],
            [0.00, 0.01, -0.01, 0.02, -0.01, -0.01],
        ],
        dtype=np.float32,
    )
    medium = short + np.array(
        [
            [-0.04, 0.06, 0.02, -0.01, -0.01, -0.02],
            [0.03, -0.05, 0.04, 0.01, -0.01, -0.02],
            [-0.02, 0.05, -0.01, -0.01, 0.00, -0.01],
            [0.04, -0.03, 0.02, -0.02, 0.00, -0.01],
        ],
        dtype=np.float32,
    )
    long = short + np.array(
        [
            [0.01, -0.03, 0.04, -0.01, 0.00, -0.01],
            [-0.02, 0.05, -0.04, 0.03, -0.01, -0.01],
            [0.04, -0.03, 0.02, -0.02, 0.00, -0.01],
            [-0.03, 0.06, -0.02, 0.01, -0.01, -0.01],
        ],
        dtype=np.float32,
    )
    return np.stack((short, medium, long))


def test_bounded_alternative_changes_only_default_topk_and_respects_cap():
    default = _scores()
    residuals = _residuals()

    alternative = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[1],
        top_k=3,
        cap=0.02,
    )

    delta = alternative.scores - default
    assert np.array_equal(delta[~alternative.topk_mask], np.zeros_like(delta)[
        ~alternative.topk_mask
    ])
    assert float(np.max(np.abs(delta))) <= 0.020002
    np.testing.assert_allclose(np.sum(delta, axis=1), 0.0, atol=1e-7)
    assert np.all(alternative.topk_mask.sum(axis=1) == 3)


def test_identical_residual_is_exact_short_fallback():
    default = _scores()
    short = _residuals()[0]

    alternative = bounded_topk_alternative(
        default,
        short,
        short,
        top_k=3,
        cap=0.02,
    )

    assert np.array_equal(alternative.scores, default)
    assert np.count_nonzero(alternative.delta) == 0


def test_hard_route_defaults_to_short_except_high_confidence_rows():
    default = _scores()
    residuals = _residuals()
    medium = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[1],
        top_k=3,
        cap=0.02,
    )
    long = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[2],
        top_k=3,
        cap=0.02,
    )
    predicted_advantages = np.array(
        [
            [0.50, 0.10],
            [0.10, 0.40],
            [-0.10, -0.20],
            [0.010, 0.009],
        ],
        dtype=np.float32,
    )

    routed = hard_high_confidence_route(
        default,
        (medium.scores, long.scores),
        predicted_advantages,
        minimum_confidence=0.05,
        maximum_route_fraction=0.50,
    )

    assert routed.route_index.tolist() == [1, 2, 0, 0]
    assert np.array_equal(routed.scores[0], medium.scores[0])
    assert np.array_equal(routed.scores[1], long.scores[1])
    assert np.array_equal(routed.scores[2:], default[2:])
    audit = audit_bounded_topk_route(
        default,
        (medium, long),
        routed,
        cap=0.02,
        maximum_route_fraction=0.50,
    )
    assert audit["passed"]
    assert audit["unrouted_rows_exact"]
    assert audit["topk_outside_exact"]


def test_router_features_are_invariant_to_candidate_permutation():
    default = _scores()
    residuals = _residuals()
    gaps = np.array(
        [
            [10.0, 11.0, 12.0, 13.0],
            [40.0, 41.0, 42.0, 43.0],
            [70.0, 71.0, 72.0, 73.0],
        ],
        dtype=np.float32,
    )
    medium = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[1],
        top_k=3,
        cap=0.02,
    )
    long = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[2],
        top_k=3,
        cap=0.02,
    )
    before, names = router_summary_features(
        default,
        residuals,
        gaps,
        (medium.scores, long.scores),
    )

    permutation = np.array([3, 0, 5, 1, 4, 2])
    permuted_default = default[:, permutation]
    permuted_residuals = residuals[:, :, permutation]
    permuted_medium = bounded_topk_alternative(
        permuted_default,
        permuted_residuals[0],
        permuted_residuals[1],
        top_k=3,
        cap=0.02,
    )
    permuted_long = bounded_topk_alternative(
        permuted_default,
        permuted_residuals[0],
        permuted_residuals[2],
        top_k=3,
        cap=0.02,
    )
    after, after_names = router_summary_features(
        permuted_default,
        permuted_residuals,
        gaps,
        (permuted_medium.scores, permuted_long.scores),
    )

    assert names == after_names
    np.testing.assert_allclose(after, before, atol=1e-6)


def test_candidate_support_features_are_invariant_to_permutation():
    default = _scores()
    residuals = _residuals()
    rng = np.random.default_rng(7)
    raw_features = rng.normal(size=(*default.shape, 3)).astype(np.float32)
    medium = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[1],
        top_k=3,
        cap=0.02,
    )
    long = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[2],
        top_k=3,
        cap=0.02,
    )
    before, names = router_candidate_support_features(
        raw_features,
        ("raw_a", "raw_b", "raw_c"),
        default,
        (medium.scores, long.scores),
    )

    permutation = np.array([3, 0, 5, 1, 4, 2])
    after, after_names = router_candidate_support_features(
        raw_features[:, permutation],
        ("raw_a", "raw_b", "raw_c"),
        default[:, permutation],
        (
            medium.scores[:, permutation],
            long.scores[:, permutation],
        ),
    )

    assert names == after_names
    assert before.shape == (default.shape[0], 15)
    np.testing.assert_allclose(after, before, atol=1e-6)


def test_timestamp_split_never_splits_equal_times():
    times = np.repeat(np.arange(20, dtype=np.int64), 5)

    split = timestamp_router_split(times)

    assert split.train_rows[0] == 0
    assert split.train_rows[1] == split.selection_rows[0]
    assert split.selection_rows[1] == split.gate_rows[0]
    assert split.gate_rows[1] == len(times)
    assert times[split.train_rows[1] - 1] < times[split.selection_rows[0]]
    assert times[split.selection_rows[1] - 1] < times[split.gate_rows[0]]


def test_reward_targets_compare_bounded_routes_to_default():
    default = _scores()
    residuals = _residuals()
    medium = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[1],
        top_k=3,
        cap=0.02,
    )
    long = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[2],
        top_k=3,
        cap=0.02,
    )

    rewards = route_reward_targets(
        default,
        (medium.scores, long.scores),
        np.zeros(default.shape[0], dtype=np.int32),
    )

    assert rewards.shape == (default.shape[0], 2)
    assert np.isfinite(rewards).all()


def test_jittor_reward_router_checkpoint_replays(tmp_path):
    rng = np.random.default_rng(60)
    features = rng.normal(size=(64, 5)).astype(np.float32)
    rewards = np.column_stack(
        (
            0.01 * features[:, 0] - 0.005 * features[:, 1],
            0.008 * features[:, 2] + 0.004 * features[:, 3],
        )
    ).astype(np.float32)
    model_config = ResidualAdvantageRouterConfig(
        input_dim=features.shape[1],
        hidden_dim=8,
        dropout=0.0,
    )
    training_config = ResidualAdvantageRouterTrainingConfig(
        epochs=2,
        batch_size=16,
        learning_rate=0.005,
        weight_decay=0.0,
        reward_scale=10.0,
        nonzero_weight=2.0,
        seed=60,
    )

    model, result = fit_residual_advantage_router(
        features,
        rewards,
        model_config=model_config,
        training_config=training_config,
        feature_names=("a", "b", "c", "d", "e"),
        verbose=False,
    )
    before = predict_residual_advantages(
        model,
        features,
        mean=result.mean,
        std=result.std,
        reward_scale=result.training_config.reward_scale,
        batch_size=16,
    )
    checkpoint = tmp_path / "router.npz"
    save_residual_advantage_router_checkpoint(checkpoint, model, result)
    loaded, loaded_result = load_residual_advantage_router_checkpoint(
        checkpoint
    )
    after = predict_residual_advantages(
        loaded,
        features,
        mean=loaded_result.mean,
        std=loaded_result.std,
        reward_scale=loaded_result.training_config.reward_scale,
        batch_size=16,
    )

    assert result.training_rows == 64
    assert loaded_result.feature_names == ("a", "b", "c", "d", "e")
    assert result.trainable_frameworks == ("jittor",)
    assert result.non_jittor_trainable_models == ()
    np.testing.assert_allclose(after, before, atol=1e-6)
