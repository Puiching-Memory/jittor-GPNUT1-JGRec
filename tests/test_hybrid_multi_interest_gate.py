import numpy as np

from jgrec.rankers.hybrid.multi_interest_gate import (
    ConfidenceGateConfig,
    blocked_temporal_folds,
    blocked_temporal_oof_gate,
    confidence_gate_descriptors,
    expert_score_descriptors,
    fit_confidence_gate,
    passes_stability_gate,
    predict_confidence_gate,
    route_query_experts,
    select_stable_high_confidence_trial,
)

RAW_FEATURE_NAMES = (
    "pair_strength",
    "repeat_rate",
    "pair_recency",
    "dst_popularity",
    "dst_recency",
    "recent_hit",
    "src_activity",
    "src_recency",
    "candidate_train_seen",
    "candidate_test_freq",
    "pair_decay_short",
    "pair_decay_long",
)


def test_query_gate_routes_whole_queries_and_preserves_exact_champion_fallback():
    champion = np.asarray(
        [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]],
        dtype=np.float64,
    )
    candidate = np.asarray(
        [[0.5, 0.4, 0.1], [0.8, 0.1, 0.1], [0.3, 0.6, 0.1]],
        dtype=np.float64,
    )

    actual = route_query_experts(
        champion,
        candidate,
        np.asarray([False, True, False]),
    )

    np.testing.assert_array_equal(actual[0], champion[0])
    np.testing.assert_array_equal(actual[1], candidate[1])
    np.testing.assert_array_equal(actual[2], champion[2])


def test_confidence_descriptors_are_label_free_and_candidate_order_invariant():
    rng = np.random.default_rng(60)
    base_features = rng.normal(size=(4, 5, len(RAW_FEATURE_NAMES))).astype(
        np.float32
    )
    base_features[..., RAW_FEATURE_NAMES.index("candidate_train_seen")] = 1.0
    proxy_features = rng.normal(size=(4, 5, 9)).astype(np.float32)
    champion = rng.random(size=(4, 5))
    candidate = rng.random(size=(4, 5))
    champion /= champion.sum(axis=1, keepdims=True)
    candidate /= candidate.sum(axis=1, keepdims=True)
    permutation = np.asarray([3, 1, 4, 0, 2])

    actual = confidence_gate_descriptors(
        base_features,
        RAW_FEATURE_NAMES,
        proxy_features,
        champion,
        candidate,
    )
    permuted = confidence_gate_descriptors(
        base_features[:, permutation],
        RAW_FEATURE_NAMES,
        proxy_features[:, permutation],
        champion[:, permutation],
        candidate[:, permutation],
    )

    np.testing.assert_allclose(actual, permuted, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        actual[:, 16:23],
        expert_score_descriptors(champion, candidate),
        rtol=0.0,
        atol=1e-6,
    )


def test_blocked_temporal_folds_hold_out_each_query_exactly_once():
    folds = blocked_temporal_folds(10, fold_count=3)
    held_out = np.concatenate([test_indices for _, test_indices in folds])

    np.testing.assert_array_equal(np.sort(held_out), np.arange(10))
    for train_indices, test_indices in folds:
        assert not np.intersect1d(train_indices, test_indices).size


def test_blocked_temporal_oof_gate_learns_query_fallback_without_self_labels():
    descriptors = np.tile(
        np.asarray([[0.0], [1.0], [1.0]], dtype=np.float32),
        (3, 1),
    )
    champion = np.tile(
        np.asarray(
            [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.1, 0.9, 0.0]]
        ),
        (3, 1),
    )
    candidate = np.tile(
        np.asarray(
            [[0.1, 0.9, 0.0], [0.9, 0.1, 0.0], [0.9, 0.1, 0.0]]
        ),
        (3, 1),
    )

    result = blocked_temporal_oof_gate(
        descriptors,
        champion,
        candidate,
        ConfidenceGateConfig(
            max_depth=1,
            min_samples_leaf=1,
            minimum_predicted_lift=0.1,
        ),
        fold_count=3,
        seed=60,
    )

    np.testing.assert_array_equal(
        result.use_candidate,
        np.tile(np.asarray([False, True, True]), 3),
    )
    assert result.full_delta > 0.0
    assert all(delta > 0.0 for delta in result.fold_deltas)


def test_stability_gate_requires_margin_and_no_fold_or_slice_regression():
    assert passes_stability_gate(
        full_delta=0.0021,
        fold_deltas=(0.001, 0.002, 0.003),
        slice_deltas=(0.001, 0.002, 0.003),
        minimum_full_delta=0.002,
    )


def test_trial_selection_rejects_low_threshold_and_excessive_coverage():
    trials = [
        {
            "passed": True,
            "config": {"minimum_predicted_lift": 0.0},
            "coverage": 0.10,
            "fold_deltas": [0.01, 0.01, 0.01],
            "full_delta": 0.01,
        },
        {
            "passed": True,
            "config": {"minimum_predicted_lift": 0.01},
            "coverage": 0.60,
            "fold_deltas": [0.009, 0.009, 0.009],
            "full_delta": 0.009,
        },
        {
            "passed": True,
            "config": {"minimum_predicted_lift": 0.005},
            "coverage": 0.20,
            "fold_deltas": [0.002, 0.003, 0.004],
            "full_delta": 0.003,
        },
        {
            "passed": True,
            "config": {"minimum_predicted_lift": 0.01},
            "coverage": 0.30,
            "fold_deltas": [0.0025, 0.0026, 0.0027],
            "full_delta": 0.0026,
        },
    ]

    selected = select_stable_high_confidence_trial(
        trials,
        minimum_predicted_lift=0.005,
        maximum_coverage=0.35,
    )

    assert selected == 3


def test_final_confidence_gate_serializes_and_applies_frozen_threshold():
    descriptors = np.asarray(
        [[0.0], [0.0], [1.0], [1.0]],
        dtype=np.float32,
    )
    rewards = np.asarray([-0.5, -0.4, 0.3, 0.4])
    config = ConfidenceGateConfig(
        max_depth=1,
        min_samples_leaf=1,
        minimum_predicted_lift=0.1,
    )

    model = fit_confidence_gate(
        descriptors,
        rewards,
        config,
        descriptor_names=("signal",),
        seed=60,
    )
    use_candidate, predicted_lift = predict_confidence_gate(
        model,
        descriptors,
        descriptor_names=("signal",),
    )

    np.testing.assert_array_equal(
        use_candidate,
        np.asarray([False, False, True, True]),
    )
    assert np.all(predicted_lift[:2] < 0.1)
    assert np.all(predicted_lift[2:] >= 0.1)
    assert not passes_stability_gate(
        full_delta=0.003,
        fold_deltas=(0.001, -0.0001, 0.004),
        slice_deltas=(0.001, 0.002, 0.003),
        minimum_full_delta=0.002,
    )
