from types import SimpleNamespace

import numpy as np

from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid import ranker as ranker_module
from jgrec.rankers.hybrid.segment_fusion import (
    QUERY_SEGMENT_FEATURE_NAMES,
    best_query_weights,
    blend_expert_probabilities,
    fit_segment_gate,
    fit_segment_policy_gate,
    predict_segment_weights,
    query_segment_features,
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


def test_query_segment_features_cover_requested_families_and_ignore_candidate_order():
    features = np.asarray(
        [
            [
                [2.0, 0.5, 0.2, 5.0, 0.2, 1.0, 7.0, 0.1, 1.0, 0.8, 0.9, 0.4],
                [0.0, 0.0, 1.0, 1.0, 0.9, 0.0, 7.0, 0.1, 0.0, 0.2, 0.0, 0.0],
                [1.0, 0.2, 0.5, 3.0, 0.5, 0.0, 7.0, 0.1, 1.0, 0.5, 0.3, 0.8],
            ],
            [
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.8, 0.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 1.0, 9.0, 0.1, 0.0, 2.0, 0.8, 1.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 0.7, 0.0, 2.0, 0.8, 1.0, 0.4, 0.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )

    actual = query_segment_features(features, RAW_FEATURE_NAMES)
    permuted = query_segment_features(features[:, [2, 0, 1]], RAW_FEATURE_NAMES)

    assert actual.shape == (2, len(QUERY_SEGMENT_FEATURE_NAMES))
    np.testing.assert_allclose(permuted, actual)
    assert any(name.startswith("repeat_") for name in QUERY_SEGMENT_FEATURE_NAMES)
    assert any(name.startswith("source_") for name in QUERY_SEGMENT_FEATURE_NAMES)
    assert any(name.startswith("target_") for name in QUERY_SEGMENT_FEATURE_NAMES)
    assert any(name.startswith("prior_") for name in QUERY_SEGMENT_FEATURE_NAMES)
    assert any(name.startswith("memory_") for name in QUERY_SEGMENT_FEATURE_NAMES)


def test_best_query_weights_choose_expert_and_prefer_global_weight_on_rank_ties():
    mlp = np.asarray(
        [
            [0.6, 0.4, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.3, 0.1],
        ]
    )
    lgbm = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.6, 0.4, 0.0],
            [0.6, 0.3, 0.1],
        ]
    )

    actual = best_query_weights(
        mlp,
        lgbm,
        candidate_weights=(0.0, 0.25, 0.5, 0.75, 1.0),
        global_weight=0.25,
    )

    np.testing.assert_array_equal(actual, np.asarray([1.0, 0.0, 0.25]))


def test_segment_gate_learns_different_query_weights_from_observable_descriptors():
    features = np.zeros((4, 2, len(RAW_FEATURE_NAMES)), dtype=np.float32)
    source_activity_index = RAW_FEATURE_NAMES.index("src_activity")
    train_seen_index = RAW_FEATURE_NAMES.index("candidate_train_seen")
    features[:, :, train_seen_index] = 1.0
    features[2:, :, source_activity_index] = 10.0
    descriptors = query_segment_features(features, RAW_FEATURE_NAMES)
    result = fit_segment_gate(
        descriptors,
        np.asarray([0.07, 0.07, 0.9, 0.9]),
        candidate_weights=(0.07, 0.5, 0.9),
        global_weight=0.5,
        max_depth=1,
        min_samples_leaf=1,
        seed=60,
        name="depth1_leaf1",
    )

    actual = predict_segment_weights(result, features, RAW_FEATURE_NAMES)

    np.testing.assert_array_equal(actual, np.asarray([0.07, 0.07, 0.9, 0.9]))


def test_policy_gate_selects_each_leaf_weight_by_reciprocal_rank_reward():
    features = np.zeros((6, 2, len(RAW_FEATURE_NAMES)), dtype=np.float32)
    source_activity_index = RAW_FEATURE_NAMES.index("src_activity")
    features[3:, :, source_activity_index] = 10.0
    descriptors = query_segment_features(features, RAW_FEATURE_NAMES)
    rewards = np.asarray(
        [
            [1.0, 0.5, 0.1],
            [1.0, 0.5, 0.1],
            [1.0, 0.5, 0.1],
            [0.1, 0.5, 1.0],
            [0.1, 0.5, 1.0],
            [0.1, 0.5, 1.0],
        ]
    )

    result = fit_segment_policy_gate(
        descriptors,
        rewards,
        candidate_weights=(0.0, 0.5, 1.0),
        global_weight=0.5,
        max_depth=1,
        min_samples_leaf=2,
        seed=60,
        name="policy_depth1_leaf2",
    )

    actual = predict_segment_weights(result, features, RAW_FEATURE_NAMES)

    np.testing.assert_array_equal(actual, np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))


def test_expert_blend_supports_legacy_scalar_and_per_query_gate_weights():
    mlp = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    lgbm = np.asarray([[0.4, 0.6], [0.9, 0.1]])

    legacy = blend_expert_probabilities(mlp, lgbm, 0.25)
    gated = blend_expert_probabilities(mlp, lgbm, np.asarray([1.0, 0.0]))

    np.testing.assert_allclose(legacy, 0.25 * mlp + 0.75 * lgbm)
    np.testing.assert_allclose(gated, np.asarray([mlp[0], lgbm[1]]))


def test_hybrid_snapshot_carries_optional_segment_gate_state(monkeypatch):
    ranker = ranker_module.TemporalHybridRanker()
    ranker.config = SimpleNamespace()
    ranker.id_map = NodeIdMap(
        src_to_id={1: 0},
        dst_to_id={2: 0},
        src_values=(1,),
        dst_values=(2,),
    )
    ranker.encoder = SimpleNamespace(snapshot=lambda: {"encoder": "state"})
    ranker.fusion = object()
    ranker.fusion_result = SimpleNamespace()
    ranker.segment_gate_result = "gate-state"
    monkeypatch.setattr(ranker_module, "get_model_state", lambda _model: {})

    snapshot = ranker.snapshot()

    assert snapshot["segment_gate_result"] == "gate-state"
