import numpy as np

from jgrec.core.types import TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.multi_interest_proxy import (
    ACTIVITY_ADAPTIVE_FEATURE_NAMES,
    MULTI_INTEREST_FEATURE_NAMES,
    MULTI_INTEREST_V2_FEATURE_NAMES,
    activity_adaptive_cluster_interests,
    adaptive_interest_affinity_features,
    append_multi_interest_features,
    cluster_interest_centers,
    exponential_interest_center,
    hierarchical_interest_centers,
    interest_affinity_features,
    multi_interest_features_for_query_array,
    temporal_interest_centers,
)


def test_temporal_interest_centers_separate_recent_and_older_history():
    history = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    centers = temporal_interest_centers(history)

    np.testing.assert_allclose(centers[0], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(centers[1], [1.0, 0.0], atol=1e-6)


def test_cluster_interest_centers_are_deterministic_and_keep_distinct_modes():
    history = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    first = cluster_interest_centers(history, k=2)
    second = cluster_interest_centers(history, k=2)

    np.testing.assert_allclose(first, second)
    assert np.max(first[:, 0]) > 0.9
    assert np.max(first[:, 1]) > 0.9


def test_interest_affinity_features_return_max_top2_and_positive_coverage():
    candidates = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    centers = np.asarray(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    features = interest_affinity_features(candidates, centers)

    assert features.shape == (3, 3)
    np.testing.assert_allclose(features[0], [1.0, 0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(features[1], [1.0, 0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(features[2], [0.0, -1.0, 0.0], atol=1e-6)


def test_query_proxy_appends_frozen_family_order_and_zeros_cold_ids():
    id_map = NodeIdMap(
        src_to_id={10: 0},
        dst_to_id={20: 0, 21: 1},
        src_values=(10,),
        dst_values=(20, 21),
    )
    state = {
        "item_embeddings": np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        "temporal2": np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "cluster2": np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "cluster4": np.asarray(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
            dtype=np.float32,
        ),
    }
    queries = TestQueryArray(
        src=np.asarray([10, 999], dtype=np.int32),
        time=np.asarray([1, 1], dtype=np.int32),
        candidates=np.asarray([[20, 21, 999], [20, 21, 999]], dtype=np.int32),
    )

    proxy = multi_interest_features_for_query_array(queries, id_map, state)
    base = np.ones((2, 3, 2), dtype=np.float32)
    augmented = append_multi_interest_features(base, queries, id_map, state)

    assert proxy.shape == (2, 3, 9)
    np.testing.assert_allclose(proxy[0, 0, :3], [1.0, 0.0, 0.5])
    np.testing.assert_allclose(proxy[0, 0, 3:6], [1.0, 0.0, 0.5])
    np.testing.assert_allclose(proxy[0, 2], 0.0)
    np.testing.assert_allclose(proxy[1], 0.0)
    np.testing.assert_allclose(augmented[..., :2], base)
    np.testing.assert_allclose(augmented[..., 2:], proxy)
    assert append_multi_interest_features(base, queries, id_map, None) is base


def test_exponential_center_is_time_shift_invariant_and_favors_recent_events():
    history = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    times = np.asarray([0, 1, 10], dtype=np.int64)

    original = exponential_interest_center(
        history,
        times,
        reference_time=10,
        half_life_events=1.0,
    )
    shifted = exponential_interest_center(
        history,
        times + 10_000,
        reference_time=10_010,
        half_life_events=1.0,
    )

    np.testing.assert_allclose(original, shifted, atol=1e-6)
    assert original[1] > original[0]
    np.testing.assert_allclose(np.linalg.norm(original), 1.0, atol=1e-6)


def test_hierarchical_centers_expose_recent16_recent64_and_full():
    history = np.concatenate(
        (
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (84, 1)),
            np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (16, 1)),
        ),
        axis=0,
    )

    centers = hierarchical_interest_centers(history)

    assert centers.shape == (3, 2)
    np.testing.assert_allclose(centers[0], [0.0, 1.0], atol=1e-6)
    assert centers[1, 0] > centers[1, 1]
    assert centers[2, 0] > centers[1, 0]


def test_high_activity_shortens_half_life_and_suppresses_old_cluster():
    low_history = np.concatenate(
        (
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (32, 1)),
            np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (32, 1)),
        ),
        axis=0,
    )
    high_history = np.concatenate(
        (
            np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (128, 1)),
            np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (128, 1)),
        ),
        axis=0,
    )

    low = activity_adaptive_cluster_interests(
        low_history,
        np.arange(low_history.shape[0], dtype=np.int64),
        k=2,
    )
    high = activity_adaptive_cluster_interests(
        high_history,
        np.arange(high_history.shape[0], dtype=np.int64),
        k=2,
    )
    low_old = int(np.argmax(low.centers[:, 0]))
    high_old = int(np.argmax(high.centers[:, 0]))

    assert high.half_life_events < low.half_life_events
    assert high.last_hit[high_old] < low.last_hit[low_old]
    assert high.weights[high_old] < low.weights[low_old]
    assert np.all((high.support >= 0.0) & (high.support <= 1.0))
    assert np.all((high.age >= 0.0) & (high.age <= 1.0))
    assert np.all((high.last_hit >= 0.0) & (high.last_hit <= 1.0))


def test_adaptive_affinity_is_invariant_to_cluster_permutation():
    candidates = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    centers = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    support = np.asarray([0.5, 0.3, 0.2], dtype=np.float32)
    age = np.asarray([0.1, 0.2, 0.9], dtype=np.float32)
    last_hit = np.asarray([0.9, 0.8, 0.1], dtype=np.float32)
    weights = np.sqrt(support) * last_hit
    permutation = np.asarray([2, 0, 1], dtype=np.int64)

    original = adaptive_interest_affinity_features(
        candidates,
        centers,
        support,
        age,
        last_hit,
        weights,
    )
    permuted = adaptive_interest_affinity_features(
        candidates,
        centers[permutation],
        support[permutation],
        age[permutation],
        last_hit[permutation],
        weights[permutation],
    )

    assert original.shape == (3, 6)
    np.testing.assert_allclose(original, permuted, atol=1e-6)
    np.testing.assert_allclose(original[0, 3:], [0.5, 0.1, 0.9], atol=1e-6)


def test_query_proxy_v2_appends_ten_channels_and_preserves_cold_contract():
    id_map = NodeIdMap(
        src_to_id={10: 0},
        dst_to_id={20: 0, 21: 1},
        src_values=(10,),
        dst_values=(20, 21),
    )
    state = {
        "item_embeddings": np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        "temporal2": np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "cluster2": np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32),
        "cluster4": np.asarray(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]],
            dtype=np.float32,
        ),
        "decay1": np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        "hierarchical3": np.asarray(
            [[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]],
            dtype=np.float32,
        ),
        "adaptive_cluster4": np.asarray(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]],
            dtype=np.float32,
        ),
        "adaptive_cluster_support": np.asarray(
            [[0.5, 0.3, 0.1, 0.1]],
            dtype=np.float32,
        ),
        "adaptive_cluster_age": np.asarray(
            [[0.1, 0.2, 0.8, 0.9]],
            dtype=np.float32,
        ),
        "adaptive_cluster_last_hit": np.asarray(
            [[0.9, 0.8, 0.2, 0.1]],
            dtype=np.float32,
        ),
        "adaptive_cluster_weight": np.asarray(
            [[0.6, 0.4, 0.1, 0.05]],
            dtype=np.float32,
        ),
    }
    queries = TestQueryArray(
        src=np.asarray([10, 999], dtype=np.int32),
        time=np.asarray([1, 1], dtype=np.int32),
        candidates=np.asarray([[20, 21, 999], [20, 21, 999]], dtype=np.int32),
    )

    proxy = multi_interest_features_for_query_array(queries, id_map, state)
    permuted_queries = TestQueryArray(
        src=np.asarray([10], dtype=np.int32),
        time=np.asarray([1], dtype=np.int32),
        candidates=np.asarray([[21, 20, 999]], dtype=np.int32),
    )
    permuted = multi_interest_features_for_query_array(
        permuted_queries,
        id_map,
        state,
    )

    assert MULTI_INTEREST_V2_FEATURE_NAMES == (
        MULTI_INTEREST_FEATURE_NAMES + ACTIVITY_ADAPTIVE_FEATURE_NAMES
    )
    assert proxy.shape == (2, 3, 19)
    np.testing.assert_allclose(proxy[0, 2], 0.0)
    np.testing.assert_allclose(proxy[1], 0.0)
    np.testing.assert_allclose(permuted[0, 0], proxy[0, 1], atol=1e-6)
    np.testing.assert_allclose(permuted[0, 1], proxy[0, 0], atol=1e-6)
