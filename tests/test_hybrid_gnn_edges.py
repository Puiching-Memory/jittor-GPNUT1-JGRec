import numpy as np
import pytest

from jgrec.rankers.hybrid.config import GraphTowerConfig
from jgrec.rankers.hybrid.gnn import (
    _dense_normalized_bipartite_adj,
    _graph_window_edges,
    _sample_edges_by_weight,
    _weighted_mapped_edges,
)


def test_none_edge_weighting_preserves_raw_tail_edges():
    mapped_edges = [
        (0, 1, 10),
        (0, 1, 20),
        (1, 2, 30),
        (2, 3, 40),
    ]

    edge_index = _graph_window_edges(
        mapped_edges,
        GraphTowerConfig(edge_weighting="none", max_graph_edges=3),
        np.random.default_rng(0),
    )

    np.testing.assert_array_equal(edge_index, np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32))


def test_repeat_edge_weighting_compresses_duplicates_and_weights_by_count():
    mapped_edges = [
        (0, 1, 10),
        (0, 1, 20),
        (1, 2, 30),
    ]

    edge_index, weights = _weighted_mapped_edges(mapped_edges, "repeat", time_decay_ratio=0.05)

    np.testing.assert_array_equal(edge_index, np.asarray([[0, 1], [1, 2]], dtype=np.int32))
    np.testing.assert_allclose(weights, np.asarray([np.log1p(2), np.log1p(1)], dtype=np.float64))


def test_time_decay_edge_weighting_prefers_recent_repeated_edges():
    mapped_edges = [
        (0, 1, 0),
        (0, 1, 10),
        (1, 2, 100),
        (1, 2, 110),
    ]

    edge_index, weights = _weighted_mapped_edges(mapped_edges, "time_decay", time_decay_ratio=0.1)

    np.testing.assert_array_equal(edge_index, np.asarray([[0, 1], [1, 2]], dtype=np.int32))
    assert weights[1] > weights[0]


def test_weighted_edge_sampling_is_reproducible_and_keeps_chronological_order():
    edge_index = np.asarray([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=np.int32)
    weights = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    first = _sample_edges_by_weight(edge_index, weights, 2, np.random.default_rng(7))
    second = _sample_edges_by_weight(edge_index, weights, 2, np.random.default_rng(7))

    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first[0]) > 0)


def test_edge_weighting_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported graph edge weighting"):
        _graph_window_edges(
            [(0, 1, 10)],
            GraphTowerConfig(edge_weighting="mystery"),
            np.random.default_rng(0),
        )


def test_dense_cpu_adjacency_builds_symmetric_normalized_bipartite_graph():
    edge_index = np.asarray([[0, 0, 1], [0, 1, 1]], dtype=np.int32)

    adj = _dense_normalized_bipartite_adj(edge_index, num_users=2, num_items=2)

    assert adj.shape == (4, 4)
    np.testing.assert_allclose(adj, adj.T)
    assert adj[0, 2] > 0.0
    assert adj[0, 3] > 0.0
    assert adj[1, 3] > 0.0
    assert adj[0, 1] == 0.0
