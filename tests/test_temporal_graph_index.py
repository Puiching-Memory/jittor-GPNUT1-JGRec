import numpy as np

from jgrec.rankers.temporal_graph.index import SafeTemporalNeighborSampler


class DummyRecentSampler:
    sample_neighbor_strategy = "recent"

    def __init__(self) -> None:
        self.nodes_neighbor_ids = [
            np.asarray([], dtype=np.int64),
            np.asarray([10, 11, 12, 13], dtype=np.int64),
            np.asarray([20, 21, 22], dtype=np.int64),
        ]
        self.nodes_edge_ids = [
            np.asarray([], dtype=np.int64),
            np.asarray([100, 101, 102, 103], dtype=np.int64),
            np.asarray([200, 201, 202], dtype=np.int64),
        ]
        self.nodes_neighbor_times = [
            np.asarray([], dtype=np.int32),
            np.asarray([100, 200, 300, 400], dtype=np.int32),
            np.asarray([129_247_799, 129_247_800, 129_247_801], dtype=np.int32),
        ]


def test_safe_temporal_neighbor_sampler_batches_recent_neighbors_left() -> None:
    sampler = SafeTemporalNeighborSampler(DummyRecentSampler())

    neighbor_ids, edge_ids, neighbor_times = sampler.get_historical_neighbors_left(
        node_ids=np.asarray([1, 1, -1, 99], dtype=np.int64),
        node_interact_times=np.asarray([300, 401, 999, 999], dtype=np.int32),
        num_neighbors=3,
    )

    np.testing.assert_array_equal(
        neighbor_ids,
        np.asarray(
            [
                [10, 11, 0],
                [11, 12, 13],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        edge_ids,
        np.asarray(
            [
                [100, 101, 0],
                [101, 102, 103],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        neighbor_times,
        np.asarray(
            [
                [100, 200, 0],
                [200, 300, 400],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )


def test_safe_temporal_neighbor_sampler_keeps_exact_timestamp_boundaries() -> None:
    sampler = SafeTemporalNeighborSampler(DummyRecentSampler())

    neighbor_ids, _, neighbor_times = sampler.get_historical_neighbors_left(
        node_ids=np.asarray([2, 2], dtype=np.int64),
        node_interact_times=np.asarray([129_247_800, 129_247_801], dtype=np.int32),
        num_neighbors=3,
    )

    np.testing.assert_array_equal(
        neighbor_ids,
        np.asarray(
            [
                [20, 0, 0],
                [20, 21, 0],
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        neighbor_times,
        np.asarray(
            [
                [129_247_799, 0, 0],
                [129_247_799, 129_247_800, 0],
            ],
            dtype=np.float32,
        ),
    )


def test_safe_temporal_neighbor_sampler_accepts_missing_interact_times() -> None:
    sampler = SafeTemporalNeighborSampler(DummyRecentSampler())

    neighbor_ids, _, _ = sampler.get_historical_neighbors_left(
        node_ids=np.asarray([1, 2], dtype=np.int64),
        node_interact_times=None,
        num_neighbors=2,
    )

    np.testing.assert_array_equal(
        neighbor_ids,
        np.asarray(
            [
                [12, 13],
                [21, 22],
            ],
            dtype=np.int64,
        ),
    )
