import numpy as np

from jgrec.core.types import InteractionTable, TestQueryArray
from jgrec.rankers.temporal_graph.index import TemporalNodeMap
from jgrec.rankers.temporal_graph.trainer import (
    TemporalTrainingBatch,
    _batch_to_jittor,
    _sample_test_like_candidate_ids,
)
from jgrec.rankers.temporal_graph.trainer import (
    TestCandidateIndex as CandidateIndex,
)


def _node_map() -> TemporalNodeMap:
    interactions = InteractionTable.from_array(
        np.asarray(
            [
                [1, 100, 10],
                [1, 101, 11],
                [1, 102, 12],
                [2, 103, 13],
                [2, 104, 14],
                [3, 105, 15],
                [3, 106, 16],
                [3, 107, 17],
            ],
            dtype=np.int32,
        )
    )
    return TemporalNodeMap.from_interactions_and_test(interactions, test_path=None)


def test_test_candidate_index_groups_candidate_rows_by_source() -> None:
    node_map = _node_map()
    queries = TestQueryArray(
        src=np.asarray([2, 1, 1], dtype=np.int32),
        time=np.asarray([20, 21, 22], dtype=np.int32),
        candidates=np.asarray(
            [
                [103, 104, 999],
                [100, 101, 999],
                [102, 103, 104],
            ],
            dtype=np.int32,
        ),
    )

    index = CandidateIndex.from_queries(queries, node_map)

    assert set(index.by_src) == {1, 2}
    assert len(index.by_src[1]) == 2
    assert len(index.by_src[2]) == 1
    assert index.global_candidates.size == 7
    assert all(row.dtype == np.int32 for rows in index.by_src.values() for row in rows)


def test_sample_test_like_candidate_ids_uses_source_rows_and_fallback_pool() -> None:
    node_map = _node_map()
    queries = TestQueryArray(
        src=np.asarray([1], dtype=np.int32),
        time=np.asarray([20], dtype=np.int32),
        candidates=np.asarray([[101, 102, 103]], dtype=np.int32),
    )
    index = CandidateIndex.from_queries(queries, node_map)
    events = InteractionTable.from_array(
        np.asarray(
            [
                [1, 100, 30],
                [99, 101, 31],
            ],
            dtype=np.int32,
        )
    )
    positives = node_map.dst_ids(events.dst)
    dst_pool = node_map.dst_ids(np.asarray([100, 101, 102, 103, 104, 105], dtype=np.int32))

    candidates = _sample_test_like_candidate_ids(
        events=events,
        positives=positives,
        candidate_index=index,
        dst_pool=dst_pool,
        num_negatives=3,
        rng=np.random.default_rng(42),
    )

    assert candidates.shape == (2, 4)
    np.testing.assert_array_equal(candidates[:, 0], positives)
    for row, positive in zip(candidates, positives):
        negatives = row[1:]
        assert 0 not in negatives
        assert int(positive) not in negatives
        assert len(set(negatives.tolist())) == len(negatives)


def test_sample_test_like_candidate_ids_keeps_full_source_row_order() -> None:
    index = CandidateIndex(
        by_src={7: (np.asarray([0, 14, 11, 12, 13, 15], dtype=np.int32),)},
        global_candidates=np.empty(0, dtype=np.int32),
    )
    events = InteractionTable.from_array(np.asarray([[7, 100, 30]], dtype=np.int32))

    candidates = _sample_test_like_candidate_ids(
        events=events,
        positives=np.asarray([14], dtype=np.int32),
        candidate_index=index,
        dst_pool=np.empty(0, dtype=np.int32),
        num_negatives=3,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(candidates, np.asarray([[14, 11, 12, 13]], dtype=np.int32))


def test_batch_to_jittor_preserves_int32_inputs() -> None:
    batch = TemporalTrainingBatch(
        src_ids=np.asarray([1, 2], dtype=np.int32),
        times=np.asarray([10, 11], dtype=np.int32),
        candidates=np.asarray([[3, 4], [5, 6]], dtype=np.int32),
        src_neighbor_ids=np.asarray([[0, 3], [4, 0]], dtype=np.int32),
        src_neighbor_times=np.asarray([[0, 9], [10, 0]], dtype=np.int32),
        candidate_neighbor_ids=np.asarray([[[0], [1]], [[2], [0]]], dtype=np.int32),
        candidate_neighbor_times=np.asarray([[[0], [8]], [[9], [0]]], dtype=np.int32),
    )

    values = _batch_to_jittor(batch)

    assert all(str(value.dtype) == "int32" for value in values)


def test_batch_to_jittor_casts_non_int32_inputs() -> None:
    batch = TemporalTrainingBatch(
        src_ids=np.asarray([1, 2], dtype=np.int64),
        times=np.asarray([10, 11], dtype=np.float32),
        candidates=np.asarray([[3, 4], [5, 6]], dtype=np.int64),
        src_neighbor_ids=np.asarray([[False, True], [True, False]], dtype=np.bool_),
        src_neighbor_times=np.asarray([[0, 9], [10, 0]], dtype=np.float64),
        candidate_neighbor_ids=np.asarray([[[0], [1]], [[2], [0]]], dtype=np.int64),
        candidate_neighbor_times=np.asarray([[[0], [8]], [[9], [0]]], dtype=np.float32),
    )

    values = _batch_to_jittor(batch)

    assert all(str(value.dtype) == "int32" for value in values)
