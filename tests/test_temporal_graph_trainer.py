import math

import numpy as np

from jgrec.core.types import InteractionTable, TestQueryArray
from jgrec.rankers.temporal_graph.index import TemporalNodeMap
from jgrec.rankers.temporal_graph.trainer import (
    CANDIDATE_PRIOR_FEATURE_NAMES,
    CandidatePriorIndex,
    TemporalTrainingBatch,
    _batch_to_jittor,
    _sample_candidate_ids,
    _sample_test_like_candidate_ids,
    build_candidate_prior_features,
    build_training_batch,
)
from jgrec.rankers.temporal_graph.trainer import (
    TestCandidateIndex as CandidateIndex,
)


class DummyNeighborSampler:
    def get_historical_neighbors_left(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = len(node_ids)
        return (
            np.zeros((rows, num_neighbors), dtype=np.int32),
            np.zeros((rows, num_neighbors), dtype=np.int32),
            np.zeros((rows, num_neighbors), dtype=np.int32),
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
    for row, positive in zip(candidates, positives, strict=True):
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


def test_sample_candidate_ids_handles_small_nonreplacement_pool() -> None:
    candidates = _sample_candidate_ids(
        positives=np.asarray([10], dtype=np.int32),
        dst_pool=np.asarray([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int32),
        num_negatives=3,
        rng=np.random.default_rng(42),
    )

    assert candidates.shape == (1, 4)
    assert candidates[0, 0] == 10
    assert 10 not in candidates[0, 1:]
    assert len(set(candidates[0, 1:].tolist())) == 3


def test_candidate_prior_features_encode_seen_unseen_and_test_frequency() -> None:
    node_map = _node_map()
    queries = TestQueryArray(
        src=np.asarray([1, 2], dtype=np.int32),
        time=np.asarray([20, 21], dtype=np.int32),
        candidates=np.asarray(
            [
                [107, 107, 100],
                [106, 107, 101],
            ],
            dtype=np.int32,
        ),
    )
    test_index = CandidateIndex.from_queries(queries, node_map)
    train_dst_ids = node_map.dst_ids(np.asarray([100, 100, 101, 102], dtype=np.int32))
    prior = CandidatePriorIndex.from_test_candidates(test_index, train_dst_ids, include_test_frequency=True)
    candidates = node_map.dst_ids(np.asarray([[100, 107, 106]], dtype=np.int32))
    names = {name: idx for idx, name in enumerate(CANDIDATE_PRIOR_FEATURE_NAMES)}

    features = build_candidate_prior_features(candidates, prior)

    assert features.shape == (1, 3, len(CANDIDATE_PRIOR_FEATURE_NAMES))
    assert features.dtype == np.float32
    assert features[0, 0, names["candidate_train_seen"]] == 1.0
    assert features[0, 1, names["candidate_train_seen"]] == 0.0
    assert features[0, 1, names["candidate_test_freq"]] > features[0, 0, names["candidate_test_freq"]]
    assert (
        features[0, 1, names["candidate_unseen_test_freq"]]
        == features[0, 1, names["candidate_test_freq"]]
    )
    assert features[0, 1, names["candidate_test_freq_row_rank"]] == 1.0
    assert features[0, 0, names["candidate_train_freq"]] > features[0, 1, names["candidate_train_freq"]]
    assert features[0, 0, names["candidate_train_freq_row_rank"]] == 1.0


def test_candidate_prior_features_encode_recent_train_windows() -> None:
    node_map = _node_map()
    queries = TestQueryArray(
        src=np.asarray([1], dtype=np.int32),
        time=np.asarray([20], dtype=np.int32),
        candidates=np.asarray([[100, 101, 102]], dtype=np.int32),
    )
    test_index = CandidateIndex.from_queries(queries, node_map)
    train_dst_ids = node_map.dst_ids(np.asarray([100, 101, 102, 100, 100, 101], dtype=np.int32))
    train_times = np.asarray([0, 20, 30, 49, 50, 60], dtype=np.int32)
    prior = CandidatePriorIndex.from_test_candidates(
        test_index,
        train_dst_ids,
        train_times=train_times,
        recent_feature_group="recency_rank",
    )
    candidates = node_map.dst_ids(np.asarray([[100, 101, 102]], dtype=np.int32))
    names = {name: idx for idx, name in enumerate(CANDIDATE_PRIOR_FEATURE_NAMES)}

    features = build_candidate_prior_features(candidates, prior)

    assert features.shape == (1, 3, len(CANDIDATE_PRIOR_FEATURE_NAMES))
    assert features[0, 0, names["candidate_train_recent_pop_w005"]] == 0.0
    assert features[0, 1, names["candidate_train_recent_pop_w005"]] == 0.0
    assert features[0, 2, names["candidate_train_recent_pop_w005"]] == 0.0
    assert features[0, 1, names["candidate_train_recent_recency_w005"]] == np.float32(math.exp(-1 / 3))
    assert features[0, 0, names["candidate_train_recent_recency_w005"]] == 0.0
    assert features[0, 1, names["candidate_train_recent_rank_w005"]] == 1.0
    assert features[0, 0, names["candidate_train_recent_rank_w005"]] == np.float32(1 / 2.5)
    assert features[0, 2, names["candidate_train_recent_rank_w005"]] == np.float32(1 / 2.5)
    assert features[0, 0, names["candidate_train_recent_pop_w020"]] == 0.0
    assert features[0, 1, names["candidate_train_recent_pop_w020"]] == 0.0
    assert features[0, 0, names["candidate_train_recent_share_w020"]] == 0.0
    assert features[0, 0, names["candidate_train_recent_rank_w020"]] == 1.0


def test_candidate_recent_feature_group_none_preserves_old_six_feature_baseline() -> None:
    node_map = _node_map()
    queries = TestQueryArray(
        src=np.asarray([1], dtype=np.int32),
        time=np.asarray([20], dtype=np.int32),
        candidates=np.asarray([[100, 101, 102]], dtype=np.int32),
    )
    test_index = CandidateIndex.from_queries(queries, node_map)
    train_dst_ids = node_map.dst_ids(np.asarray([100, 101, 102, 100, 100, 101], dtype=np.int32))
    train_times = np.asarray([0, 20, 30, 49, 50, 60], dtype=np.int32)
    prior = CandidatePriorIndex.from_test_candidates(
        test_index,
        train_dst_ids,
        train_times=train_times,
        recent_feature_group="none",
    )
    candidates = node_map.dst_ids(np.asarray([[100, 101, 102]], dtype=np.int32))

    features = build_candidate_prior_features(candidates, prior)

    assert np.any(features[:, :, :6] != 0.0)
    assert np.all(features[:, :, 6:] == 0.0)


def test_training_batch_can_use_test_like_candidates_and_prior_features() -> None:
    node_map = _node_map()
    test_index = CandidateIndex(
        by_src={
            1: (
                node_map.dst_ids(np.asarray([107, 106, 105], dtype=np.int32)),
            )
        },
        global_candidates=np.empty(0, dtype=np.int32),
    )
    train_dst_ids = node_map.dst_ids(np.asarray([100, 101, 102], dtype=np.int32))
    prior = CandidatePriorIndex.from_test_candidates(test_index, train_dst_ids, include_test_frequency=True)
    events = InteractionTable.from_array(np.asarray([[1, 100, 30]], dtype=np.int32))

    batch = build_training_batch(
        events=events,
        node_map=node_map,
        neighbor_sampler=DummyNeighborSampler(),
        dst_pool=train_dst_ids,
        num_negatives=2,
        rng=np.random.default_rng(42),
        history_len=3,
        candidate_history_len=2,
        candidate_index=test_index,
        candidate_prior_index=prior,
    )

    assert batch.candidates.shape == (1, 3)
    np.testing.assert_array_equal(batch.candidates[0, 1:], test_index.by_src[1][0][:2])
    assert batch.candidate_features.shape == (1, 3, len(CANDIDATE_PRIOR_FEATURE_NAMES))
    assert batch.candidate_features[0, 1, 1] > 0.0


def test_batch_to_jittor_preserves_int32_inputs() -> None:
    batch = TemporalTrainingBatch(
        src_ids=np.asarray([1, 2], dtype=np.int32),
        times=np.asarray([10, 11], dtype=np.int32),
        candidates=np.asarray([[3, 4], [5, 6]], dtype=np.int32),
        src_neighbor_ids=np.asarray([[0, 3], [4, 0]], dtype=np.int32),
        src_neighbor_times=np.asarray([[0, 9], [10, 0]], dtype=np.int32),
        candidate_neighbor_ids=np.asarray([[[0], [1]], [[2], [0]]], dtype=np.int32),
        candidate_neighbor_times=np.asarray([[[0], [8]], [[9], [0]]], dtype=np.int32),
        candidate_features=np.asarray([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=np.float32),
    )

    values = _batch_to_jittor(batch)

    assert all(str(value.dtype) == "int32" for value in values[:-1])
    assert str(values[-1].dtype) == "float32"


def test_batch_to_jittor_casts_non_int32_inputs() -> None:
    batch = TemporalTrainingBatch(
        src_ids=np.asarray([1, 2], dtype=np.int64),
        times=np.asarray([10, 11], dtype=np.float32),
        candidates=np.asarray([[3, 4], [5, 6]], dtype=np.int64),
        src_neighbor_ids=np.asarray([[False, True], [True, False]], dtype=np.bool_),
        src_neighbor_times=np.asarray([[0, 9], [10, 0]], dtype=np.float64),
        candidate_neighbor_ids=np.asarray([[[0], [1]], [[2], [0]]], dtype=np.int64),
        candidate_neighbor_times=np.asarray([[[0], [8]], [[9], [0]]], dtype=np.float32),
        candidate_features=np.asarray([[[0.1], [0.2]], [[0.3], [0.4]]], dtype=np.float64),
    )

    values = _batch_to_jittor(batch)

    assert all(str(value.dtype) == "int32" for value in values[:-1])
    assert str(values[-1].dtype) == "float32"
