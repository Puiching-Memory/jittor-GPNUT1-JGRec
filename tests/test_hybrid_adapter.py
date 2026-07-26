from types import SimpleNamespace

import numpy as np

from jgrec.core.types import InteractionTable, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.config import TrainingConfig
from jgrec.rankers.hybrid.ranker import (
    HybridFeatureEncoder,
    HybridRankerAdapter,
    TemporalHybridRanker,
    _build_supervised_queries,
)


class _CapturingHybridImpl:
    def __init__(self) -> None:
        self.seen_queries = None

    def predict_batch(self, queries):
        self.seen_queries = queries
        return np.zeros((len(queries), queries.candidate_count), dtype=np.float64)

    def prediction_order(self, queries):
        self.seen_queries = queries
        return np.arange(len(queries) - 1, -1, -1, dtype=np.int64)


def test_hybrid_adapter_preserves_test_query_array_for_prediction():
    adapter = HybridRankerAdapter()
    impl = _CapturingHybridImpl()
    adapter.impl = impl
    queries = TestQueryArray(
        src=np.asarray([1, 2], dtype=np.int32),
        time=np.asarray([10, 11], dtype=np.int32),
        candidates=np.asarray([[3, 4, 5], [6, 7, 8]], dtype=np.int32),
    )

    probs = adapter.predict_batch(queries)

    assert impl.seen_queries is queries
    np.testing.assert_array_equal(probs, np.zeros((2, 3), dtype=np.float64))


def test_hybrid_adapter_delegates_prediction_order():
    adapter = HybridRankerAdapter()
    impl = _CapturingHybridImpl()
    adapter.impl = impl
    queries = TestQueryArray(
        src=np.asarray([1, 2], dtype=np.int32),
        time=np.asarray([10, 11], dtype=np.int32),
        candidates=np.asarray([[3, 4, 5], [6, 7, 8]], dtype=np.int32),
    )

    order = adapter.prediction_order(queries)

    assert impl.seen_queries is queries
    np.testing.assert_array_equal(order, np.asarray([1, 0], dtype=np.int64))


def test_hybrid_prediction_order_groups_future_only_queries_by_source():
    ranker = TemporalHybridRanker()
    ranker.encoder = SimpleNamespace(stats=SimpleNamespace(max_time=100))
    queries = TestQueryArray(
        src=np.asarray([3, 1, 3, 2, 1], dtype=np.int32),
        time=np.asarray([101, 102, 103, 104, 105], dtype=np.int32),
        candidates=np.arange(15, dtype=np.int32).reshape(5, 3),
    )

    order = ranker.prediction_order(queries)

    np.testing.assert_array_equal(order, np.asarray([1, 4, 3, 0, 2], dtype=np.int64))


def test_hybrid_prediction_order_rejects_non_future_or_unfitted_queries():
    queries = TestQueryArray(
        src=np.asarray([2, 1], dtype=np.int32),
        time=np.asarray([101, 100], dtype=np.int32),
        candidates=np.arange(6, dtype=np.int32).reshape(2, 3),
    )
    ranker = TemporalHybridRanker()

    assert ranker.prediction_order(queries) is None

    ranker.encoder = SimpleNamespace(stats=SimpleNamespace(max_time=100))
    assert ranker.prediction_order(queries) is None


def test_supervised_query_builder_returns_test_query_array_batches():
    interaction_table = InteractionTable.from_array(
        np.asarray([[1, 10, 10], [2, 20, 20]], dtype=np.int32)
    )
    config = TrainingConfig(
        num_negatives=2,
        candidate_prior_enabled=False,
        structure_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
        two_tower_enabled=False,
        source_profile_enabled=False,
        negative_sampling_workers=0,
    )
    encoder = HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(interaction_table),
        recent_window=4,
        candidate_prior_config=config.candidate_prior_config(),
        structure_config=config.structure_config(),
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
        two_tower_config=config.two_tower_config(),
    )
    encoder.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)

    queries = _build_supervised_queries(
        interaction_table,
        encoder,
        np.asarray([10, 20, 30, 40], dtype=np.int64),
        config,
        np.random.default_rng(0),
    )

    assert isinstance(queries, TestQueryArray)
    np.testing.assert_array_equal(queries.src, np.asarray([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(queries.time, np.asarray([10, 20], dtype=np.int32))
    assert queries.candidates.shape == (2, 3)
    np.testing.assert_array_equal(queries.candidates[:, 0], np.asarray([10, 20], dtype=np.int32))
