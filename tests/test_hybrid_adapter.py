import numpy as np

from jgrec.core.types import Interaction, InteractionTable, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.config import TrainingConfig
from jgrec.rankers.hybrid.ranker import HybridFeatureEncoder, HybridRankerAdapter, _build_supervised_queries


class _CapturingHybridImpl:
    def __init__(self) -> None:
        self.seen_queries = None

    def predict_batch(self, queries):
        self.seen_queries = queries
        return np.zeros((len(queries), queries.candidate_count), dtype=np.float64)


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


def test_supervised_query_builder_returns_test_query_array_batches():
    positives = [
        Interaction(src=1, dst=10, time=10),
        Interaction(src=2, dst=20, time=20),
    ]
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
