import sys
from types import SimpleNamespace

import numpy as np
import pytest

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


class _FakeEncoder:
    feature_dim = 2

    def __init__(self) -> None:
        self.clear_calls = 0

    def features_for_queries(self, queries):
        return np.ones((len(queries), queries.candidate_count, self.feature_dim), dtype=np.float32)

    def clear_predict_caches(self) -> None:
        self.clear_calls += 1


class _FakeFusionResult:
    mean = np.zeros(2, dtype=np.float32)
    std = np.ones(2, dtype=np.float32)
    feature_indices = ()


def _predict_ready_ranker():
    ranker = TemporalHybridRanker()
    encoder = _FakeEncoder()
    ranker.encoder = encoder
    ranker.fusion = object()
    ranker.fusion_result = _FakeFusionResult()
    queries = TestQueryArray(
        src=np.asarray([1, 2], dtype=np.int32),
        time=np.asarray([10, 11], dtype=np.int32),
        candidates=np.asarray([[3, 4, 5], [6, 7, 8]], dtype=np.int32),
    )
    return ranker, encoder, queries


def test_hybrid_predict_batch_clears_predict_caches_after_success(monkeypatch):
    ranker, encoder, queries = _predict_ready_ranker()

    def fake_predict_logits(model, features, mean, std):
        return np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "jgrec.rankers.hybrid.fusion",
        SimpleNamespace(predict_logits=fake_predict_logits),
    )

    probs = ranker.predict_batch(queries)

    assert encoder.clear_calls == 1
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2))


def test_hybrid_predict_batch_clears_predict_caches_after_prediction_error(monkeypatch):
    ranker, encoder, queries = _predict_ready_ranker()

    def fake_predict_logits(model, features, mean, std):
        raise ValueError("fusion failed")

    monkeypatch.setitem(
        sys.modules,
        "jgrec.rankers.hybrid.fusion",
        SimpleNamespace(predict_logits=fake_predict_logits),
    )

    with pytest.raises(ValueError, match="fusion failed"):
        ranker.predict_batch(queries)

    assert encoder.clear_calls == 1
