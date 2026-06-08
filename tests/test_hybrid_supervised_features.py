from pathlib import Path

import numpy as np

from jgrec.core.types import Interaction, InteractionTable, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.config import TrainingConfig
from jgrec.rankers.hybrid.ranker import (
    HybridFeatureEncoder,
    SupervisedFeatureBuilder,
    _build_supervised_features,
    _build_supervised_queries,
)


def _interactions(count: int = 48) -> list[Interaction]:
    events: list[Interaction] = []
    for idx in range(count):
        src = idx % 7 + 1
        dst = (idx * 3) % 17 + 10
        events.append(Interaction(src=src, dst=dst, time=idx + 1))
    return events


def _encoder(interactions: InteractionTable) -> HybridFeatureEncoder:
    config = TrainingConfig(
        candidate_prior_enabled=True,
        structure_enabled=True,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
    )
    encoder = HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(interactions),
        recent_window=4,
        candidate_prior_config=config.candidate_prior_config(),
        structure_config=config.structure_config(),
        two_tower_config=config.two_tower_config(),
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
    )
    encoder.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    return encoder


def _config(**kwargs) -> TrainingConfig:
    return TrainingConfig(
        candidate_prior_enabled=True,
        structure_enabled=True,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
        num_negatives=5,
        hard_negative_ratio=0.5,
        popular_negative_ratio=0.25,
        test_candidate_negative_ratio=0.0,
        supervised_feature_batch_size=6,
        verbose=False,
        **kwargs,
    )


def test_supervised_builder_matches_reference_queries_and_features():
    interactions = InteractionTable.from_events(_interactions())
    positives = interactions[20:34]
    encoder = _encoder(interactions[:20])
    config = _config(negative_sampling_workers=0)
    dst_pool = np.unique(interactions.dst).astype(np.int64, copy=False)
    reference_rng = np.random.default_rng(42)
    fast_rng = np.random.default_rng(42)

    reference_queries = _build_supervised_queries(positives, encoder, dst_pool, config, reference_rng)
    builder = SupervisedFeatureBuilder(encoder=encoder, dst_pool=dst_pool, config=config)
    batch = builder.batch_for_events(positives, fast_rng)

    assert list(batch) == list(reference_queries)
    expected = encoder.features_for_queries(reference_queries)
    actual = encoder.features_for_query_array(batch)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_supervised_feature_matrix_matches_reference_path_with_memmap_enabled():
    interactions = InteractionTable.from_events(_interactions())
    positives = interactions[20:44]
    encoder = _encoder(interactions[:20])
    config = _config(supervised_feature_memmap=True, negative_sampling_workers=0)
    dst_pool = np.unique(interactions.dst).astype(np.int64, copy=False)

    reference_batches = []
    reference_rng = np.random.default_rng(7)
    for start in range(0, len(positives), config.supervised_feature_batch_size):
        batch_events = positives[start : start + config.supervised_feature_batch_size]
        queries = _build_supervised_queries(batch_events, encoder, dst_pool, config, reference_rng)
        reference_batches.append(encoder.features_for_queries(queries))
    expected = np.concatenate(reference_batches, axis=0)

    actual = _build_supervised_features(positives, encoder, dst_pool, config, np.random.default_rng(7))

    assert isinstance(actual, np.memmap)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


def test_supervised_builder_parallel_negatives_match_reference_order():
    interactions = InteractionTable.from_events(_interactions())
    positives = interactions[20:40]
    encoder = _encoder(interactions[:20])
    config = _config(negative_sampling_workers=2)
    dst_pool = np.unique(interactions.dst).astype(np.int64, copy=False)
    reference_rng = np.random.default_rng(19)
    fast_rng = np.random.default_rng(19)

    reference_queries = _build_supervised_queries(positives, encoder, dst_pool, config, reference_rng)
    batch = SupervisedFeatureBuilder(encoder=encoder, dst_pool=dst_pool, config=config).batch_for_events(
        positives,
        fast_rng,
    )

    assert list(batch) == list(reference_queries)


def test_encoder_query_array_keeps_disabled_tower_placeholders_zero():
    interactions = InteractionTable.from_events(_interactions())
    encoder = _encoder(interactions[:20])
    batch = TestQueryArray(
        src=np.asarray([1, 2], dtype=np.int32),
        time=np.asarray([100, 101], dtype=np.int32),
        candidates=np.asarray([[10, 11, 12], [13, 14, 15]], dtype=np.int32),
    )

    features = encoder.features_for_query_array(batch)

    assert features.shape == (2, 3, encoder.feature_dim)
    assert np.isfinite(features).all()
    assert features[:, :, -6:].sum() == 0.0


def test_benchmark_script_exists():
    assert Path("scripts/bench_supervised_features.py").exists()
