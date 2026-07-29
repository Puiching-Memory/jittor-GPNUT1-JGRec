from collections import Counter
from dataclasses import replace

import numpy as np

from jgrec.core.types import Interaction, InteractionTable, TestQuery
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES, CandidatePriorTower
from jgrec.rankers.hybrid.config import (
    TARGET_WINDOW_FEATURE_NAMES,
    CandidatePriorConfig,
    StructureTowerConfig,
    TrainingConfig,
)
from jgrec.rankers.hybrid.encoder_cache import HybridPrefixStateCache, hydrate_deterministic_state
from jgrec.rankers.hybrid.ranker import HybridFeatureEncoder, TemporalHybridRanker, _can_reuse_encoder_cache
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES, TemporalStats
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES, StructureFeatureTower
from jgrec.rankers.hybrid.target_window import TargetWindowTower


def _interactions() -> list[Interaction]:
    return [
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
        Interaction(src=2, dst=10, time=30),
        Interaction(src=10, dst=20, time=40),
        Interaction(src=2, dst=30, time=50),
        Interaction(src=3, dst=20, time=60),
        Interaction(src=1, dst=30, time=70),
        Interaction(src=30, dst=40, time=80),
        Interaction(src=2, dst=40, time=90),
        Interaction(src=1, dst=40, time=100),
    ]


def _deterministic_features(
    *,
    interactions: InteractionTable,
    queries: list[TestQuery],
    test_counts: Counter[int],
    structure_config: StructureTowerConfig,
    snapshot=None,
) -> np.ndarray:
    stats = TemporalStats(recent_window=4)
    prior = CandidatePriorTower(CandidatePriorConfig(enabled=True))
    target_window = TargetWindowTower(TrainingConfig().target_window_config())
    structure = StructureFeatureTower(structure_config)
    if snapshot is None:
        stats.fit(interactions)
        prior.fit(interactions, test_counts)
        target_window.fit(interactions)
        structure.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    else:
        hydrate_deterministic_state(
            snapshot=snapshot,
            stats=stats,
            candidate_prior=prior,
            target_window=target_window,
            structure=structure,
        )
    stat_features = stats.features_for_queries(queries)
    prior_features = prior.features_for_queries(queries, stat_features)
    target_features = target_window.features_for_queries(queries)
    structure_features = structure.features_for_queries(queries)
    return np.concatenate([stat_features, prior_features, target_features, structure_features], axis=2)


def test_encoder_prefix_cache_matches_independent_fit_for_all_prefixes():
    interactions = InteractionTable.from_events(_interactions())
    test_counts = Counter({20: 4, 40: 3, 999: 2})
    structure_config = StructureTowerConfig(future_only_transition_cooccur=True)
    cache = HybridPrefixStateCache(
        interactions,
        recent_window=4,
        candidate_prior_config=CandidatePriorConfig(enabled=True),
        structure_config=structure_config,
        test_candidate_counts=test_counts,
        verbose=False,
    )

    for prefix_end in (4, 7, len(interactions)):
        queries = [
            TestQuery(src=1, time=int(interactions.time[prefix_end - 1]) + 100, candidates=(10, 20, 40, 999)),
            TestQuery(src=2, time=int(interactions.time[prefix_end - 1]) + 100, candidates=(10, 30, 40, 999)),
        ]
        expected = _deterministic_features(
            interactions=interactions[:prefix_end],
            queries=queries,
            test_counts=test_counts,
            structure_config=structure_config,
        )
        actual = _deterministic_features(
            interactions=interactions[:prefix_end],
            queries=queries,
            test_counts=test_counts,
            structure_config=structure_config,
            snapshot=cache.snapshot_for_prefix(prefix_end),
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_encoder_prefix_cache_snapshots_are_isolated_after_structure_compaction():
    interactions = InteractionTable.from_events(_interactions())
    structure_config = StructureTowerConfig(future_only_transition_cooccur=False)
    cache = HybridPrefixStateCache(
        interactions,
        recent_window=4,
        candidate_prior_config=CandidatePriorConfig(enabled=True),
        structure_config=structure_config,
        test_candidate_counts=Counter({20: 4, 40: 3}),
        verbose=False,
    )
    first = cache.snapshot_for_prefix(7)
    second = cache.snapshot_for_prefix(len(interactions))
    first_structure = StructureFeatureTower(structure_config)
    second_structure = StructureFeatureTower(structure_config)
    first_structure.hydrate(first.structure)
    second_structure.hydrate(second.structure)

    first_structure.compact_transition_cooccur_for_future_queries()

    assert first_structure.index.future_only
    assert not second_structure.index.future_only
    assert second_structure.index.transition_times
    assert second_structure.index.cooccur_times


def test_encoder_prefix_cache_hydrates_disabled_prior_and_structure_as_zero_placeholders():
    interactions = InteractionTable.from_events(_interactions())
    snapshot = HybridPrefixStateCache(
        interactions,
        recent_window=4,
        candidate_prior_config=CandidatePriorConfig(enabled=True),
        structure_config=StructureTowerConfig(enabled=True, future_only_transition_cooccur=True),
        test_candidate_counts=Counter({20: 4, 40: 3}),
        verbose=False,
    ).snapshot_for_prefix(len(interactions))
    config = TrainingConfig(
        candidate_prior_enabled=False,
        structure_enabled=False,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
    )
    encoder = HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(interactions),
        recent_window=4,
        candidate_prior_config=config.candidate_prior_config(),
        target_window_config=config.target_window_config(),
        structure_config=config.structure_config(),
        two_tower_config=config.two_tower_config(),
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
    )

    encoder.fit(interactions, rng=np.random.default_rng(0), verbose=False, deterministic_snapshot=snapshot)
    features = encoder.features_for_queries([TestQuery(src=1, time=120, candidates=(10, 20, 999))])

    prior_start = len(STAT_FEATURE_NAMES)
    prior_end = prior_start + len(CANDIDATE_PRIOR_FEATURE_NAMES)
    target_end = prior_end + len(TARGET_WINDOW_FEATURE_NAMES)
    structure_end = target_end + len(STRUCTURE_FEATURE_NAMES)
    assert np.all(features[:, :, prior_start:prior_end] == 0.0)
    assert np.all(features[:, :, target_end:structure_end] == 0.0)


def test_final_encoder_reuses_compatible_prefix_cache_builder():
    interactions = InteractionTable.from_events(_interactions())
    config = TrainingConfig(
        candidate_prior_enabled=True,
        structure_enabled=True,
        structure_future_only_transition_cooccur=True,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
    )
    cache = HybridPrefixStateCache(
        interactions,
        recent_window=4,
        candidate_prior_config=config.candidate_prior_config(),
        structure_config=config.structure_config(),
        test_candidate_counts=Counter({20: 4, 40: 3}),
        verbose=False,
    )
    cache.snapshot_for_prefix(7)
    ranker = TemporalHybridRanker(recent_window=4)

    selected = ranker._final_encoder_cache(
        interactions=interactions,
        final_config=config,
        existing_cache=cache,
        existing_config=config,
        verbose=False,
    )

    assert selected is cache
    assert selected.snapshot_for_prefix(len(interactions)).prefix_end == len(interactions)


def test_final_encoder_drops_incompatible_prefix_cache_builder():
    interactions = InteractionTable.from_events(_interactions())
    source_config = TrainingConfig(
        candidate_prior_enabled=True,
        structure_enabled=True,
        structure_future_only_transition_cooccur=True,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
    )
    target_config = replace(source_config, structure_future_only_transition_cooccur=False)
    cache = HybridPrefixStateCache(
        interactions,
        recent_window=4,
        candidate_prior_config=source_config.candidate_prior_config(),
        structure_config=source_config.structure_config(),
        test_candidate_counts=Counter({20: 4, 40: 3}),
        verbose=False,
    )
    cache.snapshot_for_prefix(7)
    ranker = TemporalHybridRanker(recent_window=4)

    selected = ranker._final_encoder_cache(
        interactions=interactions,
        final_config=target_config,
        existing_cache=cache,
        existing_config=source_config,
        verbose=False,
    )

    assert selected is not cache
    assert len(cache.interactions) == 0
    assert selected is not None
    assert selected.snapshot_for_prefix(len(interactions)).prefix_end == len(interactions)


def test_encoder_cache_reuse_guard_matches_structure_semantics():
    base = TrainingConfig(
        candidate_prior_enabled=True,
        structure_enabled=True,
        structure_future_only_transition_cooccur=True,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
    )

    assert _can_reuse_encoder_cache(base, base)
    assert _can_reuse_encoder_cache(base, replace(base, structure_enabled=False))
    assert not _can_reuse_encoder_cache(base, replace(base, target_window_fractions=(0.02, 0.05, 0.20, 1.00)))
    assert not _can_reuse_encoder_cache(base, replace(base, structure_future_only_transition_cooccur=False))
    assert not _can_reuse_encoder_cache(
        replace(base, structure_cooccur_enabled=False),
        replace(base, structure_cooccur_enabled=True),
    )
    assert not _can_reuse_encoder_cache(
        base,
        replace(base, structure_cooccur_time_decay_enabled=True),
    )


def test_encoder_prefix_cache_preserves_time_decay_aggregate():
    interactions = InteractionTable.from_events(_interactions())
    structure_config = StructureTowerConfig(
        future_only_transition_cooccur=True,
        cooccur_time_decay_enabled=True,
        cooccur_time_decay_ratio=0.05,
        cooccur_time_decay_source_history_limit=64,
    )
    cache = HybridPrefixStateCache(
        interactions,
        recent_window=4,
        candidate_prior_config=CandidatePriorConfig(enabled=False),
        structure_config=structure_config,
        test_candidate_counts=Counter(),
        verbose=False,
    )
    snapshot = cache.snapshot_for_prefix(len(interactions))
    restored = StructureFeatureTower(structure_config)
    restored.hydrate(snapshot.structure)
    direct = StructureFeatureTower(structure_config)
    direct.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    queries = [TestQuery(src=1, time=120, candidates=(10, 20, 40, 999))]

    np.testing.assert_allclose(
        restored.time_decay_features_for_queries(queries),
        direct.time_decay_features_for_queries(queries),
        rtol=1e-6,
        atol=1e-6,
    )


def test_time_decay_feature_is_appended_without_changing_legacy_feature_columns():
    interactions = InteractionTable.from_events(_interactions())
    base = TrainingConfig(
        candidate_prior_enabled=False,
        structure_enabled=True,
        structure_future_only_transition_cooccur=True,
        source_profile_enabled=False,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
    )
    enabled = replace(
        base,
        structure_cooccur_time_decay_enabled=True,
        structure_cooccur_time_decay_ratio=0.05,
        structure_cooccur_time_decay_source_history_limit=64,
    )
    legacy_encoder = HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(interactions),
        recent_window=4,
        candidate_prior_config=base.candidate_prior_config(),
        target_window_config=base.target_window_config(),
        structure_config=base.structure_config(),
        source_profile_config=base.source_profile_config(),
        two_tower_config=base.two_tower_config(),
        graph_config=base.graph_config(),
        sequence_config=base.sequence_config(),
    )
    enabled_encoder = HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(interactions),
        recent_window=4,
        candidate_prior_config=enabled.candidate_prior_config(),
        target_window_config=enabled.target_window_config(),
        structure_config=enabled.structure_config(),
        source_profile_config=enabled.source_profile_config(),
        two_tower_config=enabled.two_tower_config(),
        graph_config=enabled.graph_config(),
        sequence_config=enabled.sequence_config(),
    )
    legacy_encoder.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    enabled_encoder.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    queries = [TestQuery(src=1, time=120, candidates=(10, 20, 40, 999))]

    legacy_features = legacy_encoder.features_for_queries(queries)
    enabled_features = enabled_encoder.features_for_queries(queries)

    assert legacy_encoder.feature_dim == 63
    assert enabled_encoder.feature_dim == 64
    assert enabled_encoder.feature_names[-1] == "cooccur_time_decay_score"
    np.testing.assert_array_equal(enabled_features[..., :63], legacy_features)
