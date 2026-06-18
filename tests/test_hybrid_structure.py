import math

import numpy as np

from jgrec.core.types import Interaction, InteractionTable
from jgrec.core.types import TestQuery as Query
from jgrec.rankers.hybrid.candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES
from jgrec.rankers.hybrid.config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SOURCE_PROFILE_FEATURE_NAMES,
    TARGET_WINDOW_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
    StructureTowerConfig,
)
from jgrec.rankers.hybrid.ranker import _feature_masks
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES, StructureFeatureTower

FEATURE = {name: idx for idx, name in enumerate(STRUCTURE_FEATURE_NAMES)}


def _table(events: list[Interaction]) -> InteractionTable:
    return InteractionTable.from_events(events)


def test_structure_features_use_temporal_cutoff():
    tower = StructureFeatureTower()
    tower.fit(
        _table([
            Interaction(src=1, dst=10, time=10),
            Interaction(src=10, dst=1, time=25),
            Interaction(src=2, dst=10, time=30),
            Interaction(src=10, dst=20, time=32),
            Interaction(src=2, dst=20, time=40),
            Interaction(src=1, dst=10, time=50),
        ]),
        rng=np.random.default_rng(0),
        verbose=False,
    )

    features = tower.features_for_queries([Query(src=1, time=35, candidates=(10, 20, 30))])[0]

    assert features[0, FEATURE["pair_decay_short"]] > 0.0
    assert features[0, FEATURE["pair_decay_medium"]] > features[0, FEATURE["pair_decay_short"]]
    assert features[0, FEATURE["pair_decay_long"]] > features[0, FEATURE["pair_decay_medium"]]
    assert features[0, FEATURE["dst_unique_src"]] == np.float32(math.log1p(2))
    assert features[0, FEATURE["dst_pop_rank"]] == np.float32(1.0 / math.log1p(3))
    assert features[0, FEATURE["reverse_log_count"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["reverse_recency"]] == np.float32(math.exp(-(35 - 25) / 40))
    assert features[0, FEATURE["common_neighbors"]] == 0.0
    assert features[0, FEATURE["jaccard"]] == 0.0

    assert features[1, FEATURE["common_neighbors"]] == np.float32(math.log1p(1))
    assert features[1, FEATURE["jaccard"]] == np.float32(1.0)
    assert features[1, FEATURE["cooccur_score"]] == 0.0
    assert features[1, FEATURE["transition_score"]] == 0.0

    assert np.all(features[2] == 0.0)


def test_structure_features_after_training_window_include_full_history():
    tower = StructureFeatureTower()
    tower.fit(
        _table([
            Interaction(src=1, dst=10, time=10),
            Interaction(src=2, dst=10, time=30),
            Interaction(src=10, dst=20, time=35),
            Interaction(src=2, dst=20, time=40),
        ]),
        rng=np.random.default_rng(0),
        verbose=False,
    )

    features = tower.features_for_queries([Query(src=1, time=50, candidates=(20,))])[0]

    assert features[0, FEATURE["common_neighbors"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["jaccard"]] == np.float32(1 / 2)
    assert features[0, FEATURE["cooccur_score"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["transition_score"]] == np.float32(math.log1p(1))


def test_structure_common_neighbors_ignore_numeric_id_collisions_in_bipartite_graph():
    tower = StructureFeatureTower()
    tower.fit(
        _table([
            Interaction(src=1, dst=10, time=10),
            Interaction(src=2, dst=20, time=20),
            Interaction(src=10, dst=20, time=30),
            Interaction(src=99, dst=30, time=50),
        ]),
        rng=np.random.default_rng(0),
        verbose=False,
    )

    cutoff_features = tower.features_for_queries([Query(src=1, time=40, candidates=(20,))])[0]
    future_features = tower.features_for_queries([Query(src=1, time=60, candidates=(20,))])[0]

    assert cutoff_features[0, FEATURE["dst_unique_src"]] == np.float32(math.log1p(2))
    assert cutoff_features[0, FEATURE["common_neighbors"]] == 0.0
    assert cutoff_features[0, FEATURE["jaccard"]] == 0.0
    assert future_features[0, FEATURE["dst_unique_src"]] == np.float32(math.log1p(2))
    assert future_features[0, FEATURE["common_neighbors"]] == 0.0
    assert future_features[0, FEATURE["jaccard"]] == 0.0


def test_structure_common_neighbors_keep_supported_bridge_nodes_in_mixed_graph():
    tower = StructureFeatureTower()
    tower.fit(
        _table([
            Interaction(src=1, dst=10, time=10),
            Interaction(src=10, dst=30, time=20),
            Interaction(src=10, dst=20, time=30),
            Interaction(src=3, dst=10, time=40),
            Interaction(src=4, dst=10, time=50),
            Interaction(src=5, dst=20, time=60),
        ]),
        rng=np.random.default_rng(0),
        verbose=False,
    )

    features = tower.features_for_queries([Query(src=1, time=70, candidates=(20,))])[0]

    assert features[0, FEATURE["common_neighbors"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["jaccard"]] == np.float32(1 / 2)


def test_future_only_structure_compaction_preserves_full_history_features():
    interactions = [
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
        Interaction(src=2, dst=10, time=30),
        Interaction(src=10, dst=20, time=35),
        Interaction(src=2, dst=20, time=40),
        Interaction(src=1, dst=30, time=50),
    ]
    query = Query(src=1, time=80, candidates=(10, 20, 30, 40))
    full_tower = StructureFeatureTower()
    compact_tower = StructureFeatureTower()
    interaction_table = _table(interactions)
    full_tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    compact_tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)

    expected = full_tower.features_for_queries([query])
    compact_tower.compact_transition_cooccur_for_future_queries()
    actual = compact_tower.features_for_queries([query])

    assert compact_tower.index.future_only
    assert compact_tower.index.transition_times == {}
    assert compact_tower.index.cooccur_times == {}
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_future_only_structure_build_preserves_full_history_features_without_time_indexes():
    interactions = [
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
        Interaction(src=2, dst=10, time=30),
        Interaction(src=10, dst=20, time=35),
        Interaction(src=2, dst=20, time=40),
        Interaction(src=1, dst=30, time=50),
    ]
    query = Query(src=1, time=80, candidates=(10, 20, 30, 40))
    full_tower = StructureFeatureTower()
    future_tower = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True))
    interaction_table = _table(interactions)
    full_tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    future_tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)

    expected = full_tower.features_for_queries([query])
    actual = future_tower.features_for_queries([query])

    assert future_tower.index.future_only
    assert future_tower.index.transition_times == {}
    assert future_tower.index.cooccur_times == {}
    assert future_tower.index.future_transition_count_maps
    assert future_tower.index.future_cooccur_count_maps
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_future_only_structure_preaggregates_large_source_cooccurs():
    interactions = [
        Interaction(src=1, dst=1000 + idx, time=idx)
        for idx in range(1, 310)
    ]
    query = Query(src=1, time=400, candidates=(1001, 1050, 1128, 1309))
    full_tower = StructureFeatureTower()
    future_tower = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True))
    interaction_table = _table(interactions)
    full_tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    future_tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)

    expected = full_tower.features_for_queries([query])
    actual = future_tower.features_for_queries([query])

    assert 1 in future_tower._full_src_cooccur_cache
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_future_only_structure_preaggregates_repeated_source_cooccurs():
    interactions = [
        Interaction(src=1, dst=1000 + idx, time=idx)
        for idx in range(1, 40)
    ]
    interactions.extend(
        Interaction(src=1000 + idx, dst=2000 + (idx % 5), time=100 + idx)
        for idx in range(1, 40)
    )
    queries = [
        Query(src=1, time=500 + idx, candidates=(2000, 2001, 2002, 9999))
        for idx in range(3)
    ]
    future_tower = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True))
    future_tower.fit(_table(interactions), rng=np.random.default_rng(0), verbose=False)

    actual = future_tower.features_for_queries(queries)

    assert 1 in future_tower._full_src_cooccur_cache
    assert np.isfinite(actual).all()


def test_future_only_structure_preaggregates_large_source_common_neighbors():
    interactions = [Interaction(src=1, dst=1000 + idx, time=idx) for idx in range(1, 310)]
    interactions.extend(
        Interaction(src=1000 + idx, dst=2000 + (idx % 7), time=400 + idx)
        for idx in range(1, 310)
    )
    query = Query(src=1, time=1000, candidates=(2000, 2001, 2002, 9999))
    full_tower = StructureFeatureTower()
    future_tower = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True))
    full_tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    future_tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)

    expected = full_tower.features_for_queries([query])
    actual = future_tower.features_for_queries([query])

    assert 1 in future_tower._full_src_common_neighbor_cache
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_structure_clear_full_history_cache_preserves_recomputed_features():
    interactions = [Interaction(src=1, dst=1000 + idx, time=idx) for idx in range(1, 310)]
    interactions.extend(
        Interaction(src=1000 + idx, dst=2000 + (idx % 7), time=400 + idx)
        for idx in range(1, 310)
    )
    tower = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True))
    tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    query = Query(src=1, time=1000, candidates=(2000, 2001, 2002, 9999))

    before = tower.features_for_queries([query])
    tower._full_dst_sources(2000)

    assert tower._full_src_neighbor_cache
    assert tower._full_dst_source_cache
    assert tower._full_src_common_neighbor_cache
    assert tower._full_src_cooccur_cache

    tower.clear_full_history_cache()

    assert not tower._full_src_neighbor_cache
    assert not tower._full_dst_source_cache
    assert not tower._full_src_common_neighbor_cache
    assert not tower._full_src_cooccur_cache

    after = tower.features_for_queries([query])

    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)


def test_structure_common_neighbor_counts_unique_neighbor_once():
    tower = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True))
    tower.fit(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=10, dst=20, time=20),
            Interaction(src=10, dst=20, time=30),
        ],
        rng=np.random.default_rng(0),
        verbose=False,
    )

    features = tower.features_for_queries([Query(src=1, time=100, candidates=(20,))])[0]

    assert features[0, FEATURE["common_neighbors"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["jaccard"]] == np.float32(1.0)


def test_full_history_dst_unique_src_counts_are_unique():
    tower = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True))
    tower.fit(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=2, dst=20, time=20),
            Interaction(src=2, dst=20, time=30),
            Interaction(src=3, dst=20, time=40),
        ],
        rng=np.random.default_rng(0),
        verbose=False,
    )

    features = tower.features_for_queries([Query(src=1, time=100, candidates=(20,))])[0]

    assert tower.index.dst_unique_src_counts[20] == 2
    assert features[0, FEATURE["dst_unique_src"]] == np.float32(math.log1p(2))
    assert features[0, FEATURE["dst_pop_rank"]] == np.float32(1.0 / math.log1p(3))


def test_structure_memory_switches_disable_heavy_features_only():
    tower = StructureFeatureTower(
        StructureTowerConfig(
            cooccur_enabled=False,
            transition_enabled=False,
            cooccur_history_limit=16,
        )
    )
    tower.fit(
        _table([
            Interaction(src=1, dst=10, time=10),
            Interaction(src=2, dst=10, time=30),
            Interaction(src=10, dst=20, time=35),
            Interaction(src=2, dst=20, time=40),
        ]),
        rng=np.random.default_rng(0),
        verbose=False,
    )

    features = tower.features_for_queries([Query(src=1, time=50, candidates=(20,))])[0]

    assert features[0, FEATURE["common_neighbors"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["jaccard"]] == np.float32(1 / 2)
    assert features[0, FEATURE["cooccur_score"]] == 0.0
    assert features[0, FEATURE["transition_score"]] == 0.0


def test_hybrid_feature_masks_include_structure_groups():
    feature_count = (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
        + len(SOURCE_PROFILE_FEATURE_NAMES)
        + len(TWO_TOWER_FEATURE_NAMES)
        + len(GRAPH_WINDOW_NAMES)
        + len(SEQUENCE_FEATURE_NAMES)
    )

    masks = _feature_masks(feature_count)

    names = [name for name, _ in masks]

    assert names == [
        "stats",
        "stats_prior",
        "stats_prior_structure",
        "stats_prior_structure_tower",
        "stats_prior_structure_tower_gnn",
        "stats_prior_structure_tower_gnn_seq",
        "stats_prior_target",
        "stats_prior_target_structure",
        "stats_prior_target_structure_profile",
        "stats_prior_target_structure_profile_tower",
        "stats_prior_target_structure_profile_tower_gnn",
        "stats_prior_target_structure_profile_tower_gnn_seq",
    ]
    assert "stats_prior_structure_tower" in names
    assert "stats_prior_structure_tower_gnn" in names
    by_name = dict(masks)
    assert len(by_name["stats_prior_structure"]) == (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
    )
    assert len(by_name["stats_prior_target_structure"]) == (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
    )
