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
        _table(
            [
                Interaction(src=1, dst=10, time=10),
                Interaction(src=10, dst=1, time=25),
                Interaction(src=2, dst=10, time=30),
                Interaction(src=10, dst=20, time=32),
                Interaction(src=2, dst=20, time=40),
                Interaction(src=1, dst=10, time=50),
            ]
        ),
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
    assert features[1, FEATURE["aa_log_rank"]] == np.float32(1.0)

    assert features[0, FEATURE["adamic_adar_log"]] == 0.0
    assert features[0, FEATURE["resource_allocation_log"]] == 0.0
    assert features[0, FEATURE["aa_log_rank"]] == 0.0

    assert np.all(features[2] == 0.0)


def test_structure_features_after_training_window_include_full_history():
    tower = StructureFeatureTower()
    tower.fit(
        _table(
            [
                Interaction(src=1, dst=10, time=10),
                Interaction(src=2, dst=10, time=30),
                Interaction(src=10, dst=20, time=35),
                Interaction(src=2, dst=20, time=40),
            ]
        ),
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
        _table(
            [
                Interaction(src=1, dst=10, time=10),
                Interaction(src=2, dst=20, time=20),
                Interaction(src=10, dst=20, time=30),
                Interaction(src=99, dst=30, time=50),
            ]
        ),
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
        _table(
            [
                Interaction(src=1, dst=10, time=10),
                Interaction(src=10, dst=30, time=20),
                Interaction(src=10, dst=20, time=30),
                Interaction(src=3, dst=10, time=40),
                Interaction(src=4, dst=10, time=50),
                Interaction(src=5, dst=20, time=60),
            ]
        ),
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
    interactions = [Interaction(src=1, dst=1000 + idx, time=idx) for idx in range(1, 310)]
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
    interactions = [Interaction(src=1, dst=1000 + idx, time=idx) for idx in range(1, 40)]
    interactions.extend(Interaction(src=1000 + idx, dst=2000 + (idx % 5), time=100 + idx) for idx in range(1, 40))
    queries = [Query(src=1, time=500 + idx, candidates=(2000, 2001, 2002, 9999)) for idx in range(3)]
    future_tower = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True))
    future_tower.fit(_table(interactions), rng=np.random.default_rng(0), verbose=False)

    actual = future_tower.features_for_queries(queries)

    assert 1 in future_tower._full_src_cooccur_cache
    assert np.isfinite(actual).all()


def test_structure_cache_byte_budget_does_not_change_features() -> None:
    interactions = [Interaction(src=1, dst=1000 + idx, time=idx) for idx in range(1, 40)]
    interactions.extend(Interaction(src=1000 + idx, dst=2000 + (idx % 5), time=100 + idx) for idx in range(1, 40))
    queries = [Query(src=1, time=500 + idx, candidates=(2000, 2001, 2002, 9999)) for idx in range(3)]
    regular = StructureFeatureTower(
        StructureTowerConfig(future_only_transition_cooccur=True, cache_max_bytes=1024 * 1024)
    )
    constrained = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True, cache_max_bytes=1))
    table = _table(interactions)
    regular.fit(table, rng=np.random.default_rng(0), verbose=False)
    constrained.fit(table, rng=np.random.default_rng(0), verbose=False)

    expected = regular.features_for_queries(queries)
    actual = constrained.features_for_queries(queries)

    np.testing.assert_array_equal(actual, expected)
    assert constrained.cache_bytes <= constrained.config.cache_max_bytes
    structure_summary = regular._full_src_structure_cache.get(1)
    cooccur_summary = regular._full_src_cooccur_cache.get(1)
    assert isinstance(structure_summary.candidate_ids, np.ndarray)
    assert isinstance(structure_summary.common_counts, np.ndarray)
    assert isinstance(structure_summary.aa_scores, np.ndarray)
    assert isinstance(structure_summary.ra_scores, np.ndarray)
    assert isinstance(cooccur_summary.candidate_ids, np.ndarray)
    assert isinstance(cooccur_summary.counts, np.ndarray)


def test_future_only_structure_preaggregates_large_source_common_neighbors():
    interactions = [Interaction(src=1, dst=1000 + idx, time=idx) for idx in range(1, 310)]
    interactions.extend(Interaction(src=1000 + idx, dst=2000 + (idx % 7), time=400 + idx) for idx in range(1, 310))
    query = Query(src=1, time=1000, candidates=(2000, 2001, 2002, 9999))
    full_tower = StructureFeatureTower()
    future_tower = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True))
    full_tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    future_tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)

    expected = full_tower.features_for_queries([query])
    actual = future_tower.features_for_queries([query])

    assert 1 in future_tower._full_src_structure_cache
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


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
        _table(
            [
                Interaction(src=1, dst=10, time=10),
                Interaction(src=2, dst=10, time=30),
                Interaction(src=10, dst=20, time=35),
                Interaction(src=2, dst=20, time=40),
            ]
        ),
        rng=np.random.default_rng(0),
        verbose=False,
    )

    features = tower.features_for_queries([Query(src=1, time=50, candidates=(20,))])[0]

    assert features[0, FEATURE["common_neighbors"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["jaccard"]] == np.float32(1 / 2)
    assert features[0, FEATURE["cooccur_score"]] == 0.0
    assert features[0, FEATURE["transition_score"]] == 0.0


def test_structure_link_prediction_features():
    tower = StructureFeatureTower()
    tower.fit(
        _table(
            [
                Interaction(src=1, dst=5, time=10),
                Interaction(src=5, dst=1, time=15),
                Interaction(src=1, dst=6, time=20),
                Interaction(src=5, dst=7, time=30),
                Interaction(src=6, dst=7, time=40),
                Interaction(src=3, dst=7, time=50),
            ]
        ),
        rng=np.random.default_rng(0),
        verbose=False,
    )

    # Full-history path (time > max_time): bridges {5, 6}
    # degree(5)=3 (out:[1,7] in:[1]), degree(6)=2 (out:[7] in:[1])
    # node_degree(7) = 0 (out) + 3 (in: 5, 6, 3) = 3
    aa_full = 1.0 / math.log1p(3) + 1.0 / math.log1p(2)
    ra_full = 1.0 / 3 + 1.0 / 2
    full = tower.features_for_queries([Query(src=1, time=60, candidates=(7,))])[0]
    assert full[0, FEATURE["adamic_adar_log"]] == np.float32(math.log1p(aa_full))
    assert full[0, FEATURE["resource_allocation_log"]] == np.float32(math.log1p(ra_full))
    assert full[0, FEATURE["dst_degree_log"]] == np.float32(math.log1p(3))
    # Single candidate -> rank stays 0
    assert full[0, FEATURE["aa_log_rank"]] == 0.0

    # Cutoff path (time=35): only bridge {5} visible for dst=7
    aa_cut = 1.0 / math.log1p(3)
    ra_cut = 1.0 / 3
    cutoff = tower.features_for_queries([Query(src=1, time=35, candidates=(7,))])[0]
    assert cutoff[0, FEATURE["adamic_adar_log"]] == np.float32(math.log1p(aa_cut))
    assert cutoff[0, FEATURE["resource_allocation_log"]] == np.float32(math.log1p(ra_cut))
    assert cutoff[0, FEATURE["dst_degree_log"]] == np.float32(math.log1p(3))
    assert cutoff[0, FEATURE["aa_log_rank"]] == 0.0


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
        len(STAT_FEATURE_NAMES) + len(CANDIDATE_PRIOR_FEATURE_NAMES) + len(STRUCTURE_FEATURE_NAMES)
    )
    assert len(by_name["stats_prior_target_structure"]) == (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
    )
