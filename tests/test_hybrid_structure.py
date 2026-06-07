import math

import numpy as np

from jgrec.core.types import Interaction
from jgrec.core.types import TestQuery as Query
from jgrec.rankers.hybrid.candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES
from jgrec.rankers.hybrid.config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
    StructureTowerConfig,
)
from jgrec.rankers.hybrid.ranker import _feature_masks
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES, StructureFeatureTower

FEATURE = {name: idx for idx, name in enumerate(STRUCTURE_FEATURE_NAMES)}


def test_structure_features_use_temporal_cutoff():
    tower = StructureFeatureTower()
    tower.fit(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=10, dst=1, time=25),
            Interaction(src=2, dst=10, time=30),
            Interaction(src=10, dst=20, time=32),
            Interaction(src=2, dst=20, time=40),
            Interaction(src=1, dst=10, time=50),
        ],
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
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=2, dst=10, time=30),
            Interaction(src=10, dst=20, time=35),
            Interaction(src=2, dst=20, time=40),
        ],
        rng=np.random.default_rng(0),
        verbose=False,
    )

    features = tower.features_for_queries([Query(src=1, time=50, candidates=(20,))])[0]

    assert features[0, FEATURE["common_neighbors"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["jaccard"]] == np.float32(1 / 2)
    assert features[0, FEATURE["cooccur_score"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["transition_score"]] == np.float32(math.log1p(1))


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
    full_tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    compact_tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)

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
    full_tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    future_tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)

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
    full_tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    future_tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)

    expected = full_tower.features_for_queries([query])
    actual = future_tower.features_for_queries([query])

    assert 1 in future_tower._full_src_cooccur_cache
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_structure_memory_switches_disable_heavy_features_only():
    tower = StructureFeatureTower(
        StructureTowerConfig(
            cooccur_enabled=False,
            transition_enabled=False,
            cooccur_history_limit=16,
        )
    )
    tower.fit(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=2, dst=10, time=30),
            Interaction(src=10, dst=20, time=35),
            Interaction(src=2, dst=20, time=40),
        ],
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
        + len(STRUCTURE_FEATURE_NAMES)
        + len(TWO_TOWER_FEATURE_NAMES)
        + len(GRAPH_WINDOW_NAMES)
        + len(SEQUENCE_FEATURE_NAMES)
    )

    masks = _feature_masks(feature_count)

    assert [name for name, _ in masks] == [
        "stats",
        "stats_prior",
        "stats_prior_structure",
        "stats_prior_structure_tower",
        "stats_prior_structure_tower_gnn",
        "stats_prior_structure_tower_gnn_seq",
    ]
    assert len(masks[2][1]) == (
        len(STAT_FEATURE_NAMES) + len(CANDIDATE_PRIOR_FEATURE_NAMES) + len(STRUCTURE_FEATURE_NAMES)
    )
