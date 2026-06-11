import math

import numpy as np

from jgrec.core.types import Interaction, InteractionTable, TestQuery, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES
from jgrec.rankers.hybrid.config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SOURCE_PROFILE_FEATURE_NAMES,
    TARGET_WINDOW_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
    TargetWindowConfig,
    TrainingConfig,
)
from jgrec.rankers.hybrid.encoder_cache import HybridPrefixStateCache, hydrate_deterministic_state
from jgrec.rankers.hybrid.ranker import HybridFeatureEncoder, _config_for_selected_features, _feature_masks
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES, TemporalStats
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES, StructureFeatureTower
from jgrec.rankers.hybrid.target_window import TargetWindowTower

FEATURE = {name: idx for idx, name in enumerate(TARGET_WINDOW_FEATURE_NAMES)}


def _interactions() -> InteractionTable:
    return InteractionTable.from_events([
        Interaction(src=1, dst=10, time=10),
        Interaction(src=2, dst=20, time=20),
        Interaction(src=3, dst=10, time=30),
        Interaction(src=4, dst=30, time=40),
        Interaction(src=5, dst=10, time=50),
        Interaction(src=6, dst=20, time=60),
        Interaction(src=7, dst=40, time=70),
        Interaction(src=8, dst=10, time=80),
    ])


def test_target_window_shape_and_finite_values():
    tower = TargetWindowTower(TargetWindowConfig(window_fractions=(0.1, 0.3, 0.6, 1.0)))
    tower.fit(_interactions())

    features = tower.features_for_queries([
        TestQuery(src=1, time=90, candidates=(10, 20, 999)),
        TestQuery(src=2, time=90, candidates=(30, 40, 999)),
    ])

    assert features.shape == (2, 3, len(TARGET_WINDOW_FEATURE_NAMES))
    assert np.all(np.isfinite(features))
    assert np.any(features[0] != 0.0)


def test_target_window_manual_window_features_and_rank():
    tower = TargetWindowTower(TargetWindowConfig(window_fractions=(0.25, 0.5, 0.75, 1.0)))
    tower.fit(_interactions())

    features = tower.features_for_queries([TestQuery(src=1, time=81, candidates=(10, 20, 40))])[0]

    assert features[0, FEATURE["target_pop_w100"]] == np.float32(math.log1p(4) / math.log1p(8))
    assert features[1, FEATURE["target_pop_w100"]] == np.float32(math.log1p(2) / math.log1p(8))
    assert features[2, FEATURE["target_pop_w100"]] == np.float32(math.log1p(1) / math.log1p(8))
    assert features[0, FEATURE["target_pop_share_w100"]] == np.float32(4 / 8)
    assert features[0, FEATURE["target_recency_w100"]] == np.float32(math.exp(-1 / 70))
    assert features[0, FEATURE["target_pop_rank_w100"]] == 1.0
    assert features[1, FEATURE["target_pop_rank_w100"]] == 0.5
    assert features[2, FEATURE["target_pop_rank_w100"]] == np.float32(1 / 3)


def test_target_window_cutoff_does_not_use_future_events():
    tower = TargetWindowTower(TargetWindowConfig(window_fractions=(0.25, 0.5, 0.75, 1.0)))
    tower.fit(_interactions())

    early = tower.features_for_queries([TestQuery(src=1, time=45, candidates=(10, 20, 30))])[0]
    late = tower.features_for_queries([TestQuery(src=1, time=81, candidates=(10, 20, 30))])[0]

    assert early[0, FEATURE["target_pop_w100"]] == np.float32(math.log1p(2) / math.log1p(4))
    assert early[1, FEATURE["target_pop_w100"]] == np.float32(math.log1p(1) / math.log1p(4))
    assert late[0, FEATURE["target_pop_w100"]] > early[0, FEATURE["target_pop_w100"]]


def test_disabled_target_window_uses_zero_placeholder_in_encoder():
    interactions = _interactions()
    config = TrainingConfig(
        target_window_enabled=False,
        structure_enabled=False,
        source_profile_enabled=False,
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
        source_profile_config=config.source_profile_config(),
        two_tower_config=config.two_tower_config(),
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
    )
    encoder.fit(interactions, rng=np.random.default_rng(0), verbose=False)

    features = encoder.features_for_queries([TestQuery(src=1, time=90, candidates=(10, 20))])

    target_start = len(STAT_FEATURE_NAMES) + len(CANDIDATE_PRIOR_FEATURE_NAMES)
    target_end = target_start + len(TARGET_WINDOW_FEATURE_NAMES)
    assert features.shape[-1] == (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
        + len(SOURCE_PROFILE_FEATURE_NAMES)
        + len(TWO_TOWER_FEATURE_NAMES)
        + len(GRAPH_WINDOW_NAMES)
        + len(SEQUENCE_FEATURE_NAMES)
    )
    assert np.all(features[:, :, target_start:target_end] == 0.0)


def test_encoder_cache_hydrates_target_window_state():
    interactions = _interactions()
    config = TargetWindowConfig(window_fractions=(0.1, 0.3, 0.6, 1.0))
    snapshot = HybridPrefixStateCache(
        interactions,
        recent_window=4,
        candidate_prior_config=TrainingConfig().candidate_prior_config(),
        target_window_config=config,
        structure_config=TrainingConfig().structure_config(),
        verbose=False,
    ).snapshot_for_prefix(len(interactions))

    independent = TargetWindowTower(config)
    independent.fit(interactions)
    hydrated = TargetWindowTower(config)
    stats = TemporalStats(recent_window=4)
    structure = StructureFeatureTower(TrainingConfig().structure_config())
    hydrate_deterministic_state(
        snapshot=snapshot,
        stats=stats,
        candidate_prior=None,
        target_window=hydrated,
        structure=structure,
    )
    queries = [TestQuery(src=1, time=90, candidates=(10, 20, 40))]

    np.testing.assert_allclose(
        hydrated.features_for_queries(queries),
        independent.features_for_queries(queries),
        rtol=1e-6,
        atol=1e-6,
    )


def test_target_window_query_array_fast_path_matches_list_path():
    tower = TargetWindowTower(TargetWindowConfig(window_fractions=(0.1, 0.3, 0.6, 1.0)))
    tower.fit(_interactions())
    queries = [
        TestQuery(src=1, time=90, candidates=(10, 20, 40)),
        TestQuery(src=2, time=50, candidates=(10, 30, 999)),
    ]

    np.testing.assert_allclose(
        tower.features_for_query_array(TestQueryArray.from_queries(queries)),
        tower.features_for_queries(queries),
        rtol=1e-6,
        atol=1e-6,
    )


def test_feature_masks_and_selected_config_include_target_window():
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

    stats_end = len(STAT_FEATURE_NAMES)
    prior_end = stats_end + len(CANDIDATE_PRIOR_FEATURE_NAMES)
    target_end = prior_end + len(TARGET_WINDOW_FEATURE_NAMES)
    structure_end = target_end + len(STRUCTURE_FEATURE_NAMES)
    config = TrainingConfig(target_window_enabled=True, structure_enabled=True, source_profile_enabled=True)

    prior_config = _config_for_selected_features(config, tuple(range(prior_end)))
    target_config = _config_for_selected_features(config, tuple(range(target_end)))
    structure_config = _config_for_selected_features(config, tuple(range(structure_end)))
    structure_no_target_config = _config_for_selected_features(
        config,
        dict(masks)["stats_prior_structure"],
    )

    assert not prior_config.target_window_enabled
    assert not prior_config.structure_enabled
    assert target_config.target_window_enabled
    assert not target_config.structure_enabled
    assert structure_config.target_window_enabled
    assert structure_config.structure_enabled
    assert not structure_no_target_config.target_window_enabled
    assert structure_no_target_config.structure_enabled
