import importlib
import math
import sys
from collections import Counter

import numpy as np
import pytest

from jgrec.core.types import Interaction, InteractionTable, TestQuery, TestQueryArray
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES
from jgrec.rankers.hybrid.config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SOURCE_PROFILE_FEATURE_NAMES,
    TARGET_WINDOW_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
    SourceProfileConfig,
    StructureTowerConfig,
    TrainingConfig,
)
from jgrec.rankers.hybrid.encoder_cache import HybridPrefixStateCache, hydrate_deterministic_state
from jgrec.rankers.hybrid.ranker import _config_for_selected_features, _feature_masks
from jgrec.rankers.hybrid.source_profile import SourceProfileTower
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES, TemporalStats
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES, StructureFeatureTower

FEATURE = {name: idx for idx, name in enumerate(SOURCE_PROFILE_FEATURE_NAMES)}


def _interactions() -> InteractionTable:
    return InteractionTable.from_events([
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
        Interaction(src=2, dst=20, time=30),
        Interaction(src=2, dst=30, time=40),
        Interaction(src=3, dst=10, time=50),
        Interaction(src=3, dst=30, time=60),
        Interaction(src=1, dst=30, time=70),
        Interaction(src=4, dst=40, time=80),
        Interaction(src=4, dst=20, time=90),
    ])


def _require_jittor() -> None:
    pytest.importorskip("jittor")


def test_source_profile_scores_have_expected_shape_and_finite_values():
    interactions = _interactions()
    tower = SourceProfileTower(
        id_map=NodeIdMap.from_interactions(interactions),
        config=SourceProfileConfig(item2vec_enabled=False, window_size=4, recent_k=2),
    )

    tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    scores = tower.scores_for_queries([
        TestQuery(src=1, time=100, candidates=(10, 20, 30, 999)),
        TestQuery(src=99, time=100, candidates=(10, 20, 30, 999)),
    ])

    assert scores.shape == (2, 4, len(SOURCE_PROFILE_FEATURE_NAMES))
    assert np.all(np.isfinite(scores))
    assert np.any(scores[0, :, :6] != 0.0)
    assert np.all(scores[1] == 0.0)


def test_source_profile_deterministic_features_match_manual_counts():
    interactions = InteractionTable.from_events([
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
        Interaction(src=2, dst=20, time=30),
        Interaction(src=2, dst=30, time=40),
    ])
    tower = SourceProfileTower(
        id_map=NodeIdMap.from_interactions(interactions),
        config=SourceProfileConfig(item2vec_enabled=False, window_size=4, recent_k=1),
    )
    tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)

    features = tower.scores_for_queries([TestQuery(src=1, time=50, candidates=(30,))])[0, 0]

    expected_cosine = 1.0 / math.sqrt(2.0 * 1.0)
    assert features[FEATURE["source_profile_cooccur_sum"]] == np.float32(math.log1p(1))
    assert features[FEATURE["source_profile_cooccur_max"]] == np.float32(math.log1p(1))
    assert features[FEATURE["source_profile_cosine_sum"]] == np.float32(expected_cosine)
    assert features[FEATURE["source_profile_cosine_max"]] == np.float32(expected_cosine)
    assert features[FEATURE["source_profile_recent_cosine_sum"]] == np.float32(expected_cosine)
    assert features[FEATURE["source_profile_recent_cosine_max"]] == np.float32(expected_cosine)


def test_source_profile_recent_features_use_recent_k_history():
    interactions = InteractionTable.from_events([
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
        Interaction(src=2, dst=10, time=30),
        Interaction(src=2, dst=30, time=40),
        Interaction(src=3, dst=20, time=50),
        Interaction(src=3, dst=40, time=60),
    ])
    tower = SourceProfileTower(
        id_map=NodeIdMap.from_interactions(interactions),
        config=SourceProfileConfig(item2vec_enabled=False, window_size=4, recent_k=1),
    )
    tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)

    features = tower.scores_for_queries([TestQuery(src=1, time=70, candidates=(30, 40))])[0]

    assert features[0, FEATURE["source_profile_cosine_sum"]] > 0.0
    assert features[0, FEATURE["source_profile_recent_cosine_sum"]] == 0.0
    assert features[1, FEATURE["source_profile_cosine_sum"]] > 0.0
    assert features[1, FEATURE["source_profile_recent_cosine_sum"]] > 0.0


def test_source_profile_item2vec_outputs_signal_and_score_batch_size_is_stable():
    _require_jittor()
    interactions = _interactions()
    tower = SourceProfileTower(
        id_map=NodeIdMap.from_interactions(interactions),
        config=SourceProfileConfig(
            deterministic_enabled=False,
            item2vec_enabled=True,
            embedding_dim=8,
            epochs=1,
            batch_size=4,
            score_batch_size=8,
            max_samples=16,
            window_size=4,
        ),
    )
    tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    queries = [
        TestQuery(src=1, time=100, candidates=(10, 20, 30)),
        TestQuery(src=2, time=100, candidates=(10, 30, 999)),
    ]
    full_scores = tower.scores_for_queries(queries)
    tower.config = SourceProfileConfig(
        deterministic_enabled=False,
        item2vec_enabled=True,
        embedding_dim=8,
        epochs=1,
        batch_size=4,
        score_batch_size=1,
        max_samples=16,
        window_size=4,
    )
    small_batch_scores = tower.scores_for_queries(queries)

    assert np.any(full_scores[:, :, 6:] != 0.0)
    np.testing.assert_allclose(small_batch_scores, full_scores, rtol=1e-6, atol=1e-6)


def test_disabled_source_profile_uses_zero_features_without_importing_module():
    interactions = _interactions()
    config = TrainingConfig(
        gnn_enabled=False,
        seq_enabled=False,
        two_tower_enabled=False,
        source_profile_enabled=False,
    )
    sys.modules.pop("jgrec.rankers.hybrid.source_profile", None)
    ranker_module = importlib.import_module("jgrec.rankers.hybrid.ranker")

    encoder = ranker_module.HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(interactions),
        recent_window=4,
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
        two_tower_config=config.two_tower_config(),
        source_profile_config=config.source_profile_config(),
    )
    encoder.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    features = encoder.features_for_queries([TestQuery(src=1, time=100, candidates=(10, 20))])

    profile_start = (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
    )
    profile_end = profile_start + len(SOURCE_PROFILE_FEATURE_NAMES)
    assert "jgrec.rankers.hybrid.source_profile" not in sys.modules
    assert np.all(features[:, :, profile_start:profile_end] == 0.0)


def test_source_profile_branch_disables_zero_their_feature_halves():
    interactions = _interactions()
    deterministic_only = SourceProfileTower(
        id_map=NodeIdMap.from_interactions(interactions),
        config=SourceProfileConfig(item2vec_enabled=False),
    )
    deterministic_only.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    deterministic_scores = deterministic_only.scores_for_queries([TestQuery(src=1, time=100, candidates=(10, 20, 30))])

    assert np.any(deterministic_scores[:, :, :6] != 0.0)
    assert np.all(deterministic_scores[:, :, 6:] == 0.0)
    _require_jittor()
    item2vec_only = SourceProfileTower(
        id_map=NodeIdMap.from_interactions(interactions),
        config=SourceProfileConfig(
            deterministic_enabled=False,
            item2vec_enabled=True,
            embedding_dim=8,
            epochs=1,
            batch_size=4,
            max_samples=16,
        ),
    )
    item2vec_only.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    item2vec_scores = item2vec_only.scores_for_queries([TestQuery(src=1, time=100, candidates=(10, 20, 30))])

    assert np.all(item2vec_scores[:, :, :6] == 0.0)
    assert np.any(item2vec_scores[:, :, 6:] != 0.0)


def test_encoder_cache_hydrates_source_profile_deterministic_state():
    interactions = _interactions()
    config = SourceProfileConfig(item2vec_enabled=False, window_size=4, recent_k=2)
    snapshot = HybridPrefixStateCache(
        interactions,
        recent_window=4,
        candidate_prior_config=TrainingConfig().candidate_prior_config(),
        structure_config=StructureTowerConfig(future_only_transition_cooccur=True),
        source_profile_config=config,
        test_candidate_counts=Counter({30: 2}),
        verbose=False,
    ).snapshot_for_prefix(len(interactions))

    independent = SourceProfileTower(NodeIdMap.from_interactions(interactions), config)
    independent.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    hydrated = SourceProfileTower(NodeIdMap.from_interactions(interactions), config)
    stats = TemporalStats(recent_window=4)
    prior = None
    structure = StructureFeatureTower(StructureTowerConfig(future_only_transition_cooccur=True))
    hydrate_deterministic_state(
        snapshot=snapshot,
        stats=stats,
        candidate_prior=prior,
        structure=structure,
        source_profile=hydrated,
    )
    hydrated.index = structure.index
    queries = [TestQuery(src=1, time=100, candidates=(10, 20, 30))]

    np.testing.assert_allclose(
        hydrated.scores_for_queries(queries)[:, :, :6],
        independent.scores_for_queries(queries)[:, :, :6],
        rtol=1e-6,
        atol=1e-6,
    )


def test_feature_masks_and_selected_config_include_source_profile():
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

    stats_end = len(STAT_FEATURE_NAMES)
    prior_end = stats_end + len(CANDIDATE_PRIOR_FEATURE_NAMES)
    target_end = prior_end + len(TARGET_WINDOW_FEATURE_NAMES)
    structure_end = target_end + len(STRUCTURE_FEATURE_NAMES)
    profile_end = structure_end + len(SOURCE_PROFILE_FEATURE_NAMES)
    config = TrainingConfig(source_profile_enabled=True, two_tower_enabled=True, gnn_enabled=True, seq_enabled=True)

    structure_config = _config_for_selected_features(config, tuple(range(structure_end)))
    profile_config = _config_for_selected_features(config, tuple(range(profile_end)))
    tower_no_profile_config = _config_for_selected_features(config, dict(masks)["stats_prior_structure_tower"])

    assert not structure_config.source_profile_enabled
    assert not structure_config.two_tower_enabled
    assert profile_config.source_profile_enabled
    assert not profile_config.two_tower_enabled
    assert not tower_no_profile_config.source_profile_enabled
    assert tower_no_profile_config.two_tower_enabled


def test_source_profile_query_array_fast_path_matches_list_path():
    interactions = _interactions()
    tower = SourceProfileTower(
        id_map=NodeIdMap.from_interactions(interactions),
        config=SourceProfileConfig(item2vec_enabled=False, window_size=4, recent_k=2),
    )
    tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    queries = [
        TestQuery(src=1, time=100, candidates=(10, 20, 30)),
        TestQuery(src=2, time=100, candidates=(10, 30, 40)),
    ]
    query_array = TestQueryArray.from_queries(queries)

    np.testing.assert_allclose(
        tower.scores_for_query_array(query_array),
        tower.scores_for_queries(queries),
        rtol=1e-6,
        atol=1e-6,
    )
