import importlib
import sys
from dataclasses import replace

import numpy as np

from jgrec.core.types import Interaction, InteractionTable, TestQuery
from jgrec.idmap import NodeIdMap
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex
from jgrec.rankers.hybrid.candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES
from jgrec.rankers.hybrid.config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
    TrainingConfig,
    TwoTowerConfig,
)
from jgrec.rankers.hybrid.ranker import (
    _config_for_selected_features,
    _feature_masks,
)
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES


def _interactions() -> list[Interaction]:
    return [
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
        Interaction(src=2, dst=20, time=30),
        Interaction(src=2, dst=30, time=40),
        Interaction(src=3, dst=10, time=50),
        Interaction(src=3, dst=30, time=60),
        Interaction(src=1, dst=30, time=70),
        Interaction(src=2, dst=10, time=80),
        Interaction(src=4, dst=40, time=90),
        Interaction(src=4, dst=20, time=100),
    ]


def test_two_tower_scores_have_expected_shape_and_signal():
    from jgrec.rankers.hybrid.two_tower import TwoTower

    interactions = _interactions()
    interaction_table = InteractionTable.from_events(interactions)
    tower = TwoTower(
        id_map=NodeIdMap.from_interactions(interaction_table),
        config=TwoTowerConfig(
            embedding_dim=8,
            hidden_dim=8,
            epochs=1,
            batch_size=4,
            max_samples=8,
            num_negatives=2,
        ),
    )

    tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    scores = tower.scores_for_queries(
        [
            TestQuery(src=1, time=110, candidates=(10, 20, 40)),
            TestQuery(src=2, time=110, candidates=(10, 30, 50)),
        ]
    )

    assert scores.shape == (2, 3, len(TWO_TOWER_FEATURE_NAMES))
    assert np.all(np.isfinite(scores))
    assert np.any(scores != 0.0)
    assert np.all(scores[:, :, 1] >= -1.0001)
    assert np.all(scores[:, :, 1] <= 1.0001)
    assert scores[1, 2, 0] == 0.0
    assert scores[1, 2, 1] == 0.0


def test_two_tower_scoring_batch_size_preserves_scores():
    from jgrec.rankers.hybrid.two_tower import TwoTower

    interactions = _interactions()
    interaction_table = InteractionTable.from_events(interactions)
    queries = [
        TestQuery(src=1, time=110, candidates=(10, 20, 40)),
        TestQuery(src=2, time=110, candidates=(10, 30, 50)),
        TestQuery(src=4, time=110, candidates=(20, 30, 40)),
    ]
    common_config = dict(
        embedding_dim=8,
        hidden_dim=8,
        epochs=1,
        batch_size=4,
        max_samples=8,
        num_negatives=2,
    )
    tower = TwoTower(
        id_map=NodeIdMap.from_interactions(interaction_table),
        config=TwoTowerConfig(score_batch_size=32, **common_config),
    )
    tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    full_scores = tower.scores_for_queries(queries)
    tower.config = replace(tower.config, score_batch_size=1)

    np.testing.assert_allclose(
        tower.scores_for_queries(queries),
        full_scores,
        rtol=1e-5,
        atol=1e-5,
    )


def test_two_tower_reuses_future_only_structure_index():
    from jgrec.rankers.hybrid.two_tower import TwoTower

    interactions = _interactions()
    interaction_table = InteractionTable.from_events(interactions)
    shared_index = TemporalInteractionIndex()
    shared_index.fit(
        interaction_table,
        build_transitions=True,
        build_cooccurs=True,
        future_only_transition_cooccur=True,
    )
    tower = TwoTower(
        id_map=NodeIdMap.from_interactions(interaction_table),
        config=TwoTowerConfig(
            embedding_dim=8,
            hidden_dim=8,
            epochs=1,
            batch_size=4,
            max_samples=8,
            num_negatives=2,
        ),
    )

    tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False, shared_index=shared_index)
    scores = tower.scores_for_queries([TestQuery(src=1, time=110, candidates=(10, 20, 40))])

    assert tower.index is shared_index
    assert shared_index.future_only
    assert shared_index.transition_times == {}
    assert shared_index.cooccur_times == {}
    assert scores.shape == (1, 3, len(TWO_TOWER_FEATURE_NAMES))
    assert np.any(scores != 0.0)


def test_disabled_two_tower_uses_zero_features_without_importing_tower_module():
    interactions = _interactions()
    interaction_table = InteractionTable.from_events(interactions)
    config = TrainingConfig(gnn_enabled=False, seq_enabled=False, two_tower_enabled=False)
    sys.modules.pop("jgrec.rankers.hybrid.two_tower", None)
    ranker_module = importlib.import_module("jgrec.rankers.hybrid.ranker")

    encoder = ranker_module.HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(interaction_table),
        recent_window=4,
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
        two_tower_config=config.two_tower_config(),
    )
    encoder.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    features = encoder.features_for_queries([TestQuery(src=1, time=110, candidates=(10, 20))])

    tower_start = len(STAT_FEATURE_NAMES) + len(CANDIDATE_PRIOR_FEATURE_NAMES) + len(STRUCTURE_FEATURE_NAMES)
    tower_end = tower_start + len(TWO_TOWER_FEATURE_NAMES)
    assert "jgrec.rankers.hybrid.two_tower" not in sys.modules
    assert features.shape[-1] == (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
        + len(TWO_TOWER_FEATURE_NAMES)
        + len(GRAPH_WINDOW_NAMES)
        + len(SEQUENCE_FEATURE_NAMES)
    )
    assert np.all(features[:, :, tower_start:tower_end] == 0.0)


def test_hybrid_feature_masks_include_two_tower_groups():
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


def test_selected_feature_config_disables_unused_two_tower():
    stats_end = len(STAT_FEATURE_NAMES)
    prior_end = stats_end + len(CANDIDATE_PRIOR_FEATURE_NAMES)
    structure_end = prior_end + len(STRUCTURE_FEATURE_NAMES)
    tower_end = structure_end + len(TWO_TOWER_FEATURE_NAMES)
    config = TrainingConfig(two_tower_enabled=True, gnn_enabled=True, seq_enabled=True)

    stats_config = _config_for_selected_features(config, tuple(range(stats_end)))
    prior_config = _config_for_selected_features(config, tuple(range(prior_end)))
    tower_config = _config_for_selected_features(config, tuple(range(tower_end)))

    assert not stats_config.candidate_prior_enabled
    assert not stats_config.two_tower_enabled
    assert not stats_config.gnn_enabled
    assert not stats_config.seq_enabled
    assert prior_config.candidate_prior_enabled
    assert not prior_config.two_tower_enabled
    assert tower_config.two_tower_enabled
    assert not tower_config.gnn_enabled
    assert not tower_config.seq_enabled
