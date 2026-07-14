import importlib
import sys

import numpy as np
import pytest

from jgrec.core.types import Interaction, InteractionTable, TestQuery
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES
from jgrec.rankers.hybrid.config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SOURCE_PROFILE_FEATURE_NAMES,
    TARGET_WINDOW_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
    SequenceTowerConfig,
    TrainingConfig,
)
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES


def _interactions() -> list[Interaction]:
    return [
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
        Interaction(src=1, dst=30, time=30),
        Interaction(src=2, dst=20, time=40),
        Interaction(src=2, dst=30, time=50),
        Interaction(src=2, dst=10, time=60),
        Interaction(src=3, dst=30, time=70),
        Interaction(src=3, dst=10, time=80),
        Interaction(src=3, dst=20, time=90),
    ]


def _require_jittor() -> None:
    pytest.importorskip("jittor")


def test_gru_sequence_scores_have_expected_shape_and_signal():
    _require_jittor()
    from jgrec.rankers.hybrid.sequence import SequenceTower  # noqa: PLC0415

    interactions = _interactions()
    interaction_table = InteractionTable.from_events(interactions)
    tower = SequenceTower(
        id_map=NodeIdMap.from_interactions(interaction_table),
        config=SequenceTowerConfig(
            epochs=1,
            batch_size=4,
            score_batch_size=1,
            max_samples=8,
            max_seq_len=4,
            hidden_size=8,
            layers=1,
            dropout=0.0,
        ),
    )

    tower.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    scores = tower.scores_for_queries(
        [
            TestQuery(src=1, time=100, candidates=(10, 20, 40)),
            TestQuery(src=2, time=100, candidates=(10, 30, 50)),
            TestQuery(src=99, time=100, candidates=(10, 20, 30)),
        ]
    )

    assert SEQUENCE_FEATURE_NAMES == ("gru_dot", "gru_cosine", "gru_decay_dot")
    assert scores.shape == (3, 3, len(SEQUENCE_FEATURE_NAMES))
    assert np.all(np.isfinite(scores))
    assert np.any(scores[:2, :2, 0] != 0.0)
    assert scores[0, 2, 0] == 0.0
    assert scores[1, 2, 0] == 0.0
    assert np.all(scores[2, :, 0] == 0.0)


def test_disabled_sequence_uses_zero_features_without_importing_sequence_module():
    interactions = _interactions()
    interaction_table = InteractionTable.from_events(interactions)
    config = TrainingConfig(gnn_enabled=False, seq_enabled=False, two_tower_enabled=False, source_profile_enabled=False)
    sys.modules.pop("jgrec.rankers.hybrid.sequence", None)
    ranker_module = importlib.import_module("jgrec.rankers.hybrid.ranker")

    encoder = ranker_module.HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(interaction_table),
        recent_window=4,
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
        two_tower_config=config.two_tower_config(),
    )
    encoder.fit(interaction_table, rng=np.random.default_rng(0), verbose=False)
    features = encoder.features_for_queries([TestQuery(src=1, time=100, candidates=(10, 20))])

    sequence_start = (
        len(STAT_FEATURE_NAMES)
        + len(CANDIDATE_PRIOR_FEATURE_NAMES)
        + len(TARGET_WINDOW_FEATURE_NAMES)
        + len(STRUCTURE_FEATURE_NAMES)
        + len(SOURCE_PROFILE_FEATURE_NAMES)
        + len(TWO_TOWER_FEATURE_NAMES)
        + len(GRAPH_WINDOW_NAMES)
    )
    assert "jgrec.rankers.hybrid.sequence" not in sys.modules
    assert features.shape[-1] == sequence_start + len(SEQUENCE_FEATURE_NAMES)
    assert np.all(features[:, :, sequence_start:] == 0.0)
