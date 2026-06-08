from collections import Counter

import numpy as np

from jgrec.core.types import Interaction, InteractionTable, TestQuery
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.candidate_prior import CANDIDATE_PRIOR_FEATURE_NAMES, CandidatePriorTower
from jgrec.rankers.hybrid.config import CandidatePriorConfig, TrainingConfig
from jgrec.rankers.hybrid.stats import STAT_FEATURE_DIM, STAT_FEATURE_NAMES


def test_candidate_prior_features_cover_seen_unseen_and_row_ranks():
    tower = CandidatePriorTower(CandidatePriorConfig(enabled=True))
    tower.fit(
        InteractionTable.from_events([
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=20, time=20),
        ]),
        Counter({999: 50, 20: 25, 10: 5}),
    )
    queries = [TestQuery(src=1, time=30, candidates=(999, 20, 10))]
    stat_features = np.zeros((1, 3, STAT_FEATURE_DIM), dtype=np.float32)
    pop_idx = STAT_FEATURE_NAMES.index("dst_popularity")
    recency_idx = STAT_FEATURE_NAMES.index("dst_recency")
    stat_features[0, :, pop_idx] = [0.1, 0.9, 0.2]
    stat_features[0, :, recency_idx] = [0.5, 0.4, 0.9]

    features = tower.features_for_queries(queries, stat_features)[0]
    names = {name: idx for idx, name in enumerate(CANDIDATE_PRIOR_FEATURE_NAMES)}

    assert features[0, names["candidate_train_seen"]] == 0.0
    assert features[1, names["candidate_train_seen"]] == 1.0
    assert features[0, names["candidate_test_freq"]] > features[1, names["candidate_test_freq"]]
    assert features[0, names["candidate_unseen_test_freq"]] > 0.0
    assert features[1, names["candidate_unseen_test_freq"]] == 0.0
    assert features[1, names["candidate_dst_pop_row_rank"]] == 1.0
    assert features[2, names["candidate_dst_recency_row_rank"]] == 1.0
    assert features[0, names["candidate_test_freq_row_rank"]] == 1.0


def test_disabled_candidate_prior_uses_zero_features_in_encoder():
    from jgrec.rankers.hybrid.ranker import HybridFeatureEncoder  # noqa: PLC0415

    interactions = [
        Interaction(src=1, dst=10, time=10),
        Interaction(src=1, dst=20, time=20),
    ]
    config = TrainingConfig(
        candidate_prior_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
        two_tower_enabled=False,
        source_profile_enabled=False,
    )
    encoder = HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(InteractionTable.from_events(interactions)),
        recent_window=4,
        candidate_prior_config=config.candidate_prior_config(),
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
        two_tower_config=config.two_tower_config(),
    )
    encoder.fit(InteractionTable.from_events(interactions), rng=np.random.default_rng(0), verbose=False)
    features = encoder.features_for_queries([TestQuery(src=1, time=30, candidates=(10, 20))])

    prior_start = len(STAT_FEATURE_NAMES)
    prior_end = prior_start + len(CANDIDATE_PRIOR_FEATURE_NAMES)
    assert features.shape[-1] > prior_end
    assert np.all(features[:, :, prior_start:prior_end] == 0.0)
