import math

import numpy as np

from jgrec.core.types import Interaction, InteractionTable
from jgrec.core.types import TestQuery as Query
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES, TemporalStats

FEATURE = {name: idx for idx, name in enumerate(STAT_FEATURE_NAMES)}


def _table(events: list[Interaction]) -> InteractionTable:
    return InteractionTable.from_events(events)


def test_temporal_stats_cutoff_ignores_future_pair_events():
    stats = TemporalStats(recent_window=4)
    stats.fit(
        _table([
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=20, time=20),
            Interaction(src=1, dst=10, time=30),
            Interaction(src=2, dst=10, time=40),
        ])
    )

    features = stats.features_for_queries([Query(src=1, time=25, candidates=(10, 20, 30))])[0]
    graph_span = 10

    assert features[0, FEATURE["pair_strength"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["repeat_rate"]] == np.float32(0.5)
    assert features[0, FEATURE["pair_recency"]] == np.float32(math.exp(-(25 - 10) / graph_span))
    assert features[0, FEATURE["dst_popularity"]] == np.float32(math.log1p(1) / math.log1p(2))
    assert features[0, FEATURE["dst_recency"]] == np.float32(math.exp(-(25 - 10) / graph_span))
    assert features[0, FEATURE["recent_hit"]] == np.float32(0.5)

    assert features[1, FEATURE["pair_strength"]] == np.float32(math.log1p(1))
    assert features[1, FEATURE["repeat_rate"]] == np.float32(0.5)
    assert features[1, FEATURE["pair_recency"]] == np.float32(math.exp(-(25 - 20) / graph_span))
    assert features[1, FEATURE["dst_popularity"]] == np.float32(math.log1p(1) / math.log1p(2))
    assert features[1, FEATURE["recent_hit"]] == np.float32(1.0)

    assert features[2, FEATURE["pair_strength"]] == 0.0
    assert features[2, FEATURE["dst_popularity"]] == 0.0
    assert np.all(features[:, FEATURE["src_activity"]] == np.float32(math.log1p(2) / math.log1p(2)))
    assert np.all(features[:, FEATURE["src_recency"]] == np.float32(math.exp(-(25 - 20) / graph_span)))


def test_temporal_stats_cutoff_excludes_events_at_query_time():
    stats = TemporalStats(recent_window=4)
    stats.fit(
        _table([
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=10, time=20),
        ])
    )

    features = stats.features_for_queries([Query(src=1, time=20, candidates=(10,))])[0]

    assert features[0, FEATURE["pair_strength"]] == np.float32(math.log1p(1))
    assert features[0, FEATURE["repeat_rate"]] == np.float32(1.0)
    assert features[0, FEATURE["pair_recency"]] == np.float32(math.exp(-(20 - 10)))
    assert features[0, FEATURE["dst_popularity"]] == np.float32(1.0)
    assert features[0, FEATURE["dst_recency"]] == np.float32(math.exp(-(20 - 10)))


def test_temporal_stats_uses_aggregate_fast_path_after_training_window():
    stats = TemporalStats(recent_window=4)
    stats.fit(
        _table([
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=20, time=20),
            Interaction(src=1, dst=10, time=30),
            Interaction(src=2, dst=10, time=40),
        ])
    )

    batch_features = stats.features_for_queries([Query(src=1, time=50, candidates=(10, 20))])[0]
    single_features = np.empty((2, len(STAT_FEATURE_NAMES)), dtype=np.float32)
    stats.fill_features(Query(src=1, time=50, candidates=(10, 20)), single_features)

    np.testing.assert_allclose(batch_features, single_features)
