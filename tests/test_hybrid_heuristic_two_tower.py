import numpy as np

from jgrec.core.types import Interaction, InteractionTable, TestQuery
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid_heuristic.config import TwoTowerConfig
from jgrec.rankers.hybrid_heuristic.two_tower import (
    TWO_TOWER_FEATURE_NAMES,
    TwoTower,
    _precompute_tower_batches,
    _TowerTrainingContext,
)


def _interactions(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return InteractionTable.from_events(
        [
            Interaction(src=int(rng.integers(0, 40)), dst=int(rng.integers(0, 200)), time=int(t))
            for t in range(n)
        ]
    )


def test_precompute_matches_training_batches_shape():
    iv = _interactions()
    id_map = NodeIdMap.from_interactions(iv)
    config = TwoTowerConfig(num_negatives=7, max_samples=400)
    tower = TwoTower(id_map=id_map, config=config)
    tower.index.fit(iv, build_transitions=True, build_cooccurs=True)
    ctx = _TowerTrainingContext.from_interactions(iv, id_map, tower.index)
    seeds = np.arange(len(iv), dtype=np.uint32)
    pre = _precompute_tower_batches(iv, seeds, ctx, config)
    assert pre.src_ids.shape == (len(iv),)
    assert pre.dst_ids.shape == (len(iv), config.num_negatives + 1)
    assert pre.dst_popularity_buckets.shape == (len(iv), config.num_negatives + 1)
    # 正例总在第 0 列且为有效 dst
    assert np.all(pre.dst_ids[:, 0] >= 0)


def test_two_tower_fit_produces_scores():
    iv = _interactions()
    id_map = NodeIdMap.from_interactions(iv)
    config = TwoTowerConfig(epochs=2, max_samples=400, num_negatives=7, batch_size=64, early_stop_val_ratio=0.2)
    tower = TwoTower(id_map=id_map, config=config)
    tower.fit(iv, rng=np.random.default_rng(0), verbose=False)
    assert tower.model is not None
    scores = tower.scores_for_queries([TestQuery(src=1, time=10_000, candidates=(5, 6, 7))])
    assert scores.shape == (1, 3, len(TWO_TOWER_FEATURE_NAMES))
    assert np.isfinite(scores).all()


def test_two_tower_precompute_is_deterministic_for_same_seeds():
    iv = _interactions()
    id_map = NodeIdMap.from_interactions(iv)
    config = TwoTowerConfig(num_negatives=5, max_samples=400)
    tower = TwoTower(id_map=id_map, config=config)
    tower.index.fit(iv, build_transitions=True, build_cooccurs=True)
    ctx = _TowerTrainingContext.from_interactions(iv, id_map, tower.index)
    seeds = np.full(len(iv), 12345, dtype=np.uint32)
    a = _precompute_tower_batches(iv, seeds, ctx, config)
    b = _precompute_tower_batches(iv, seeds.copy(), ctx, config)
    np.testing.assert_array_equal(a.dst_ids, b.dst_ids)
    np.testing.assert_array_equal(a.src_ids, b.src_ids)
