import numpy as np
import pytest

from jgrec.core.types import Interaction, InteractionTable, TestQuery, TestQueryArray
from jgrec.rankers.hybrid_heuristic.config import HeuristicTowerConfig
from jgrec.rankers.hybrid_heuristic.heuristic import HEURISTIC_FEATURE_NAMES, HeuristicTower


def _names():
    return {name: idx for idx, name in enumerate(HEURISTIC_FEATURE_NAMES)}


def test_heuristic_local_recency_and_frequency():
    # src=1 与 dst=10 交互 2 次，dst=20 交互 0 次 -> 候选10的 LP/LR 更高
    iv = InteractionTable.from_events(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=10, time=20),
            Interaction(src=1, dst=30, time=30),
        ]
    )
    tower = HeuristicTower(HeuristicTowerConfig(enabled=True, hop2_enabled=False, cooccur_time_decay=False))
    tower.fit(iv)
    q = TestQuery(src=1, time=40, candidates=(10, 20))
    feats = tower.features_for_queries([q])[0]
    n = _names()
    assert feats[0, n["heur_lp_log"]] > feats[1, n["heur_lp_log"]]
    assert feats[0, n["heur_lr_freq"]] > feats[1, n["heur_lr_freq"]]
    assert feats[0, n["heur_lr_recency"]] > feats[1, n["heur_lr_recency"]]
    # dst=10 全局入边多于 dst=20
    assert feats[0, n["heur_gp_log"]] > feats[1, n["heur_gp_log"]]
    # combined 候选10更高
    assert feats[0, n["heur_combined"]] > feats[1, n["heur_combined"]]


def test_heuristic_respects_time_cutoff():
    # 查询时间早于部分交互时，应只看到 cutoff 之前的历史
    iv = InteractionTable.from_events(
        [
            Interaction(src=1, dst=10, time=10),
            Interaction(src=1, dst=10, time=90),
        ]
    )
    tower = HeuristicTower(HeuristicTowerConfig(enabled=True, hop2_enabled=False, cooccur_time_decay=False))
    tower.fit(iv)
    q = TestQuery(src=1, time=50, candidates=(10,))
    feats = tower.features_for_queries([q])[0]
    n = _names()
    # 只看到 time=10 那一次
    assert feats[0, n["heur_lp_log"]] == pytest.approx(float(np.log1p(1)), rel=1e-5)


def test_heuristic_common_neighbor_window():
    # src=1 近期与 bridge=5 交互；bridge=5 也与 dst=77 交互 -> 候选77有共同邻居
    # 拉大时间跨度使短窗足够宽
    iv = InteractionTable.from_events(
        [
            Interaction(src=9, dst=1, time=0),
            Interaction(src=1, dst=5, time=1000),
            Interaction(src=5, dst=77, time=1001),
            Interaction(src=5, dst=88, time=1002),
        ]
    )
    tower = HeuristicTower(HeuristicTowerConfig(enabled=True, hop2_enabled=False, cooccur_time_decay=False))
    tower.fit(iv)
    q = TestQuery(src=1, time=1003, candidates=(77, 42))
    feats = tower.features_for_queries([q])[0]
    n = _names()
    assert feats[0, n["heur_cn_long"]] > feats[1, n["heur_cn_long"]]
    assert feats[0, n["heur_cn_decay"]] > feats[1, n["heur_cn_decay"]]


def test_heuristic_directional_cooccur():
    # src 序列 10->20->10->20，候选20正序共现应>=逆序
    iv = InteractionTable.from_events(
        [
            Interaction(src=1, dst=10, time=1),
            Interaction(src=1, dst=20, time=2),
            Interaction(src=1, dst=10, time=3),
            Interaction(src=1, dst=20, time=4),
        ]
    )
    tower = HeuristicTower(HeuristicTowerConfig(enabled=True, hop2_enabled=False, cooccur_time_decay=True))
    tower.fit(iv)
    q = TestQuery(src=1, time=5, candidates=(20,))
    feats = tower.features_for_queries([q])[0]
    n = _names()
    assert feats[0, n["heur_cooccur_fwd"]] > 0.0


def test_heuristic_disabled_returns_zeros():
    from jgrec.rankers.hybrid_heuristic.ranker import _DisabledHeuristicTower  # noqa: PLC0415

    tower = _DisabledHeuristicTower()
    q = TestQueryArray.from_queries([TestQuery(src=1, time=5, candidates=(1, 2))])
    feats = tower.features_for_query_array(q)
    assert feats.shape == (1, 2, len(HEURISTIC_FEATURE_NAMES))
    assert np.all(feats == 0.0)
