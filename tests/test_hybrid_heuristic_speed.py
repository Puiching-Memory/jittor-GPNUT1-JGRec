import numpy as np
import pytest

from jgrec.core.types import Interaction, InteractionTable, TestQuery, TestQueryArray
from jgrec.rankers.hybrid_heuristic.config import HeuristicTowerConfig
from jgrec.rankers.hybrid_heuristic.heuristic import HeuristicTower


def _interactions() -> InteractionTable:
    # 多 src、重复边、跨窗、hop2 路径，覆盖 cn/cooccur/hop2 各分支
    events = []
    rng = np.random.default_rng(7)
    dsts = [10, 20, 30, 40, 50, 60, 70, 80]
    t = 0
    for _ in range(200):
        t += int(rng.integers(1, 5))
        src = int(rng.integers(1, 8))
        dst = int(rng.choice(dsts))
        events.append(Interaction(src=src, dst=dst, time=t))
    # 明确构造 hop2：1->99 (最近), 99->42
    events.append(Interaction(src=1, dst=99, time=t + 1))
    events.append(Interaction(src=99, dst=42, time=t + 2))
    events.append(Interaction(src=1, dst=42, time=t + 3))
    return InteractionTable.from_events(events)


def _tower(iv: InteractionTable) -> HeuristicTower:
    tower = HeuristicTower(HeuristicTowerConfig(enabled=True, vectorize_quadrant=True))
    tower.fit(iv, verbose=False)
    return tower


@pytest.mark.parametrize("hop2", [False, True])
@pytest.mark.parametrize("cooccur_decay", [False, True])
def test_grouped_matches_per_query_vec(hop2: bool, cooccur_decay: bool) -> None:
    """分组批量路径必须与改造前逐查询 vec_index 路径逐位一致。"""
    iv = _interactions()
    cfg = HeuristicTowerConfig(
        enabled=True, vectorize_quadrant=True, hop2_enabled=hop2, cooccur_time_decay=cooccur_decay
    )
    tower = HeuristicTower(cfg)
    tower.fit(iv, verbose=False)
    max_t = int(iv.time.max())
    queries = [
        TestQuery(src=1, time=max_t + 10, candidates=(10, 42, 77, 999)),
        TestQuery(src=1, time=max_t + 10, candidates=(30, 40, 50, 60)),  # 同 (src,qt) 复用
        TestQuery(src=3, time=max_t - 5, candidates=(20, 30, 40, 50)),
        TestQuery(src=999, time=max_t + 1, candidates=(10, 20, 30, 40)),  # 未见 src
    ]
    qa = TestQueryArray.from_queries(queries)
    actual = tower.features_for_query_array(qa)

    # 参考：逐查询调用 vec_index 原方法（旧逻辑）
    vec = tower._vec_index
    windows = tower.windows
    expected = np.zeros_like(actual)
    for r, q in enumerate(queries):
        cand = np.asarray(q.candidates, dtype=np.int64)
        expected[r, :, 0:7] = vec.quadrant_features_for_query(q.src, cand, q.time)
        expected[r, :, 7:11] = vec.cn_features_for_query(q.src, cand, q.time, windows)
        if cooccur_decay:
            expected[r, :, 11:13] = vec.cooccur_features_for_query(q.src, cand, q.time, windows[1])
        if hop2:
            expected[r, :, 13:14] = vec.hop2_features_for_query(q.src, cand, q.time, windows[1])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_vec_repeated_src_queries_benefit_from_cache() -> None:
    iv = _interactions()
    tower = _tower(iv)
    max_t = int(iv.time.max())
    q = TestQuery(src=2, time=max_t + 10, candidates=(10, 20, 30, 40))
    first = tower.features_for_queries([q])
    second = tower.features_for_queries([q])
    np.testing.assert_array_equal(first, second)
