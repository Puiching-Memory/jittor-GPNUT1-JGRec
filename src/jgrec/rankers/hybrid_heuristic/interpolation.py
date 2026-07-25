"""Base3 风格的无训练插值兜底打分器。

EdgeBank（边重现记忆）+ PopTrack（节点流行度）+ tCoMem（时序共现记忆）
的凸组合插值。完全无梯度训练，仅通过小规模网格搜索在验证集上拟合
(α, β, γ)。用途：

1. 每次提交前的 sanity floor —— 任何融合模型都不应低于它；
2. 融合训练异常时的应急提交；
3. 作为融合层的一个参考特征来源（可选）。

参考：Base3 (Kondrup, 2025) 与 On the Power of Heuristics in Temporal
Graphs (King AI Labs, 2025)。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray


@dataclass(frozen=True)
class InterpolationWeights:
    alpha: float = 1.0  # EdgeBank（局部边重现）
    beta: float = 0.0  # PopTrack（全局流行度）
    gamma: float = 0.0  # tCoMem（时序共现）


class InterpolationScorer:
    """对每行候选输出插值分数；分数越高排名越靠前。"""

    def __init__(self, weights: InterpolationWeights | None = None) -> None:
        self.weights = weights or InterpolationWeights()
        self.min_time = 0
        self.max_time = 0
        self.graph_span = 1
        self.total_edges = 0
        # EdgeBank: (src, dst) -> 最近交互时间 + 次数
        self._pair_last: dict[tuple[int, int], int] = {}
        self._pair_count: Counter[tuple[int, int]] = Counter()
        # PopTrack: dst -> 总入边次数
        self._dst_count: Counter[int] = Counter()
        # tCoMem: (src, dst) -> 衰减共现强度（基于 src 出边序列的时序共现）
        self._comem: dict[tuple[int, int], float] = {}

    def fit(self, interactions: InteractionTable) -> None:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")
        interactions = interactions.sort_by_time()
        self.min_time = int(interactions.time[0])
        self.max_time = int(interactions.time[-1])
        self.graph_span = max(self.max_time - self.min_time, 1)
        self.total_edges = len(interactions)

        src = interactions.src.astype(np.int64, copy=False)
        dst = interactions.dst.astype(np.int64, copy=False)
        times = interactions.time.astype(np.int64, copy=False)
        for s, d, t in zip(src, dst, times, strict=True):
            key = (int(s), int(d))
            self._pair_last[key] = int(t)
            self._pair_count[key] += 1
            self._dst_count[int(d)] += 1

        # tCoMem：对每个 src 的出边序列，统计 (历史项, 目标) 的时间衰减共现
        by_src: dict[int, list[tuple[int, int]]] = {}
        for s, d, t in zip(src, dst, times, strict=True):
            by_src.setdefault(int(s), []).append((int(d), int(t)))
        decay = max(self.graph_span * 0.2, 1.0)
        for s, seq in by_src.items():
            seen: dict[int, int] = {}
            for d, t in seq:
                for prev_d, prev_t in seen.items():
                    w = math.exp(-max(t - prev_t, 0) / decay)
                    key = (s, d)
                    self._comem[key] = self._comem.get(key, 0.0) + w
                seen[d] = t
                if len(seen) > 128:
                    # 保留最近 128 个，控制内存
                    oldest = min(seen, key=lambda k: seen[k])
                    del seen[oldest]

    def scores_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if isinstance(queries, TestQueryArray):
            query_list = list(queries)
        else:
            query_list = queries
        if not query_list:
            return np.empty((0, 0), dtype=np.float32)
        n_cand = len(query_list[0].candidates)
        out = np.zeros((len(query_list), n_cand), dtype=np.float32)
        for i, q in enumerate(query_list):
            out[i] = self._score_row(int(q.src), int(q.time), [int(c) for c in q.candidates])
        return out

    def _score_row(self, src: int, qt: int, candidates: list[int]) -> np.ndarray:
        n = len(candidates)
        edge = np.zeros(n, dtype=np.float64)
        pop = np.zeros(n, dtype=np.float64)
        comem = np.zeros(n, dtype=np.float64)
        for j, d in enumerate(candidates):
            key = (src, d)
            last = self._pair_last.get(key)
            if last is not None:
                recency = math.exp(-max(qt - last, 0) / self.graph_span)
                freq = math.log1p(self._pair_count[key])
                edge[j] = 0.7 * recency + 0.3 * freq
            pop[j] = math.log1p(self._dst_count.get(d, 0))
            comem[j] = math.log1p(self._comem.get(key, 0.0))
        # 行内归一到 [0,1]，避免量纲主导
        edge = _row_norm(edge)
        pop = _row_norm(pop)
        comem = _row_norm(comem)
        w = self.weights
        return (w.alpha * edge + w.beta * pop + w.gamma * comem).astype(np.float32)


def _row_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def fit_weights_on_validation(
    scorer: InterpolationScorer,
    val_queries: TestQueryArray | list[TestQuery],
    val_targets: np.ndarray,
    grid_step: float = 0.25,
) -> InterpolationWeights:
    """在验证集上网格搜索 (α,β,γ) 使 MRR 最大。

    ``val_targets`` 为每行正例在候选中的列索引（shape=(n_rows,)）。
    """
    if isinstance(val_queries, TestQueryArray):
        query_list = list(val_queries)
    else:
        query_list = val_queries
    if not query_list:
        return scorer.weights

    steps = np.arange(0.0, 1.0 + 1e-9, grid_step)
    best_w = scorer.weights
    best_mrr = -1.0
    for alpha in steps:
        for beta in steps:
            for gamma in steps:
                if alpha + beta + gamma <= 0:
                    continue
                scorer.weights = InterpolationWeights(alpha, beta, gamma)
                scores = scorer.scores_for_queries(query_list)
                mrr = _mrr(scores, val_targets)
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_w = scorer.weights
    scorer.weights = best_w
    return best_w


def _mrr(scores: np.ndarray, targets: np.ndarray) -> float:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    cols = np.arange(scores.shape[1])[None, :]
    ranks[rows, order] = cols + 1
    target_ranks = ranks[rows[:, 0], targets]
    return float(np.mean(1.0 / target_ranks))
