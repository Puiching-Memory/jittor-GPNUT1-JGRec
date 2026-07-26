from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from jgrec.core.types import InteractionTable, TestQuery, TestQueryArray
from jgrec.rankers.common.byte_budget_lru import ByteBudgetLRU
from jgrec.rankers.common.temporal_index import TemporalInteractionIndex

from .config import HEURISTIC_FEATURE_DIM, HEURISTIC_FEATURE_NAMES, HeuristicTowerConfig

# 四象限：local=src 出边记忆, global=dst 入边记忆
# LR  = local recency   (u,v) 最近一次交互的近期度
# LP  = local popularity (u,v) 交互次数
# GR  = global recency  v 最近交互时间
# GP  = global popularity v 总交互次数


class HeuristicTower:
    """成对启发式特征塔：四象限启发式 + 时间窗共同邻居 + 方向化衰减共现/2跳路径分。

    该塔复用 structure 塔的 ``TemporalInteractionIndex``（通过 ``shared_index`` 注入），
    不重复建图；特征维度固定为 ``HEURISTIC_FEATURE_DIM``。
    """

    def __init__(self, config: HeuristicTowerConfig | None = None) -> None:
        self.config = config or HeuristicTowerConfig()
        self.index = TemporalInteractionIndex()
        self.min_time = 0
        self.max_time = 0
        self.graph_span = 1
        self.total_edges = 0
        self.windows: tuple[float, float, float] = (1.0, 1.0, 1.0)
        cache_bytes = max(int(self.config.cache_max_bytes), 0)
        self._cn_cache: ByteBudgetLRU[int, tuple[np.ndarray, np.ndarray]] = ByteBudgetLRU(
            max_bytes=cache_bytes // 2,
            max_entries=256,
        )
        self._cn_decay_cache: ByteBudgetLRU[int, np.ndarray] = ByteBudgetLRU(
            max_bytes=cache_bytes // 2,
            max_entries=256,
        )
        self._vec_index: Any = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return HEURISTIC_FEATURE_NAMES

    def fit(
        self,
        interactions: InteractionTable,
        rng: np.random.Generator | None = None,
        verbose: bool = True,
        shared_index: TemporalInteractionIndex | None = None,
    ) -> None:
        if len(interactions) == 0:
            raise ValueError("training interactions are empty")
        interactions = interactions.sort_by_time()
        if shared_index is not None:
            self.index = shared_index
        else:
            self.index = TemporalInteractionIndex()
            self.index.fit(interactions, build_transitions=False, build_cooccurs=False)
        self.min_time = int(interactions.time[0])
        self.max_time = int(interactions.time[-1])
        self.graph_span = max(self.max_time - self.min_time, 1)
        self.total_edges = len(interactions)
        self.windows = (
            max(self.graph_span * self.config.window_short_ratio, 1.0),
            max(self.graph_span * self.config.window_medium_ratio, 1.0),
            max(self.graph_span * self.config.window_long_ratio, 1.0),
        )
        self._vec_index = None
        if self.config.vectorize_quadrant:
            from jgrec.rankers.common.vec_heuristic import VectorizedHeuristicIndex  # noqa: PLC0415

            self._vec_index = VectorizedHeuristicIndex()
            self._vec_index.fit(interactions)
        self._cn_cache.clear()
        self._cn_decay_cache.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "index": self.index.shallow_copy(),
            "min_time": self.min_time,
            "max_time": self.max_time,
            "graph_span": self.graph_span,
            "total_edges": self.total_edges,
            "windows": self.windows,
        }

    def hydrate(self, snapshot: dict[str, Any]) -> None:
        self.index = snapshot["index"].shallow_copy()
        self.min_time = int(snapshot["min_time"])
        self.max_time = int(snapshot["max_time"])
        self.graph_span = int(snapshot["graph_span"])
        self.total_edges = int(snapshot["total_edges"])
        self.windows = tuple(float(value) for value in snapshot["windows"])
        self._cn_cache.clear()
        self._cn_decay_cache.clear()

    def features_for_queries(self, queries: TestQueryArray | list[TestQuery]) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, HEURISTIC_FEATURE_DIM), dtype=np.float32)
        return self.features_for_query_array(TestQueryArray.from_queries(queries))

    def features_for_query_array(self, queries: TestQueryArray) -> np.ndarray:
        if not queries:
            return np.empty((0, 0, HEURISTIC_FEATURE_DIM), dtype=np.float32)
        features = np.zeros((len(queries), queries.candidate_count, HEURISTIC_FEATURE_DIM), dtype=np.float32)
        if self._vec_index is not None:
            for row_idx, query in enumerate(queries):
                src = int(query.src)
                qt = int(query.time)
                cand = np.asarray([int(c) for c in query.candidates], dtype=np.int64)
                features[row_idx, :, 0:7] = self._vec_index.quadrant_features_for_query(src, cand, qt)
                features[row_idx, :, 7:11] = self._vec_index.cn_features_for_query(src, cand, qt, self.windows)
                if self.config.cooccur_time_decay:
                    features[row_idx, :, 11:13] = self._vec_index.cooccur_features_for_query(src, cand, qt, self.windows[1])
                if self.config.hop2_enabled:
                    features[row_idx, :, 13:14] = self._vec_index.hop2_features_for_query(src, cand, qt, self.windows[1])
            return features
        for row_idx, query in enumerate(queries):
            self._fill_query_features(query, features[row_idx])
        return features

    # ------------------------------------------------------------------ 内部

    def _fill_query_features(self, query: TestQuery, output: np.ndarray) -> None:
        qt = int(query.time)
        src = int(query.src)
        n = len(query.candidates)
        cand = np.asarray([int(c) for c in query.candidates], dtype=np.int64)

        src_view = self.index.source_view(src, qt)
        src_dsts = src_view.visible_dsts  # 时间升序
        src_total = len(src_dsts)

        # src 侧局部记忆：候选 v 的 (u,v) 次数、最近一次时间
        lr_freq = np.zeros(n, dtype=np.float32)
        lr_recency = np.zeros(n, dtype=np.float32)
        lp_log = np.zeros(n, dtype=np.float32)
        # dst 侧全局记忆：候选 v 的总入边、最近入边时间
        gr_freq = np.zeros(n, dtype=np.float32)
        gr_recency = np.zeros(n, dtype=np.float32)
        gp_log = np.zeros(n, dtype=np.float32)

        for j, dst in enumerate(cand):
            dst_int = int(dst)
            pair_times = self.index.pair_times_before(src, dst_int, qt)
            cnt = pair_times.size
            if cnt:
                lp_log[j] = math.log1p(cnt)
                lr_freq[j] = cnt / max(src_total, 1)
                last = int(pair_times[-1])
                lr_recency[j] = math.exp(-max(qt - last, 0) / self.graph_span)
            dst_view = self.index.destination_view(dst_int, qt)
            gcnt = dst_view.cutoff
            if gcnt:
                gp_log[j] = math.log1p(gcnt)
                gr_freq[j] = gcnt / max(self.total_edges, 1)
                glast = int(dst_view.visible_times[-1])
                gr_recency[j] = math.exp(-max(qt - glast, 0) / self.graph_span)

        # 归一化 combined：LR 主导，热度用于打破并列（King AI Labs 思路）
        combined = (
            2.0 * lr_recency
            + 1.0 * lp_log / max(1.0, math.log1p(max(src_total, 1)))
            + 1.0 * gr_recency
            + 0.5 * gp_log / max(1.0, math.log1p(max(self.total_edges, 1)))
        ).astype(np.float32)

        output[:, 0] = lr_freq
        output[:, 1] = lr_recency
        output[:, 2] = gr_freq
        output[:, 3] = gr_recency
        output[:, 4] = lp_log
        output[:, 5] = gp_log
        output[:, 6] = combined

        self._fill_query_features_tail(query, output)

    def _fill_query_features_tail(self, query: TestQuery, output: np.ndarray) -> None:
        """只计算特征 7-13（时间窗共同邻居 + 方向共现 + 2 跳路径分）。"""
        qt = int(query.time)
        src = int(query.src)
        n = len(query.candidates)
        cand = np.asarray([int(c) for c in query.candidates], dtype=np.int64)

        src_view = self.index.source_view(src, qt)
        src_dsts = src_view.visible_dsts
        src_times = src_view.visible_times
        src_total = len(src_dsts)
        last_visible_dst = int(src_dsts[-1]) if src_total else None

        # 时间窗共同邻居（短/中/长窗 + 衰减）
        cn_short = np.zeros(n, dtype=np.float32)
        cn_medium = np.zeros(n, dtype=np.float32)
        cn_long = np.zeros(n, dtype=np.float32)
        cn_decay = np.zeros(n, dtype=np.float32)
        if src_total:
            # src 近邻按窗口取不同规模
            w_short, w_medium, w_long = self.windows
            short_cut = qt - w_short
            medium_cut = qt - w_medium
            long_cut = qt - w_long
            short_set = {int(d) for d, t in zip(src_dsts, src_times, strict=True) if t >= short_cut}
            medium_set = {int(d) for d, t in zip(src_dsts, src_times, strict=True) if t >= medium_cut}
            long_set = {int(d) for d, t in zip(src_dsts, src_times, strict=True) if t >= long_cut}
            # 衰减用 src 全历史带权
            decay_weight_by_dst: dict[int, float] = {}
            for d, t in zip(src_dsts, src_times, strict=True):
                w = math.exp(-max(qt - int(t), 0) / w_medium)
                di = int(d)
                decay_weight_by_dst[di] = decay_weight_by_dst.get(di, 0.0) + w
            for j, dst in enumerate(cand):
                dst_int = int(dst)
                dst_view = self.index.destination_view(dst_int, qt)
                dst_srcs = dst_view.visible_srcs
                if dst_srcs.size == 0:
                    continue
                dst_src_set = {int(s) for s in dst_srcs}
                if short_set:
                    cn_short[j] = math.log1p(len(short_set & dst_src_set))
                if medium_set:
                    cn_medium[j] = math.log1p(len(medium_set & dst_src_set))
                if long_set:
                    cn_long[j] = math.log1p(len(long_set & dst_src_set))
                # 衰减版：src 到 v 的邻居 z 的衰减权重和（z ∈ src_dsts ∩ dst_srcs）
                s = 0.0
                for z in decay_weight_by_dst.keys() & dst_src_set:
                    s += decay_weight_by_dst[z]
                cn_decay[j] = math.log1p(s)
        output[:, 7] = cn_short
        output[:, 8] = cn_medium
        output[:, 9] = cn_long
        output[:, 10] = cn_decay

        # 方向化衰减共现 + 2跳路径分
        cooccur_fwd = np.zeros(n, dtype=np.float32)
        cooccur_bwd = np.zeros(n, dtype=np.float32)
        hop2_score = np.zeros(n, dtype=np.float32)
        if self.config.cooccur_time_decay and src_total:
            # (i,j) 共现：i 与 j 在同一 src 序列中相邻出现；i->j 为正序（j 在 i 之后）
            fwd = Counter()
            bwd = Counter()
            for i_idx in range(src_total - 1):
                a = int(src_dsts[i_idx])
                b = int(src_dsts[i_idx + 1])
                ta = float(src_times[i_idx + 1])
                w = math.exp(-max(qt - ta, 0) / self.windows[1])
                fwd[(a, b)] += w
                bwd[(b, a)] += w
            for j, dst in enumerate(cand):
                dst_int = int(dst)
                # 候选与 src 历史各项的加权共现：sum over i in history of w(i, cand)
                s_f = 0.0
                s_b = 0.0
                for h in {int(d) for d in src_dsts}:
                    s_f += fwd.get((h, dst_int), 0.0)
                    s_b += bwd.get((h, dst_int), 0.0)
                cooccur_fwd[j] = math.log1p(s_f)
                cooccur_bwd[j] = math.log1p(s_b)
        output[:, 11] = cooccur_fwd
        output[:, 12] = cooccur_bwd

        if self.config.hop2_enabled and last_visible_dst is not None:
            # 2 跳路径分：src -> z -> cand，z 为 src 最近一跳邻居
            z = last_visible_dst
            z_view = self.index.source_view(z, qt)
            z_dsts = z_view.visible_dsts
            z_times = z_view.visible_times
            if z_dsts.size:
                edge_w = {}
                for i in range(len(z_dsts) - 1):
                    a = int(z_dsts[i])
                    b = int(z_dsts[i + 1])
                    t = float(z_times[i + 1])
                    edge_w[(a, b)] = edge_w.get((a, b), 0.0) + math.exp(-max(qt - t, 0) / self.windows[1])
                for j, dst in enumerate(cand):
                    dst_int = int(dst)
                    score = 0.0
                    # 直接从 z 出发的转移
                    score += edge_w.get((z, dst_int), 0.0)
                    # 经由 z 邻居的二跳（z->m->cand），限于最近若干
                    for m in {int(d) for d in z_dsts[-64:]}:
                        m_view = self.index.source_view(m, qt)
                        m_dsts = m_view.visible_dsts
                        if m_dsts.size == 0:
                            continue
                        w1 = edge_w.get((z, m), 0.0)
                        if w1 <= 0.0:
                            continue
                        for i2 in range(len(m_dsts) - 1):
                            if int(m_dsts[i2]) == m and int(m_dsts[i2 + 1]) == dst_int:
                                w2 = math.exp(-max(qt - float(m_view.visible_times[i2 + 1]), 0) / self.windows[1])
                                score += w1 * w2
                    hop2_score[j] = math.log1p(score)
        output[:, 13] = hop2_score
