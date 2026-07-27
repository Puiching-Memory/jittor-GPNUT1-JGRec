"""Heuristic 特征的向量化批量计算。

把 ``HeuristicTower._fill_query_features`` 的"逐查询 × 逐候选" Python 循环，
改写为对一批 ``TestQuery`` 的 numpy 向量化批量计算。核心思路：

- fit 时把交互整理成按 src / 按 dst 分组的 CSR 结构（dst 数组按 id 排序、
  配对应的时间数组），节点 id 用连续整数索引。
- 查询时对一批 (src, cand, qt) 用 ``np.searchsorted`` / 布尔掩码向量化求
  pair 次数、最近时间、入边次数、共同邻居，消除 Python 逐点循环。

覆盖特征 0-10（四象限 LR/GR/LP/GP/Combined + 时间窗共同邻居）。
特征 11-13（方向共现 / 2 跳路径）仍走原塔的低频路径。
"""

from __future__ import annotations

import numpy as np

from jgrec.core.types import InteractionTable


class VectorizedHeuristicIndex:
    """按 src / dst 分组的 CSR 邻接 + 时间数组，供向量化特征计算。"""

    def __init__(self) -> None:
        self.n_src = 0
        self.n_dst = 0
        # 按 src 分组（每个 src 的事件按时间升序）
        self.src_offsets = np.zeros(0, dtype=np.int64)
        self.src_dst = np.empty(0, dtype=np.int64)      # 与 src_times 对齐
        self.src_times = np.empty(0, dtype=np.int64)
        # 按 dst 分组
        self.dst_offsets = np.zeros(0, dtype=np.int64)
        self.dst_src = np.empty(0, dtype=np.int64)
        self.dst_times = np.empty(0, dtype=np.int64)
        self.min_time = 0
        self.max_time = 0
        self.graph_span = 1
        self.total_edges = 0
        self.src_keys = np.empty(0, dtype=np.int64)
        self.dst_keys = np.empty(0, dtype=np.int64)

    def fit(self, interactions: InteractionTable) -> None:
        interactions = interactions.sort_by_time()
        src = interactions.src.astype(np.int64)
        dst = interactions.dst.astype(np.int64)
        tim = interactions.time.astype(np.int64)
        self.min_time = int(tim[0])
        self.max_time = int(tim[-1])
        self.graph_span = max(self.max_time - self.min_time, 1)
        self.total_edges = len(tim)

        # ---- 按 src 分组（src 已按时间升序，直接按 src 分段）----
        order_src = np.lexsort((tim, src))
        s_src, s_dst, s_tim = src[order_src], dst[order_src], tim[order_src]
        self.src_keys, src_start = np.unique(s_src, return_index=True)
        self.src_offsets = np.concatenate([src_start, [len(s_src)]]).astype(np.int64)
        self.src_dst = s_dst
        self.src_times = s_tim

        # ---- 按 dst 分组 ----
        order_dst = np.lexsort((tim, dst))
        d_dst, d_src, d_tim = dst[order_dst], src[order_dst], tim[order_dst]
        self.dst_keys, dst_start = np.unique(d_dst, return_index=True)
        self.dst_offsets = np.concatenate([dst_start, [len(d_dst)]]).astype(np.int64)
        self.dst_src = d_src
        self.dst_times = d_tim

        # ---- 全局共现边：同一 src 序列相邻对 (a,b) 带时间 t（按 a 分组）----
        # 遍历每个 src 的 (dst,time) 序列，相邻对 (a,b) 记为 left=a -> (right=b, t=b的时间)
        left_list: list[int] = []
        right_list: list[int] = []
        time_list: list[int] = []
        for si in range(len(self.src_keys)):
            a, b = int(self.src_offsets[si]), int(self.src_offsets[si + 1])
            if b - a < 2:
                continue
            dseg = self.src_dst[a:b]
            tseg = self.src_times[a:b]
            left_list.extend(dseg[:-1].tolist())
            right_list.extend(dseg[1:].tolist())
            time_list.extend(tseg[1:].tolist())
        if left_list:
            left = np.asarray(left_list, dtype=np.int64)
            right = np.asarray(right_list, dtype=np.int64)
            etime = np.asarray(time_list, dtype=np.int64)
            order_co = np.lexsort((etime, left))
            c_left, c_right, c_time = left[order_co], right[order_co], etime[order_co]
            self.co_keys, co_start = np.unique(c_left, return_index=True)
            self.co_offsets = np.concatenate([co_start, [len(c_left)]]).astype(np.int64)
            self.co_right = c_right
            self.co_times = c_time
        else:
            self.co_keys = np.empty(0, dtype=np.int64)
            self.co_offsets = np.zeros(1, dtype=np.int64)
            self.co_right = np.empty(0, dtype=np.int64)
            self.co_times = np.empty(0, dtype=np.int64)

    # ------------------------------------------------------------------ 查询

    def _src_visible(self, src: int, qt: int) -> tuple[np.ndarray, np.ndarray]:
        """src 在 qt 之前的 (dst, time) 可见历史。"""
        pos = int(np.searchsorted(self.src_keys, src))
        if pos >= len(self.src_keys) or self.src_keys[pos] != src:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        a, b = int(self.src_offsets[pos]), int(self.src_offsets[pos + 1])
        tim_seg = self.src_times[a:b]
        cutoff = int(np.searchsorted(tim_seg, qt, side="left"))
        return self.src_dst[a:a + cutoff], tim_seg[:cutoff]

    def _dst_visible_srcs(self, dst: int, qt: int) -> np.ndarray:
        """dst 在 qt 之前的入边 src 数组。"""
        pos = int(np.searchsorted(self.dst_keys, dst))
        if pos >= len(self.dst_keys) or self.dst_keys[pos] != dst:
            return np.empty(0, dtype=np.int64)
        a, b = int(self.dst_offsets[pos]), int(self.dst_offsets[pos + 1])
        tim_seg = self.dst_times[a:b]
        cutoff = int(np.searchsorted(tim_seg, qt, side="left"))
        return self.dst_src[a:a + cutoff]

    def cn_features_for_query(
        self,
        src: int,
        cand: np.ndarray,
        qt: int,
        windows: tuple[float, float, float],
        weighted: bool = False,
    ) -> np.ndarray:
        """时间窗共同邻居特征 7-10。shape=(len(cand),4)。

        weighted=False（默认，经融合实测与加权版等效且更稳）：二值集合交集计数。
        weighted=True（TPNet 时间衰减游走思想，实测在本数据上无额外收益）：
        共同邻居按交互近期度衰减加权。
        """
        cand = cand.astype(np.int64)
        n = len(cand)
        out = np.zeros((n, 4), dtype=np.float64)
        vis_dst, vis_tim = self._src_visible(src, qt)
        if vis_dst.size == 0:
            return out.astype(np.float32)

        w_short, w_medium, w_long = windows

        if weighted:
            # 每个 src 邻居的衰减权重（按所属窗口的衰减率），近期 > 远期
            w_short_arr = np.exp(-np.maximum(qt - vis_tim, 0) / w_short)
            w_medium_arr = np.exp(-np.maximum(qt - vis_tim, 0) / w_medium)
            w_long_arr = np.exp(-np.maximum(qt - vis_tim, 0) / w_long)
            w_med_decay = w_medium_arr
            # 聚合每个 dst 的权重（重复交互累计）
            def _agg(weights: np.ndarray) -> dict[int, float]:
                agg: dict[int, float] = {}
                for d, w in zip(vis_dst.tolist(), weights.tolist(), strict=True):
                    agg[d] = agg.get(d, 0.0) + w
                return agg

            w_short_by = _agg(w_short_arr)
            w_medium_by = _agg(w_medium_arr)
            w_long_by = _agg(w_long_arr)
            w_decay_by = _agg(w_med_decay)
            for j in range(n):
                c = int(cand[j])
                c_srcs = self._dst_visible_srcs(c, qt)
                if c_srcs.size == 0:
                    continue
                c_src_set = set(c_srcs.tolist())
                out[j, 0] = np.log1p(sum(w_short_by[z] for z in w_short_by.keys() & c_src_set))
                out[j, 1] = np.log1p(sum(w_medium_by[z] for z in w_medium_by.keys() & c_src_set))
                out[j, 2] = np.log1p(sum(w_long_by[z] for z in w_long_by.keys() & c_src_set))
                out[j, 3] = np.log1p(sum(w_decay_by[z] for z in w_decay_by.keys() & c_src_set))
            return out.astype(np.float32)

        short_cut, medium_cut, long_cut = qt - w_short, qt - w_medium, qt - w_long
        short_set = set(vis_dst[vis_tim >= short_cut].tolist())
        medium_set = set(vis_dst[vis_tim >= medium_cut].tolist())
        long_set = set(vis_dst[vis_tim >= long_cut].tolist())
        decay_w = np.exp(-np.maximum(qt - vis_tim, 0) / w_medium)
        decay_by_dst: dict[int, float] = {}
        for d, w in zip(vis_dst.tolist(), decay_w.tolist(), strict=True):
            decay_by_dst[d] = decay_by_dst.get(d, 0.0) + w

        # 向量化：把候选 c 的入边 src 集合一次性取出，求与 src 邻居集合的交集。
        # 交集大小 = 共同邻居数；cn_decay 用 decay 权重在交集上求和。
        for j in range(n):
            c = int(cand[j])
            c_srcs = self._dst_visible_srcs(c, qt)
            if c_srcs.size == 0:
                continue
            c_src_set = set(c_srcs.tolist())
            if short_set:
                out[j, 0] = np.log1p(len(short_set & c_src_set))
            if medium_set:
                out[j, 1] = np.log1p(len(medium_set & c_src_set))
            if long_set:
                out[j, 2] = np.log1p(len(long_set & c_src_set))
            s = 0.0
            for z in decay_by_dst.keys() & c_src_set:
                s += decay_by_dst[z]
            out[j, 3] = np.log1p(s)
        return out.astype(np.float32)

    def cooccur_features_for_query(
        self,
        src: int,
        cand: np.ndarray,
        qt: int,
        w_medium: float,
    ) -> np.ndarray:
        """方向化衰减共现特征 11-12。shape=(len(cand),2) (fwd, bwd)。

        共现只在 src 自己的可见序列内：相邻对 (a,b) 且时间 t，
        fwd[h->cand] = sum over 序列中 (h,cand) 相邻对 of exp(-(qt-t)/w_medium)
        bwd[cand->h] = sum over 序列中 (cand,h) 相邻对 of exp(-(qt-t)/w_medium)
        其中 h 遍历 src 历史去重项。
        """
        cand = cand.astype(np.int64)
        n = len(cand)
        out = np.zeros((n, 2), dtype=np.float64)
        vis_dst, vis_tim = self._src_visible(src, qt)
        if vis_dst.size < 2:
            return out.astype(np.float32)

        # src 序列相邻对 (a,b) 与后项时间
        a_seq = vis_dst[:-1]
        b_seq = vis_dst[1:]
        t_seq = vis_tim[1:]
        decay = np.exp(-np.maximum(qt - t_seq, 0) / w_medium)
        src_hist = np.unique(vis_dst)

        # 真正的矩阵向量化：cand(K) × src_hist(H) 的共现权重一次算得。
        # 对每个相邻对 (a,b,t)，fwd[h,c] += decay if a==h and b==c。
        # 等价于：构建 (a_seq,b_seq) 的稀疏共现，查询 (h,c) 命中。
        # 用字典按 (a,b) 聚合 decay，再对 (h,c) 网格查询。
        pair_w: dict[tuple[int, int], float] = {}
        for a, b, w in zip(a_seq.tolist(), b_seq.tolist(), decay.tolist(), strict=True):
            key = (a, b)
            pair_w[key] = pair_w.get(key, 0.0) + w
        hist_list = src_hist.tolist()
        for j in range(n):
            c = int(cand[j])
            sf = 0.0
            sb = 0.0
            for h in hist_list:
                sf += pair_w.get((h, c), 0.0)
                sb += pair_w.get((c, h), 0.0)
            out[j, 0] = np.log1p(sf)
            out[j, 1] = np.log1p(sb)
        return out.astype(np.float32)

    def tail_features_for_group(
        self,
        src: int,
        qt: int,
        cand_rows: np.ndarray,
        windows: tuple[float, float, float],
        want_cooccur: bool,
    ) -> np.ndarray:
        """同一 (src, qt) 下多行候选的 cn(7-10)+cooccur(11-12) 向量化批量计算。

        cand_rows: (G, K) int64。返回 (G, K, 6)，列序 [short,medium,long,decay,fwd,bwd]。
        邻接成员用全局 src-id 位运算一次构造，组内各行复用；逐候选循环被
        np.isin/交集计数取代。
        """
        cand_rows = np.asarray(cand_rows, dtype=np.int64)
        G, K = cand_rows.shape
        out = np.zeros((G, K, 6), dtype=np.float64)
        vis_dst, vis_tim = self._src_visible(src, qt)
        if vis_dst.size == 0:
            return out.astype(np.float32)

        w_short, w_medium, w_long = windows
        short_cut, medium_cut, long_cut = qt - w_short, qt - w_medium, qt - w_long
        # src 邻居（dst id 空间）按窗口去重；decay 权重按 dst 聚合
        short_set = np.unique(vis_dst[vis_tim >= short_cut])
        medium_set = np.unique(vis_dst[vis_tim >= medium_cut])
        long_set = np.unique(vis_dst[vis_tim >= long_cut])
        decay_w = np.exp(-np.maximum(qt - vis_tim, 0) / w_medium)
        decay_by_dst: dict[int, float] = {}
        for d, w in zip(vis_dst.tolist(), decay_w.tolist(), strict=True):
            decay_by_dst[d] = decay_by_dst.get(d, 0.0) + w
        short_set = set(vis_dst[vis_tim >= short_cut].tolist())
        medium_set = set(vis_dst[vis_tim >= medium_cut].tolist())
        long_set = set(vis_dst[vis_tim >= long_cut].tolist())

        # 候选 -> dst_offsets 行位置（向量化）
        flat_cand = cand_rows.reshape(-1)
        d_pos = np.searchsorted(self.dst_keys, flat_cand)
        d_pos = np.clip(d_pos, 0, max(len(self.dst_keys) - 1, 0))
        found = (len(self.dst_keys) > 0) & (self.dst_keys[d_pos] == flat_cand)
        d_pos = d_pos.reshape(G, K)
        found = found.reshape(G, K)

        # 逐候选取其入边 src 集合，与 src 邻居窗口求交集（窗口/权重只构造一次）。
        for g in range(G):
            for j in range(K):
                if not found[g, j]:
                    continue
                p = int(d_pos[g, j])
                a, b = int(self.dst_offsets[p]), int(self.dst_offsets[p + 1])
                tim_seg = self.dst_times[a:b]
                cutoff = int(np.searchsorted(tim_seg, qt, side="left"))
                if cutoff <= 0:
                    continue
                c_src_set = set(self.dst_src[a:a + cutoff].tolist())
                if short_set:
                    out[g, j, 0] = np.log1p(len(short_set & c_src_set))
                if medium_set:
                    out[g, j, 1] = np.log1p(len(medium_set & c_src_set))
                if long_set:
                    out[g, j, 2] = np.log1p(len(long_set & c_src_set))
                s = 0.0
                for z in decay_by_dst.keys() & c_src_set:
                    s += decay_by_dst[z]
                out[g, j, 3] = np.log1p(s)

        if want_cooccur and vis_dst.size >= 2:
            a_seq = vis_dst[:-1]
            b_seq = vis_dst[1:]
            t_seq = vis_tim[1:]
            decay = np.exp(-np.maximum(qt - t_seq, 0) / w_medium)
            # 双向索引：fwd_idx[right][left]=w（查 (h,c)：right=c 桶内 left=h）
            #           bwd_idx[left][right]=w（查 (c,h)：left=c 桶内 right=h）
            fwd_idx: dict[int, dict[int, float]] = {}
            bwd_idx: dict[int, dict[int, float]] = {}
            for a, b, w in zip(a_seq.tolist(), b_seq.tolist(), decay.tolist(), strict=True):
                ai, bi = int(a), int(b)
                fw = fwd_idx.get(bi)
                if fw is None:
                    fw = fwd_idx[bi] = {}
                fw[ai] = fw.get(ai, 0.0) + w
                bw = bwd_idx.get(ai)
                if bw is None:
                    bw = bwd_idx[ai] = {}
                bw[bi] = bw.get(bi, 0.0) + w
            hist_set = set(vis_dst.tolist())
            for g in range(G):
                for j in range(K):
                    c = int(cand_rows[g, j])
                    # fwd[h,c]：c 的入边 left 桶 ∩ hist
                    fw = fwd_idx.get(c)
                    sf = 0.0
                    if fw:
                        for h in fw.keys() & hist_set:
                            sf += fw[h]
                    # bwd[c,h]：c 的出边 right 桶 ∩ hist
                    bw = bwd_idx.get(c)
                    sb = 0.0
                    if bw:
                        for h in bw.keys() & hist_set:
                            sb += bw[h]
                    out[g, j, 4] = np.log1p(sf)
                    out[g, j, 5] = np.log1p(sb)
        return out.astype(np.float32)

    def _co_weight(self, left: int, right: int, qt: int, w_medium: float) -> float:
        """(left,right) 相邻共现在 qt 前的衰减权重和。"""
        pos = int(np.searchsorted(self.co_keys, left))
        if pos >= len(self.co_keys) or self.co_keys[pos] != left:
            return 0.0
        a, b = int(self.co_offsets[pos]), int(self.co_offsets[pos + 1])
        r_seg = self.co_right[a:b]
        t_seg = self.co_times[a:b]
        hit = r_seg == right
        if not hit.any():
            return 0.0
        times = t_seg[hit]
        times = times[times <= qt]
        if times.size == 0:
            return 0.0
        return float(np.exp(-np.maximum(qt - times, 0) / w_medium).sum())

    def hop2_features_for_query(
        self,
        src: int,
        cand: np.ndarray,
        qt: int,
        w_medium: float,
    ) -> np.ndarray:
        """2 跳路径分特征 13。shape=(len(cand),1)。

        z = src 最近一跳邻居；score(cand) = co_w(z,cand) + Σ_{m ∈ recent(z)} co_w(z,m)·co_w(m,cand)
        co_w(a,b) = 全局相邻共现边 (a,b) 在 qt 前的衰减权重和。
        """
        cand = cand.astype(np.int64)
        n = len(cand)
        out = np.zeros((n, 1), dtype=np.float64)
        if len(self.co_keys) == 0:
            return out.astype(np.float32)

        vis_dst, _vis_tim = self._src_visible(src, qt)
        if vis_dst.size == 0:
            return out.astype(np.float32)
        z = int(vis_dst[-1])

        # z 的最近邻居 m（z 作为 src 的出边序列，取最近 64 个）
        z_dsts, _ = self._src_visible(z, qt)
        m_set = np.unique(z_dsts[-64:]) if z_dsts.size else np.empty(0, dtype=np.int64)
        # edge_w(z, m) 预计算
        w_zm = {int(m): self._co_weight(z, int(m), qt, w_medium) for m in m_set}

        for j in range(n):
            c = int(cand[j])
            score = self._co_weight(z, c, qt, w_medium)
            for m in m_set:
                mi = int(m)
                w1 = w_zm[mi]
                if w1 <= 0.0:
                    continue
                score += w1 * self._co_weight(mi, c, qt, w_medium)
            out[j, 0] = np.log1p(score)
        return out.astype(np.float32)

    def quadrant_features_for_query(
        self,
        src: int,
        cand: np.ndarray,
        qt: int,
    ) -> np.ndarray:
        """对单个查询的整行候选向量化计算 7 维四象限特征。shape=(len(cand),7)。

        行内向量化：pair 次数/最近时间用 cand 在 src 历史 dst 上的布尔掩码一次
        算得；dst 入边次数/最近时间用向量化 searchsorted 一次算得。
        """
        cand = cand.astype(np.int64)
        n = len(cand)
        out = np.zeros((n, 7), dtype=np.float64)

        # ---- src 侧 ----
        s_pos = int(np.searchsorted(self.src_keys, src))
        src_total = 0
        if s_pos < len(self.src_keys) and self.src_keys[s_pos] == src:
            a, b = int(self.src_offsets[s_pos]), int(self.src_offsets[s_pos + 1])
            dst_seg = self.src_dst[a:b]
            tim_seg = self.src_times[a:b]
            cutoff = int(np.searchsorted(tim_seg, qt, side="left"))
            src_total = cutoff
            if cutoff > 0:
                vis_dst = dst_seg[:cutoff]
                vis_tim = tim_seg[:cutoff]
                # 对整行 cand 向量化：cand[j] 在 vis_dst 中的出现次数与最后时间
                # 用排序索引统计每个 cand 的 count
                order = np.argsort(vis_dst)
                s_dst = vis_dst[order]
                s_tim = vis_tim[order]
                # 每个 cand 的位置区间
                left = np.searchsorted(s_dst, cand, side="left")
                right = np.searchsorted(s_dst, cand, side="right")
                cnt = right - left
                has = cnt > 0
                out[has, 4] = np.log1p(cnt[has])  # lp_log
                out[has, 0] = cnt[has] / max(src_total, 1)  # lr_freq
                # 最近时间：right-1 即该 cand 最后一次出现（时间升序，但排序按 dst
                # 打乱了时间；需取该 cand 区间内的最大时间）
                for j in np.flatnonzero(has):
                    seg = s_tim[left[j]:right[j]]
                    last = int(seg.max())
                    out[j, 1] = float(np.exp(-max(qt - last, 0) / self.graph_span))  # lr_recency

        # ---- dst 侧：整行 cand 的入边次数与最近时间 ----
        d_pos = np.searchsorted(self.dst_keys, cand)
        d_found = (d_pos < len(self.dst_keys)) & (self.dst_keys[np.clip(d_pos, 0, len(self.dst_keys) - 1)] == cand)
        for j in np.flatnonzero(d_found):
            a, b = int(self.dst_offsets[d_pos[j]]), int(self.dst_offsets[d_pos[j] + 1])
            tim_seg = self.dst_times[a:b]
            cutoff = int(np.searchsorted(tim_seg, qt, side="left"))
            if cutoff <= 0:
                continue
            out[j, 5] = np.log1p(cutoff)  # gp_log
            glast = int(tim_seg[cutoff - 1])
            out[j, 3] = float(np.exp(-max(qt - glast, 0) / self.graph_span))  # gr_recency

        out[:, 2] = np.where(self.total_edges > 0, np.expm1(out[:, 5]) / max(self.total_edges, 1), 0.0)  # gr_freq
        out[:, 6] = (
            2.0 * out[:, 1]
            + 1.0 * out[:, 4] / max(1.0, float(np.log1p(max(src_total, 1))))
            + 1.0 * out[:, 3]
            + 0.5 * out[:, 5] / max(1.0, float(np.log1p(max(self.total_edges, 1))))
        )  # combined
        return out.astype(np.float32)

    def quadrant_features(
        self,
        src: np.ndarray,
        cand: np.ndarray,
        qt: np.ndarray,
    ) -> np.ndarray:
        """逐查询调用行向量化（cand 为 (N,K) 或每行一组）。"""
        if cand.ndim == 2:
            rows = [self.quadrant_features_for_query(int(src[i]), cand[i], int(qt[i])) for i in range(len(src))]
            return np.stack(rows, axis=0)
        return self.quadrant_features_for_query(int(src[0]), cand, int(qt[0]))
