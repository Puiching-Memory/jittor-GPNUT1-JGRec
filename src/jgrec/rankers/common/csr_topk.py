"""CSR 稀疏计数图的批量 top-k 候选查询算子（Jittor CUDA/CPU 双路径）。

把 ``SparseCountMap._top_sparse_items`` 的"对一组 left key 各取 value 最大的
k 个 (col, value)"从 Python 逐行循环下沉到一次 Jittor 算子调用。用于
``TemporalInteractionIndex.cooccur_candidates`` / ``transition_candidates``
的负采样候选生成，绕开 Python 循环与 GIL。

数据布局沿用 ``SparseCountMap`` 的 CSR：
- row_keys:    (R,) int32   已排序的 left 键
- row_offsets: (R+1,) int32 每行在 col_indices/values 的起止
- col_indices: (E,) int32   列键（right 节点 id）
- values:      (E,) int32   计数（越大越优先）

输出：
- out_cols: (Q, k) int32    每个查询 left 的 top-k 列 id，不足补 -1
- out_vals: (Q, k) int32    对应的计数，不足补 0

排序规则与 Python 版一致：value 降序，value 相同按 col 升序。
"""

from __future__ import annotations

import numpy as np

try:
    import jittor as jt
except Exception:  # pragma: no cover - jittor 仅在训练环境可用
    jt = None  # type: ignore

_COMMON = r"""
// 对单个 left：先 searchsorted 定位行，再按 value 降序（稳定，平手保持 CSR
// 存储顺序）选 top-k，与 temporal_index._top_sparse_items 语义一致。
struct CsrTopk {
    static inline
#ifdef __CUDACC__
    __host__ __device__
#endif
    void run_one(
        int query_id,
        const int* __restrict__ row_keys, int R,
        const int* __restrict__ row_offsets,
        const int* __restrict__ col_indices,
        const int* __restrict__ values,
        const int* __restrict__ query_keys,
        int k,
        int* __restrict__ out_cols,
        int* __restrict__ out_vals) {

        int key = query_keys[query_id];
        int lo = 0, hi = R;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (row_keys[mid] < key) lo = mid + 1; else hi = mid;
        }
        int* oc = out_cols + (long long)query_id * k;
        int* ov = out_vals + (long long)query_id * k;
        for (int i = 0; i < k; i++) { oc[i] = -1; ov[i] = 0; }
        if (lo >= R || row_keys[lo] != key) return;
        int start = row_offsets[lo];
        int end = row_offsets[lo + 1];
        for (int e = start; e < end; e++) {
            int v = values[e];
            int c = col_indices[e];
            // 仅当 v 严格大于当前第 k-1 名时才可能插入（稳定：平手保持先来者）
            if (ov[k-1] >= v) continue;
            int pos = 0;
            while (pos < k && ov[pos] >= v) pos++;
            if (pos >= k) continue;
            for (int j = k - 1; j > pos; j--) { ov[j] = ov[j-1]; oc[j] = oc[j-1]; }
            ov[pos] = v; oc[pos] = c;
        }
    }
};
"""

_CUDA_HEADER = _COMMON + r"""
__global__ void csr_topk_kernel(
    const int* row_keys, int R,
    const int* row_offsets,
    const int* col_indices,
    const int* values,
    const int* query_keys, int Q, int k,
    int* out_cols, int* out_vals) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid < Q) {
        CsrTopk::run_one(qid, row_keys, R, row_offsets, col_indices, values, query_keys, k, out_cols, out_vals);
    }
}
"""

_CPU_HEADER = "#include <algorithm>\n" + _COMMON

_CPU_SRC = r"""
@alias(row_keys, in0) @alias(row_offsets, in1) @alias(col_indices, in2)
@alias(values, in3) @alias(query_keys, in4)
@alias(out_cols, out0) @alias(out_vals, out1)
int R = row_keys_shape0, k = out_cols_shape1, Q = query_keys_shape0;
for (int qid = 0; qid < Q; qid++) {
    CsrTopk::run_one(qid, row_keys_p, R, row_offsets_p, col_indices_p, values_p, query_keys_p, k, out_cols_p, out_vals_p);
}
"""

_CUDA_SRC = r"""
@alias(row_keys, in0) @alias(row_offsets, in1) @alias(col_indices, in2)
@alias(values, in3) @alias(query_keys, in4)
@alias(out_cols, out0) @alias(out_vals, out1)
int R = row_keys_shape0, k = out_cols_shape1, Q = query_keys_shape0;
int threads = 128;
int blocks = (Q + threads - 1) / threads;
csr_topk_kernel<<<blocks, threads>>>(row_keys_p, R, row_offsets_p, col_indices_p, values_p, query_keys_p, Q, k, out_cols_p, out_vals_p);
"""


def csr_topk(
    row_keys: "jt.Var",
    row_offsets: "jt.Var",
    col_indices: "jt.Var",
    values: "jt.Var",
    query_keys: "jt.Var",
    k: int,
) -> tuple["jt.Var", "jt.Var"]:
    """对每个 query key 返回 top-k (col, value)。shape=(Q,k)。"""
    if jt is None:
        raise RuntimeError("jittor is required for csr_topk")
    q = query_keys.shape[0]
    outs = jt.code(
        [(q, k), (q, k)], ["int32", "int32"],
        [row_keys, row_offsets, col_indices, values, query_keys],
        cpu_header=_CPU_HEADER, cpu_src=_CPU_SRC,
        cuda_header=_CUDA_HEADER, cuda_src=_CUDA_SRC,
    )
    return outs[0], outs[1]


def csr_topk_numpy(
    row_keys: np.ndarray,
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    values: np.ndarray,
    query_keys: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """numpy 参考实现（用于正确性对照与无 jittor 环境回退）。"""
    q = len(query_keys)
    out_cols = np.full((q, k), -1, dtype=np.int32)
    out_vals = np.zeros((q, k), dtype=np.int32)
    for qi, key in enumerate(query_keys):
        lo = int(np.searchsorted(row_keys, key))
        if lo >= len(row_keys) or row_keys[lo] != key:
            continue
        start, end = int(row_offsets[lo]), int(row_offsets[lo + 1])
        if start == end:
            continue
        cols = col_indices[start:end]
        vals = values[start:end]
        # value 降序，稳定（平手保持 CSR 存储顺序）
        order = np.argsort(-vals, kind="stable")[:k]
        n = len(order)
        out_cols[qi, :n] = cols[order]
        out_vals[qi, :n] = vals[order]
    return out_cols, out_vals
