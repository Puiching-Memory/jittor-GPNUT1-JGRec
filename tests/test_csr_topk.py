import numpy as np
import pytest

from jgrec.rankers.common.csr_topk import csr_topk_numpy
from jgrec.rankers.common.sparse_counts import SparseCountMap


def _map():
    return SparseCountMap.from_nested_dict({
        1: {5: 3, 2: 9, 7: 1, 3: 9},
        4: {8: 5, 1: 5},
        9: {2: 7, 3: 7, 4: 7, 5: 7},
    })


def test_csr_topk_numpy_basic():
    m = _map()
    qk = np.array([1, 4, 99], dtype=np.int32)
    cols, vals = csr_topk_numpy(m.row_keys, m.row_offsets, m.col_indices, m.values, qk, 3)
    # left=1: value 降序, 平手 col 升序 -> (2,9),(3,9),(5,3)
    assert cols[0].tolist() == [2, 3, 5]
    assert vals[0].tolist() == [9, 9, 3]
    # left=4: 只有 2 个 -> (1,5),(8,5)
    assert cols[1].tolist() == [1, 8, -1]
    assert vals[1].tolist() == [5, 5, 0]
    # 不存在 -> 全 -1/0
    assert cols[2].tolist() == [-1, -1, -1]
    assert vals[2].tolist() == [0, 0, 0]


def test_csr_topk_numpy_tie_break_by_col():
    m = _map()
    qk = np.array([9], dtype=np.int32)
    cols, vals = csr_topk_numpy(m.row_keys, m.row_offsets, m.col_indices, m.values, qk, 4)
    # 全 value=7 -> 按 col 升序
    assert cols[0].tolist() == [2, 3, 4, 5]
    assert vals[0].tolist() == [7, 7, 7, 7]


def test_csr_topk_matches_sparse_map_get_row():
    m = _map()
    qk = np.array([1], dtype=np.int32)
    cols, vals = csr_topk_numpy(m.row_keys, m.row_offsets, m.col_indices, m.values, qk, 2)
    row = m.get_row(1)
    assert row is not None
    rc, rv = row
    order = np.lexsort((rc, -rv))[:2]
    assert cols[0].tolist() == rc[order].tolist()
    assert vals[0].tolist() == rv[order].tolist()


def test_csr_topk_cuda_matches_numpy():
    jt = pytest.importorskip("jittor")
    from jgrec.rankers.common.csr_topk import csr_topk

    jt.flags.use_cuda = 1
    m = _map()
    qk = np.array([1, 4, 9, 99], dtype=np.int32)
    k = 3
    nc, nv = csr_topk_numpy(m.row_keys, m.row_offsets, m.col_indices, m.values, qk, k)
    outs = csr_topk(
        jt.array(m.row_keys), jt.array(m.row_offsets),
        jt.array(m.col_indices), jt.array(m.values), jt.array(qk), k,
    )
    oc, ov = outs[0].numpy(), outs[1].numpy()
    np.testing.assert_array_equal(nc, oc)
    np.testing.assert_array_equal(nv, ov)
