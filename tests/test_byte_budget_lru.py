from __future__ import annotations

import numpy as np

from jgrec.rankers.common.byte_budget_lru import ByteBudgetLRU


def test_byte_budget_lru_evicts_oldest_values_before_exceeding_budget() -> None:
    cache: ByteBudgetLRU[int, np.ndarray] = ByteBudgetLRU(max_bytes=12, max_entries=8)
    first = np.asarray([1, 2], dtype=np.int32)
    second = np.asarray([3, 4], dtype=np.int32)

    cache.put(1, first, size_bytes=first.nbytes)
    cache.put(2, second, size_bytes=second.nbytes)

    assert 1 not in cache
    assert cache.get(2) is second
    assert cache.current_bytes == second.nbytes
    assert cache.current_bytes <= cache.max_bytes


def test_byte_budget_lru_rejects_single_value_larger_than_budget() -> None:
    cache: ByteBudgetLRU[int, np.ndarray] = ByteBudgetLRU(max_bytes=4, max_entries=8)
    oversized = np.asarray([1, 2], dtype=np.int32)

    cache.put(1, oversized, size_bytes=oversized.nbytes)

    assert 1 not in cache
    assert cache.current_bytes == 0
