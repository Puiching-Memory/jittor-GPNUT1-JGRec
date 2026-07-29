from __future__ import annotations

import numpy as np
import pytest

from jgrec.rankers.hybrid.new_link_diagnostics import (
    historical_pair_mask,
    new_link_error_report,
    passes_new_link_concentration_gate,
)


def test_historical_pair_mask_marks_only_pairs_present_in_fixed_context() -> None:
    context_src = np.asarray([1, 1, 2, 2], dtype=np.int64)
    context_dst = np.asarray([10, 10, 20, 21], dtype=np.int64)
    query_src = np.asarray([1, 1, 2, 3], dtype=np.int64)
    positive_dst = np.asarray([10, 11, 21, 10], dtype=np.int64)
    original = tuple(array.copy() for array in (context_src, context_dst, query_src, positive_dst))

    actual = historical_pair_mask(context_src, context_dst, query_src, positive_dst)

    np.testing.assert_array_equal(actual, np.asarray([True, False, True, False]))
    for values, expected in zip((context_src, context_dst, query_src, positive_dst), original, strict=True):
        np.testing.assert_array_equal(values, expected)


def test_new_link_error_report_computes_exact_rank_regret_and_segment_shares() -> None:
    scores = _scores_for_ranks([1, 2, 1, 3, 1, 2])
    repeat_mask = np.asarray([True, False, True, False, True, False])

    report = new_link_error_report(scores, repeat_mask, slices=(slice(0, 3), slice(3, 6)))

    assert report.repeat.rows == 3
    assert report.repeat.mrr == pytest.approx(1.0)
    assert report.repeat.top1_error_rate == pytest.approx(0.0)
    assert report.new.rows == 3
    assert report.new.mrr == pytest.approx((0.5 + 1.0 / 3.0 + 0.5) / 3.0)
    assert report.new.top1_error_rate == pytest.approx(1.0)
    assert report.new_row_share == pytest.approx(0.5)
    assert report.new_regret_share == pytest.approx(1.0)
    assert report.repeat_minus_new_mrr == pytest.approx(1.0 - report.new.mrr)


def test_new_link_concentration_gate_requires_stable_difficulty_in_every_slice() -> None:
    repeat_mask = np.asarray([True, False] * 6)
    passing = new_link_error_report(
        _scores_for_ranks([1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3]),
        repeat_mask,
        slices=(slice(0, 4), slice(4, 8), slice(8, 12)),
    )
    unstable = new_link_error_report(
        _scores_for_ranks([1, 2, 1, 3, 1, 1, 1, 1, 1, 2, 1, 3]),
        repeat_mask,
        slices=(slice(0, 4), slice(4, 8), slice(8, 12)),
    )

    assert passes_new_link_concentration_gate(passing, min_rows_per_segment=2)
    assert not passes_new_link_concentration_gate(unstable, min_rows_per_segment=2)


def test_new_link_error_report_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        new_link_error_report(
            np.zeros((3, 4), dtype=np.float32),
            np.asarray([True, False]),
            slices=(slice(0, 3),),
        )


def _scores_for_ranks(ranks: list[int]) -> np.ndarray:
    scores = np.zeros((len(ranks), 3), dtype=np.float64)
    for row, rank in enumerate(ranks):
        if rank == 1:
            scores[row] = (2.0, 1.0, 0.0)
        elif rank == 2:
            scores[row] = (1.0, 2.0, 0.0)
        elif rank == 3:
            scores[row] = (0.0, 2.0, 1.0)
        else:
            raise ValueError("test helper supports ranks one through three")
    return scores
