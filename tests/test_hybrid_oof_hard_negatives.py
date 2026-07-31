import numpy as np
import pytest

from jgrec.rankers.hybrid.oof_hard_negatives import (
    contiguous_oof_folds,
    passes_temporal_mrr_gate,
    select_hard_negative_features,
    select_hard_negative_positions,
)


def test_contiguous_oof_folds_cover_each_row_once_and_exclude_held_rows():
    folds = contiguous_oof_folds(row_count=10, fold_count=3)

    held_rows: list[int] = []
    for fold in folds:
        held = set(range(fold.holdout.start, fold.holdout.stop))
        fitted = {
            row
            for fit_slice in fold.fit_slices
            for row in range(fit_slice.start, fit_slice.stop)
        }
        held_rows.extend(sorted(held))

        assert held
        assert held.isdisjoint(fitted)
        assert held | fitted == set(range(10))

    assert sorted(held_rows) == list(range(10))


def test_hard_negative_positions_keep_positive_and_rank_ties_stably():
    scores = np.asarray(
        [
            [100.0, 0.2, 0.9, 0.9, 0.1],
            [-10.0, 0.7, 0.1, 0.8, 0.6],
        ],
        dtype=np.float64,
    )

    actual = select_hard_negative_positions(scores, keep_negatives=2)

    np.testing.assert_array_equal(
        actual,
        np.asarray(
            [
                [0, 2, 3],
                [0, 3, 1],
            ]
        ),
    )


def test_hard_negative_features_follow_selected_positions_without_mutating_input():
    features = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    original = features.copy()
    scores = np.asarray(
        [
            [0.0, 0.2, 0.9, 0.8, 0.1],
            [0.0, 0.7, 0.1, 0.8, 0.6],
        ]
    )

    selected = select_hard_negative_features(features, scores, keep_negatives=2)

    np.testing.assert_array_equal(selected[0], features[0, [0, 2, 3]])
    np.testing.assert_array_equal(selected[1], features[1, [0, 3, 1]])
    np.testing.assert_array_equal(features, original)


@pytest.mark.parametrize(
    ("row_count", "fold_count"),
    [(0, 3), (2, 3), (10, 1)],
)
def test_oof_folds_reject_empty_or_impossible_partitions(row_count: int, fold_count: int):
    with pytest.raises(ValueError):
        contiguous_oof_folds(row_count=row_count, fold_count=fold_count)


@pytest.mark.parametrize("keep_negatives", [0, 5])
def test_hard_negative_selection_rejects_invalid_keep_count(keep_negatives: int):
    scores = np.zeros((2, 5), dtype=np.float32)

    with pytest.raises(ValueError):
        select_hard_negative_positions(scores, keep_negatives=keep_negatives)


def test_temporal_gate_requires_every_slice_and_minimum_full_delta():
    baseline = (0.50, 0.48, 0.46)

    assert passes_temporal_mrr_gate(
        candidate_slices=(0.503, 0.482, 0.461),
        baseline_slices=baseline,
        candidate_full_mrr=0.482,
        baseline_full_mrr=0.480,
        min_full_delta=0.002,
    )
    assert not passes_temporal_mrr_gate(
        candidate_slices=(0.503, 0.479, 0.465),
        baseline_slices=baseline,
        candidate_full_mrr=0.483,
        baseline_full_mrr=0.480,
        min_full_delta=0.002,
    )
    assert not passes_temporal_mrr_gate(
        candidate_slices=(0.501, 0.481, 0.461),
        baseline_slices=baseline,
        candidate_full_mrr=0.481,
        baseline_full_mrr=0.480,
        min_full_delta=0.002,
    )


def test_temporal_gate_allows_unchanged_slice_when_other_slices_supply_full_gain():
    assert passes_temporal_mrr_gate(
        candidate_slices=(0.503, 0.480, 0.465),
        baseline_slices=(0.500, 0.480, 0.460),
        candidate_full_mrr=0.482,
        baseline_full_mrr=0.480,
        min_full_delta=0.002,
    )
