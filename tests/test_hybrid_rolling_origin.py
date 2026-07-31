import numpy as np
import pytest

from jgrec.rankers.hybrid.rolling_origin import (
    passes_rolling_origin_gate,
    select_candidate_on_rolling_origins,
    sliding_rolling_origin_folds,
    validate_rolling_origin_times,
)


def test_sliding_origins_cover_last_horizons_once_without_future_rows():
    folds = sliding_rolling_origin_folds(
        row_count=200_000,
        train_window_rows=100_000,
        score_rows=25_000,
        fold_count=4,
        step_rows=25_000,
        selection_fold_count=3,
    )

    assert [
        (fold.train_rows, fold.score_rows, fold.role)
        for fold in folds
    ] == [
        ((0, 100_000), (100_000, 125_000), "selection"),
        ((25_000, 125_000), (125_000, 150_000), "selection"),
        ((50_000, 150_000), (150_000, 175_000), "selection"),
        ((75_000, 175_000), (175_000, 200_000), "gate"),
    ]
    scored = np.concatenate(
        [
            np.arange(*fold.score_rows, dtype=np.int64)
            for fold in folds
        ]
    )
    np.testing.assert_array_equal(
        scored,
        np.arange(100_000, 200_000, dtype=np.int64),
    )
    assert all(
        fold.train_rows[1] <= fold.score_rows[0]
        for fold in folds
    )


def test_sliding_origins_reject_overlap_or_insufficient_history():
    with pytest.raises(ValueError, match="overlap"):
        sliding_rolling_origin_folds(
            row_count=200,
            train_window_rows=100,
            score_rows=30,
            fold_count=3,
            step_rows=20,
            selection_fold_count=2,
        )
    with pytest.raises(ValueError, match="history"):
        sliding_rolling_origin_folds(
            row_count=100,
            train_window_rows=80,
            score_rows=20,
            fold_count=2,
            step_rows=20,
            selection_fold_count=1,
        )


def test_fold_time_validation_requires_chronological_sidecar():
    folds = sliding_rolling_origin_folds(
        row_count=8,
        train_window_rows=4,
        score_rows=2,
        fold_count=2,
        step_rows=2,
        selection_fold_count=1,
    )
    times = np.asarray([1, 1, 2, 3, 3, 4, 5, 6])

    boundaries = validate_rolling_origin_times(times, folds)

    assert boundaries[0].train_time_max == 3
    assert boundaries[0].score_time_min == 3
    assert boundaries[1].score_time_max == 6
    broken = times.copy()
    broken[5] = 2
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_rolling_origin_times(broken, folds)


def test_selection_rejects_any_regressing_fold_without_forward_metrics():
    selection = select_candidate_on_rolling_origins(
        baseline_mrrs=(0.70, 0.70, 0.70),
        candidate_mrrs={
            "gamma_0.5": (0.701, 0.702, 0.703),
            "regressing": (0.710, 0.720, 0.699),
        },
        minimum_mean_delta=0.0002,
        tie_break_order=("regressing", "gamma_0.5"),
    )

    assert selection.selected_name == "gamma_0.5"
    assert selection.forward_metrics_read is False
    assert selection.selection_fold_count == 3
    trials = {trial.name: trial for trial in selection.trials}
    assert trials["gamma_0.5"].eligible is True
    assert trials["regressing"].eligible is False
    assert trials["regressing"].fold_deltas[-1] < 0.0


def test_gate_requires_nonnegative_forward_fold_and_overall_mean():
    accepted = passes_rolling_origin_gate(
        selection_fold_deltas=(0.001, 0.002, 0.003),
        baseline_forward_mrr=0.70,
        candidate_forward_mrr=0.701,
        minimum_overall_mean_delta=0.0002,
    )
    rejected_forward = passes_rolling_origin_gate(
        selection_fold_deltas=(0.010, 0.010, 0.010),
        baseline_forward_mrr=0.70,
        candidate_forward_mrr=0.699,
        minimum_overall_mean_delta=0.0002,
    )
    rejected_mean = passes_rolling_origin_gate(
        selection_fold_deltas=(0.0001, 0.0001, 0.0001),
        baseline_forward_mrr=0.70,
        candidate_forward_mrr=0.7001,
        minimum_overall_mean_delta=0.0002,
    )

    assert accepted.passed is True
    assert accepted.forward_delta == pytest.approx(0.001)
    assert rejected_forward.passed is False
    assert rejected_forward.forward_delta < 0.0
    assert rejected_mean.passed is False
    assert rejected_mean.overall_mean_delta < 0.0002
