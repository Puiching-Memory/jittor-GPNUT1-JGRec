import numpy as np
import pytest

from jgrec.core.prediction_ties import break_prediction_ties


def test_tie_break_handles_zero_one_boundaries_and_preserves_distinct_levels():
    scores = np.asarray([[0.0, 0.0, 0.5, 0.5, 1.0, 1.0]])
    priorities = np.asarray([[2.0, 1.0, 1.0, 2.0, 1.0, 2.0]])
    candidates = np.asarray([[20, 10, 30, 40, 50, 60]])

    stabilized, report = break_prediction_ties(
        scores,
        priorities=priorities,
        candidate_ids=candidates,
    )

    assert np.unique(stabilized[0]).size == scores.shape[1]
    assert np.all((stabilized >= 0.0) & (stabilized <= 1.0))
    assert stabilized[0, 0] > stabilized[0, 1]
    assert stabilized[0, 3] > stabilized[0, 2]
    assert stabilized[0, 5] > stabilized[0, 4]
    assert stabilized[0, :2].max() < stabilized[0, 2:4].min()
    assert stabilized[0, 2:4].max() < stabilized[0, 4:].min()
    assert report.rows_with_ties == 1
    assert report.tied_groups == 3
    assert report.tied_candidates == 6


def test_tie_break_leaves_rows_without_ties_bit_identical():
    scores = np.asarray([[0.8, 0.6, 0.2]], dtype=np.float64)

    stabilized, report = break_prediction_ties(scores)

    np.testing.assert_array_equal(stabilized, scores)
    assert report.rows_with_ties == 0


def test_tie_break_rejects_mismatched_prior_shape():
    with pytest.raises(ValueError, match="priorities"):
        break_prediction_ties(
            np.asarray([[0.5, 0.5]]),
            priorities=np.asarray([[1.0]]),
        )


def test_tie_break_uses_rank_fallback_when_no_float_exists_between_neighbors():
    lower = np.float64(0.5)
    tied = np.nextafter(lower, np.inf)
    upper = np.nextafter(tied, np.inf)
    scores = np.asarray([[upper, tied, tied, lower]])
    priorities = np.asarray([[0.0, 1.0, 2.0, 0.0]])

    stabilized, report = break_prediction_ties(
        scores,
        priorities=priorities,
    )

    assert np.unique(stabilized[0]).size == 4
    assert stabilized[0, 0] > stabilized[0, 2] > stabilized[0, 1]
    assert stabilized[0, 1] > stabilized[0, 3]
    assert report.rank_fallback_rows == 1


def test_tie_break_avoids_subnormal_perturbations():
    tiny = np.finfo(np.float64).tiny
    scores = np.asarray([[2.0 * tiny, tiny, tiny, 0.0]])

    stabilized, report = break_prediction_ties(scores)

    assert np.unique(stabilized[0]).size == 4
    assert not np.any(
        (stabilized > 0.0) & (stabilized < tiny)
    )
    assert stabilized[0, 0] > stabilized[0, 1]
    assert stabilized[0, 2] > stabilized[0, 3]
    assert report.rows_with_ties == 1
