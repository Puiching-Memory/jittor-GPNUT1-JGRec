from __future__ import annotations

import numpy as np
import pytest

from jgrec.rankers.hybrid.conservative_window_blend import (
    conservative_window_scores,
    evaluate_conservative_window_gate,
    select_conservative_window_blend_on_prefix,
)


def _two_candidate_scores(margins: list[float]) -> np.ndarray:
    values = np.asarray(margins, dtype=np.float64)
    return np.column_stack((values, np.zeros_like(values)))


def test_conservative_scores_shrink_window_residual_toward_champion() -> None:
    champion = np.asarray([[0.8, 0.2], [0.4, 0.6]], dtype=np.float64)
    window = np.asarray([[0.2, 0.8], [0.6, 0.4]], dtype=np.float64)

    actual = conservative_window_scores(champion, window, alpha=0.20)

    np.testing.assert_allclose(actual, champion + 0.20 * (window - champion))
    np.testing.assert_array_equal(
        conservative_window_scores(champion, window, alpha=0.0),
        champion,
    )
    np.testing.assert_array_equal(
        conservative_window_scores(champion, window, alpha=1.0),
        window,
    )
    with pytest.raises(ValueError, match="between zero and one"):
        conservative_window_scores(champion, window, alpha=1.01)
    with pytest.raises(ValueError, match="same shape"):
        conservative_window_scores(champion, window[:, :1], alpha=0.20)


def test_prefix_selection_ignores_forward_and_prefers_smallest_robust_alpha() -> None:
    champion = _two_candidate_scores(
        [0.30, -0.05, 0.30, 0.30, np.nan, np.nan]
    )
    window = _two_candidate_scores(
        [1.00, 1.00, -1.00, 1.00, np.nan, np.nan]
    )

    selection = select_conservative_window_blend_on_prefix(
        champion,
        window,
        alphas=(0.10, 0.20, 0.30),
        first_slice_stop=2,
        selection_stop=4,
        minimum_prefix_delta=0.01,
    )

    reports = {report.alpha: report for report in selection.candidates}
    assert selection.forward_metrics_read is False
    assert selection.selected_alpha == pytest.approx(0.10)
    assert reports[0.10].eligible is True
    assert reports[0.20].eligible is True
    assert reports[0.30].eligible is False
    assert reports[0.30].slice_deltas[1] < 0.0


def test_prefix_selection_returns_no_candidate_when_visible_slice_regresses() -> None:
    champion = _two_candidate_scores(
        [0.30, 0.30, 0.10, 0.10, np.nan, np.nan]
    )
    window = _two_candidate_scores(
        [1.00, 1.00, -1.00, -1.00, np.nan, np.nan]
    )

    selection = select_conservative_window_blend_on_prefix(
        champion,
        window,
        alphas=(0.20, 0.30),
        first_slice_stop=2,
        selection_stop=4,
        minimum_prefix_delta=0.0,
    )

    assert selection.selected_alpha is None
    assert all(not report.eligible for report in selection.candidates)


def test_gate_requires_forward_slice_and_full_gain() -> None:
    champion = _two_candidate_scores(
        [0.30, 0.30, 0.30, 0.30, -0.05, 0.30]
    )
    robust_window = _two_candidate_scores(
        [0.30, 0.30, 0.30, 0.30, 1.00, 0.30]
    )
    forward_regression = _two_candidate_scores(
        [1.00, 1.00, 1.00, 1.00, -1.00, -1.00]
    )

    accepted = evaluate_conservative_window_gate(
        champion,
        robust_window,
        selected_alpha=0.10,
        first_slice_stop=2,
        selection_stop=4,
        minimum_full_delta=0.01,
    )
    rejected_forward = evaluate_conservative_window_gate(
        champion,
        forward_regression,
        selected_alpha=0.50,
        first_slice_stop=2,
        selection_stop=4,
        minimum_full_delta=0.0,
    )
    rejected_gain = evaluate_conservative_window_gate(
        champion,
        robust_window,
        selected_alpha=0.01,
        first_slice_stop=2,
        selection_stop=4,
        minimum_full_delta=0.01,
    )

    assert accepted.passed is True
    assert accepted.slice_deltas[2] > 0.0
    assert rejected_forward.passed is False
    assert rejected_forward.slice_deltas[2] < 0.0
    assert rejected_gain.passed is False
    assert rejected_gain.full_delta < 0.01
