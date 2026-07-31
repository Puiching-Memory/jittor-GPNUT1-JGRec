from __future__ import annotations

import numpy as np
import pytest

from jgrec.rankers.hybrid.window_diversity import (
    blend_expert_subset,
    normalized_exponential_recency_weights,
    recent_window_view,
    select_uniform_subset_on_prefix,
)

EXPERT_ORDER = (
    "recent50k",
    "recent100k",
    "recent200k",
    "recent200k_decay100k",
)


def _two_candidate_scores(margins: list[float]) -> np.ndarray:
    values = np.asarray(margins, dtype=np.float64)
    return np.column_stack((values, np.zeros_like(values)))


def test_recent_window_view_returns_exact_shared_tail() -> None:
    features = np.arange(20, dtype=np.float32).reshape(10, 2)

    selected = recent_window_view(features, 4)

    np.testing.assert_array_equal(selected, features[6:])
    assert np.shares_memory(selected, features)
    with pytest.raises(ValueError, match="positive"):
        recent_window_view(features, 0)
    with pytest.raises(ValueError, match="only 10"):
        recent_window_view(features, 11)


def test_exponential_recency_weights_are_normalized_and_obey_half_life() -> None:
    weights = normalized_exponential_recency_weights(
        row_count=5,
        half_life_rows=2,
    )

    assert weights.dtype == np.float32
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert np.all(np.diff(weights) > 0.0)
    assert float(np.mean(weights, dtype=np.float64)) == pytest.approx(
        1.0,
        abs=1e-7,
    )
    assert float(weights[0] / weights[2]) == pytest.approx(0.5, abs=1e-7)
    assert float(weights[2] / weights[4]) == pytest.approx(0.5, abs=1e-7)


def test_prefix_selection_finds_complementary_window_subset_without_forward_rows() -> None:
    first = _two_candidate_scores([2.0, 2.0, -1.0, -1.0, np.nan, np.nan])
    second = _two_candidate_scores([-1.0, -1.0, 2.0, 2.0, np.nan, np.nan])
    always_wrong = _two_candidate_scores(
        [-4.0, -4.0, -4.0, -4.0, np.nan, np.nan]
    )
    experts = {
        "recent50k": first,
        "recent100k": second,
        "recent200k": always_wrong,
        "recent200k_decay100k": always_wrong,
    }
    lightgbm = np.full((6, 2), 0.5, dtype=np.float64)

    result = select_uniform_subset_on_prefix(
        experts,
        lightgbm,
        selection_stop=4,
        expert_weight=0.80,
        expert_order=EXPERT_ORDER,
    )

    assert result.selected_experts == ("recent50k", "recent100k")
    assert result.selection_mrr == pytest.approx(1.0)
    assert len(result.candidates) == 15


def test_prefix_selection_tie_breaks_by_fewer_experts_then_frozen_order() -> None:
    shared = _two_candidate_scores([2.0, 2.0, 2.0, 2.0, np.nan, np.nan])
    experts = {name: shared.copy() for name in EXPERT_ORDER}
    lightgbm = np.full((6, 2), 0.5, dtype=np.float64)

    result = select_uniform_subset_on_prefix(
        experts,
        lightgbm,
        selection_stop=4,
        expert_weight=0.80,
        expert_order=EXPERT_ORDER,
    )

    assert result.selected_experts == ("recent50k",)
    assert result.selection_mrr == pytest.approx(1.0)


def test_blend_expert_subset_uses_uniform_probability_mean_and_fixed_outer_weight() -> None:
    first = np.asarray([[0.8, 0.2], [0.6, 0.4]], dtype=np.float64)
    second = np.asarray([[0.4, 0.6], [0.2, 0.8]], dtype=np.float64)
    lightgbm = np.asarray([[0.5, 0.5], [0.7, 0.3]], dtype=np.float64)

    actual = blend_expert_subset(
        {"recent50k": first, "recent100k": second},
        lightgbm,
        selected_experts=("recent50k", "recent100k"),
        expert_weight=0.80,
    )

    expected = 0.80 * ((first + second) / 2.0) + 0.20 * lightgbm
    np.testing.assert_allclose(actual, expected)
