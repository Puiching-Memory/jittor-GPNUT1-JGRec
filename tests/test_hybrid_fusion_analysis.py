import numpy as np
import pytest

from jgrec.rankers.hybrid.fusion_analysis import (
    authorized_setwise_weight,
    inclusive_weight_grid,
    ranking_mrr_slices,
    ranking_mrr_three_slices,
    scan_high_weight_blend_on_prefix,
    scan_probability_blend,
    scan_rank_blend_on_prefix,
    select_setwise_model_blend_on_prefix,
    uniform_rank_average,
)


def test_ranking_mrr_slices_reports_full_and_temporal_halves():
    scores = np.asarray(
        [
            [3.0, 2.0, 1.0],
            [2.0, 3.0, 1.0],
            [1.0, 3.0, 2.0],
            [3.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )

    actual = ranking_mrr_slices(scores)

    assert actual == {
        "full": np.mean([1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0]),
        "early": np.mean([1.0, 1.0 / 2.0]),
        "late": np.mean([1.0 / 3.0, 1.0]),
    }


def test_scan_probability_blend_tests_101_weights_and_prefers_reference_on_ties():
    reference = np.asarray(
        [
            [0.7, 0.3],
            [0.4, 0.6],
        ],
        dtype=np.float64,
    )
    alternate = np.asarray(
        [
            [0.2, 0.8],
            [0.9, 0.1],
        ],
        dtype=np.float64,
    )

    result = scan_probability_blend(reference, alternate)

    assert result.weights_tested == 101
    assert result.reference_weight == 0.8
    assert result.mrr["full"] == 1.0


def test_ranking_mrr_three_slices_covers_balanced_chronological_parts():
    scores = np.asarray(
        [
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )

    actual = ranking_mrr_three_slices(scores)

    assert actual == {
        "full": np.mean([1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 0.5]),
        "slice_0": np.mean([1.0, 0.5, 1.0]),
        "slice_1": np.mean([0.5, 1.0]),
        "slice_2": np.mean([1.0, 0.5]),
    }


def test_ranking_mrr_three_slices_uses_tie_neutral_average_rank():
    tied = np.ones((3, 2), dtype=np.float64)

    actual = ranking_mrr_three_slices(tied)

    assert actual == {
        "full": 1.0 / 1.5,
        "slice_0": 1.0 / 1.5,
        "slice_1": 1.0 / 1.5,
        "slice_2": 1.0 / 1.5,
    }


def test_rank_blend_selects_on_prefix_only_and_tests_101_weights():
    reference = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    alternate = np.asarray(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.9, 0.1],
            [0.9, 0.1],
        ],
        dtype=np.float64,
    )

    result = scan_rank_blend_on_prefix(
        reference,
        alternate,
        selection_stop=4,
        method="probability",
    )

    assert result.reference_weight == 1.0
    assert result.selection_mrr == 1.0
    assert result.weights_tested == 101
    assert result.mrr["slice_2"] == 0.5


@pytest.mark.parametrize("method", ["row_zscore", "rank_percentile"])
def test_rank_blend_normalizations_are_finite_for_constant_rows(method: str):
    reference = np.asarray([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    alternate = np.asarray([[3.0, 3.0], [1.0, 2.0], [2.0, 1.0]])

    result = scan_rank_blend_on_prefix(
        reference,
        alternate,
        selection_stop=2,
        method=method,
    )

    assert np.isfinite(result.selection_mrr)
    assert all(np.isfinite(value) for value in result.mrr.values())


def test_rank_blend_rejects_misaligned_or_non_finite_scores():
    with pytest.raises(ValueError, match="same shape"):
        scan_rank_blend_on_prefix(
            np.zeros((3, 2)),
            np.zeros((3, 3)),
            selection_stop=2,
        )
    with pytest.raises(ValueError, match="finite"):
        scan_rank_blend_on_prefix(
            np.asarray([[1.0, np.nan], [1.0, 0.0], [1.0, 0.0]]),
            np.zeros((3, 2)),
            selection_stop=2,
        )


def test_high_weight_scan_includes_one_and_ignores_forward_rows_for_selection():
    setwise = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    lightgbm = np.asarray(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.9, 0.1],
            [0.9, 0.1],
        ],
        dtype=np.float64,
    )

    first = scan_high_weight_blend_on_prefix(
        setwise,
        lightgbm,
        selection_stop=4,
    )
    changed_forward = setwise.copy()
    changed_forward[4:] = lightgbm[4:]
    second = scan_high_weight_blend_on_prefix(
        changed_forward,
        lightgbm,
        selection_stop=4,
    )

    assert first.weights_tested == 21
    assert first.primary_weight == 1.0
    assert second.primary_weight == first.primary_weight
    assert second.selection_mrr == first.selection_mrr


def test_setwise_model_and_weight_selection_ignores_forward_rows():
    champion = np.asarray(
        [
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.9, 0.1],
        ],
        dtype=np.float64,
    )
    recent_100k = np.asarray(
        [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.9, 0.1],
        ],
        dtype=np.float64,
    )
    recent_200k = np.asarray(
        [
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    candidates = {
        "recent_100k": recent_100k,
        "recent_200k": recent_200k,
    }

    first = select_setwise_model_blend_on_prefix(
        candidates,
        champion,
        selection_stop=4,
        primary_weights=(0.5, 1.0),
        model_tie_break_order=("recent_100k", "recent_200k"),
    )
    changed_forward = {
        name: np.concatenate(
            (scores[:4], np.full_like(scores[4:], np.nan)),
            axis=0,
        )
        for name, scores in candidates.items()
    }
    second = select_setwise_model_blend_on_prefix(
        changed_forward,
        champion,
        selection_stop=4,
        primary_weights=(0.5, 1.0),
        model_tie_break_order=("recent_100k", "recent_200k"),
    )

    assert first.model_name == "recent_200k"
    assert first.primary_weight == 1.0
    assert first.models_tested == 2
    assert first.weights_tested == 2
    assert second.model_name == first.model_name
    assert second.primary_weight == first.primary_weight
    assert second.selection_mrr == first.selection_mrr


def test_refined_weight_scan_uses_31_point_grid_and_ignores_forward_rows():
    weights = inclusive_weight_grid(0.75, 0.90, 0.005)
    assert len(weights) == 31
    assert weights[0] == 0.75
    assert weights[10] == 0.80
    assert weights[-1] == 0.90

    setwise = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    lightgbm = 1.0 - setwise
    first = scan_high_weight_blend_on_prefix(
        setwise,
        lightgbm,
        selection_stop=4,
        primary_weights=weights,
    )
    changed_forward = setwise.copy()
    changed_forward[4:] = lightgbm[4:]
    second = scan_high_weight_blend_on_prefix(
        changed_forward,
        lightgbm,
        selection_stop=4,
        primary_weights=weights,
    )

    assert first.weights_tested == 31
    assert first.primary_weight == 0.90
    assert second.primary_weight == first.primary_weight
    assert second.selection_mrr == first.selection_mrr


def test_authorized_setwise_weight_requires_passing_setwise_report():
    report = {
        "status": "passed",
        "gate_passed": True,
        "package_authorized": True,
        "winner": "setwise",
        "setwise": {
            "gate_passed": True,
            "selected_weight": 0.96,
        },
    }

    assert authorized_setwise_weight(report) == 0.96

    invalid = {
        **report,
        "setwise": {
            **report["setwise"],
            "selected_weight": 1.01,
        },
    }
    with pytest.raises(ValueError, match="between zero and one"):
        authorized_setwise_weight(invalid)


def test_uniform_rank_average_uses_equal_per_model_query_local_ranks():
    first = np.asarray(
        [
            [0.9, 0.8, 0.1],
            [0.1, 0.9, 0.8],
        ],
        dtype=np.float64,
    )
    second = np.asarray(
        [
            [0.1, 0.9, 0.8],
            [0.8, 0.1, 0.9],
        ],
        dtype=np.float64,
    )
    third = np.asarray(
        [
            [0.8, 0.1, 0.9],
            [0.9, 0.8, 0.1],
        ],
        dtype=np.float64,
    )

    averaged = uniform_rank_average((first, second, third))

    assert averaged.shape == first.shape
    assert averaged.dtype == np.float64
    np.testing.assert_allclose(averaged, np.full_like(first, 0.5))


def test_uniform_rank_average_rejects_misaligned_or_non_finite_models():
    with pytest.raises(ValueError, match="same shape"):
        uniform_rank_average(
            (
                np.zeros((2, 3)),
                np.zeros((2, 2)),
                np.zeros((2, 3)),
            )
        )
    with pytest.raises(ValueError, match="finite"):
        uniform_rank_average(
            (
                np.zeros((2, 3)),
                np.asarray([[0.0, np.nan, 0.0], [0.0, 0.0, 0.0]]),
                np.zeros((2, 3)),
            )
        )
