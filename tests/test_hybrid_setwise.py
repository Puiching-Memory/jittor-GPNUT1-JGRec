from __future__ import annotations

import numpy as np
import pytest

from jgrec.rankers.hybrid.setwise import (
    SetwiseFeatureView,
    setwise_context_features,
)


def test_setwise_context_v1_remains_the_default_and_is_explicitly_selectable() -> None:
    values = np.asarray([[[1.0, 4.0], [3.0, 2.0]]], dtype=np.float32)
    expected = np.asarray(
        [
            [
                [1.0, 4.0, -1.0, 1.0, -2.0, 0.0],
                [3.0, 2.0, 1.0, -1.0, 0.0, -2.0],
            ]
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(setwise_context_features(values), expected)
    np.testing.assert_array_equal(
        setwise_context_features(values, transform_version=1),
        expected,
    )


def test_setwise_context_v2_adds_tie_neutral_percentile_and_robust_zscore() -> None:
    values = np.asarray(
        [
            [
                [1.0, 7.0],
                [1.0, 7.0],
                [3.0, 7.0],
                [5.0, 7.0],
            ]
        ],
        dtype=np.float32,
    )
    transformed = setwise_context_features(values, transform_version=2)

    expected_percentile = np.asarray(
        [
            [
                [1.0 / 6.0, 0.5],
                [1.0 / 6.0, 0.5],
                [2.0 / 3.0, 0.5],
                [1.0, 0.5],
            ]
        ],
        dtype=np.float32,
    )
    robust_scale = np.float32(1.4826)
    expected_robust = np.asarray(
        [
            [
                [-1.0 / robust_scale, 0.0],
                [-1.0 / robust_scale, 0.0],
                [1.0 / robust_scale, 0.0],
                [3.0 / robust_scale, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    assert transformed.shape == (1, 4, 10)
    np.testing.assert_allclose(
        transformed[..., 6:8],
        expected_percentile,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        transformed[..., 8:10],
        expected_robust,
        rtol=0.0,
        atol=1e-6,
    )


def test_setwise_context_v2_percentiles_follow_values_not_candidate_positions() -> None:
    original = np.asarray(
        [[[2.0], [2.0], [8.0], [-1.0]]],
        dtype=np.float32,
    )
    permutation = np.asarray([2, 0, 3, 1])

    transformed = setwise_context_features(original, transform_version=2)
    permuted = setwise_context_features(
        original[:, permutation],
        transform_version=2,
    )

    np.testing.assert_array_equal(
        permuted,
        transformed[:, permutation],
    )
    assert transformed[0, 0, 3] == transformed[0, 1, 3]


def test_setwise_feature_view_v2_reports_five_times_source_width() -> None:
    values = np.asarray(
        [[[1.0, 4.0], [3.0, 2.0], [2.0, 3.0]]],
        dtype=np.float32,
    )

    view = SetwiseFeatureView(values, transform_version=2)

    assert view.shape == (1, 3, 10)
    np.testing.assert_array_equal(
        view[:],
        setwise_context_features(values, transform_version=2),
    )


def test_setwise_context_rejects_unknown_transform_version() -> None:
    values = np.zeros((1, 3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="transform version"):
        setwise_context_features(values, transform_version=3)

