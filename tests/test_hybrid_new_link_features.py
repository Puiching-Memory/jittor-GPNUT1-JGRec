from __future__ import annotations

import numpy as np
import pytest

from jgrec.rankers.hybrid.new_link_features import (
    NEW_LINK_GROWTH_FEATURE_NAMES,
    append_new_link_growth_features,
)


def test_append_new_link_growth_features_derives_ratio_and_activity_cross_without_mutation() -> None:
    feature_names = (
        "src_activity",
        "target_pop_share_w001",
        "target_pop_share_w100",
        "unrelated",
    )
    features = np.asarray(
        [
            [[0.5, 0.02, 0.01, 7.0], [0.25, 0.00, 0.00, 8.0]],
            [[0.8, 0.03, 0.06, 9.0], [0.10, 0.04, 0.02, 10.0]],
        ],
        dtype=np.float32,
    )
    original = features.copy()

    augmented, augmented_names = append_new_link_growth_features(features, feature_names)

    expected_growth = np.log1p(np.asarray([[2.0, 0.0], [0.5, 2.0]], dtype=np.float64))
    np.testing.assert_allclose(augmented[..., -2], expected_growth, rtol=1e-6)
    np.testing.assert_allclose(augmented[..., -1], features[..., 0] * expected_growth, rtol=1e-6)
    np.testing.assert_array_equal(augmented[..., : features.shape[-1]], original)
    np.testing.assert_array_equal(features, original)
    assert augmented_names == feature_names + NEW_LINK_GROWTH_FEATURE_NAMES
    assert augmented.dtype == np.float32


def test_append_new_link_growth_features_rejects_missing_or_misaligned_names() -> None:
    features = np.zeros((2, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="feature names"):
        append_new_link_growth_features(features, ("src_activity",))
    with pytest.raises(ValueError, match="required"):
        append_new_link_growth_features(features, ("src_activity", "a", "b"))
