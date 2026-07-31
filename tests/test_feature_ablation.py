import numpy as np

from jgrec.rankers.hybrid.feature_ablation import (
    neutralize_feature_columns,
    permute_candidate_feature_columns,
    replace_feature_columns,
    retained_context_feature_indices,
)


def test_neutralize_feature_columns_copies_input_and_sets_selected_means():
    features = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    original = features.copy()

    actual = neutralize_feature_columns(
        features,
        columns=(1, 3),
        neutral_values=(10.5, -2.0),
    )

    np.testing.assert_array_equal(features, original)
    np.testing.assert_array_equal(actual[..., 0], features[..., 0])
    np.testing.assert_array_equal(actual[..., 2], features[..., 2])
    np.testing.assert_array_equal(actual[..., 1], 10.5)
    np.testing.assert_array_equal(actual[..., 3], -2.0)


def test_replace_feature_columns_copies_input_and_uses_aligned_replacement():
    features = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    replacement = features + 100.0
    original = features.copy()

    actual = replace_feature_columns(
        features,
        replacement,
        columns=(0, 2),
    )

    np.testing.assert_array_equal(features, original)
    np.testing.assert_array_equal(actual[..., 0], replacement[..., 0])
    np.testing.assert_array_equal(actual[..., 2], replacement[..., 2])
    np.testing.assert_array_equal(actual[..., 1], features[..., 1])
    np.testing.assert_array_equal(actual[..., 3], features[..., 3])


def test_permute_candidate_feature_columns_breaks_candidate_alignment_only():
    features = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    original = features.copy()
    permutations = np.asarray([[2, 0, 1], [1, 2, 0]])

    actual = permute_candidate_feature_columns(
        features,
        columns=(1, 3),
        permutations=permutations,
    )

    expected = np.take_along_axis(
        features,
        permutations[..., np.newaxis],
        axis=1,
    )
    np.testing.assert_array_equal(features, original)
    np.testing.assert_array_equal(actual[..., 0], features[..., 0])
    np.testing.assert_array_equal(actual[..., 2], features[..., 2])
    np.testing.assert_array_equal(actual[..., 1], expected[..., 1])
    np.testing.assert_array_equal(actual[..., 3], expected[..., 3])


def test_retained_context_feature_indices_removes_each_derived_copy():
    actual = retained_context_feature_indices(
        source_feature_count=4,
        excluded_source_indices=(1, 3),
        context_copies=3,
    )

    assert actual == (0, 2, 4, 6, 8, 10)
