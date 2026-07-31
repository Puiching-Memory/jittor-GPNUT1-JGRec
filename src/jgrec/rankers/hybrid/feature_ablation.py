from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def retained_context_feature_indices(
    *,
    source_feature_count: int,
    excluded_source_indices: Sequence[int],
    context_copies: int,
) -> tuple[int, ...]:
    if source_feature_count <= 0 or context_copies <= 0:
        raise ValueError("feature count and context copies must be positive")
    excluded = tuple(int(index) for index in excluded_source_indices)
    if len(set(excluded)) != len(excluded):
        raise ValueError("excluded source indices must be unique")
    if excluded and (
        min(excluded) < 0 or max(excluded) >= source_feature_count
    ):
        raise ValueError("excluded source feature is out of bounds")
    excluded_context = {
        index + copy * source_feature_count
        for copy in range(context_copies)
        for index in excluded
    }
    return tuple(
        index
        for index in range(source_feature_count * context_copies)
        if index not in excluded_context
    )


def neutralize_feature_columns(
    features: np.ndarray,
    *,
    columns: Sequence[int],
    neutral_values: Sequence[float],
) -> np.ndarray:
    values = np.asarray(features)
    selected = _validated_columns(values, columns)
    replacements = np.asarray(neutral_values, dtype=values.dtype)
    if replacements.shape != (len(selected),):
        raise ValueError("neutral values must match selected columns")
    result = np.array(values, copy=True)
    result[..., selected] = replacements
    return result


def replace_feature_columns(
    features: np.ndarray,
    replacement: np.ndarray,
    *,
    columns: Sequence[int],
) -> np.ndarray:
    values = np.asarray(features)
    replacement_values = np.asarray(replacement)
    if values.shape != replacement_values.shape:
        raise ValueError("replacement features must have the same shape")
    selected = _validated_columns(values, columns)
    result = np.array(values, copy=True)
    result[..., selected] = replacement_values[..., selected]
    return result


def permute_candidate_feature_columns(
    features: np.ndarray,
    *,
    columns: Sequence[int],
    permutations: np.ndarray,
) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 3:
        raise ValueError("candidate permutation requires three-dimensional features")
    candidate_permutations = np.asarray(permutations)
    if candidate_permutations.shape != values.shape[:2]:
        raise ValueError("candidate permutations must match query and candidate axes")
    expected = np.arange(values.shape[1])
    if not np.all(np.sort(candidate_permutations, axis=1) == expected):
        raise ValueError("each candidate row must contain a complete permutation")
    replacement = np.take_along_axis(
        values,
        candidate_permutations[..., np.newaxis],
        axis=1,
    )
    return replace_feature_columns(
        values,
        replacement,
        columns=columns,
    )


def _validated_columns(
    features: np.ndarray,
    columns: Sequence[int],
) -> tuple[int, ...]:
    if features.ndim < 2:
        raise ValueError("features must have a feature dimension")
    selected = tuple(int(column) for column in columns)
    if not selected:
        raise ValueError("at least one feature column is required")
    if len(set(selected)) != len(selected):
        raise ValueError("feature columns must be unique")
    if min(selected) < 0 or max(selected) >= features.shape[-1]:
        raise ValueError("feature column is out of bounds")
    return selected
