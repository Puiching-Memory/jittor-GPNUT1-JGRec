from __future__ import annotations

from typing import Any

import numpy as np


def setwise_context_features(
    features: np.ndarray,
    *,
    transform_version: int = 1,
) -> np.ndarray:
    """Add the requested candidate-set-relative channels to every candidate."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(
            "setwise context features require [queries, candidates, features]"
        )
    multiplier = _context_feature_multiplier(transform_version)
    row_mean = values.mean(axis=1, keepdims=True, dtype=np.float32)
    row_max = values.max(axis=1, keepdims=True)
    channels = [values, values - row_mean, values - row_max]
    if multiplier == 5:
        channels.extend(
            (
                _tie_neutral_percentile_rank(values),
                _robust_row_zscore(values),
            )
        )
    return np.concatenate(
        channels,
        axis=-1,
        dtype=np.float32,
    )


class SetwiseFeatureView:
    """Lazy row-wise set-context view over a cache or memmap."""

    def __init__(
        self,
        features: Any,
        *,
        transform_version: int = 1,
    ) -> None:
        if len(features.shape) != 3:
            raise ValueError(
                "setwise feature view requires [queries, candidates, features]"
            )
        multiplier = _context_feature_multiplier(transform_version)
        self._features = features
        self._transform_version = transform_version
        self.shape = (
            int(features.shape[0]),
            int(features.shape[1]),
            int(features.shape[2]) * multiplier,
        )
        self.ndim = 3
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key: Any) -> np.ndarray:
        selected = np.asarray(self._features[key], dtype=np.float32)
        if selected.ndim == 2:
            return setwise_context_features(
                selected[np.newaxis, ...],
                transform_version=self._transform_version,
            )[0]
        return setwise_context_features(
            selected,
            transform_version=self._transform_version,
        )


def _context_feature_multiplier(transform_version: int) -> int:
    if transform_version == 1:
        return 3
    if transform_version == 2:
        return 5
    raise ValueError(
        f"unsupported Setwise context transform version: {transform_version}"
    )


def _tie_neutral_percentile_rank(values: np.ndarray) -> np.ndarray:
    candidate_count = int(values.shape[1])
    if candidate_count == 1:
        return np.full_like(values, 0.5, dtype=np.float32)

    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    positions = np.arange(candidate_count, dtype=np.float32).reshape(
        1,
        candidate_count,
        1,
    )
    group_start = np.empty_like(sorted_values, dtype=bool)
    group_start[:, 0, :] = True
    group_start[:, 1:, :] = (
        sorted_values[:, 1:, :] != sorted_values[:, :-1, :]
    )
    group_end = np.empty_like(sorted_values, dtype=bool)
    group_end[:, -1, :] = True
    group_end[:, :-1, :] = (
        sorted_values[:, :-1, :] != sorted_values[:, 1:, :]
    )
    lower = np.maximum.accumulate(
        np.where(group_start, positions, 0.0),
        axis=1,
    )
    upper = np.minimum.accumulate(
        np.where(
            group_end,
            positions,
            np.float32(candidate_count - 1),
        )[:, ::-1, :],
        axis=1,
    )[:, ::-1, :]
    sorted_percentile = (
        (lower + upper) * np.float32(0.5 / (candidate_count - 1))
    )
    percentile = np.empty_like(sorted_percentile, dtype=np.float32)
    np.put_along_axis(percentile, order, sorted_percentile, axis=1)
    return percentile


def _robust_row_zscore(values: np.ndarray) -> np.ndarray:
    row_median = np.median(values, axis=1, keepdims=True).astype(
        np.float32,
        copy=False,
    )
    centered = values - row_median
    row_mad = np.median(
        np.abs(centered),
        axis=1,
        keepdims=True,
    ).astype(np.float32, copy=False)
    robust = np.zeros_like(values, dtype=np.float32)
    np.divide(
        centered,
        row_mad * np.float32(1.4826),
        out=robust,
        where=row_mad > np.float32(1e-6),
    )
    return robust
