from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from jgrec.rankers.hybrid.cooccur_lift import BASE_FEATURE_COUNT

FULL_ONLY_FEATURE_COUNT = 64
GAP_AWARE_FEATURE_COUNT = 66


class _CooccurLiftSuccessorView:
    def __init__(
        self,
        source: Any,
        *,
        short_none_scores: np.ndarray,
        gnn_short_column: int,
        lift_features: np.ndarray,
        feature_count: int,
    ) -> None:
        if len(source.shape) != 3 or int(source.shape[-1]) != BASE_FEATURE_COUNT:
            raise ValueError("source cache must have shape [rows, candidates, 63]")
        matrix_shape = tuple(int(value) for value in source.shape[:2])
        if tuple(short_none_scores.shape) != matrix_shape:
            raise ValueError("short_none scores must match source rows and candidates")
        if tuple(lift_features.shape) != (*matrix_shape, 2):
            raise ValueError("lift features must have shape [rows, candidates, 2]")
        column = int(gnn_short_column)
        if not 0 <= column < BASE_FEATURE_COUNT:
            raise ValueError("gnn_short_column is outside the 63 base columns")
        self._source = source
        self._short_none_scores = short_none_scores
        self._gnn_short_column = column
        self._lift_features = lift_features
        self.shape = (*matrix_shape, int(feature_count))
        self.ndim = 3
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def _base(self, key: Any) -> np.ndarray:
        base = np.array(self._source[key], dtype=np.float32, copy=True)
        base[..., self._gnn_short_column] = self._short_none_scores[key]
        return base


class CooccurLiftFullOnlyView(_CooccurLiftSuccessorView):
    """The v2 view that retains only the full-history lift channel."""

    def __init__(
        self,
        source: Any,
        *,
        short_none_scores: np.ndarray,
        gnn_short_column: int,
        lift_features: np.ndarray,
    ) -> None:
        super().__init__(
            source,
            short_none_scores=short_none_scores,
            gnn_short_column=gnn_short_column,
            lift_features=lift_features,
            feature_count=FULL_ONLY_FEATURE_COUNT,
        )

    def __getitem__(self, key: Any) -> np.ndarray:
        base = self._base(key)
        full_lift = np.asarray(
            self._lift_features[key][..., :1],
            dtype=np.float32,
        )
        return np.concatenate((base, full_lift), axis=-1, dtype=np.float32)


class CooccurLiftGapAwareView(_CooccurLiftSuccessorView):
    """The v2 view with full/short lift and explicit row-level support."""

    def __init__(
        self,
        source: Any,
        *,
        short_none_scores: np.ndarray,
        gnn_short_column: int,
        lift_features: np.ndarray,
        short_window_supported: np.ndarray,
    ) -> None:
        super().__init__(
            source,
            short_none_scores=short_none_scores,
            gnn_short_column=gnn_short_column,
            lift_features=lift_features,
            feature_count=GAP_AWARE_FEATURE_COUNT,
        )
        support = np.asarray(short_window_supported, dtype=np.float32)
        if support.shape != (self.shape[0],):
            raise ValueError("short-window support must contain one value per row")
        if not np.all((support == 0.0) | (support == 1.0)):
            raise ValueError("short-window support values must be binary")
        self._short_window_supported = support

    def __getitem__(self, key: Any) -> np.ndarray:
        base = self._base(key)
        lift = np.asarray(self._lift_features[key], dtype=np.float32)
        support = np.asarray(
            self._short_window_supported[key],
            dtype=np.float32,
        )
        if base.ndim == 2:
            support_column = np.full(
                (*base.shape[:-1], 1),
                float(support),
                dtype=np.float32,
            )
        else:
            support_column = np.broadcast_to(
                support[..., None, None],
                (*base.shape[:-1], 1),
            )
        return np.concatenate(
            (base, lift, support_column),
            axis=-1,
            dtype=np.float32,
        )


class ConcatenatedFeatureView:
    """Lazy row-wise concatenation for weighted near/stale training copies."""

    def __init__(self, sources: Sequence[Any]) -> None:
        if not sources:
            raise ValueError("at least one feature source is required")
        tail = tuple(int(value) for value in sources[0].shape[1:])
        if len(tail) != 2:
            raise ValueError("feature sources must be three-dimensional")
        for source in sources:
            if len(source.shape) != 3 or tuple(source.shape[1:]) != tail:
                raise ValueError("concatenated feature sources must share a schema")
        self._sources = tuple(sources)
        lengths = np.asarray(
            [int(source.shape[0]) for source in self._sources],
            dtype=np.int64,
        )
        self._stops = np.cumsum(lengths)
        self.shape = (int(self._stops[-1]), *tail)
        self.ndim = 3
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key: Any) -> np.ndarray:
        selected = np.arange(self.shape[0], dtype=np.int64)[key]
        if np.ndim(selected) == 0:
            row = int(selected)
            source_index = int(np.searchsorted(self._stops, row, side="right"))
            start = 0 if source_index == 0 else int(self._stops[source_index - 1])
            return np.asarray(
                self._sources[source_index][row - start],
                dtype=np.float32,
            )

        selected_array = np.asarray(selected, dtype=np.int64)
        flat = selected_array.reshape(-1)
        output = np.empty(
            (len(flat), *self.shape[1:]),
            dtype=np.float32,
        )
        start = 0
        for source, stop in zip(self._sources, self._stops, strict=True):
            mask = (flat >= start) & (flat < stop)
            if np.any(mask):
                output[mask] = source[flat[mask] - start]
            start = int(stop)
        return output.reshape((*selected_array.shape, *self.shape[1:]))


def short_window_support(
    query_time: np.ndarray,
    availability_time: np.ndarray,
    *,
    short_window_seconds: int,
) -> np.ndarray:
    query_values = np.asarray(query_time, dtype=np.int64)
    availability_values = np.asarray(availability_time, dtype=np.int64)
    if query_values.shape != availability_values.shape:
        raise ValueError("query and availability times must have the same shape")
    if short_window_seconds <= 0:
        raise ValueError("short_window_seconds must be positive")
    if np.any(availability_values > query_values):
        raise ValueError("availability time must not exceed query time")
    return (
        query_values - availability_values < int(short_window_seconds)
    ).astype(np.float32)
