from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from jgrec.core.types import InteractionTable

from .sequence_model import time_delta_bucket


@dataclass(frozen=True)
class SourceSequenceRows:
    items: np.ndarray
    time_buckets: np.ndarray
    lengths: np.ndarray


@dataclass(frozen=True)
class SourceConditionedFold:
    index: int
    train_rows: tuple[int, int]
    score_rows: tuple[int, int]
    role: Literal["selection", "gate"]
    train_time_max: int
    score_time_min: int
    score_time_max: int


def expanding_timestamp_abcd_folds(
    query_times: np.ndarray,
) -> tuple[SourceConditionedFold, ...]:
    """Build 40/20, 60/20, and 80/20 expanding timestamp folds."""
    times = np.asarray(query_times)
    if (
        times.ndim != 1
        or times.size < 5
        or not np.issubdtype(times.dtype, np.integer)
    ):
        raise ValueError(
            "ABCD rolling-origin times must be a non-empty integer vector"
        )
    times64 = times.astype(np.int64, copy=False)
    if np.any(np.diff(times64) < 0):
        raise ValueError("ABCD rolling-origin times must be non-decreasing")

    boundaries = [
        _timestamp_boundary(times64, fraction)
        for fraction in (0.4, 0.6, 0.8)
    ]
    if (
        boundaries[0] <= 0
        or not boundaries[0] < boundaries[1] < boundaries[2] < times.size
    ):
        raise ValueError(
            "timestamp groups are too coarse for three ABCD folds"
        )

    score_stops = (boundaries[1], boundaries[2], int(times.size))
    folds: list[SourceConditionedFold] = []
    for index, (score_start, score_stop) in enumerate(
        zip(boundaries, score_stops, strict=True)
    ):
        if times64[score_start - 1] >= times64[score_start]:
            raise RuntimeError("ABCD fold splits an equal timestamp")
        folds.append(
            SourceConditionedFold(
                index=index,
                train_rows=(0, score_start),
                score_rows=(score_start, score_stop),
                role="selection" if index < 2 else "gate",
                train_time_max=int(times64[score_start - 1]),
                score_time_min=int(times64[score_start]),
                score_time_max=int(times64[score_stop - 1]),
            )
        )
    return tuple(folds)


def build_causal_source_sequences(
    interactions: InteractionTable,
    *,
    query_src: np.ndarray,
    query_time: np.ndarray,
    max_length: int,
    history_time_limit: int | None = None,
) -> SourceSequenceRows:
    """Build right-padded histories using only strictly earlier events.

    ``history_time_limit`` freezes the available graph at an origin. Both the
    query time and the origin are strict upper bounds, so equal-time events
    never enter the history.
    """
    sources = np.asarray(query_src, dtype=np.int32)
    times = np.asarray(query_time, dtype=np.int64)
    if sources.ndim != 1 or times.ndim != 1 or sources.shape != times.shape:
        raise ValueError("source sequence queries require aligned vectors")
    if max_length <= 0:
        raise ValueError("source sequence max_length must be positive")
    if history_time_limit is not None and not isinstance(
        history_time_limit,
        (int, np.integer),
    ):
        raise ValueError("history_time_limit must be an integer")

    item_rows = np.zeros(
        (sources.size, int(max_length)),
        dtype=np.int32,
    )
    bucket_rows = np.zeros_like(item_rows)
    lengths = np.zeros(sources.size, dtype=np.int32)
    if sources.size == 0 or len(interactions) == 0:
        return SourceSequenceRows(item_rows, bucket_rows, lengths)

    interaction_src = np.asarray(interactions.src, dtype=np.int32)
    interaction_dst = np.asarray(interactions.dst, dtype=np.int32)
    interaction_time = np.asarray(interactions.time, dtype=np.int64)
    original_rows = np.arange(len(interactions), dtype=np.int64)
    order = np.lexsort((original_rows, interaction_time, interaction_src))
    grouped_src = interaction_src[order]
    grouped_dst = interaction_dst[order]
    grouped_time = interaction_time[order]
    unique_src, starts, counts = np.unique(
        grouped_src,
        return_index=True,
        return_counts=True,
    )
    source_groups = {
        int(src): (int(start), int(start + count))
        for src, start, count in zip(
            unique_src,
            starts,
            counts,
            strict=True,
        )
    }
    frozen_limit = (
        None if history_time_limit is None else int(history_time_limit)
    )

    for row, (source, query_at) in enumerate(
        zip(sources, times, strict=True)
    ):
        group = source_groups.get(int(source))
        if group is None:
            continue
        start, stop = group
        upper_time = int(query_at)
        if frozen_limit is not None:
            upper_time = min(upper_time, frozen_limit)
        local_stop = int(
            np.searchsorted(
                grouped_time[start:stop],
                upper_time,
                side="left",
            )
        )
        take_start = max(0, local_stop - int(max_length))
        selected = slice(start + take_start, start + local_stop)
        selected_items = grouped_dst[selected]
        selected_times = grouped_time[selected]
        length = int(selected_items.size)
        if length == 0:
            continue
        item_rows[row, :length] = selected_items
        deltas = np.maximum(int(query_at) - selected_times, 0)
        bucket_rows[row, :length] = time_delta_bucket(deltas)
        lengths[row] = length

    return SourceSequenceRows(
        items=item_rows,
        time_buckets=bucket_rows,
        lengths=lengths,
    )


def _timestamp_boundary(times: np.ndarray, fraction: float) -> int:
    target = min(max(int(times.size * fraction), 1), times.size - 1)
    return int(np.searchsorted(times, times[target], side="left"))

