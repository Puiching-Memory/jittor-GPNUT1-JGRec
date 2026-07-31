from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class OOFStackingFold:
    index: int
    train_rows: tuple[int, int]
    score_rows: tuple[int, int]
    role: Literal["meta_train", "meta_validation"]
    train_time_max: int
    score_time_min: int
    score_time_max: int


_EXPERT_LOGIT_CHANNELS = (
    "percentile_rank",
    "robust_z",
    "robust_margin_to_top",
    "top1_support",
    "normalized_entropy",
    "robust_top_gap",
)
_ENSEMBLE_LOGIT_CHANNELS = (
    "mean_percentile_rank",
    "std_percentile_rank",
    "mean_robust_z",
    "std_robust_z",
    "mean_top1_support",
)
_ROBUST_LOGIT_QUANTIZATION_DECIMALS = 3
_ROBUST_TIE_TOLERANCE = 1.5e-3
STABLE_EXPERT_LOGIT_FEATURE_VERSION = 2


def stable_expert_logit_feature_names(
    expert_names: Sequence[str],
) -> tuple[str, ...]:
    """Return the deterministic feature order used by the OOF meta model."""
    names = tuple(str(name).strip() for name in expert_names)
    if not names or any(not name for name in names):
        raise ValueError("OOF stacking requires non-empty expert names")
    if len(set(names)) != len(names):
        raise ValueError("OOF stacking expert names must be unique")
    return (
        *(
            f"{expert}__{channel}"
            for expert in names
            for channel in _EXPERT_LOGIT_CHANNELS
        ),
        *(f"experts__{channel}" for channel in _ENSEMBLE_LOGIT_CHANNELS),
    )


def stable_expert_logit_features(logits: np.ndarray) -> np.ndarray:
    """Convert expert logits into affine-stable candidate-set features.

    Args:
        logits: ``[experts, queries, candidates]`` raw expert outputs.

    Returns:
        ``[queries, candidates, 6 * experts + 5]`` float32 features.
    """
    values = np.asarray(logits)
    if values.ndim != 3:
        raise ValueError(
            "OOF expert logits require [experts, queries, candidates]"
        )
    expert_count, query_count, candidate_count = map(int, values.shape)
    if expert_count == 0 or query_count == 0 or candidate_count == 0:
        raise ValueError("OOF expert logits cannot contain an empty axis")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("OOF expert logits must be numeric")
    work = np.transpose(
        values.astype(np.float64, copy=False),
        (1, 2, 0),
    )
    if not np.all(np.isfinite(work)):
        raise ValueError("OOF expert logits must be finite")

    median = np.median(work, axis=1, keepdims=True)
    centered = work - median
    mad = np.median(np.abs(centered), axis=1, keepdims=True)
    scale = mad * 1.4826
    fallback = np.mean(np.abs(centered), axis=1, keepdims=True) * 1.253314
    scale = np.where(scale > 0.0, scale, fallback)

    robust_z = np.zeros_like(work)
    np.divide(centered, scale, out=robust_z, where=scale > 0.0)
    robust_z = np.clip(robust_z, -8.0, 8.0)
    row_span = np.max(work, axis=1, keepdims=True) - np.min(
        work,
        axis=1,
        keepdims=True,
    )
    magnitude = np.max(np.abs(work), axis=1, keepdims=True)
    numerical_noise_floor = np.maximum(1e-5, magnitude * 1e-6)
    stable_score = np.round(
        robust_z,
        decimals=_ROBUST_LOGIT_QUANTIZATION_DECIMALS,
    )
    stable_score = np.where(
        row_span <= numerical_noise_floor,
        0.0,
        stable_score,
    )
    percentile = _tie_neutral_candidate_percentile(stable_score)

    robust_margin = stable_score - np.max(
        stable_score,
        axis=1,
        keepdims=True,
    )

    is_top = (
        np.max(stable_score, axis=1, keepdims=True) - stable_score
        <= _ROBUST_TIE_TOLERANCE
    )
    top1_support = is_top / np.sum(is_top, axis=1, keepdims=True)

    shifted = stable_score - np.max(stable_score, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    if candidate_count == 1:
        entropy = np.zeros((query_count, 1, expert_count), dtype=np.float64)
        top_gap = np.zeros_like(entropy)
    else:
        entropy = -np.sum(
            probabilities * np.log(np.maximum(probabilities, 1e-300)),
            axis=1,
            keepdims=True,
        ) / np.log(float(candidate_count))
        top_two = np.partition(
            stable_score,
            candidate_count - 2,
            axis=1,
        )[
            :, -2:, :
        ]
        gap = np.max(top_two, axis=1, keepdims=True) - np.min(
            top_two,
            axis=1,
            keepdims=True,
        )
        gap = np.where(gap <= _ROBUST_TIE_TOLERANCE, 0.0, gap)
        top_gap = np.clip(gap, 0.0, 8.0)
    entropy = np.broadcast_to(entropy, work.shape)
    top_gap = np.broadcast_to(top_gap, work.shape)

    expert_channels = np.stack(
        (
            percentile,
            stable_score,
            robust_margin,
            top1_support,
            entropy,
            top_gap,
        ),
        axis=-1,
    ).reshape(query_count, candidate_count, expert_count * 6)
    ensemble_channels = np.stack(
        (
            percentile.mean(axis=2),
            percentile.std(axis=2),
            stable_score.mean(axis=2),
            stable_score.std(axis=2),
            top1_support.mean(axis=2),
        ),
        axis=-1,
    )
    return np.concatenate(
        (expert_channels, ensemble_channels),
        axis=-1,
    ).astype(np.float32, copy=False)


def tie_neutral_mrr(
    scores: np.ndarray,
    positive_indices: np.ndarray,
) -> float:
    """Return MRR with every exact tie assigned its average rank.

    Candidate-set validation commonly stores the positive at column zero.
    Counting only strictly greater negatives would therefore reward a model
    that emits ties. Average ranks make the metric invariant to where the
    positive happens to appear in the candidate row.
    """
    values = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positive_indices)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(
            "tie-neutral MRR requires non-empty 2D candidate scores"
        )
    if positives.shape != (values.shape[0],) or not np.issubdtype(
        positives.dtype,
        np.integer,
    ):
        raise ValueError(
            "tie-neutral MRR requires one integer positive index per row"
        )
    positives = positives.astype(np.int64, copy=False)
    if np.any(positives < 0) or np.any(positives >= values.shape[1]):
        raise ValueError("tie-neutral MRR positive index is out of range")
    if not np.all(np.isfinite(values)):
        raise ValueError("tie-neutral MRR scores must be finite")
    positive_scores = np.take_along_axis(
        values,
        positives[:, None],
        axis=1,
    )
    greater = np.sum(values > positive_scores, axis=1)
    equal = np.sum(values == positive_scores, axis=1) - 1
    average_rank = 1.0 + greater + 0.5 * equal
    return float(np.mean(1.0 / average_rank))


class StableExpertLogitFeatureView:
    """Lazy stable-feature view over expert logits without a large cache."""

    def __init__(
        self,
        logits: Any,
        *,
        row_start: int = 0,
        row_stop: int | None = None,
    ) -> None:
        if len(logits.shape) != 3:
            raise ValueError(
                "stable logit view requires "
                "[experts, queries, candidates]"
            )
        start = int(row_start)
        stop = int(logits.shape[1] if row_stop is None else row_stop)
        if not 0 <= start < stop <= int(logits.shape[1]):
            raise ValueError("stable logit view row bounds are invalid")
        self._logits = logits
        self._row_start = start
        self._row_stop = stop
        self.shape = (
            stop - start,
            int(logits.shape[2]),
            int(logits.shape[0]) * len(_EXPERT_LOGIT_CHANNELS)
            + len(_ENSEMBLE_LOGIT_CHANNELS),
        )
        self.ndim = 3
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if index < 0:
                index += self.shape[0]
            if not 0 <= index < self.shape[0]:
                raise IndexError("stable logit view index is out of range")
            selected = np.asarray(
                self._logits[:, self._row_start + index, :],
            )
            return stable_expert_logit_features(selected[:, None, :])[0]
        if isinstance(key, slice):
            local_start, local_stop, step = key.indices(self.shape[0])
            source = slice(
                self._row_start + local_start,
                self._row_start + local_stop,
                step,
            )
            return stable_expert_logit_features(
                np.asarray(self._logits[:, source, :])
            )
        local_indices = np.asarray(key)
        if not np.issubdtype(local_indices.dtype, np.integer):
            raise IndexError("stable logit view indices must be integers")
        local_indices = local_indices.astype(np.int64, copy=False)
        if np.any(local_indices < 0):
            local_indices = np.where(
                local_indices < 0,
                local_indices + self.shape[0],
                local_indices,
            )
        if np.any(local_indices < 0) or np.any(
            local_indices >= self.shape[0]
        ):
            raise IndexError("stable logit view index is out of range")
        return stable_expert_logit_features(
            np.asarray(
                self._logits[
                    :,
                    local_indices + self._row_start,
                    :,
                ]
            )
        )


def oof_row_fold_assignments(
    total_rows: int,
    folds: Sequence[OOFStackingFold],
) -> np.ndarray:
    """Map each scored row to exactly one OOF fold; warmup rows stay -1."""
    row_count = int(total_rows)
    ordered = tuple(folds)
    if row_count <= 0:
        raise ValueError("OOF total_rows must be positive")
    if not ordered:
        raise ValueError("OOF assignments require at least one fold")
    assignments = np.full(row_count, -1, dtype=np.int16)
    expected_start = int(ordered[0].score_rows[0])
    if not 0 < expected_start < row_count:
        raise ValueError("OOF warmup boundary is outside the cache")
    for expected_index, fold in enumerate(ordered):
        if fold.index != expected_index:
            raise ValueError("OOF fold indices must be contiguous")
        start, stop = map(int, fold.score_rows)
        if start < expected_start:
            raise ValueError("OOF score folds overlap")
        if start > expected_start:
            raise ValueError("OOF score folds contain a gap")
        if stop <= start or stop > row_count:
            raise ValueError("OOF score fold boundary is invalid")
        if fold.train_rows != (0, start):
            raise ValueError(
                "OOF fold training rows must be the expanding prefix"
            )
        assignments[start:stop] = expected_index
        expected_start = stop
    if expected_start != row_count:
        raise ValueError("OOF score folds contain a trailing gap")
    return assignments


def expanding_timestamp_oof_folds(
    query_times: np.ndarray,
    *,
    warmup_rows: int,
    fold_rows: int,
    fold_count: int,
    meta_train_fold_count: int,
) -> tuple[OOFStackingFold, ...]:
    """Build expanding folds whose score blocks start at new timestamps."""
    times = np.asarray(query_times)
    warmup = int(warmup_rows)
    width = int(fold_rows)
    count = int(fold_count)
    train_fold_count = int(meta_train_fold_count)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("OOF query times must be a non-empty vector")
    if not np.issubdtype(times.dtype, np.integer):
        raise ValueError("OOF query times must use an integer dtype")
    times64 = times.astype(np.int64, copy=False)
    if np.any(np.diff(times64) < 0):
        raise ValueError("OOF query times must be non-decreasing")
    if warmup <= 0 or width <= 0 or count < 2:
        raise ValueError("OOF row counts must be positive")
    if not 0 < train_fold_count < count:
        raise ValueError(
            "OOF meta-train folds must leave a meta-validation fold"
        )
    target_boundaries = tuple(
        warmup + index * width for index in range(count)
    )
    if target_boundaries[-1] >= times64.size:
        raise ValueError("OOF cache is too short for requested folds")
    aligned = tuple(
        int(np.searchsorted(times64, times64[target], side="left"))
        for target in target_boundaries
    )
    if aligned[0] <= 0 or any(
        right <= left for left, right in pairwise(aligned)
    ):
        raise ValueError(
            "OOF timestamp groups are too large for requested folds"
        )
    stops = (*aligned[1:], int(times64.size))
    folds = tuple(
        OOFStackingFold(
            index=index,
            train_rows=(0, score_start),
            score_rows=(score_start, score_stop),
            role=(
                "meta_train"
                if index < train_fold_count
                else "meta_validation"
            ),
            train_time_max=int(times64[score_start - 1]),
            score_time_min=int(times64[score_start]),
            score_time_max=int(times64[score_stop - 1]),
        )
        for index, (score_start, score_stop) in enumerate(
            zip(aligned, stops, strict=True)
        )
    )
    if any(
        fold.train_time_max >= fold.score_time_min for fold in folds
    ):
        raise RuntimeError(
            "OOF timestamp alignment failed to separate origins"
        )
    return folds


def _tie_neutral_candidate_percentile(values: np.ndarray) -> np.ndarray:
    candidate_count = int(values.shape[1])
    if candidate_count == 1:
        return np.full_like(values, 0.5, dtype=np.float64)

    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    positions = np.arange(candidate_count, dtype=np.float64).reshape(
        1,
        candidate_count,
        1,
    )
    group_start = np.empty_like(sorted_values, dtype=bool)
    group_start[:, 0, :] = True
    group_start[:, 1:, :] = (
        np.abs(sorted_values[:, 1:, :] - sorted_values[:, :-1, :])
        > _ROBUST_TIE_TOLERANCE
    )
    group_end = np.empty_like(sorted_values, dtype=bool)
    group_end[:, -1, :] = True
    group_end[:, :-1, :] = (
        np.abs(sorted_values[:, :-1, :] - sorted_values[:, 1:, :])
        > _ROBUST_TIE_TOLERANCE
    )
    lower = np.maximum.accumulate(
        np.where(group_start, positions, 0.0),
        axis=1,
    )
    upper = np.minimum.accumulate(
        np.where(group_end, positions, float(candidate_count - 1))[
            :, ::-1, :
        ],
        axis=1,
    )[:, ::-1, :]
    sorted_percentile = (lower + upper) * (
        0.5 / float(candidate_count - 1)
    )
    percentile = np.empty_like(sorted_percentile)
    np.put_along_axis(percentile, order, sorted_percentile, axis=1)
    return percentile
