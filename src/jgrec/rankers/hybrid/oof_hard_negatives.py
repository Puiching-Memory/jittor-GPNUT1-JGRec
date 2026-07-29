from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class ContiguousOOFFold:
    index: int
    holdout: slice
    fit_slices: tuple[slice, ...]


def contiguous_oof_folds(*, row_count: int, fold_count: int) -> tuple[ContiguousOOFFold, ...]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    if row_count < fold_count:
        raise ValueError("row_count must provide at least one row per fold")

    base_size, remainder = divmod(row_count, fold_count)
    boundaries = [0]
    for index in range(fold_count):
        boundaries.append(boundaries[-1] + base_size + (1 if index < remainder else 0))

    folds: list[ContiguousOOFFold] = []
    for index, (start, stop) in enumerate(pairwise(boundaries)):
        fit_slices = tuple(
            current
            for current in (slice(0, start), slice(stop, row_count))
            if current.start < current.stop
        )
        folds.append(
            ContiguousOOFFold(
                index=index,
                holdout=slice(start, stop),
                fit_slices=fit_slices,
            )
        )
    return tuple(folds)


def select_hard_negative_positions(
    scores: np.ndarray,
    *,
    keep_negatives: int,
) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must contain query groups with one positive and at least one negative")
    negative_count = values.shape[1] - 1
    if keep_negatives < 1 or keep_negatives > negative_count:
        raise ValueError("keep_negatives must be between one and the available negative count")

    ranked_negatives = np.argsort(-values[:, 1:], axis=1, kind="stable")[:, :keep_negatives] + 1
    positive_positions = np.zeros((values.shape[0], 1), dtype=np.int64)
    return np.concatenate((positive_positions, ranked_negatives.astype(np.int64, copy=False)), axis=1)


def select_hard_negative_features(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    keep_negatives: int,
) -> np.ndarray:
    values = np.asarray(features)
    score_values = np.asarray(scores)
    if values.ndim != 3 or values.shape[:2] != score_values.shape:
        raise ValueError("features and scores must share query and candidate dimensions")
    positions = select_hard_negative_positions(score_values, keep_negatives=keep_negatives)
    return np.take_along_axis(values, positions[:, :, None], axis=1)


def passes_temporal_mrr_gate(
    *,
    candidate_slices: tuple[float, ...],
    baseline_slices: tuple[float, ...],
    candidate_full_mrr: float,
    baseline_full_mrr: float,
    min_full_delta: float,
) -> bool:
    candidate = np.asarray(candidate_slices, dtype=np.float64)
    baseline = np.asarray(baseline_slices, dtype=np.float64)
    scalars = np.asarray(
        [candidate_full_mrr, baseline_full_mrr, min_full_delta],
        dtype=np.float64,
    )
    if (
        candidate.ndim != 1
        or candidate.size < 2
        or candidate.shape != baseline.shape
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(scalars))
        or min_full_delta < 0.0
    ):
        raise ValueError("temporal MRR gate inputs are invalid")
    full_delta = float(candidate_full_mrr) - float(baseline_full_mrr)
    return bool(np.all(candidate >= baseline) and full_delta + 1e-12 >= min_full_delta)
