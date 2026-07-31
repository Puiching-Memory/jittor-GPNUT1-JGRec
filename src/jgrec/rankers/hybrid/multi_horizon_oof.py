from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .source_sequence_cache import SourceConditionedFold

HORIZON_NAMES = ("short", "medium", "long")
SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class MultiHorizonSlice:
    horizon: str
    horizon_index: int
    origin_index: int
    target_fold_index: int
    train_rows: tuple[int, int]
    score_rows: tuple[int, int]
    train_stop: int
    train_time_max: int
    history_time_limit: int
    score_time_min: int
    score_time_max: int


@dataclass(frozen=True)
class MultiHorizonOOFPiece:
    horizon: str
    origin_index: int
    score_rows: tuple[int, int]
    train_stop: int
    train_time_max: int
    history_time_limit: int
    query_times: Any
    base_logits: Any
    corrected_logits: Any


@dataclass
class MultiHorizonOOFArtifact:
    residuals: np.ndarray
    base_logits: np.ndarray
    corrected_logits: np.ndarray
    valid_mask: np.ndarray
    origin_index: np.ndarray
    gap_days: np.ndarray


def canonical_multi_horizon_slices(
    folds: Iterable[SourceConditionedFold],
) -> tuple[MultiHorizonSlice, ...]:
    """Map expanding folds to short/medium/long frozen-origin slices."""
    rows = tuple(folds)
    if len(rows) < len(HORIZON_NAMES):
        raise ValueError("multi-horizon OOF requires at least three folds")
    _validate_folds(rows)

    result: list[MultiHorizonSlice] = []
    for horizon_index, horizon in enumerate(HORIZON_NAMES):
        for origin_position in range(len(rows) - horizon_index):
            origin = rows[origin_position]
            target = rows[origin_position + horizon_index]
            train_stop = int(origin.train_rows[1])
            score_start, score_stop = (
                int(target.score_rows[0]),
                int(target.score_rows[1]),
            )
            if train_stop > score_start:
                raise ValueError("multi-horizon train and score rows overlap")
            if int(origin.train_time_max) >= int(target.score_time_min):
                raise ValueError(
                    "multi-horizon score times must be strictly after training"
                )
            result.append(
                MultiHorizonSlice(
                    horizon=horizon,
                    horizon_index=horizon_index,
                    origin_index=int(origin.index),
                    target_fold_index=int(target.index),
                    train_rows=(
                        int(origin.train_rows[0]),
                        int(origin.train_rows[1]),
                    ),
                    score_rows=(score_start, score_stop),
                    train_stop=train_stop,
                    train_time_max=int(origin.train_time_max),
                    history_time_limit=int(origin.score_time_min),
                    score_time_min=int(target.score_time_min),
                    score_time_max=int(target.score_time_max),
                )
            )
    return tuple(result)


def assemble_multi_horizon_oof(
    pieces: Iterable[MultiHorizonOOFPiece],
    *,
    row_count: int,
    candidate_count: int,
) -> MultiHorizonOOFArtifact:
    """Assemble disjoint slices into a fixed horizon-major OOF tensor."""
    if row_count <= 0 or candidate_count <= 0:
        raise ValueError("multi-horizon output dimensions must be positive")
    shape = (len(HORIZON_NAMES), int(row_count), int(candidate_count))
    base = np.zeros(shape, dtype=np.float32)
    corrected = np.zeros(shape, dtype=np.float32)
    residuals = np.zeros(shape, dtype=np.float32)
    valid = np.zeros(shape[:2], dtype=bool)
    origins = np.full(shape[:2], -1, dtype=np.int16)
    gaps = np.full(shape[:2], np.nan, dtype=np.float32)
    horizon_to_index = {
        name: index for index, name in enumerate(HORIZON_NAMES)
    }

    for piece in pieces:
        if piece.horizon not in horizon_to_index:
            raise ValueError(f"unknown OOF horizon: {piece.horizon}")
        horizon_index = horizon_to_index[piece.horizon]
        start, stop = (int(piece.score_rows[0]), int(piece.score_rows[1]))
        if not 0 <= start < stop <= row_count:
            raise ValueError("multi-horizon score rows are out of bounds")
        if int(piece.train_stop) > start:
            raise ValueError("multi-horizon train and score rows overlap")
        target = (horizon_index, slice(start, stop))
        if np.any(valid[target]):
            raise ValueError(
                f"multi-horizon overlap for {piece.horizon} rows "
                f"[{start}, {stop})"
            )

        rows = stop - start
        piece_base = np.asarray(piece.base_logits, dtype=np.float32)
        piece_corrected = np.asarray(
            piece.corrected_logits,
            dtype=np.float32,
        )
        query_times = np.asarray(piece.query_times, dtype=np.int64)
        if (
            piece_base.shape != (rows, candidate_count)
            or piece_corrected.shape != piece_base.shape
            or query_times.shape != (rows,)
        ):
            raise ValueError("multi-horizon piece arrays do not align")
        if (
            not np.isfinite(piece_base).all()
            or not np.isfinite(piece_corrected).all()
        ):
            raise ValueError("multi-horizon logits must be finite")
        if np.any(np.diff(query_times) < 0):
            raise ValueError("multi-horizon query times must be non-decreasing")

        train_time_max = int(piece.train_time_max)
        history_limit = int(piece.history_time_limit)
        if (
            train_time_max >= int(query_times[0])
            or train_time_max >= history_limit
            or history_limit > int(query_times[0])
        ):
            raise ValueError(
                "multi-horizon score must be strictly after its training "
                "origin and use an origin-frozen history"
            )

        piece_residual = piece_corrected - piece_base
        base[target] = piece_base
        corrected[target] = piece_corrected
        residuals[target] = piece_residual
        valid[target] = True
        origins[target] = int(piece.origin_index)
        gaps[target] = (
            query_times.astype(np.float64) - float(train_time_max)
        ) / SECONDS_PER_DAY

    return MultiHorizonOOFArtifact(
        residuals=residuals,
        base_logits=base,
        corrected_logits=corrected,
        valid_mask=valid,
        origin_index=origins,
        gap_days=gaps,
    )


def audit_multi_horizon_oof(
    artifact: MultiHorizonOOFArtifact,
    *,
    cap: float,
    tolerance: float = 2e-6,
) -> dict[str, Any]:
    """Audit replay, hard cap, zero-fill and temporal metadata."""
    limit = float(cap)
    allowed_error = float(tolerance)
    if (
        not math.isfinite(limit)
        or not 0.0 <= limit <= 0.10
        or not math.isfinite(allowed_error)
        or allowed_error < 0.0
    ):
        raise ValueError("multi-horizon audit bounds are invalid")

    residuals = np.asarray(artifact.residuals)
    base = np.asarray(artifact.base_logits)
    corrected = np.asarray(artifact.corrected_logits)
    valid = np.asarray(artifact.valid_mask)
    origins = np.asarray(artifact.origin_index)
    gaps = np.asarray(artifact.gap_days)
    expected_shape = residuals.shape
    if (
        residuals.ndim != 3
        or residuals.shape[0] != len(HORIZON_NAMES)
        or base.shape != expected_shape
        or corrected.shape != expected_shape
        or valid.shape != expected_shape[:2]
        or origins.shape != expected_shape[:2]
        or gaps.shape != expected_shape[:2]
    ):
        raise ValueError("multi-horizon artifact arrays do not align")

    valid_values = residuals[valid]
    maximum = (
        float(np.max(np.abs(valid_values))) if valid_values.size else 0.0
    )
    replay_values = (base + residuals - corrected)[valid]
    replay_error = (
        float(np.max(np.abs(replay_values)))
        if replay_values.size
        else 0.0
    )
    row_means = np.mean(residuals[valid], axis=1) if np.any(valid) else []
    row_mean_error = (
        float(np.max(np.abs(row_means))) if len(row_means) else 0.0
    )
    invalid_zero = bool(
        np.count_nonzero(residuals[~valid]) == 0
        and np.count_nonzero(base[~valid]) == 0
        and np.count_nonzero(corrected[~valid]) == 0
    )
    finite = bool(
        np.isfinite(residuals[valid]).all()
        and np.isfinite(base[valid]).all()
        and np.isfinite(corrected[valid]).all()
        and np.isfinite(gaps[valid]).all()
    )
    metadata_passed = bool(
        np.all(origins[valid] >= 0)
        and np.all(origins[~valid] == -1)
        and np.all(gaps[valid] > 0.0)
        and np.isnan(gaps[~valid]).all()
    )
    cap_passed = maximum <= limit + allowed_error
    replay_passed = replay_error <= allowed_error
    row_centered_passed = row_mean_error <= allowed_error
    coverage = {
        name: int(np.sum(valid[index]))
        for index, name in enumerate(HORIZON_NAMES)
    }

    return {
        "passed": bool(
            cap_passed
            and replay_passed
            and row_centered_passed
            and invalid_zero
            and finite
            and metadata_passed
        ),
        "cap": limit,
        "tolerance": allowed_error,
        "max_absolute_residual": maximum,
        "max_cap_violation": maximum - limit,
        "cap_passed": bool(cap_passed),
        "max_replay_error": replay_error,
        "replay_passed": bool(replay_passed),
        "max_absolute_row_mean": row_mean_error,
        "row_centered_passed": bool(row_centered_passed),
        "invalid_zero_passed": invalid_zero,
        "finite_passed": finite,
        "metadata_passed": metadata_passed,
        "coverage_rows": coverage,
    }


def _validate_folds(folds: tuple[SourceConditionedFold, ...]) -> None:
    previous_score_stop = -1
    for expected_index, fold in enumerate(folds):
        train_start, train_stop = map(int, fold.train_rows)
        score_start, score_stop = map(int, fold.score_rows)
        if (
            int(fold.index) != expected_index
            or train_start != 0
            or train_stop != score_start
            or not train_start < train_stop < score_stop
            or int(fold.train_time_max) >= int(fold.score_time_min)
            or int(fold.score_time_min) > int(fold.score_time_max)
        ):
            raise ValueError("folds do not form a strict expanding origin")
        if previous_score_stop >= 0 and score_start != previous_score_stop:
            raise ValueError("fold score windows must be contiguous")
        previous_score_stop = score_stop
