from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpertBlendCalibration:
    mode: str
    mlp_temperature: float = 1.0
    lgbm_temperature: float = 1.0
    rrf_k: float = 60.0


def blend_expert_logits(
    mlp_logits: np.ndarray,
    lgbm_logits: np.ndarray,
    mlp_weight: float | np.ndarray,
    *,
    calibration: ExpertBlendCalibration,
) -> np.ndarray:
    mlp = _validated_logits(mlp_logits, "MLP")
    lgbm = _validated_logits(lgbm_logits, "LGBM")
    if mlp.shape != lgbm.shape:
        raise ValueError("expert logits must have matching query-by-candidate shapes")
    weights = _query_weights(mlp_weight, mlp.shape[0])
    mode = calibration.mode.strip().lower()

    if mode == "probability":
        mlp_scores = _softmax(mlp)
        lgbm_scores = _softmax(lgbm)
    elif mode == "temperature":
        mlp_scores = _softmax(
            mlp / _positive_temperature(calibration.mlp_temperature, "MLP")
        )
        lgbm_scores = _softmax(
            lgbm / _positive_temperature(calibration.lgbm_temperature, "LGBM")
        )
    elif mode == "rrf":
        rrf_k = float(calibration.rrf_k)
        if not np.isfinite(rrf_k) or rrf_k <= 0.0:
            raise ValueError("RRF k must be finite and positive")
        mlp_scores = 1.0 / (rrf_k + _average_descending_ranks(mlp))
        lgbm_scores = 1.0 / (rrf_k + _average_descending_ranks(lgbm))
    else:
        raise ValueError(f"unsupported expert blend mode: {calibration.mode!r}")

    blended = (
        weights[:, None] * mlp_scores
        + (1.0 - weights[:, None]) * lgbm_scores
    )
    row_sums = blended.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
        raise ValueError("expert blend produced an invalid row sum")
    return blended / row_sums


def fit_positive_column_temperature(logits: np.ndarray) -> float:
    values = _validated_logits(logits, "calibration")
    candidates = np.geomspace(0.05, 20.0, num=161)
    losses = np.asarray(
        [positive_column_nll(values, float(value)) for value in candidates]
    )
    return float(candidates[int(np.argmin(losses))])


def positive_column_nll(logits: np.ndarray, temperature: float = 1.0) -> float:
    values = _validated_logits(logits, "calibration")
    scale = _positive_temperature(temperature, "calibration")
    scaled = values / scale
    row_max = scaled.max(axis=1, keepdims=True)
    log_normalizer = (
        row_max[:, 0]
        + np.log(np.exp(scaled - row_max).sum(axis=1))
    )
    return float(np.mean(log_normalizer - scaled[:, 0]))


def _validated_logits(values: np.ndarray, name: str) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    if (
        logits.ndim != 2
        or logits.shape[1] < 2
        or not np.all(np.isfinite(logits))
    ):
        raise ValueError(
            f"{name} logits must be a finite query-by-candidate matrix"
        )
    return logits


def _query_weights(values: float | np.ndarray, row_count: int) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.ndim == 0:
        weights = np.full(row_count, float(weights), dtype=np.float64)
    if weights.shape != (row_count,) or not np.all(np.isfinite(weights)):
        raise ValueError("MLP weights must contain one finite value per query")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("MLP weights must be between zero and one")
    return weights


def _positive_temperature(value: float, name: str) -> float:
    temperature = float(value)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{name} temperature must be finite and positive")
    return temperature


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _average_descending_ranks(values: np.ndarray) -> np.ndarray:
    n_rows, n_cols = values.shape
    order = np.argsort(-values, axis=1, kind="mergesort")
    sorted_values = np.take_along_axis(values, order, axis=1)
    positions = np.broadcast_to(
        np.arange(1, n_cols + 1, dtype=np.float64),
        (n_rows, n_cols),
    )
    group_starts = np.ones((n_rows, n_cols), dtype=bool)
    group_starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group_ids = np.cumsum(group_starts, axis=1) - 1
    flat_groups = (np.arange(n_rows)[:, None] * n_cols + group_ids).ravel()
    group_slots = n_rows * n_cols
    sums = np.bincount(
        flat_groups,
        weights=positions.ravel(),
        minlength=group_slots,
    )
    counts = np.bincount(flat_groups, minlength=group_slots)
    average = sums / np.maximum(counts, 1)
    sorted_ranks = average[flat_groups].reshape(n_rows, n_cols)
    ranks = np.empty((n_rows, n_cols), dtype=np.float64)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    return ranks
