from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .config import TwoTowerConfig

TWO_TOWER_SCREEN_ARMS = (
    "control",
    "optimizer_only",
    "inbatch_only",
    "combined",
)


def two_tower_screen_config(
    base: TwoTowerConfig,
    arm: str,
) -> TwoTowerConfig:
    normalized = str(arm).lower()
    if normalized not in TWO_TOWER_SCREEN_ARMS:
        raise ValueError(f"unsupported Two-Tower screen arm: {arm}")
    optimizer_enabled = normalized in {"optimizer_only", "combined"}
    in_batch_enabled = normalized in {"inbatch_only", "combined"}
    return replace(
        base,
        lr_schedule="cosine" if optimizer_enabled else "constant",
        min_lr_ratio=0.1 if optimizer_enabled else 0.0,
        weight_decay=1e-4 if optimizer_enabled else 0.0,
        in_batch_negatives=in_batch_enabled,
    )


def positive_ranks(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("candidate scores must have shape [queries, candidates]")
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate scores must be finite")
    positive = values[:, :1]
    greater = np.sum(
        values[:, 1:] > positive,
        axis=1,
        dtype=np.int32,
    )
    equal = np.sum(
        values[:, 1:] == positive,
        axis=1,
        dtype=np.int32,
    )
    return (
        1.0
        + greater.astype(np.float64)
        + 0.5 * equal.astype(np.float64)
    )


def ranking_metrics(scores: np.ndarray) -> dict[str, float]:
    ranks = positive_ranks(scores)
    reciprocal_ranks = 1.0 / ranks
    ndcg = np.where(
        ranks <= 10,
        1.0 / np.log2(ranks.astype(np.float64) + 1.0),
        0.0,
    )
    return {
        "mrr": float(np.mean(reciprocal_ranks)),
        "hit_at_1": float(np.mean(ranks <= 1)),
        "hit_at_3": float(np.mean(ranks <= 3)),
        "hit_at_10": float(np.mean(ranks <= 10)),
        "ndcg_at_10": float(np.mean(ndcg)),
        "mean_rank": float(np.mean(ranks)),
    }


def paired_rank_movements(
    control_ranks: np.ndarray,
    candidate_ranks: np.ndarray,
) -> dict[str, Any]:
    control = np.asarray(control_ranks)
    candidate = np.asarray(candidate_ranks)
    if control.ndim != 1 or candidate.shape != control.shape:
        raise ValueError("paired ranks must be aligned one-dimensional arrays")
    improved = int(np.sum(candidate < control))
    worsened = int(np.sum(candidate > control))
    unchanged = int(control.size - improved - worsened)
    return {
        "improved_queries": improved,
        "worsened_queries": worsened,
        "unchanged_queries": unchanged,
        "net_improved_queries": improved - worsened,
        "mean_rank_delta": float(
            np.mean(candidate.astype(np.float64) - control)
        ),
    }


def two_tower_screen_gate(
    control: dict[str, float],
    candidate: dict[str, float],
    control_slices: list[dict[str, float]],
    candidate_slices: list[dict[str, float]],
    movements: dict[str, Any],
) -> dict[str, Any]:
    if len(control_slices) != len(candidate_slices):
        raise ValueError("control and candidate slice metrics must align")

    tolerance = 1e-12
    checks: dict[str, bool] = {}
    for metric in (
        "mrr",
        "hit_at_1",
        "hit_at_3",
        "hit_at_10",
        "ndcg_at_10",
    ):
        checks[f"full_{metric}"] = bool(
            float(candidate[metric]) + tolerance >= float(control[metric])
        )
    checks["full_mean_rank"] = bool(
        float(candidate["mean_rank"])
        <= float(control["mean_rank"]) + tolerance
    )
    for index, (control_part, candidate_part) in enumerate(
        zip(control_slices, candidate_slices, strict=True)
    ):
        for metric in ("mrr", "ndcg_at_10"):
            checks[f"slice_{index}_{metric}"] = bool(
                float(candidate_part[metric]) + tolerance
                >= float(control_part[metric])
            )
    checks["query_movements"] = bool(
        int(movements["improved_queries"])
        > int(movements["worsened_queries"])
    )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }
