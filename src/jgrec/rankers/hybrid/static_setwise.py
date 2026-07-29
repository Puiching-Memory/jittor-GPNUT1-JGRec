from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

_DATASET1_STATIC_WEIGHTS = tuple(
    index / 100.0 for index in range(5, 81, 5)
)


def static_setwise_weight_grid() -> tuple[float, ...]:
    """Return the preregistered Dataset1 static Setwise weights."""

    return _DATASET1_STATIC_WEIGHTS


def blend_static_setwise(
    backbone_scores: np.ndarray,
    setwise_scores: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Blend one query-invariant Setwise weight into backbone scores."""

    backbone = np.asarray(backbone_scores, dtype=np.float64)
    setwise = np.asarray(setwise_scores, dtype=np.float64)
    scalar = float(weight)
    if backbone.shape != setwise.shape or backbone.ndim != 2:
        raise ValueError("static Setwise score matrices must align")
    if (
        not np.isfinite(backbone).all()
        or not np.isfinite(setwise).all()
        or not np.isfinite(scalar)
        or scalar not in _DATASET1_STATIC_WEIGHTS
    ):
        raise ValueError("invalid preregistered static Setwise blend")
    return (1.0 - scalar) * backbone + scalar * setwise


def apply_prediction_history_limit(config: Any, *, limit: int) -> Any:
    """Return a config with both prediction-history limits changed."""

    value = int(limit)
    if value <= 0:
        raise ValueError("prediction history limit must be positive")
    return replace(
        config,
        structure_predict_neighbor_limit=value,
        source_profile_predict_history_limit=value,
    )


def select_dual_horizon_static_weight(
    trials: Mapping[float, Mapping[str, Sequence[float]]],
) -> dict[str, Any]:
    """Apply the preregistered near/gapped eligibility and order."""

    reports: dict[str, dict[str, Any]] = {}
    eligible: list[tuple[tuple[float, ...], float]] = []
    required = (
        "near_mrr",
        "near_ndcg_at_10",
        "gapped_mrr",
        "gapped_ndcg_at_10",
    )
    for raw_weight, raw_metrics in sorted(trials.items()):
        weight = float(raw_weight)
        if weight not in _DATASET1_STATIC_WEIGHTS:
            raise ValueError("trial weight is outside the preregistered grid")
        metrics = {
            name: tuple(float(value) for value in raw_metrics[name])
            for name in required
        }
        if any(len(values) != 3 for values in metrics.values()):
            raise ValueError("dual-horizon selection requires three folds")
        if not all(
            np.isfinite(value)
            for values in metrics.values()
            for value in values
        ):
            raise ValueError("dual-horizon deltas must be finite")
        failed = []
        if any(value < 0.0 for value in metrics["near_mrr"]):
            failed.append("near_mrr")
        if any(value < 0.0 for value in metrics["near_ndcg_at_10"]):
            failed.append("near_ndcg_at_10")
        if any(value <= 0.0 for value in metrics["gapped_mrr"]):
            failed.append("gapped_mrr_strict")
        if any(value < 0.0 for value in metrics["gapped_ndcg_at_10"]):
            failed.append("gapped_ndcg_at_10")
        trial_report = {
            **{name: list(values) for name, values in metrics.items()},
            "eligible": not failed,
            "failed_gates": failed,
            "mean_gapped_mrr_delta": float(
                np.mean(metrics["gapped_mrr"])
            ),
            "worst_gapped_mrr_delta": min(metrics["gapped_mrr"]),
            "mean_near_mrr_delta": float(np.mean(metrics["near_mrr"])),
        }
        reports[f"{weight:.2f}"] = trial_report
        if not failed:
            order = (
                trial_report["mean_gapped_mrr_delta"],
                trial_report["worst_gapped_mrr_delta"],
                trial_report["mean_near_mrr_delta"],
                weight,
            )
            eligible.append((order, weight))
    selected = max(eligible)[1] if eligible else None
    return {
        "status": "selected" if selected is not None else "no_candidate",
        "selected_weight": selected,
        "eligible_weights": [
            weight for _, weight in sorted(eligible, key=lambda item: item[1])
        ],
        "trials": reports,
    }


def evaluate_external_safety_deltas(
    deltas: Mapping[str, float],
    *,
    improved: int,
    worsened: int,
) -> dict[str, Any]:
    """Evaluate the preregistered seven external direction gates."""

    values = {
        name: float(deltas[name])
        for name in (
            "mrr",
            "hit_at_1",
            "hit_at_3",
            "hit_at_10",
            "ndcg_at_10",
            "mean_rank",
        )
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("external metric deltas must be finite")
    gates = {
        "mrr_strictly_increases": values["mrr"] > 0.0,
        "hit_at_1_non_decreasing": values["hit_at_1"] >= 0.0,
        "hit_at_3_non_decreasing": values["hit_at_3"] >= 0.0,
        "hit_at_10_non_decreasing": values["hit_at_10"] >= 0.0,
        "ndcg_at_10_non_decreasing": values["ndcg_at_10"] >= 0.0,
        "mean_rank_non_increasing": values["mean_rank"] <= 0.0,
        "improved_exceeds_worsened": int(improved) > int(worsened),
    }
    return {"accepted": all(gates.values()), "gates": gates}
