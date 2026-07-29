from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.rankers.hybrid.static_setwise import (
    static_setwise_weight_grid,
)

QUANTILES = (0.75, 0.90, 1.00)
NEAR_SCORE_STARTS = (100_000, 125_000, 150_000)
GAPPED_SCORE_STARTS = (170_000, 180_000, 190_000)
TRAIN_ROWS = 100_000
SCORE_ROWS = 25_000
GAPPED_SCORE_ROWS = 10_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-times", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sidecar = args.output.with_suffix(f"{args.output.suffix}.sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    cache_times = np.load(
        args.cache_times,
        mmap_mode="r",
        allow_pickle=False,
    )
    if cache_times.shape != (200_000,) or np.any(np.diff(cache_times) < 0):
        raise ValueError("Dataset1 plan requires sorted recent-200k times")
    train_time_max = max(_csv_times(args.train_csv, column=2))
    test_times = np.asarray(
        tuple(_csv_times(args.test_csv, column=1)),
        dtype=np.int64,
    )
    horizons = test_times - train_time_max
    if horizons.size == 0 or int(horizons.min()) <= 0:
        raise ValueError("Dataset1 deployment horizons must be positive")
    gaps = tuple(
        math.ceil(
            float(np.quantile(horizons, quantile, method="higher"))
        )
        for quantile in QUANTILES
    )

    near_folds = [
        _near_fold(cache_times, index, score_start)
        for index, score_start in enumerate(NEAR_SCORE_STARTS)
    ]
    gapped_folds = [
        _gapped_fold(
            cache_times,
            quantile=quantile,
            gap_seconds=gap,
            score_start=score_start,
        )
        for quantile, gap, score_start in zip(
            QUANTILES,
            gaps,
            GAPPED_SCORE_STARTS,
            strict=True,
        )
    ]
    weights = static_setwise_weight_grid()
    candidates = []
    for priority, weight in enumerate(reversed(weights)):
        payload = {
            "integration": "static_setwise_over_exact_fold_backbone",
            "prediction_history_limit": 256,
            "setwise_weight": weight,
        }
        candidates.append(
            {
                "candidate_id": f"static_setwise_w{int(weight * 100):03d}",
                "config": payload,
                "config_sha256": _json_sha256(payload),
                "tie_break_priority": priority,
            }
        )
    candidates.sort(key=lambda value: float(value["config"]["setwise_weight"]))
    plan = {
        "schema_version": 1,
        "protocol": "dataset1_k256_static_setwise_dual_horizon_v1",
        "status": "frozen_before_training_or_metric_read",
        "experiment_id": "dataset1_k256_static_setwise_dual_horizon_20260730",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "cache_times": str(args.cache_times.resolve()),
        "cache_times_sha256": _sha256(args.cache_times),
        "external_labels_read": False,
        "prediction_limits": {
            "structure_predict_neighbor_limit": 256,
            "source_profile_predict_history_limit": 256,
        },
        "cache_contract": {
            "train_rows": 200_000,
            "candidate_count": 100,
            "feature_count": 63,
        },
        "head_training": {
            "fit_and_tune_within_fold": True,
            "inner_tune_rows": 20_000,
            "seed": 60,
            "setwise_epochs": 10,
            "setwise_patience": 2,
            "setwise_batch_size": 256,
            "setwise_hidden_dim": 32,
            "setwise_learning_rate": 0.001,
        },
        "baseline": {
            "integration": "exact_fold_backbone_to_setwise_time_ramp",
            "time_ramp_power": 0.5,
            "global_time_bounds": [
                int(cache_times[NEAR_SCORE_STARTS[0]]),
                int(cache_times[-1]),
            ],
        },
        "candidate_space": candidates,
        "near_folds": near_folds,
        "deployment_reference_time": train_time_max,
        "deployment_horizon_seconds": {
            f"p{int(quantile * 100)}": gap
            for quantile, gap in zip(QUANTILES, gaps, strict=True)
        },
        "gapped_folds": gapped_folds,
        "eligibility": {
            "near_per_fold_minimum_deltas": {
                "mrr": 0.0,
                "ndcg_at_10": 0.0,
            },
            "gapped_per_fold_strictly_increasing": ["mrr"],
            "gapped_per_fold_minimum_deltas": {"ndcg_at_10": 0.0},
            "selection_order": [
                "maximum_mean_gapped_mrr_delta",
                "maximum_worst_gapped_mrr_delta",
                "maximum_mean_near_mrr_delta",
                "higher_static_weight",
            ],
        },
        "external_gate": {
            "decision_role": "safety_gate_only",
            "effect_size_estimation_authorized": False,
            "minimum_deltas": {
                "mrr": 0.0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_10": 0.0,
                "ndcg_at_10": 0.0,
            },
            "strictly_increasing_metrics": ["mrr"],
            "maximum_deltas": {"mean_rank": 0.0},
            "minimum_improved_minus_worsened": 1,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    digest = _sha256(args.output)
    sidecar.write_text(f"{digest}  {args.output.name}\n", encoding="ascii")
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"plan_sha256={digest}")
    return 0


def _near_fold(
    times: np.ndarray,
    index: int,
    score_start: int,
) -> dict[str, Any]:
    train_start = score_start - TRAIN_ROWS
    score_stop = score_start + SCORE_ROWS
    return {
        "fold_id": f"near-{index}",
        "train_rows": [train_start, score_start],
        "score_rows": [score_start, score_stop],
        "train_time_max": int(times[score_start - 1]),
        "score_time_min": int(times[score_start]),
        "score_time_max": int(times[score_stop - 1]),
    }


def _gapped_fold(
    times: np.ndarray,
    *,
    quantile: float,
    gap_seconds: int,
    score_start: int,
) -> dict[str, Any]:
    score_stop = score_start + GAPPED_SCORE_ROWS
    score_time_min = int(times[score_start])
    cutoff = score_time_min - gap_seconds
    train_stop = min(
        score_start,
        int(np.searchsorted(times, cutoff, side="right")),
    )
    train_start = train_stop - TRAIN_ROWS
    if train_start < 0:
        raise ValueError(
            f"gapped p{int(quantile * 100)} cannot fit a 100k train window"
        )
    actual_gap = score_time_min - int(times[train_stop - 1])
    if actual_gap < gap_seconds:
        raise RuntimeError("constructed gapped fold misses its minimum gap")
    return {
        "fold_id": f"gapped-p{int(quantile * 100)}",
        "deployment_horizon_quantile": quantile,
        "minimum_gap_seconds": gap_seconds,
        "actual_gap_seconds": actual_gap,
        "train_rows": [train_start, train_stop],
        "score_rows": [score_start, score_stop],
        "train_time_max": int(times[train_stop - 1]),
        "score_time_min": score_time_min,
        "score_time_max": int(times[score_stop - 1]),
    }


def _csv_times(path: Path, *, column: int):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.reader(handle)
        next(rows)
        for row in rows:
            yield int(row[column])


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
