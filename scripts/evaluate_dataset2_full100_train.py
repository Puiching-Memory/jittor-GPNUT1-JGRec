from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.rankers.hybrid.full100_training import passes_full100_gate
from jgrec.rankers.hybrid.fusion import build_fusion_from_state, predict_logits
from jgrec.rankers.hybrid.fusion_lgbm import (
    _flatten_for_ranking,
    predict_logits_lgbm,
)
from jgrec.rankers.hybrid.lgbm_tuning import predeclared_dataset2_lgbm_grid

EXPECTED_BASELINE_MRR = 0.5428303297309955


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train frozen Dataset2 LightGBM on full-100 groups and run the exact gate."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-cache-prefix", required=True, type=Path)
    parser.add_argument("--full100-prefix", required=True, type=Path)
    parser.add_argument("--cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--boost-rounds", type=int, default=308)
    parser.add_argument("--mlp-weight", type=float, default=0.07)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "dataset2-full100-lgbm.txt"
    report_path = args.output_dir / "full100-evaluation.json"
    frozen_path = args.output_dir / "full100-frozen-config.json"
    if model_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite a completed full-100 evaluation")

    started = time.time()
    cache_report = json.loads(args.cache_report.read_text(encoding="utf-8"))
    if cache_report.get("status") != "complete":
        raise RuntimeError("full-100 cache build is incomplete")
    train_path = Path(f"{args.full100_prefix}.train.npy")
    train_features = np.load(train_path, mmap_mode="r", allow_pickle=False)
    val_path = args.source_cache_prefix.with_suffix(".val.npy")
    val_features = np.load(val_path, mmap_mode="r", allow_pickle=False)
    if train_features.shape != (50_000, 100, 63):
        raise ValueError(f"full-100 training feature shape mismatch: {train_features.shape}")
    if val_features.shape != (20_000, 100, 63):
        raise ValueError(f"frozen validation feature shape mismatch: {val_features.shape}")
    if _sha256(train_path) != cache_report["artifacts"]["features"]["sha256"]:
        raise ValueError("full-100 training feature hash differs from the build report")
    if _sha256(val_path) != cache_report["source_validation_sha256"]:
        raise ValueError("frozen validation tensor changed after the cache build")

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    fusion_result = state["fusion_result"]
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("Dataset2 checkpoint has no LightGBM expert")
    feature_indices = tuple(int(index) for index in fusion_result.feature_indices)
    if feature_names != tuple(cache_report["feature_names"]):
        raise ValueError("cache feature schema differs from the checkpoint")
    if feature_indices != tuple(range(63)):
        raise ValueError("the frozen champion MLP must use all 63 features")
    if tuple(int(index) for index in lgbm_result.feature_indices) != feature_indices:
        raise ValueError("the frozen champion experts must share all 63 features")
    if int(config.seed) != args.seed:
        raise ValueError("evaluation seed differs from the checkpoint seed")
    if abs(float(lgbm_result.mlp_weight) - args.mlp_weight) > 1e-12:
        raise ValueError("fixed blend weight differs from the source champion")

    params = dict(
        dict(
            predeclared_dataset2_lgbm_grid(
                seed=args.seed,
                num_threads=args.num_threads,
            )
        )["lr003"]
    )
    frozen = {
        "status": "frozen_before_training",
        "training_scope": "Dataset2 LightGBM only; MLP and every tower frozen",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_path": str(train_path.resolve()),
        "train_shape": list(train_features.shape),
        "train_sha256": cache_report["artifacts"]["features"]["sha256"],
        "validation_path": str(val_path.resolve()),
        "validation_shape": list(val_features.shape),
        "validation_sha256": cache_report["source_validation_sha256"],
        "feature_names": list(feature_names),
        "params": params,
        "boost_rounds": args.boost_rounds,
        "mlp_weight": args.mlp_weight,
        "min_full_delta": args.min_full_delta,
        "expected_baseline_mrr": EXPECTED_BASELINE_MRR,
        "selection": "none; one frozen fit without validation tuning or early stopping",
        "validation_slices": [[0, 6667], [6667, 13334], [13334, 20000]],
    }
    _write_json_atomic(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    import lightgbm as lgb  # noqa: PLC0415

    flatten_started = time.time()
    train_X, train_y, train_group = _flatten_for_ranking(
        train_features,
        feature_indices,
    )
    train_ds = lgb.Dataset(
        train_X,
        label=train_y,
        group=train_group,
        feature_name=list(feature_names),
        params={"feature_pre_filter": False},
        free_raw_data=True,
    )
    train_ds.construct()
    del train_X, train_y, train_group
    gc.collect()
    print(
        f"[full100-eval] LightGBM dataset constructed seconds={time.time() - flatten_started:.1f}",
        flush=True,
    )

    training_started = time.time()
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=args.boost_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    candidate_model_text = booster.model_to_string(num_iteration=args.boost_rounds)
    model_path.write_text(candidate_model_text, encoding="utf-8")
    training_seconds = time.time() - training_started
    del train_ds, train_features
    gc.collect()
    print(
        f"[full100-eval] LightGBM training complete seconds={training_seconds:.1f}",
        flush=True,
    )

    mlp_model = build_fusion_from_state(
        input_dim=len(feature_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    mlp = _softmax(
        predict_logits(
            mlp_model,
            val_features,
            fusion_result.mean,
            fusion_result.std,
        )
    )
    baseline_lgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, val_features)
    )
    candidate_lgbm = _softmax(
        predict_logits_lgbm(candidate_model_text, val_features)
    )
    baseline_blend = args.mlp_weight * mlp + (1.0 - args.mlp_weight) * baseline_lgbm
    candidate_blend = args.mlp_weight * mlp + (1.0 - args.mlp_weight) * candidate_lgbm
    slices = (slice(0, 6667), slice(6667, 13334), slice(13334, 20000))
    baseline_metrics = _temporal_mrr(baseline_blend, slices)
    if abs(float(baseline_metrics["full"]) - EXPECTED_BASELINE_MRR) > 1e-12:
        raise RuntimeError(
            "source champion baseline mismatch: "
            f"actual={baseline_metrics['full']:.16f} expected={EXPECTED_BASELINE_MRR:.16f}"
        )
    candidate_metrics = _temporal_mrr(candidate_blend, slices)
    baseline_slices = tuple(float(item["mrr"]) for item in baseline_metrics["slices"])
    candidate_slices = tuple(float(item["mrr"]) for item in candidate_metrics["slices"])
    passed = passes_full100_gate(
        baseline_full_mrr=float(baseline_metrics["full"]),
        candidate_full_mrr=float(candidate_metrics["full"]),
        baseline_slice_mrrs=baseline_slices,
        candidate_slice_mrrs=candidate_slices,
        min_full_delta=args.min_full_delta,
    )
    slice_deltas = [
        candidate - baseline
        for baseline, candidate in zip(
            baseline_slices,
            candidate_slices,
            strict=True,
        )
    ]
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "package_authorized": passed,
        "package_generated": False,
        "frozen_config": frozen,
        "baseline": {
            "lgbm": _temporal_mrr(baseline_lgbm, slices),
            "fixed_blend": baseline_metrics,
        },
        "candidate": {
            "lgbm": _temporal_mrr(candidate_lgbm, slices),
            "fixed_blend": candidate_metrics,
            "blend_full_delta": float(
                candidate_metrics["full"] - baseline_metrics["full"]
            ),
            "blend_slice_deltas": slice_deltas,
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
        },
        "gate": {
            "min_full_delta": args.min_full_delta,
            "full_delta_passed": bool(
                candidate_metrics["full"] - baseline_metrics["full"] + 1e-12
                >= args.min_full_delta
            ),
            "all_slices_non_decreasing": bool(
                all(delta >= 0.0 for delta in slice_deltas)
            ),
        },
        "training_seconds": training_seconds,
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _mrr(scores: np.ndarray) -> float:
    values = np.asarray(scores)
    ranks = 1 + np.sum(values[:, 1:] > values[:, :1], axis=1)
    return float(np.mean(1.0 / ranks))


def _temporal_mrr(
    scores: np.ndarray,
    slices: tuple[slice, ...],
) -> dict[str, Any]:
    return {
        "full": _mrr(scores),
        "slices": [
            {
                "start": int(part.start or 0),
                "stop": int(part.stop or len(scores)),
                "rows": int((part.stop or len(scores)) - (part.start or 0)),
                "mrr": _mrr(scores[part]),
            }
            for part in slices
        ],
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
