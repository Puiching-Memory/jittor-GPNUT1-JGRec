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
from jgrec.rankers.hybrid.full100_training import passes_matched_control_gate
from jgrec.rankers.hybrid.fusion import build_fusion_from_state, predict_logits
from jgrec.rankers.hybrid.fusion_lgbm import (
    _flatten_for_ranking,
    predict_logits_lgbm,
)
from jgrec.rankers.hybrid.lgbm_tuning import predeclared_dataset2_lgbm_grid

EXPECTED_BASELINE_MRR = 0.5428303297309955
SLICES = (slice(0, 6667), slice(6667, 13334), slice(13334, 20000))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and compare matched Dataset2 32- and 100-candidate LightGBMs."
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
    control_model_path = args.output_dir / "dataset2-matched32-lgbm.txt"
    full_model_path = args.output_dir / "dataset2-full100-lgbm.txt"
    report_path = args.output_dir / "matched-full100-evaluation.json"
    frozen_path = args.output_dir / "matched-full100-frozen-config.json"
    if any(
        path.exists()
        for path in (control_model_path, full_model_path, report_path, frozen_path)
    ):
        raise FileExistsError("refusing to overwrite a started matched evaluation")

    started = time.time()
    cache_report = json.loads(args.cache_report.read_text(encoding="utf-8"))
    if cache_report.get("status") != "complete":
        raise RuntimeError("matched cache build is incomplete")
    full_path = Path(f"{args.full100_prefix}.train.npy")
    val_path = args.source_cache_prefix.with_suffix(".val.npy")
    full_features = np.load(full_path, mmap_mode="r", allow_pickle=False)
    control_features = full_features[:, :32]
    val_features = np.load(val_path, mmap_mode="r", allow_pickle=False)
    if control_features.shape != (50_000, 32, 63):
        raise ValueError(f"matched control shape differs: {control_features.shape}")
    if full_features.shape != (50_000, 100, 63):
        raise ValueError(f"full-100 training shape differs: {full_features.shape}")
    if val_features.shape != (20_000, 100, 63):
        raise ValueError(f"frozen validation shape differs: {val_features.shape}")
    _require_hash(
        full_path,
        cache_report["artifacts"]["features"]["sha256"],
        "full-100 training",
    )
    _require_hash(
        val_path,
        cache_report["source_validation_sha256"],
        "frozen validation",
    )

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    fusion_result = state["fusion_result"]
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("Dataset2 checkpoint has no LightGBM expert")
    feature_indices = tuple(int(index) for index in fusion_result.feature_indices)
    if feature_names != tuple(cache_report["feature_names"]):
        raise ValueError("matched cache schema differs from the checkpoint")
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
        "protocol": cache_report["protocol"],
        "training_scope": "Dataset2 LightGBM only; MLP and all towers frozen",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "control_train_shape": list(control_features.shape),
        "control_derived_from_full100_positions": [0, 32],
        "full100_train_shape": list(full_features.shape),
        "full100_train_sha256": cache_report["artifacts"]["features"]["sha256"],
        "validation_shape": list(val_features.shape),
        "validation_sha256": cache_report["source_validation_sha256"],
        "feature_names": list(feature_names),
        "params": params,
        "boost_rounds": args.boost_rounds,
        "mlp_weight": args.mlp_weight,
        "min_full_delta": args.min_full_delta,
        "expected_baseline_mrr": EXPECTED_BASELINE_MRR,
        "selection": "none; two frozen fits without validation tuning or early stopping",
        "gate": (
            "full100 vs champion delta >= 0.002; all three slices >= champion; "
            "full100 full MRR >= matched32 full MRR"
        ),
    }
    _write_json_atomic(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    control_text, control_seconds = _train_lgbm(
        label="matched32",
        features=control_features,
        feature_indices=feature_indices,
        feature_names=feature_names,
        params=params,
        boost_rounds=args.boost_rounds,
    )
    control_model_path.write_text(control_text, encoding="utf-8")
    del control_features
    gc.collect()
    full_text, full_seconds = _train_lgbm(
        label="full100",
        features=full_features,
        feature_indices=feature_indices,
        feature_names=feature_names,
        params=params,
        boost_rounds=args.boost_rounds,
    )
    full_model_path.write_text(full_text, encoding="utf-8")
    del full_features
    gc.collect()

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
    control_lgbm = _softmax(predict_logits_lgbm(control_text, val_features))
    full_lgbm = _softmax(predict_logits_lgbm(full_text, val_features))
    baseline_blend = args.mlp_weight * mlp + (1.0 - args.mlp_weight) * baseline_lgbm
    control_blend = args.mlp_weight * mlp + (1.0 - args.mlp_weight) * control_lgbm
    full_blend = args.mlp_weight * mlp + (1.0 - args.mlp_weight) * full_lgbm
    baseline_metrics = _temporal_mrr(baseline_blend)
    if abs(float(baseline_metrics["full"]) - EXPECTED_BASELINE_MRR) > 1e-12:
        raise RuntimeError(
            "source champion baseline mismatch: "
            f"actual={baseline_metrics['full']:.16f} expected={EXPECTED_BASELINE_MRR:.16f}"
        )
    control_metrics = _temporal_mrr(control_blend)
    full_metrics = _temporal_mrr(full_blend)
    baseline_slices = tuple(float(item["mrr"]) for item in baseline_metrics["slices"])
    full_slices = tuple(float(item["mrr"]) for item in full_metrics["slices"])
    passed = passes_matched_control_gate(
        baseline_full_mrr=float(baseline_metrics["full"]),
        control_full_mrr=float(control_metrics["full"]),
        candidate_full_mrr=float(full_metrics["full"]),
        baseline_slice_mrrs=baseline_slices,
        candidate_slice_mrrs=full_slices,
        min_full_delta=args.min_full_delta,
    )
    champion_slice_deltas = [
        candidate - baseline
        for baseline, candidate in zip(
            baseline_slices,
            full_slices,
            strict=True,
        )
    ]
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "package_authorized": passed,
        "package_generated": False,
        "frozen_config": frozen,
        "baseline_champion": {
            "lgbm": _temporal_mrr(baseline_lgbm),
            "fixed_blend": baseline_metrics,
        },
        "matched32_control": {
            "lgbm": _temporal_mrr(control_lgbm),
            "fixed_blend": control_metrics,
            "blend_full_delta_vs_champion": float(
                control_metrics["full"] - baseline_metrics["full"]
            ),
            "model_path": str(control_model_path.resolve()),
            "model_sha256": _sha256(control_model_path),
        },
        "full100_candidate": {
            "lgbm": _temporal_mrr(full_lgbm),
            "fixed_blend": full_metrics,
            "blend_full_delta_vs_champion": float(
                full_metrics["full"] - baseline_metrics["full"]
            ),
            "blend_full_delta_vs_matched32": float(
                full_metrics["full"] - control_metrics["full"]
            ),
            "blend_slice_deltas_vs_champion": champion_slice_deltas,
            "model_path": str(full_model_path.resolve()),
            "model_sha256": _sha256(full_model_path),
        },
        "gate": {
            "min_full_delta": args.min_full_delta,
            "full_delta_vs_champion_passed": bool(
                full_metrics["full"] - baseline_metrics["full"] + 1e-12
                >= args.min_full_delta
            ),
            "all_slices_vs_champion_non_decreasing": bool(
                all(delta >= 0.0 for delta in champion_slice_deltas)
            ),
            "full100_not_below_matched32": bool(
                full_metrics["full"] + 1e-12 >= control_metrics["full"]
            ),
        },
        "training_seconds": {
            "matched32": control_seconds,
            "full100": full_seconds,
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _train_lgbm(
    *,
    label: str,
    features: np.ndarray,
    feature_indices: tuple[int, ...],
    feature_names: tuple[str, ...],
    params: dict[str, Any],
    boost_rounds: int,
) -> tuple[str, float]:
    import lightgbm as lgb  # noqa: PLC0415

    started = time.time()
    train_X, train_y, train_group = _flatten_for_ranking(features, feature_indices)
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
        f"[matched-eval] {label} dataset constructed seconds={time.time() - started:.1f}",
        flush=True,
    )
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=boost_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    model_text = booster.model_to_string(num_iteration=boost_rounds)
    elapsed = time.time() - started
    del booster, train_ds
    gc.collect()
    print(f"[matched-eval] {label} training complete seconds={elapsed:.1f}", flush=True)
    return model_text, elapsed


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash differs from the cache report")


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _mrr(scores: np.ndarray) -> float:
    values = np.asarray(scores)
    ranks = 1 + np.sum(values[:, 1:] > values[:, :1], axis=1)
    return float(np.mean(1.0 / ranks))


def _temporal_mrr(scores: np.ndarray) -> dict[str, Any]:
    return {
        "full": _mrr(scores),
        "slices": [
            {
                "start": int(part.start or 0),
                "stop": int(part.stop or len(scores)),
                "rows": int((part.stop or len(scores)) - (part.start or 0)),
                "mrr": _mrr(scores[part]),
            }
            for part in SLICES
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
