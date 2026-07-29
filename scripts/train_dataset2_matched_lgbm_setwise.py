from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.rankers.hybrid.full100_training import (
    passes_full100_gate,
    validate_joint_cache_reports,
)
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    build_fusion_from_state,
    fit_fusion_mlp_listwise_streaming,
)
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_three_slices
from jgrec.rankers.hybrid.fusion_lgbm import (
    _flatten_for_ranking,
    _full_candidate_mrr_evaluator,
    predict_logits_lgbm,
)
from jgrec.rankers.hybrid.lgbm_tuning import predeclared_dataset2_lgbm_grid
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train matched recent-200k/full-100 Dataset2 LightGBM and "
            "set-context listwise rerankers."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--lgbm-max-rounds", type=int, default=800)
    parser.add_argument("--lgbm-patience", type=int, default=60)
    parser.add_argument("--setwise-epochs", type=int, default=10)
    parser.add_argument("--setwise-patience", type=int, default=2)
    parser.add_argument("--setwise-batch-size", type=int, default=256)
    parser.add_argument("--setwise-hidden-dim", type=int, default=32)
    parser.add_argument("--setwise-learning-rate", type=float, default=0.001)
    parser.add_argument("--mlp-weight", type=float, default=0.07)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite matched reranker experiment: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    frozen_path = args.output_dir / "frozen-config.json"
    progress_path = args.output_dir / "training-progress.json"
    report_path = args.output_dir / "evaluation-report.json"
    lgbm_model_path = args.output_dir / "dataset2-matched-lgbm.txt"
    setwise_model_path = args.output_dir / "dataset2-setwise.npz"

    train_report = json.loads(
        args.train_cache_report.read_text(encoding="utf-8")
    )
    val_report = json.loads(
        args.validation_cache_report.read_text(encoding="utf-8")
    )
    if train_report.get("status") != "complete":
        raise RuntimeError("recent-200k training cache is incomplete")
    if val_report.get("status") != "complete":
        raise RuntimeError("matched validation cache is incomplete")
    joint_cache_contract = validate_joint_cache_reports(
        train_report,
        val_report,
    )
    if (
        val_report.get("train_feature_sha256")
        != train_report["artifacts"]["features"]["sha256"]
    ):
        raise ValueError("validation cache was not matched to this training cache")

    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    val_path = Path(f"{args.validation_cache_prefix}.val.npy")
    train_features = np.load(train_path, mmap_mode="r", allow_pickle=False)
    val_features = np.load(val_path, mmap_mode="r", allow_pickle=False)
    if train_features.shape != (200_000, 100, 63):
        raise ValueError(f"training feature shape mismatch: {train_features.shape}")
    if val_features.shape != (20_000, 100, 63):
        raise ValueError(f"validation feature shape mismatch: {val_features.shape}")
    _require_hash(
        train_path,
        train_report["artifacts"]["features"]["sha256"],
        "training features",
    )
    _require_hash(
        val_path,
        val_report["artifacts"]["features"]["sha256"],
        "validation features",
    )

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    fusion_result = state["fusion_result"]
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("Dataset2 champion has no LightGBM expert")
    feature_indices = tuple(int(index) for index in fusion_result.feature_indices)
    if feature_names != tuple(train_report["feature_names"]):
        raise ValueError("checkpoint feature schema differs from matched caches")
    if feature_indices != tuple(range(63)):
        raise ValueError("frozen matched experiment requires all 63 features")
    if tuple(int(index) for index in lgbm_result.feature_indices) != feature_indices:
        raise ValueError("champion MLP and LightGBM feature selections differ")
    if int(config.seed) != args.seed:
        raise ValueError("experiment seed differs from champion")
    if abs(float(lgbm_result.mlp_weight) - args.mlp_weight) > 1e-12:
        raise ValueError("fixed blend weight differs from champion")

    lgbm_params = dict(
        dict(
            predeclared_dataset2_lgbm_grid(
                seed=args.seed,
                num_threads=args.num_threads,
            )
        )["lr003"]
    )
    setwise_config = FusionConfig(
        epochs=args.setwise_epochs,
        batch_size=args.setwise_batch_size,
        lr=args.setwise_learning_rate,
        weight_decay=0.0,
        hidden_dim=args.setwise_hidden_dim,
        selection_metric="mrr",
        early_stop_patience=args.setwise_patience,
    )
    frozen = {
        "status": "frozen_before_training_and_validation_predictions",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_cache_report": str(args.train_cache_report.resolve()),
        "train_cache_report_sha256": _sha256(args.train_cache_report),
        "joint_cache_contract": joint_cache_contract,
        "validation_cache_report": str(args.validation_cache_report.resolve()),
        "validation_cache_report_sha256": _sha256(
            args.validation_cache_report
        ),
        "train_shape": list(train_features.shape),
        "validation_shape": list(val_features.shape),
        "feature_names": list(feature_names),
        "feature_indices": list(feature_indices),
        "positive_candidate_position": 0,
        "baseline": "champion fixed 0.07 MLP + 0.93 LightGBM",
        "lightgbm": {
            "params": lgbm_params,
            "max_rounds": args.lgbm_max_rounds,
            "early_stopping_rounds": args.lgbm_patience,
            "selection_metric": "full_candidate_mrr",
            "candidate_blend": "0.07 champion MLP + 0.93 candidate LightGBM",
        },
        "setwise": {
            "context_transform": [
                "raw",
                "candidate_minus_row_mean",
                "candidate_minus_row_max",
            ],
            "input_dim": int(train_features.shape[-1]) * 3,
            "epochs": setwise_config.epochs,
            "patience": setwise_config.early_stop_patience,
            "batch_size": setwise_config.batch_size,
            "hidden_dim": setwise_config.hidden_dim,
            "learning_rate": setwise_config.lr,
            "objective": "negative_log_softmax_of_candidate_zero",
            "selection_metric": "full_candidate_mrr",
            "candidate_blend": "0.07 candidate Setwise + 0.93 champion LightGBM",
        },
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "all_three_slices_non_decreasing": True,
            "package_only_best_passing_candidate": True,
        },
    }
    _write_json_atomic(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    started = time.time()
    baseline_model = build_fusion_from_state(
        input_dim=len(feature_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    baseline_mlp_logits = _predict_mlp_streaming(
        baseline_model,
        val_features,
        fusion_result.mean,
        fusion_result.std,
        feature_indices=feature_indices,
        batch_size=512,
    )
    baseline_mlp = _softmax(baseline_mlp_logits)
    del baseline_mlp_logits, baseline_model
    baseline_lgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, val_features)
    )
    baseline_blend = (
        args.mlp_weight * baseline_mlp
        + (1.0 - args.mlp_weight) * baseline_lgbm
    )
    baseline_metrics = ranking_mrr_three_slices(baseline_blend)
    _write_json_atomic(
        progress_path,
        {
            "status": "baseline_complete",
            "baseline": baseline_metrics,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(
        f"[matched-rerankers] baseline_mrr={baseline_metrics['full']:.8f}",
        flush=True,
    )

    import lightgbm as lgb  # noqa: PLC0415

    lgbm_started = time.time()
    train_X, train_y, train_group = _flatten_for_ranking(
        train_features,
        feature_indices,
    )
    val_X, val_y, val_group = _flatten_for_ranking(
        val_features,
        feature_indices,
    )
    dataset_params = {"feature_pre_filter": False}
    train_ds = lgb.Dataset(
        train_X,
        label=train_y,
        group=train_group,
        feature_name=list(feature_names),
        params=dataset_params,
        free_raw_data=True,
    )
    val_ds = lgb.Dataset(
        val_X,
        label=val_y,
        group=val_group,
        reference=train_ds,
        feature_name=list(feature_names),
        params=dataset_params,
        free_raw_data=True,
    )
    train_ds.construct()
    val_ds.construct()
    del train_X, train_y, train_group, val_X, val_y, val_group
    gc.collect()
    booster = lgb.train(
        lgbm_params,
        train_ds,
        num_boost_round=args.lgbm_max_rounds,
        valid_sets=[val_ds],
        feval=_full_candidate_mrr_evaluator(100),
        callbacks=[
            lgb.early_stopping(
                args.lgbm_patience,
                first_metric_only=True,
                verbose=True,
            ),
            lgb.log_evaluation(10),
        ],
    )
    lgbm_best_iteration = int(booster.best_iteration)
    candidate_lgbm_text = booster.model_to_string(
        num_iteration=lgbm_best_iteration
    )
    lgbm_model_path.write_text(candidate_lgbm_text, encoding="utf-8")
    del booster, train_ds, val_ds
    gc.collect()
    candidate_lgbm = _softmax(
        predict_logits_lgbm(candidate_lgbm_text, val_features)
    )
    candidate_lgbm_blend = (
        args.mlp_weight * baseline_mlp
        + (1.0 - args.mlp_weight) * candidate_lgbm
    )
    lgbm_metrics = ranking_mrr_three_slices(candidate_lgbm_blend)
    lgbm_gate = _gate(
        baseline_metrics,
        lgbm_metrics,
        min_full_delta=args.min_full_delta,
    )
    lgbm_seconds = time.time() - lgbm_started
    _write_json_atomic(
        progress_path,
        {
            "status": "lightgbm_complete",
            "baseline": baseline_metrics,
            "lightgbm": {
                "best_iteration": lgbm_best_iteration,
                "metrics": lgbm_metrics,
                "gate_passed": lgbm_gate,
                "seconds": lgbm_seconds,
            },
            "elapsed_seconds": time.time() - started,
        },
    )
    print(
        f"[matched-rerankers] LightGBM iter={lgbm_best_iteration} "
        f"mrr={lgbm_metrics['full']:.8f} "
        f"delta={lgbm_metrics['full'] - baseline_metrics['full']:+.8f} "
        f"gate={lgbm_gate}",
        flush=True,
    )

    setwise_started = time.time()
    setwise_train = SetwiseFeatureView(train_features)
    setwise_val = SetwiseFeatureView(val_features)
    setwise_model, setwise_result, setwise_history = (
        fit_fusion_mlp_listwise_streaming(
            setwise_train,
            setwise_val,
            setwise_config,
            np.random.default_rng(args.seed),
            verbose=True,
            feature_indices=tuple(range(setwise_train.shape[-1])),
            candidate_name="dataset2_setwise_recent200k_full100",
        )
    )
    _save_setwise_model(
        setwise_model_path,
        result=setwise_result,
        hidden_dim=args.setwise_hidden_dim,
        source_feature_count=train_features.shape[-1],
    )
    setwise_logits = _predict_mlp_streaming(
        setwise_model,
        setwise_val,
        setwise_result.mean,
        setwise_result.std,
        feature_indices=setwise_result.feature_indices,
        batch_size=args.setwise_batch_size,
    )
    setwise_probabilities = _softmax(setwise_logits)
    setwise_blend = (
        args.mlp_weight * setwise_probabilities
        + (1.0 - args.mlp_weight) * baseline_lgbm
    )
    setwise_metrics = ranking_mrr_three_slices(setwise_blend)
    setwise_gate = _gate(
        baseline_metrics,
        setwise_metrics,
        min_full_delta=args.min_full_delta,
    )
    setwise_seconds = time.time() - setwise_started
    passing = [
        (name, metrics)
        for name, metrics, passed in (
            ("lightgbm", lgbm_metrics, lgbm_gate),
            ("setwise", setwise_metrics, setwise_gate),
        )
        if passed
    ]
    winner = (
        max(passing, key=lambda item: item[1]["full"])[0]
        if passing
        else None
    )
    report = {
        "status": "passed" if winner is not None else "rejected",
        "gate_passed": winner is not None,
        "package_authorized": winner is not None,
        "package_generated": False,
        "winner": winner,
        "frozen_config": frozen,
        "baseline": {
            "fixed_blend": baseline_metrics,
            "mlp": ranking_mrr_three_slices(baseline_mlp),
            "lightgbm": ranking_mrr_three_slices(baseline_lgbm),
        },
        "lightgbm": {
            "gate_passed": lgbm_gate,
            "best_iteration": lgbm_best_iteration,
            "fixed_blend": lgbm_metrics,
            "expert": ranking_mrr_three_slices(candidate_lgbm),
            "full_delta": lgbm_metrics["full"] - baseline_metrics["full"],
            "slice_deltas": _slice_deltas(baseline_metrics, lgbm_metrics),
            "model_path": str(lgbm_model_path.resolve()),
            "model_sha256": _sha256(lgbm_model_path),
            "seconds": lgbm_seconds,
        },
        "setwise": {
            "gate_passed": setwise_gate,
            "fixed_blend": setwise_metrics,
            "expert": ranking_mrr_three_slices(setwise_probabilities),
            "full_delta": setwise_metrics["full"] - baseline_metrics["full"],
            "slice_deltas": _slice_deltas(baseline_metrics, setwise_metrics),
            "best_val_ap": setwise_result.best_val_ap,
            "best_val_mrr": setwise_result.best_val_mrr,
            "history": list(setwise_history),
            "model_path": str(setwise_model_path.resolve()),
            "model_sha256": _sha256(setwise_model_path),
            "seconds": setwise_seconds,
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(report_path, report)
    _write_json_atomic(
        progress_path,
        {
            "status": "complete",
            "winner": winner,
            "gate_passed": winner is not None,
            "elapsed_seconds": report["elapsed_seconds"],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if winner is not None else 2


def _predict_mlp_streaming(
    model: Any,
    features: Any,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    feature_indices: tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    step = max(int(batch_size), 1)
    with jt.no_grad():
        for start in range(0, features.shape[0], step):
            end = min(start + step, features.shape[0])
            batch = np.asarray(features[start:end], dtype=np.float32)
            if feature_indices != tuple(range(batch.shape[-1])):
                batch = batch[..., feature_indices]
            normalized = ((batch - mean) / std).astype(np.float32, copy=False)
            logits = model(jt.array(normalized, dtype=jt.float32))
            scores[start:end] = np.asarray(logits.numpy(), dtype=np.float32)
            del batch, normalized, logits
    return scores


def _gate(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    min_full_delta: float,
) -> bool:
    return passes_full100_gate(
        baseline_full_mrr=baseline["full"],
        candidate_full_mrr=candidate["full"],
        baseline_slice_mrrs=tuple(
            baseline[f"slice_{index}"] for index in range(3)
        ),
        candidate_slice_mrrs=tuple(
            candidate[f"slice_{index}"] for index in range(3)
        ),
        min_full_delta=min_full_delta,
    )


def _slice_deltas(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> list[float]:
    return [
        candidate[f"slice_{index}"] - baseline[f"slice_{index}"]
        for index in range(3)
    ]


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _save_setwise_model(
    path: Path,
    *,
    result: Any,
    hidden_dim: int,
    source_feature_count: int,
) -> None:
    payload = {
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
        "feature_indices": np.asarray(result.feature_indices, dtype=np.int32),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
        "source_feature_count": np.asarray(
            [source_feature_count],
            dtype=np.int32,
        ),
        "context_transform_version": np.asarray([1], dtype=np.int32),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: actual={actual} expected={expected}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
