from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.rankers.hybrid.feature_ablation import (
    retained_context_feature_indices,
)
from jgrec.rankers.hybrid.full100_training import validate_joint_cache_reports
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    fit_fusion_mlp_listwise_streaming,
)
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_three_slices
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

EXPECTED_TRAIN_SHAPE = (200_000, 100, 63)
EXPECTED_VAL_SHAPE = (20_000, 100, 63)
GNN_NAMES = ("gnn_full", "gnn_recent", "gnn_short")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a fixed-hyperparameter Dataset2 Setwise control without GNN."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--perturbation-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--setwise-weight", type=float, default=0.80)
    parser.add_argument("--min-full-degradation", type=float, default=0.001)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite no-GNN control: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    frozen_path = args.output_dir / "frozen-config.json"
    report_path = args.output_dir / "no-gnn-control-report.json"
    model_path = args.output_dir / "dataset2-setwise-no-gnn.npz"
    started = time.time()

    perturbation = json.loads(
        args.perturbation_report.read_text(encoding="utf-8")
    )
    if not perturbation["gate"]["no_gnn_retrain_authorized"]:
        raise RuntimeError("perturbation report did not authorize no-GNN retraining")
    train_report = json.loads(
        args.train_cache_report.read_text(encoding="utf-8")
    )
    val_report = json.loads(
        args.validation_cache_report.read_text(encoding="utf-8")
    )
    joint_cache_contract = validate_joint_cache_reports(
        train_report,
        val_report,
    )
    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    val_path = Path(f"{args.validation_cache_prefix}.val.npy")
    train_sha_before = _sha256(train_path)
    val_sha_before = _sha256(val_path)
    if train_sha_before != train_report["artifacts"]["features"]["sha256"]:
        raise ValueError("training cache hash differs from report")
    if val_sha_before != val_report["artifacts"]["features"]["sha256"]:
        raise ValueError("validation cache hash differs from report")

    train_features = np.load(train_path, mmap_mode="r", allow_pickle=False)
    val_features = np.load(val_path, mmap_mode="r", allow_pickle=False)
    if train_features.shape != EXPECTED_TRAIN_SHAPE:
        raise ValueError(f"training feature shape differs: {train_features.shape}")
    if val_features.shape != EXPECTED_VAL_SHAPE:
        raise ValueError(f"validation feature shape differs: {val_features.shape}")
    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if feature_names != tuple(train_report["feature_names"]):
        raise ValueError("checkpoint and cache feature schemas differ")
    excluded_source_indices = tuple(
        feature_names.index(name)
        for name in GNN_NAMES
    )
    retained_indices = retained_context_feature_indices(
        source_feature_count=len(feature_names),
        excluded_source_indices=excluded_source_indices,
        context_copies=3,
    )
    if len(retained_indices) != 180:
        raise RuntimeError("no-GNN Setwise control must retain 180 features")
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("checkpoint has no Dataset2 LightGBM")

    config = FusionConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=0.0,
        hidden_dim=args.hidden_dim,
        selection_metric="mrr",
        early_stop_patience=args.patience,
    )
    frozen = {
        "status": "frozen_before_training",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_cache": str(train_path.resolve()),
        "train_cache_sha256": train_sha_before,
        "validation_cache": str(val_path.resolve()),
        "validation_cache_sha256": val_sha_before,
        "joint_cache_contract": joint_cache_contract,
        "perturbation_report": str(args.perturbation_report.resolve()),
        "perturbation_report_sha256": _sha256(args.perturbation_report),
        "excluded_source_features": list(GNN_NAMES),
        "excluded_source_indices": list(excluded_source_indices),
        "retained_context_indices": list(retained_indices),
        "retained_input_dim": len(retained_indices),
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "setwise_weight": args.setwise_weight,
        "minimum_full_degradation": args.min_full_degradation,
    }
    _write_json_atomic(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2), flush=True)

    setwise_train = SetwiseFeatureView(train_features)
    setwise_val = SetwiseFeatureView(val_features)
    model, result, history = fit_fusion_mlp_listwise_streaming(
        setwise_train,
        setwise_val,
        config,
        np.random.default_rng(args.seed),
        verbose=True,
        feature_indices=retained_indices,
        candidate_name="dataset2_setwise_no_gnn_control",
    )
    _save_model(
        model_path,
        result=result,
        hidden_dim=args.hidden_dim,
        source_feature_count=len(feature_names),
        excluded_source_indices=excluded_source_indices,
    )
    logits = _predict_streaming(
        model,
        setwise_val,
        result.mean,
        result.std,
        feature_indices=result.feature_indices,
        batch_size=args.batch_size,
    )
    probabilities = _softmax(logits)
    lightgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, val_features)
    )
    lightgbm_metrics = ranking_mrr_three_slices(lightgbm)
    _require_metrics_close(
        lightgbm_metrics,
        perturbation["baseline"]["lightgbm_expert"],
        "LightGBM expert",
    )
    blend = (
        args.setwise_weight * probabilities
        + (1.0 - args.setwise_weight) * lightgbm
    )
    control_metrics = ranking_mrr_three_slices(blend)
    baseline_metrics = {
        key: float(value)
        for key, value in perturbation["baseline"]["fixed_blend"].items()
    }
    degradation = {
        key: baseline_metrics[key] - control_metrics[key]
        for key in baseline_metrics
    }
    evidence_passed = bool(
        degradation["full"] + 1e-12 >= args.min_full_degradation
        and all(degradation[f"slice_{index}"] >= 0.0 for index in range(3))
    )
    train_sha_after = _sha256(train_path)
    val_sha_after = _sha256(val_path)
    if train_sha_after != train_sha_before or val_sha_after != val_sha_before:
        raise RuntimeError("source cache changed during no-GNN training")

    report = {
        "status": "complete",
        "frozen_config": frozen,
        "baseline": baseline_metrics,
        "no_gnn_setwise_expert": ranking_mrr_three_slices(probabilities),
        "no_gnn_fixed_blend": control_metrics,
        "degradation": degradation,
        "history": list(history),
        "best_val_ap": result.best_val_ap,
        "best_val_mrr": result.best_val_mrr,
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "gate": {
            "minimum_full_degradation": args.min_full_degradation,
            "all_three_slices_non_improving": True,
            "gnn_contribution_confirmed": evidence_passed,
            "gnn_improvement_authorized": evidence_passed,
        },
        "train_cache_sha256_before": train_sha_before,
        "train_cache_sha256_after": train_sha_after,
        "validation_cache_sha256_before": val_sha_before,
        "validation_cache_sha256_after": val_sha_after,
        "source_caches_unchanged": True,
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _predict_streaming(
    model: Any,
    features: Any,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    feature_indices: tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    with jt.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            batch = np.asarray(features[start:end], dtype=np.float32)
            batch = batch[..., feature_indices]
            normalized = ((batch - mean) / std).astype(
                np.float32,
                copy=False,
            )
            logits = model(jt.array(normalized, dtype=jt.float32))
            scores[start:end] = np.asarray(logits.numpy(), dtype=np.float32)
            del batch, normalized, logits
    return scores


def _save_model(
    path: Path,
    *,
    result: Any,
    hidden_dim: int,
    source_feature_count: int,
    excluded_source_indices: tuple[int, ...],
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
        "excluded_source_indices": np.asarray(
            excluded_source_indices,
            dtype=np.int32,
        ),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _require_metrics_close(
    actual: dict[str, float],
    expected: dict[str, float],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if abs(float(actual[key]) - float(expected_value)) > 1e-10:
            raise RuntimeError(
                f"{label} reproduction failed for {key}: "
                f"actual={actual[key]:.16f} expected={expected_value:.16f}"
            )


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
