from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    build_fusion_from_state,
    fit_fusion_mlp_listwise_fixed,
    predict_logits,
)
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.oof_hard_negatives import contiguous_oof_folds, passes_temporal_mrr_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one fixed Dataset2 listwise MLP from cached features.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-prefix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--mlp-weight", type=float, default=0.07)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=60)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite listwise experiment: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    frozen_path = args.output_dir / "frozen-config.json"
    report_path = args.output_dir / "listwise-mlp-report.json"
    model_path = args.output_dir / "dataset2-listwise-mlp.npz"

    manifest_path = args.cache_prefix.with_suffix(".json")
    train_path = args.cache_prefix.with_suffix(".train.npy")
    val_path = args.cache_prefix.with_suffix(".val.npy")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_features = np.load(train_path, mmap_mode="r", allow_pickle=False)
    val_features = np.load(val_path, mmap_mode="r", allow_pickle=False)
    if list(train_features.shape) != manifest["train"]["shape"]:
        raise ValueError("training cache shape does not match manifest")
    if list(val_features.shape) != manifest["val"]["shape"]:
        raise ValueError("validation cache shape does not match manifest")
    if train_features.ndim != 3 or train_features.shape[1] < 2:
        raise ValueError("training cache must contain grouped candidates")
    if val_features.ndim != 3 or val_features.shape[1] < 2:
        raise ValueError("validation cache must contain grouped candidates")

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    fusion_result = state["fusion_result"]
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("Dataset2 checkpoint has no LightGBM expert")
    feature_indices = tuple(int(index) for index in fusion_result.feature_indices)
    if feature_indices != tuple(int(index) for index in lgbm_result.feature_indices):
        raise ValueError("Dataset2 MLP and LightGBM feature selections differ")
    if not feature_indices:
        raise ValueError("Dataset2 checkpoint selected no supervised features")
    if abs(float(lgbm_result.mlp_weight) - args.mlp_weight) > 1e-12:
        raise ValueError("fixed MLP weight does not match the champion checkpoint")
    if len(state["feature_names"]) != train_features.shape[-1]:
        raise ValueError("checkpoint and cache feature counts differ")

    hidden_dim = int(state["fusion_hidden_dim"])
    baseline_model = build_fusion_from_state(
        input_dim=len(feature_indices),
        hidden_dim=hidden_dim,
        state=state["fusion_state"],
    )
    current_lgbm_text = str(lgbm_result.model_text)
    validation_slices = contiguous_oof_folds(row_count=int(val_features.shape[0]), fold_count=3)
    config = FusionConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=0.0,
        hidden_dim=hidden_dim,
        selection_metric="mrr",
        early_stop_patience=0,
    )
    frozen = {
        "status": "frozen_before_training_and_validation_predictions",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "cache_key": manifest["key"],
        "cache_manifest_sha256": _sha256(manifest_path),
        "train_shape": list(train_features.shape),
        "validation_shape": list(val_features.shape),
        "feature_indices": list(feature_indices),
        "hidden_dim": hidden_dim,
        "objective": "negative_log_softmax_of_candidate_zero",
        "positive_candidate_position": 0,
        "initialization": "fresh_deterministic_from_seed",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.0,
        "mlp_weight": args.mlp_weight,
        "min_full_delta": args.min_full_delta,
        "validation_selection": "none; evaluated once after all epochs",
        "validation_slices": [
            [fold.holdout.start, fold.holdout.stop] for fold in validation_slices
        ],
    }
    _write_json(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    started = time.time()
    candidate_model, candidate_result, epoch_losses = fit_fusion_mlp_listwise_fixed(
        train_features,
        val_features,
        config,
        np.random.default_rng(args.seed),
        verbose=True,
        feature_indices=feature_indices,
        candidate_name="dataset2_listwise_fixed",
    )
    _save_model(model_path, candidate_result)
    training_seconds = time.time() - started
    print(
        f"[listwise-mlp] training_complete seconds={training_seconds:.1f} "
        f"loss={epoch_losses[-1]:.6f}; scoring fixed validation",
        flush=True,
    )

    selected_val = np.asarray(val_features[..., feature_indices])
    baseline_mlp = _softmax(
        predict_logits(baseline_model, selected_val, fusion_result.mean, fusion_result.std)
    )
    candidate_mlp = _softmax(
        predict_logits(candidate_model, selected_val, candidate_result.mean, candidate_result.std)
    )
    lgbm = _softmax(predict_logits_lgbm(current_lgbm_text, selected_val))
    baseline_blend = args.mlp_weight * baseline_mlp + (1.0 - args.mlp_weight) * lgbm
    candidate_blend = args.mlp_weight * candidate_mlp + (1.0 - args.mlp_weight) * lgbm

    baseline_mlp_metrics = _temporal_mrr(baseline_mlp, validation_slices)
    candidate_mlp_metrics = _temporal_mrr(candidate_mlp, validation_slices)
    lgbm_metrics = _temporal_mrr(lgbm, validation_slices)
    baseline_blend_metrics = _temporal_mrr(baseline_blend, validation_slices)
    candidate_blend_metrics = _temporal_mrr(candidate_blend, validation_slices)
    baseline_slices = tuple(item["mrr"] for item in baseline_blend_metrics["slices"])
    candidate_slices = tuple(item["mrr"] for item in candidate_blend_metrics["slices"])
    passed = passes_temporal_mrr_gate(
        candidate_slices=candidate_slices,
        baseline_slices=baseline_slices,
        candidate_full_mrr=float(candidate_blend_metrics["full"]),
        baseline_full_mrr=float(baseline_blend_metrics["full"]),
        min_full_delta=args.min_full_delta,
    )
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "frozen_config": frozen,
        "training": {
            "epoch_losses": list(epoch_losses),
            "epochs_completed": len(epoch_losses),
            "seconds": training_seconds,
            "post_training_val_ap": candidate_result.best_val_ap,
            "post_training_val_mrr": candidate_result.best_val_mrr,
        },
        "baseline": {
            "mlp": baseline_mlp_metrics,
            "lgbm": lgbm_metrics,
            "fixed_blend": baseline_blend_metrics,
        },
        "candidate": {
            "mlp": candidate_mlp_metrics,
            "fixed_blend": candidate_blend_metrics,
            "pure_mlp_full_delta": candidate_mlp_metrics["full"] - baseline_mlp_metrics["full"],
            "blend_full_delta": candidate_blend_metrics["full"] - baseline_blend_metrics["full"],
            "blend_slice_deltas": [
                candidate_value - baseline_value
                for candidate_value, baseline_value in zip(candidate_slices, baseline_slices, strict=True)
            ],
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    del state, baseline_model, candidate_model, selected_val
    gc.collect()
    return 0 if passed else 2


def _save_model(path: Path, result) -> None:
    payload = {
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
        "feature_indices": np.asarray(result.feature_indices, dtype=np.int32),
    }
    payload.update(
        {f"state__{key}": np.asarray(value, dtype=np.float32) for key, value in result.state.items()}
    )
    np.savez_compressed(path, **payload)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _mrr(scores: np.ndarray) -> float:
    values = np.asarray(scores)
    ranks = 1 + (values[:, 1:] > values[:, 0:1]).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _temporal_mrr(scores: np.ndarray, folds) -> dict:
    return {
        "full": _mrr(scores),
        "slices": [
            {
                "index": fold.index,
                "rows": [fold.holdout.start, fold.holdout.stop],
                "mrr": _mrr(scores[fold.holdout]),
            }
            for fold in folds
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
