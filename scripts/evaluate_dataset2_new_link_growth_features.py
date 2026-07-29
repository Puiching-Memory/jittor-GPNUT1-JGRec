from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.rankers.hybrid.fusion import build_fusion_from_state, predict_logits
from jgrec.rankers.hybrid.fusion_lgbm import _flatten_for_ranking, predict_logits_lgbm
from jgrec.rankers.hybrid.lgbm_tuning import predeclared_dataset2_lgbm_grid
from jgrec.rankers.hybrid.new_link_features import (
    NEW_LINK_GROWTH_FEATURE_NAMES,
    append_new_link_growth_features,
)
from jgrec.rankers.hybrid.oof_hard_negatives import contiguous_oof_folds, passes_temporal_mrr_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate two cached Dataset2 new-link growth features.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-prefix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--boost-rounds", type=int, default=308)
    parser.add_argument("--mlp-weight", type=float, default=0.07)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite growth-feature experiment: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    frozen_path = args.output_dir / "frozen-config.json"
    report_path = args.output_dir / "new-link-growth-report.json"
    model_path = args.output_dir / "dataset2-new-link-growth-lgbm.txt"

    manifest_path = args.cache_prefix.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_features = np.load(args.cache_prefix.with_suffix(".train.npy"), mmap_mode="r", allow_pickle=False)
    val_features = np.load(args.cache_prefix.with_suffix(".val.npy"), mmap_mode="r", allow_pickle=False)
    if list(train_features.shape) != manifest["train"]["shape"]:
        raise ValueError("training cache shape does not match manifest")
    if list(val_features.shape) != manifest["val"]["shape"]:
        raise ValueError("validation cache shape does not match manifest")

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if len(feature_names) != train_features.shape[-1]:
        raise ValueError("checkpoint and cache feature counts differ")
    fusion_result = state["fusion_result"]
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("Dataset2 checkpoint has no LightGBM expert")
    feature_indices = tuple(int(index) for index in fusion_result.feature_indices)
    if feature_indices != tuple(range(len(feature_names))):
        raise ValueError("frozen growth experiment requires the champion to use every cached feature")
    if feature_indices != tuple(int(index) for index in lgbm_result.feature_indices):
        raise ValueError("Dataset2 MLP and LightGBM feature selections differ")
    if abs(float(lgbm_result.mlp_weight) - args.mlp_weight) > 1e-12:
        raise ValueError("fixed MLP weight does not match the champion checkpoint")

    params = dict(
        dict(predeclared_dataset2_lgbm_grid(seed=args.seed, num_threads=args.num_threads))["lr003"]
    )
    validation_folds = contiguous_oof_folds(row_count=val_features.shape[0], fold_count=3)
    frozen = {
        "status": "frozen_before_training_and_validation_predictions",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "cache_key": manifest["key"],
        "cache_manifest_sha256": _sha256(manifest_path),
        "train_shape": list(train_features.shape),
        "validation_shape": list(val_features.shape),
        "original_feature_count": len(feature_names),
        "derived_features": list(NEW_LINK_GROWTH_FEATURE_NAMES),
        "derived_formulas": {
            NEW_LINK_GROWTH_FEATURE_NAMES[0]: (
                "log1p(target_pop_share_w001 / max(target_pop_share_w100, 1e-12))"
            ),
            NEW_LINK_GROWTH_FEATURE_NAMES[1]: (
                "src_activity * target_short_vs_long_growth_log1p_ratio"
            ),
        },
        "boost_rounds": args.boost_rounds,
        "mlp_weight": args.mlp_weight,
        "min_full_delta": args.min_full_delta,
        "seed": args.seed,
        "params": params,
        "validation_selection": "none; fixed model scored once after training",
        "validation_slices": [
            [fold.holdout.start, fold.holdout.stop] for fold in validation_folds
        ],
    }
    _write_json(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    started = time.time()
    augmented_train, augmented_names = append_new_link_growth_features(train_features, feature_names)
    all_indices = tuple(range(len(augmented_names)))
    train_X, train_y, train_group = _flatten_for_ranking(augmented_train, all_indices)
    del augmented_train
    gc.collect()

    import lightgbm as lgb  # noqa: PLC0415

    train_ds = lgb.Dataset(
        train_X,
        label=train_y,
        group=train_group,
        feature_name=list(augmented_names),
        params={"feature_pre_filter": False},
        free_raw_data=True,
    )
    candidate = lgb.train(
        params,
        train_ds,
        num_boost_round=args.boost_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    candidate_text = candidate.model_to_string(num_iteration=args.boost_rounds)
    model_path.write_text(candidate_text, encoding="utf-8")
    training_seconds = time.time() - started
    del train_X, train_y, train_group, train_ds
    gc.collect()
    print(f"[new-link-growth] training_complete seconds={training_seconds:.1f}; scoring", flush=True)

    augmented_val, actual_augmented_names = append_new_link_growth_features(val_features, feature_names)
    if actual_augmented_names != augmented_names:
        raise RuntimeError("train and validation derived feature names differ")
    mlp_model = build_fusion_from_state(
        input_dim=len(feature_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    original_val = augmented_val[..., : len(feature_names)]
    mlp = _softmax(predict_logits(mlp_model, original_val, fusion_result.mean, fusion_result.std))
    baseline_lgbm = _softmax(predict_logits_lgbm(lgbm_result.model_text, original_val))
    candidate_lgbm = _softmax(predict_logits_lgbm(candidate_text, augmented_val))
    baseline_blend = args.mlp_weight * mlp + (1.0 - args.mlp_weight) * baseline_lgbm
    candidate_blend = args.mlp_weight * mlp + (1.0 - args.mlp_weight) * candidate_lgbm
    baseline_metrics = _temporal_mrr(baseline_blend, validation_folds)
    candidate_metrics = _temporal_mrr(candidate_blend, validation_folds)
    baseline_slices = tuple(item["mrr"] for item in baseline_metrics["slices"])
    candidate_slices = tuple(item["mrr"] for item in candidate_metrics["slices"])
    passed = passes_temporal_mrr_gate(
        candidate_slices=candidate_slices,
        baseline_slices=baseline_slices,
        candidate_full_mrr=float(candidate_metrics["full"]),
        baseline_full_mrr=float(baseline_metrics["full"]),
        min_full_delta=args.min_full_delta,
    )
    importances = {
        name: {
            "gain": float(gain),
            "split": int(split),
        }
        for name, gain, split in zip(
            augmented_names,
            candidate.feature_importance(importance_type="gain"),
            candidate.feature_importance(importance_type="split"),
            strict=True,
        )
        if name in NEW_LINK_GROWTH_FEATURE_NAMES
    }
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "frozen_config": frozen,
        "baseline": {
            "lgbm": _temporal_mrr(baseline_lgbm, validation_folds),
            "fixed_blend": baseline_metrics,
        },
        "candidate": {
            "lgbm": _temporal_mrr(candidate_lgbm, validation_folds),
            "fixed_blend": candidate_metrics,
            "blend_full_delta": candidate_metrics["full"] - baseline_metrics["full"],
            "blend_slice_deltas": [
                candidate_value - baseline_value
                for candidate_value, baseline_value in zip(candidate_slices, baseline_slices, strict=True)
            ],
            "derived_feature_importance": importances,
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
        },
        "training_seconds": training_seconds,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    del state, candidate, augmented_val, mlp_model, mlp, baseline_lgbm, candidate_lgbm
    gc.collect()
    return 0 if passed else 2


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _mrr(scores: np.ndarray) -> float:
    ranks = 1 + (scores[:, 1:] > scores[:, 0:1]).sum(axis=1)
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
