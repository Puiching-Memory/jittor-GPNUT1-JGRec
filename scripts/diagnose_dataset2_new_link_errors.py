from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions
from jgrec.rankers.hybrid.fusion import build_fusion_from_state, predict_logits
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.new_link_diagnostics import (
    historical_pair_mask,
    new_link_error_report,
    passes_new_link_concentration_gate,
)
from jgrec.rankers.hybrid.oof_hard_negatives import contiguous_oof_folds
from jgrec.rankers.hybrid.ranker import _sample_events


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose exact champion errors on Dataset2 new positive edges.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-prefix", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-baseline-mrr", type=float, default=0.5428303297309955)
    parser.add_argument("--min-rows-per-segment", type=int, default=100)
    parser.add_argument("--min-new-row-share", type=float, default=0.50)
    parser.add_argument("--min-new-regret-share", type=float, default=0.65)
    parser.add_argument("--min-full-mrr-gap", type=float, default=0.03)
    parser.add_argument("--min-slice-mrr-gap", type=float, default=0.02)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite new-link diagnosis: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    frozen_path = args.output_dir / "frozen-config.json"
    report_path = args.output_dir / "new-link-error-report.json"

    manifest_path = args.cache_prefix.with_suffix(".json")
    val_path = args.cache_prefix.with_suffix(".val.npy")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_features = np.load(val_path, mmap_mode="r", allow_pickle=False)
    if list(val_features.shape) != manifest["val"]["shape"]:
        raise ValueError("validation cache shape does not match manifest")

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    config = state["config"]
    interactions = read_interactions(args.train_csv).sort_by_time()
    original_rows = len(interactions)
    if config.max_fit_events > 0 and len(interactions) > config.max_fit_events:
        interactions = interactions.tail(config.max_fit_events)
    n_events = len(interactions)
    val_size = max(1, int(n_events * config.val_ratio))
    train_end = max(2, n_events - val_size)
    context_end = max(1, min(train_end - 1, int(train_end * config.context_ratio)))
    train_pool = interactions[context_end:train_end]
    val_pool = interactions[train_end:]
    rng = np.random.default_rng(config.seed)
    sampled_train = _sample_events(train_pool, config.max_train_events, rng)
    sampled_val = _sample_events(val_pool, config.max_val_events, rng)
    if len(sampled_val) != val_features.shape[0]:
        raise ValueError(
            f"reconstructed validation rows differ from cache: events={len(sampled_val)} cache={val_features.shape[0]}"
        )
    if len(sampled_train) != manifest["train"]["shape"][0]:
        raise ValueError("reconstructed training rows differ from cache")

    validation_folds = contiguous_oof_folds(row_count=len(sampled_val), fold_count=3)
    thresholds = {
        "min_rows_per_segment": args.min_rows_per_segment,
        "min_new_row_share": args.min_new_row_share,
        "min_new_regret_share": args.min_new_regret_share,
        "min_full_mrr_gap": args.min_full_mrr_gap,
        "min_slice_mrr_gap": args.min_slice_mrr_gap,
    }
    frozen = {
        "status": "frozen_before_champion_scoring",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "cache_key": manifest["key"],
        "cache_manifest_sha256": _sha256(manifest_path),
        "validation_shape": list(val_features.shape),
        "train_csv": str(args.train_csv.resolve()),
        "train_csv_sha256": _sha256(args.train_csv),
        "original_interaction_rows": original_rows,
        "fit_interaction_rows": n_events,
        "context_end": context_end,
        "train_end": train_end,
        "sampled_train_rows": len(sampled_train),
        "sampled_validation_rows": len(sampled_val),
        "sampled_validation_time_range": [int(sampled_val.time.min()), int(sampled_val.time.max())],
        "seed": int(config.seed),
        "max_fit_events": int(config.max_fit_events),
        "max_train_events": int(config.max_train_events),
        "max_val_events": int(config.max_val_events),
        "val_ratio": float(config.val_ratio),
        "context_ratio": float(config.context_ratio),
        "repeat_definition": "positive (src,dst) occurs in fixed interactions[0:train_end]",
        "validation_slices": [
            [fold.holdout.start, fold.holdout.stop] for fold in validation_folds
        ],
        "expected_baseline_mrr": args.expected_baseline_mrr,
        "thresholds": thresholds,
    }
    _write_json(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    repeat_mask = historical_pair_mask(
        interactions[:train_end].src,
        interactions[:train_end].dst,
        sampled_val.src,
        sampled_val.dst,
    )
    fusion_result = state["fusion_result"]
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("Dataset2 checkpoint has no LightGBM expert")
    feature_indices = tuple(int(index) for index in fusion_result.feature_indices)
    if feature_indices != tuple(int(index) for index in lgbm_result.feature_indices):
        raise ValueError("Dataset2 MLP and LightGBM feature selections differ")
    selected_val = np.asarray(val_features[..., feature_indices])
    model = build_fusion_from_state(
        input_dim=len(feature_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    mlp = _softmax(predict_logits(model, selected_val, fusion_result.mean, fusion_result.std))
    lgbm = _softmax(predict_logits_lgbm(lgbm_result.model_text, selected_val))
    mlp_weight = float(lgbm_result.mlp_weight)
    champion = mlp_weight * mlp + (1.0 - mlp_weight) * lgbm
    slices = tuple(fold.holdout for fold in validation_folds)
    diagnostic = new_link_error_report(champion, repeat_mask, slices=slices)
    baseline_mrr = _mrr(champion)
    if abs(baseline_mrr - args.expected_baseline_mrr) > 1e-12:
        raise RuntimeError(
            f"champion/cache alignment failed: expected={args.expected_baseline_mrr:.16f} actual={baseline_mrr:.16f}"
        )
    passed = passes_new_link_concentration_gate(diagnostic, **thresholds)
    diagnostic_dict = asdict(diagnostic)
    diagnostic_dict["new_top1_error_share"] = _share(
        diagnostic.new.top1_errors,
        diagnostic.new.top1_errors + diagnostic.repeat.top1_errors,
    )
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "frozen_config": frozen,
        "alignment": {
            "baseline_mrr": baseline_mrr,
            "expected_baseline_mrr": args.expected_baseline_mrr,
            "absolute_delta": abs(baseline_mrr - args.expected_baseline_mrr),
            "feature_indices": list(feature_indices),
            "mlp_weight": mlp_weight,
        },
        "diagnostic": diagnostic_dict,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    del state, model, selected_val, mlp, lgbm, champion
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


def _share(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


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
