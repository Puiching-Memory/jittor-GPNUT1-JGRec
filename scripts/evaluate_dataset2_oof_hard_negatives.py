from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.rankers.hybrid.fusion import build_fusion_from_state, predict_logits
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_slices
from jgrec.rankers.hybrid.fusion_lgbm import _flatten_for_ranking, predict_logits_lgbm
from jgrec.rankers.hybrid.lgbm_tuning import predeclared_dataset2_lgbm_grid
from jgrec.rankers.hybrid.oof_hard_negatives import (
    contiguous_oof_folds,
    passes_temporal_mrr_gate,
    select_hard_negative_features,
    select_hard_negative_positions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed Dataset2 OOF hard-negative LambdaRank.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-prefix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--keep-negatives", type=int, default=16)
    parser.add_argument("--boost-rounds", type=int, default=308)
    parser.add_argument("--mlp-weight", type=float, default=0.07)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite OOF experiment: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    progress_path = args.output_dir / "progress.json"
    frozen_path = args.output_dir / "frozen-config.json"
    report_path = args.output_dir / "oof-hard-negative-report.json"
    model_path = args.output_dir / "dataset2-oof-hard-negative-lgbm.txt"
    oof_score_path = args.output_dir / "oof-scores.npy"
    position_path = args.output_dir / "selected-positions.npy"
    mined_path = args.output_dir / "mined-train-selected.npy"

    manifest = json.loads(args.cache_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    train_features = np.load(args.cache_prefix.with_suffix(".train.npy"), mmap_mode="r")
    val_features = np.load(args.cache_prefix.with_suffix(".val.npy"), mmap_mode="r")
    if list(train_features.shape) != manifest["train"]["shape"]:
        raise ValueError("training cache shape does not match manifest")
    if list(val_features.shape) != manifest["val"]["shape"]:
        raise ValueError("validation cache shape does not match manifest")
    if args.keep_negatives >= train_features.shape[1]:
        raise ValueError("keep_negatives must be smaller than the cached candidate count")

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    fusion_result = state["fusion_result"]
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("Dataset2 checkpoint has no LightGBM expert")
    feature_indices = tuple(int(index) for index in lgbm_result.feature_indices)
    if feature_indices != tuple(int(index) for index in fusion_result.feature_indices):
        raise ValueError("Dataset2 MLP and LightGBM feature selections differ")
    if not feature_indices:
        raise ValueError("Dataset2 checkpoint selected no supervised features")
    if abs(float(lgbm_result.mlp_weight) - args.mlp_weight) > 1e-12:
        raise ValueError("fixed MLP weight does not match the champion checkpoint")
    current_lgbm_text = str(lgbm_result.model_text)
    mlp_model = build_fusion_from_state(
        input_dim=len(feature_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    del state
    gc.collect()

    params_by_name = dict(
        predeclared_dataset2_lgbm_grid(seed=args.seed, num_threads=args.num_threads)
    )
    params = dict(params_by_name["lr003"])
    folds = contiguous_oof_folds(
        row_count=int(train_features.shape[0]),
        fold_count=args.fold_count,
    )
    validation_slices = contiguous_oof_folds(
        row_count=int(val_features.shape[0]),
        fold_count=3,
    )
    frozen = {
        "status": "frozen_before_validation_predictions",
        "checkpoint": str(args.checkpoint.resolve()),
        "cache_key": manifest["key"],
        "train_shape": list(train_features.shape),
        "validation_shape": list(val_features.shape),
        "feature_indices": list(feature_indices),
        "fold_count": args.fold_count,
        "keep_negatives": args.keep_negatives,
        "boost_rounds": args.boost_rounds,
        "mlp_weight": args.mlp_weight,
        "min_full_delta": args.min_full_delta,
        "params": params,
        "negative_source": "existing public-unlabeled-candidate cache; mechanism test only",
        "validation_selection": "none",
    }
    _write_json(frozen_path, frozen)

    import lightgbm as lgb  # noqa: PLC0415

    started = time.time()
    oof_scores = np.lib.format.open_memmap(
        oof_score_path,
        mode="w+",
        dtype=np.float32,
        shape=train_features.shape[:2],
    )
    coverage = np.zeros(train_features.shape[0], dtype=np.uint8)
    fold_reports = []
    all_selected = tuple(range(len(feature_indices)))
    for fold in folds:
        fold_started = time.time()
        fit_features = np.concatenate(
            [train_features[fit_slice][..., feature_indices] for fit_slice in fold.fit_slices],
            axis=0,
        )
        fit_X, fit_y, fit_group = _flatten_for_ranking(fit_features, all_selected)
        train_ds = lgb.Dataset(
            fit_X,
            label=fit_y,
            group=fit_group,
            params={"feature_pre_filter": False},
            free_raw_data=True,
        )
        miner = lgb.train(
            params,
            train_ds,
            num_boost_round=args.boost_rounds,
            callbacks=[lgb.log_evaluation(0)],
        )
        held_selected = np.ascontiguousarray(
            train_features[fold.holdout][..., feature_indices],
            dtype=np.float32,
        )
        held_scores = miner.predict(
            held_selected.reshape(-1, held_selected.shape[-1]),
            num_iteration=args.boost_rounds,
        ).reshape(held_selected.shape[:2])
        oof_scores[fold.holdout] = held_scores.astype(np.float32, copy=False)
        coverage[fold.holdout] += 1
        fold_report = {
            "fold": fold.index,
            "holdout": [fold.holdout.start, fold.holdout.stop],
            "fit_slices": [[part.start, part.stop] for part in fold.fit_slices],
            "fit_rows": int(fit_features.shape[0]),
            "holdout_rows": int(held_selected.shape[0]),
            "elapsed_seconds": time.time() - fold_started,
        }
        fold_reports.append(fold_report)
        _write_json(
            progress_path,
            {
                "status": "mining",
                "completed_folds": len(fold_reports),
                "total_folds": len(folds),
                "elapsed_seconds": time.time() - started,
                "folds": fold_reports,
            },
        )
        print(
            f"[oof-hard-negative] fold={fold.index + 1}/{len(folds)} "
            f"fit={fit_features.shape[0]} held={held_selected.shape[0]} "
            f"elapsed={time.time() - fold_started:.1f}s",
            flush=True,
        )
        del fit_features, fit_X, fit_y, fit_group, train_ds, miner, held_selected, held_scores
        gc.collect()
    oof_scores.flush()
    if not np.all(coverage == 1):
        raise RuntimeError("OOF coverage must score every training row exactly once")

    positions = np.lib.format.open_memmap(
        position_path,
        mode="w+",
        dtype=np.int16,
        shape=(train_features.shape[0], args.keep_negatives + 1),
    )
    mined = np.lib.format.open_memmap(
        mined_path,
        mode="w+",
        dtype=np.float32,
        shape=(train_features.shape[0], args.keep_negatives + 1, len(feature_indices)),
    )
    selected_score_samples: list[np.ndarray] = []
    all_negative_score_samples: list[np.ndarray] = []
    position_counts: Counter[int] = Counter()
    chunk_size = 2_000
    for start in range(0, train_features.shape[0], chunk_size):
        stop = min(start + chunk_size, train_features.shape[0])
        chunk_scores = np.asarray(oof_scores[start:stop])
        chunk_positions = select_hard_negative_positions(
            chunk_scores,
            keep_negatives=args.keep_negatives,
        )
        selected_chunk = select_hard_negative_features(
            np.asarray(train_features[start:stop][..., feature_indices]),
            chunk_scores,
            keep_negatives=args.keep_negatives,
        )
        positions[start:stop] = chunk_positions.astype(np.int16, copy=False)
        mined[start:stop] = selected_chunk
        selected_score_samples.append(
            np.take_along_axis(chunk_scores, chunk_positions[:, 1:], axis=1).reshape(-1)
        )
        all_negative_score_samples.append(chunk_scores[:, 1:].reshape(-1))
        position_counts.update(int(value) for value in chunk_positions[:, 1:].reshape(-1))
    positions.flush()
    mined.flush()
    selected_scores = np.concatenate(selected_score_samples)
    all_negative_scores = np.concatenate(all_negative_score_samples)
    del selected_score_samples, all_negative_score_samples
    gc.collect()

    final_X, final_y, final_group = _flatten_for_ranking(mined, all_selected)
    final_ds = lgb.Dataset(
        final_X,
        label=final_y,
        group=final_group,
        params={"feature_pre_filter": False},
        free_raw_data=True,
    )
    candidate = lgb.train(
        params,
        final_ds,
        num_boost_round=args.boost_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    candidate_text = candidate.model_to_string(num_iteration=args.boost_rounds)
    model_path.write_text(candidate_text, encoding="utf-8")
    del final_X, final_y, final_group, final_ds, candidate
    gc.collect()

    selected_val = np.asarray(val_features[..., feature_indices])
    mlp_val = _softmax(
        predict_logits(mlp_model, selected_val, fusion_result.mean, fusion_result.std)
    )
    baseline_lgbm = _softmax(predict_logits_lgbm(current_lgbm_text, selected_val))
    candidate_lgbm = _softmax(predict_logits_lgbm(candidate_text, selected_val))
    baseline_blend = args.mlp_weight * mlp_val + (1.0 - args.mlp_weight) * baseline_lgbm
    candidate_blend = args.mlp_weight * mlp_val + (1.0 - args.mlp_weight) * candidate_lgbm
    baseline_full = _mrr(baseline_blend)
    candidate_full = _mrr(candidate_blend)
    baseline_slices = tuple(_mrr(baseline_blend[part.holdout]) for part in validation_slices)
    candidate_slices = tuple(_mrr(candidate_blend[part.holdout]) for part in validation_slices)
    passed = passes_temporal_mrr_gate(
        candidate_slices=candidate_slices,
        baseline_slices=baseline_slices,
        candidate_full_mrr=candidate_full,
        baseline_full_mrr=baseline_full,
        min_full_delta=args.min_full_delta,
    )
    slice_reports = [
        {
            "slice": index,
            "rows": [part.holdout.start, part.holdout.stop],
            "baseline_mrr": baseline_mrr,
            "candidate_mrr": candidate_mrr,
            "delta": candidate_mrr - baseline_mrr,
        }
        for index, (part, baseline_mrr, candidate_mrr) in enumerate(
            zip(validation_slices, baseline_slices, candidate_slices, strict=True)
        )
    ]
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "frozen_config": frozen,
        "oof": {
            "coverage_min": int(coverage.min()),
            "coverage_max": int(coverage.max()),
            "folds": fold_reports,
            "selected_position_counts": {
                str(position): count for position, count in sorted(position_counts.items())
            },
            "all_negative_score_quantiles": _quantiles(all_negative_scores),
            "selected_negative_score_quantiles": _quantiles(selected_scores),
            "position_sha256": _sha256(position_path),
        },
        "baseline": {
            "lgbm_full_mrr": _mrr(baseline_lgbm),
            "blend_full_mrr": baseline_full,
            "blend_slices": list(baseline_slices),
            "mlp_weight": args.mlp_weight,
        },
        "candidate": {
            "lgbm_full_mrr": _mrr(candidate_lgbm),
            "blend_full_mrr": candidate_full,
            "blend_slices": list(candidate_slices),
            "full_delta": candidate_full - baseline_full,
            "slice_results": slice_reports,
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _mrr(probabilities: np.ndarray) -> float:
    return float(ranking_mrr_slices(probabilities)["full"])


def _quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    quantiles = np.quantile(values, levels)
    return {f"q{int(level * 100):03d}": float(value) for level, value in zip(levels, quantiles, strict=True)}


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
