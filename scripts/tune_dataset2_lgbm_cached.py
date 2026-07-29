from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.rankers.hybrid.fusion import build_fusion_from_state, predict_logits
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_slices, scan_probability_blend
from jgrec.rankers.hybrid.fusion_lgbm import (
    _flatten_for_ranking,
    _full_candidate_mrr_evaluator,
    predict_logits_lgbm,
)
from jgrec.rankers.hybrid.lgbm_tuning import (
    Dataset2LGBMTrial,
    chronological_validation_slices,
    passes_robustness_gate,
    predeclared_dataset2_lgbm_grid,
    select_tune_winner,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune only Dataset2 LightGBM on cached supervised features.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-prefix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tune-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "tuning-progress.json"
    selection_path = output_dir / "frozen-selection.json"
    report_path = output_dir / "tuning-report.json"
    model_path = output_dir / "dataset2-lgbm.txt"

    manifest_path = args.cache_prefix.with_suffix(".json")
    train_path = args.cache_prefix.with_suffix(".train.npy")
    val_path = args.cache_prefix.with_suffix(".val.npy")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_features = np.load(train_path, mmap_mode="r")
    val_features = np.load(val_path, mmap_mode="r")
    tune_slice, pseudo_b_slice = chronological_validation_slices(
        int(val_features.shape[0]), tune_rows=args.tune_rows,
    )
    if list(train_features.shape) != manifest["train"]["shape"]:
        raise ValueError("training cache shape does not match manifest")
    if list(val_features.shape) != manifest["val"]["shape"]:
        raise ValueError("validation cache shape does not match manifest")

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    fusion_result = state["fusion_result"]
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("Dataset2 checkpoint has no LightGBM result")
    feature_indices = tuple(int(index) for index in fusion_result.feature_indices)
    if tuple(int(index) for index in lgbm_result.feature_indices) != feature_indices:
        raise ValueError("Dataset2 MLP and LightGBM feature selections differ")
    if not feature_indices:
        raise ValueError("Dataset2 checkpoint selected no fusion features")
    model = build_fusion_from_state(
        input_dim=len(feature_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    current_lgbm_text = str(lgbm_result.model_text)
    current_mlp_weight = float(lgbm_result.mlp_weight)
    del state
    gc.collect()

    tune_features = val_features[tune_slice]
    selected_tune = _select_columns(tune_features, feature_indices)
    mlp_tune = _softmax(predict_logits(model, selected_tune, fusion_result.mean, fusion_result.std))
    current_lgbm_tune = _softmax(predict_logits_lgbm(current_lgbm_text, selected_tune))
    current_stored_tune = current_mlp_weight * mlp_tune + (1.0 - current_mlp_weight) * current_lgbm_tune
    baseline = {
        "stored_mlp_weight": current_mlp_weight,
        "mlp_tune_mrr": ranking_mrr_slices(mlp_tune)["full"],
        "lgbm_tune_mrr": ranking_mrr_slices(current_lgbm_tune)["full"],
        "stored_tune_mrr": ranking_mrr_slices(current_stored_tune)["full"],
    }

    import lightgbm as lgb  # noqa: PLC0415

    train_X, train_y, train_group = _flatten_for_ranking(train_features, feature_indices)
    tune_X, tune_y, tune_group = _flatten_for_ranking(tune_features, feature_indices)
    dataset_params = {"feature_pre_filter": False}
    train_ds = lgb.Dataset(
        train_X,
        label=train_y,
        group=train_group,
        params=dataset_params,
        free_raw_data=True,
    )
    tune_ds = lgb.Dataset(
        tune_X,
        label=tune_y,
        group=tune_group,
        reference=train_ds,
        params=dataset_params,
        free_raw_data=True,
    )
    train_ds.construct()
    tune_ds.construct()
    del train_X, train_y, tune_y
    gc.collect()

    grid = predeclared_dataset2_lgbm_grid(seed=args.seed, num_threads=args.num_threads)
    trials: list[Dataset2LGBMTrial] = []
    started = time.time()
    for grid_index, (name, params) in enumerate(grid):
        trial_started = time.time()
        booster = lgb.train(
            params,
            train_ds,
            num_boost_round=args.num_boost_round,
            valid_sets=[tune_ds],
            feval=_full_candidate_mrr_evaluator(tune_features.shape[1]),
            callbacks=[
                lgb.early_stopping(
                    args.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(0),
            ],
        )
        tune_scores = booster.predict(tune_X, num_iteration=booster.best_iteration).reshape(tune_features.shape[:2])
        lgbm_tune = _softmax(tune_scores)
        blend = scan_probability_blend(mlp_tune, lgbm_tune)
        trial = Dataset2LGBMTrial(
            grid_index=grid_index,
            name=name,
            params=dict(params),
            best_iteration=int(booster.best_iteration),
            lgbm_tune_mrr=ranking_mrr_slices(lgbm_tune)["full"],
            blend_tune_mrr=float(blend.mrr["full"]),
            mlp_weight=float(blend.reference_weight),
            model_text=booster.model_to_string(num_iteration=booster.best_iteration),
        )
        trials.append(trial)
        del booster, tune_scores, lgbm_tune
        gc.collect()
        progress = {
            "status": "tuning",
            "cache_key": manifest["key"],
            "checkpoint": str(args.checkpoint.resolve()),
            "grid_size": len(grid),
            "completed": len(trials),
            "elapsed_seconds": time.time() - started,
            "baseline": baseline,
            "trials": [_public_trial(item, elapsed=None) for item in trials],
            "last_trial_seconds": time.time() - trial_started,
        }
        _write_json(progress_path, progress)
        print(
            f"[dataset2-lgbm] {grid_index + 1}/{len(grid)} {name} "
            f"iter={trial.best_iteration} lgbm={trial.lgbm_tune_mrr:.8f} "
            f"blend={trial.blend_tune_mrr:.8f} mlp_w={trial.mlp_weight:.2f} "
            f"elapsed={time.time() - trial_started:.1f}s",
            flush=True,
        )

    winner = select_tune_winner(trials)
    model_path.write_text(winner.model_text, encoding="utf-8")
    frozen_selection = {
        "status": "frozen_before_pseudo_b",
        "selection_source": f"validation[0:{args.tune_rows}]",
        "pseudo_b_source": f"validation[{args.tune_rows}:{val_features.shape[0]}]",
        "winner": _public_trial(winner, elapsed=None),
        "baseline_tune": baseline,
    }
    _write_json(selection_path, frozen_selection)
    print(
        f"[dataset2-lgbm] frozen winner={winner.name} blend={winner.blend_tune_mrr:.8f} "
        f"mlp_w={winner.mlp_weight:.2f}; evaluating pseudo-B now",
        flush=True,
    )

    pseudo_b_features = val_features[pseudo_b_slice]
    selected_pseudo_b = _select_columns(pseudo_b_features, feature_indices)
    mlp_pseudo_b = _softmax(predict_logits(model, selected_pseudo_b, fusion_result.mean, fusion_result.std))
    current_lgbm_pseudo_b = _softmax(predict_logits_lgbm(current_lgbm_text, selected_pseudo_b))
    current_stored_pseudo_b = (
        current_mlp_weight * mlp_pseudo_b + (1.0 - current_mlp_weight) * current_lgbm_pseudo_b
    )
    winner_lgbm_pseudo_b = _softmax(predict_logits_lgbm(winner.model_text, selected_pseudo_b))
    winner_blend_pseudo_b = (
        winner.mlp_weight * mlp_pseudo_b + (1.0 - winner.mlp_weight) * winner_lgbm_pseudo_b
    )
    winner_blend_tune = winner.mlp_weight * mlp_tune + (1.0 - winner.mlp_weight) * _softmax(
        predict_logits_lgbm(winner.model_text, selected_tune)
    )
    baseline_pseudo_b_mrr = ranking_mrr_slices(current_stored_pseudo_b)["full"]
    winner_pseudo_b_mrr = ranking_mrr_slices(winner_blend_pseudo_b)["full"]
    gate_passed = passes_robustness_gate(
        candidate_tune_mrr=winner.blend_tune_mrr,
        candidate_pseudo_b_mrr=winner_pseudo_b_mrr,
        baseline_tune_mrr=baseline["stored_tune_mrr"],
        baseline_pseudo_b_mrr=baseline_pseudo_b_mrr,
    )
    report = {
        "status": "passed" if gate_passed else "rejected",
        "gate_passed": gate_passed,
        "selection_protocol": {
            "grid_size": len(grid),
            "tune_slice": [tune_slice.start, tune_slice.stop],
            "pseudo_b_slice": [pseudo_b_slice.start, pseudo_b_slice.stop],
            "pseudo_b_used_during_selection": False,
            "blend_step": 0.01,
        },
        "cache": {
            "key": manifest["key"],
            "train_shape": list(train_features.shape),
            "validation_shape": list(val_features.shape),
            "feature_indices": list(feature_indices),
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "baseline": {
            **baseline,
            "stored_pseudo_b_mrr": baseline_pseudo_b_mrr,
        },
        "winner": {
            **_public_trial(winner, elapsed=None),
            "lgbm_pseudo_b_mrr": ranking_mrr_slices(winner_lgbm_pseudo_b)["full"],
            "blend_pseudo_b_mrr": winner_pseudo_b_mrr,
            "blend_full_mrr": ranking_mrr_slices(
                np.concatenate((winner_blend_tune, winner_blend_pseudo_b), axis=0)
            )["full"],
            "tune_delta_vs_stored": winner.blend_tune_mrr - baseline["stored_tune_mrr"],
            "pseudo_b_delta_vs_stored": winner_pseudo_b_mrr - baseline_pseudo_b_mrr,
            "model_path": str(model_path.resolve()),
        },
        "trials": [_public_trial(item, elapsed=None) for item in trials],
        "elapsed_seconds": time.time() - started,
    }
    _write_json(report_path, report)
    print(json.dumps(report["winner"], ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[dataset2-lgbm] status={report['status']} report={report_path}", flush=True)
    return 0 if gate_passed else 2


def _select_columns(features: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    if indices == tuple(range(features.shape[-1])):
        return features
    return features[..., indices]


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _public_trial(trial: Dataset2LGBMTrial, *, elapsed: float | None) -> dict:
    values = asdict(trial)
    values.pop("model_text")
    if elapsed is not None:
        values["elapsed_seconds"] = elapsed
    return values


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
