from __future__ import annotations

import argparse
import gc
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.tree import export_text

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.rankers.hybrid.fusion import build_fusion_from_state, predict_logits
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_slices
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.segment_fusion import (
    QUERY_SEGMENT_FEATURE_NAMES,
    MRRPolicyTree,
    SegmentGateResult,
    best_query_weights,
    blend_expert_probabilities,
    fit_segment_policy_gate,
    predict_segment_weights,
    query_segment_features,
)

FIT_SLICE = slice(0, 10_000)
CALIBRATION_SLICE = slice(10_000, 15_000)
FINAL_SLICE = slice(15_000, 20_000)
TREE_GRID = (
    ("depth2_leaf1000", 2, 1_000),
    ("depth2_leaf500", 2, 500),
    ("depth3_leaf1000", 3, 1_000),
    ("depth3_leaf500", 3, 500),
    ("depth4_leaf1000", 4, 1_000),
    ("depth4_leaf500", 4, 500),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune query-segment MLP/LightGBM gates on cached validation features.")
    parser.add_argument("--dataset1-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset1-cache-prefix", required=True, type=Path)
    parser.add_argument("--dataset2-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset2-cache-prefix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    specifications = (
        ("dataset1", args.dataset1_checkpoint, args.dataset1_cache_prefix),
        ("dataset2", args.dataset2_checkpoint, args.dataset2_cache_prefix),
    )
    reports = {}
    started = time.time()
    for dataset_name, checkpoint, cache_prefix in specifications:
        reports[dataset_name] = _tune_dataset(
            dataset_name=dataset_name,
            checkpoint=checkpoint,
            cache_prefix=cache_prefix,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        _write_json(
            args.output_dir / "segment-fusion-progress.json",
            {
                "status": "running",
                "elapsed_seconds": time.time() - started,
                "datasets": reports,
            },
        )

    accepted = [name for name, report in reports.items() if report["accepted"]]
    final_report = {
        "status": "passed" if accepted else "rejected",
        "accepted_datasets": accepted,
        "protocol": {
            "fit_slice": [FIT_SLICE.start, FIT_SLICE.stop],
            "calibration_slice": [CALIBRATION_SLICE.start, CALIBRATION_SLICE.stop],
            "final_slice": [FINAL_SLICE.start, FINAL_SLICE.stop],
            "tree_grid": [
                {"name": name, "max_depth": depth, "min_samples_leaf": leaf}
                for name, depth, leaf in TREE_GRID
            ],
            "final_used_during_selection": False,
            "gate_granularity": "one_mlp_weight_per_query",
            "optimization_target": "leaf-level full-candidate reciprocal-rank reward",
        },
        "datasets": reports,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "segment-fusion-report.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if accepted else 2


def _tune_dataset(
    *,
    dataset_name: str,
    checkpoint: Path,
    cache_prefix: Path,
    output_dir: Path,
    seed: int,
) -> dict:
    manifest = json.loads(cache_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    val_features = np.load(cache_prefix.with_suffix(".val.npy"), mmap_mode="r")
    if list(val_features.shape) != manifest["val"]["shape"] or val_features.shape[0] != FINAL_SLICE.stop:
        raise ValueError(f"{dataset_name} validation cache shape does not match the frozen protocol")

    state = load_checkpoint_dataset(checkpoint, dataset_name)
    fusion_result = state["fusion_result"]
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError(f"{dataset_name} checkpoint has no LightGBM expert")
    feature_names = tuple(state["feature_names"])
    selected_indices = tuple(int(index) for index in fusion_result.feature_indices)
    if tuple(int(index) for index in lgbm_result.feature_indices) != selected_indices:
        raise ValueError(f"{dataset_name} expert feature selections differ")
    model = build_fusion_from_state(
        input_dim=len(selected_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    global_weight = float(lgbm_result.mlp_weight)
    lgbm_model_text = str(lgbm_result.model_text)
    del state
    gc.collect()

    fit_features = val_features[FIT_SLICE]
    fit_mlp, fit_lgbm = _expert_probabilities(
        model, fusion_result, lgbm_model_text, fit_features, selected_indices,
    )
    fit_descriptors = query_segment_features(fit_features, feature_names)
    candidate_weights = _candidate_weights(global_weight)
    oracle_weights = best_query_weights(
        fit_mlp,
        fit_lgbm,
        candidate_weights=candidate_weights,
        global_weight=global_weight,
    )
    global_fit = blend_expert_probabilities(fit_mlp, fit_lgbm, global_weight)
    fit_rewards = _candidate_reward_matrix(fit_mlp, fit_lgbm, candidate_weights)

    calibration_features = val_features[CALIBRATION_SLICE]
    calibration_mlp, calibration_lgbm = _expert_probabilities(
        model, fusion_result, lgbm_model_text, calibration_features, selected_indices,
    )
    baseline_calibration = blend_expert_probabilities(calibration_mlp, calibration_lgbm, global_weight)
    baseline_calibration_mrr = _mrr(baseline_calibration)
    trials = []
    fitted_results: list[SegmentGateResult] = []
    for grid_index, (name, max_depth, min_samples_leaf) in enumerate(TREE_GRID):
        gate = fit_segment_policy_gate(
            fit_descriptors,
            fit_rewards,
            candidate_weights=candidate_weights,
            global_weight=global_weight,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            seed=seed,
            name=name,
        )
        calibration_weights = predict_segment_weights(gate, calibration_features, feature_names)
        calibration_blend = blend_expert_probabilities(
            calibration_mlp,
            calibration_lgbm,
            calibration_weights,
        )
        fitted_results.append(gate)
        trials.append(
            {
                "grid_index": grid_index,
                "name": name,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "calibration_mrr": _mrr(calibration_blend),
                "calibration_delta": _mrr(calibration_blend) - baseline_calibration_mrr,
                "calibration_weight_counts": _weight_counts(calibration_weights),
            }
        )
        print(
            f"[segment-fusion:{dataset_name}] {grid_index + 1}/{len(TREE_GRID)} {name} "
            f"calibration_mrr={trials[-1]['calibration_mrr']:.8f} "
            f"delta={trials[-1]['calibration_delta']:+.8f} "
            f"weights={trials[-1]['calibration_weight_counts']}",
            flush=True,
        )

    winner_index = max(
        range(len(trials)),
        key=lambda index: (trials[index]["calibration_mrr"], -trials[index]["grid_index"]),
    )
    winner = fitted_results[winner_index]
    winner_trial = trials[winner_index]
    frozen_path = output_dir / f"{dataset_name}-frozen-selection.json"
    _write_json(
        frozen_path,
        {
            "status": "frozen_before_final",
            "dataset": dataset_name,
            "cache_key": manifest["key"],
            "checkpoint": str(checkpoint.resolve()),
            "candidate_weights": list(candidate_weights),
            "global_weight": global_weight,
            "baseline_calibration_mrr": baseline_calibration_mrr,
            "winner": winner_trial,
            "trials": trials,
        },
    )
    print(
        f"[segment-fusion:{dataset_name}] frozen={winner.name}; reading final holdout now",
        flush=True,
    )

    final_features = val_features[FINAL_SLICE]
    final_mlp, final_lgbm = _expert_probabilities(
        model, fusion_result, lgbm_model_text, final_features, selected_indices,
    )
    baseline_final = blend_expert_probabilities(final_mlp, final_lgbm, global_weight)
    final_weights = predict_segment_weights(winner, final_features, feature_names)
    gated_final = blend_expert_probabilities(final_mlp, final_lgbm, final_weights)
    baseline_final_mrr = _mrr(baseline_final)
    gated_final_mrr = _mrr(gated_final)
    calibration_distinct = len(winner_trial["calibration_weight_counts"])
    final_distinct = len(_weight_counts(final_weights))
    accepted = (
        winner_trial["calibration_mrr"] > baseline_calibration_mrr
        and gated_final_mrr > baseline_final_mrr
        and calibration_distinct > 1
        and final_distinct > 1
    )

    fit_weights = predict_segment_weights(winner, fit_features, feature_names)
    gated_fit = blend_expert_probabilities(fit_mlp, fit_lgbm, fit_weights)
    calibration_weights = predict_segment_weights(winner, calibration_features, feature_names)
    gated_calibration = blend_expert_probabilities(
        calibration_mlp,
        calibration_lgbm,
        calibration_weights,
    )
    model_object = pickle.loads(winner.model_bytes)
    if not isinstance(model_object, MRRPolicyTree):
        raise TypeError("frozen segment gate is not an MRR policy tree")
    rules = export_text(model_object.estimator, feature_names=list(QUERY_SEGMENT_FEATURE_NAMES))
    rules += "\nleaf_weight_indices=" + json.dumps(
        model_object.leaf_weight_indices,
        ensure_ascii=False,
        sort_keys=True,
    )
    (output_dir / f"{dataset_name}-gate-rules.txt").write_text(rules, encoding="utf-8")
    with (output_dir / f"{dataset_name}-gate.pkl").open("wb") as handle:
        pickle.dump(winner, handle, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "accepted": accepted,
        "dataset": dataset_name,
        "checkpoint": str(checkpoint.resolve()),
        "cache_key": manifest["key"],
        "validation_shape": list(val_features.shape),
        "feature_names": list(feature_names),
        "descriptor_names": list(QUERY_SEGMENT_FEATURE_NAMES),
        "candidate_weights": list(candidate_weights),
        "global_weight": global_weight,
        "oracle_fit_weight_counts": _weight_counts(oracle_weights),
        "baseline": {
            "fit_mrr": _mrr(global_fit),
            "calibration_mrr": baseline_calibration_mrr,
            "final_mrr": baseline_final_mrr,
        },
        "winner": {
            **winner_trial,
            "final_mrr": gated_final_mrr,
            "final_delta": gated_final_mrr - baseline_final_mrr,
            "final_weight_counts": _weight_counts(final_weights),
            "fit_mrr": _mrr(gated_fit),
            "full_mrr": _weighted_mrr(
                (_mrr(gated_fit), len(gated_fit)),
                (_mrr(gated_calibration), len(gated_calibration)),
                (gated_final_mrr, len(gated_final)),
            ),
            "rules_path": str((output_dir / f"{dataset_name}-gate-rules.txt").resolve()),
            "gate_path": str((output_dir / f"{dataset_name}-gate.pkl").resolve()),
        },
        "trials": trials,
    }
    print(
        f"[segment-fusion:{dataset_name}] accepted={accepted} "
        f"final_mrr={gated_final_mrr:.8f} delta={gated_final_mrr - baseline_final_mrr:+.8f}",
        flush=True,
    )
    del model, fit_mlp, fit_lgbm, calibration_mlp, calibration_lgbm, final_mlp, final_lgbm
    gc.collect()
    return report


def _expert_probabilities(model, fusion_result, lgbm_model_text, features, selected_indices):
    selected = features[..., selected_indices] if selected_indices else features
    mlp_logits = predict_logits(model, selected, fusion_result.mean, fusion_result.std)
    lgbm_logits = predict_logits_lgbm(lgbm_model_text, selected)
    return _softmax(mlp_logits), _softmax(lgbm_logits)


def _candidate_weights(global_weight: float) -> tuple[float, ...]:
    return tuple(sorted({0.0, 0.25, 0.5, 0.75, 1.0, float(global_weight)}))


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _reciprocal_ranks(probabilities: np.ndarray) -> np.ndarray:
    ranks = 1 + (probabilities[:, 1:] > probabilities[:, 0:1]).sum(axis=1)
    return 1.0 / ranks


def _candidate_reward_matrix(
    mlp_probs: np.ndarray,
    lgbm_probs: np.ndarray,
    candidate_weights: tuple[float, ...],
) -> np.ndarray:
    return np.column_stack(
        [
            _reciprocal_ranks(
                blend_expert_probabilities(mlp_probs, lgbm_probs, weight)
            )
            for weight in candidate_weights
        ]
    )


def _mrr(probabilities: np.ndarray) -> float:
    return float(ranking_mrr_slices(probabilities)["full"])


def _weighted_mrr(*parts: tuple[float, int]) -> float:
    total_rows = sum(rows for _, rows in parts)
    return float(sum(mrr * rows for mrr, rows in parts) / total_rows)


def _weight_counts(weights: np.ndarray) -> dict[str, int]:
    counts = Counter(float(weight) for weight in weights)
    return {f"{weight:.2f}": count for weight, count in sorted(counts.items())}


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
