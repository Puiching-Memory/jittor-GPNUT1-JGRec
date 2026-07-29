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

from jgrec.contest_checkpoint import load_checkpoint_dataset, set_model_state
from jgrec.rankers.hybrid.feature_ablation import (
    neutralize_feature_columns,
    permute_candidate_feature_columns,
)
from jgrec.rankers.hybrid.fusion import FusionMLP, predict_logits
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_three_slices
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

EXPECTED_SHAPE = (20_000, 100, 63)
GNN_NAMES = ("gnn_full", "gnn_recent", "gnn_short")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the marginal contribution of Dataset2 GNN features."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-features", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--source-evaluation-report", required=True, type=Path)
    parser.add_argument("--setwise-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--setwise-weight", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--min-full-degradation", type=float, default=0.002)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite ablation: {args.output_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not 0.0 <= args.setwise_weight <= 1.0:
        raise ValueError("Setwise weight must be between zero and one")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "gnn-ablation-report.json"
    frozen_path = args.output_dir / "frozen-config.json"
    started = time.time()

    validation_report = json.loads(
        args.validation_cache_report.read_text(encoding="utf-8")
    )
    source_evaluation = json.loads(
        args.source_evaluation_report.read_text(encoding="utf-8")
    )
    expected_validation_sha = validation_report["artifacts"]["features"]["sha256"]
    validation_sha_before = _sha256(args.validation_features)
    if validation_sha_before != expected_validation_sha:
        raise ValueError("validation feature hash differs from cache report")
    _require_hash(
        args.checkpoint,
        source_evaluation["frozen_config"]["checkpoint_sha256"],
        "champion checkpoint",
    )
    _require_hash(
        args.setwise_model,
        source_evaluation["setwise"]["model_sha256"],
        "Setwise model",
    )

    val_features = np.load(
        args.validation_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    if val_features.shape != EXPECTED_SHAPE:
        raise ValueError(f"validation tensor shape differs: {val_features.shape}")
    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if tuple(validation_report["feature_names"]) != feature_names:
        raise ValueError("validation and checkpoint feature schemas differ")
    missing = tuple(name for name in GNN_NAMES if name not in feature_names)
    if missing:
        raise ValueError(f"missing GNN features: {missing}")
    raw_gnn_indices = {
        name: feature_names.index(name)
        for name in GNN_NAMES
    }

    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("checkpoint has no Dataset2 LightGBM")
    champion_lgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, val_features)
    )
    lgbm_metrics = ranking_mrr_three_slices(champion_lgbm)
    _require_metrics_close(
        lgbm_metrics,
        source_evaluation["baseline"]["lightgbm"],
        "champion LightGBM",
    )

    payload = np.load(args.setwise_model, allow_pickle=False)
    source_feature_count = int(payload["source_feature_count"][0])
    if source_feature_count != EXPECTED_SHAPE[-1]:
        raise ValueError("Setwise source feature count differs")
    if int(payload["context_transform_version"][0]) != 1:
        raise ValueError("unsupported Setwise context transform")
    feature_indices = tuple(int(value) for value in payload["feature_indices"])
    expected_context_count = source_feature_count * 3
    if feature_indices != tuple(range(expected_context_count)):
        raise ValueError("GNN ablation requires all Setwise context features")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    model = FusionMLP(
        input_dim=len(feature_indices),
        hidden_dim=int(payload["hidden_dim"][0]),
    )
    set_model_state(
        model,
        {
            key.removeprefix("state__"): np.asarray(
                payload[key],
                dtype=np.float32,
            )
            for key in payload.files
            if key.startswith("state__")
        },
    )
    view = SetwiseFeatureView(val_features)
    baseline_logits = _predict_variant(
        model,
        view,
        mean,
        std,
        feature_indices=feature_indices,
        batch_size=args.batch_size,
    )
    baseline_setwise = _softmax(baseline_logits)
    baseline_setwise_metrics = ranking_mrr_three_slices(baseline_setwise)
    _require_metrics_close(
        baseline_setwise_metrics,
        source_evaluation["setwise"]["expert"],
        "Setwise expert",
    )
    baseline_blend = (
        args.setwise_weight * baseline_setwise
        + (1.0 - args.setwise_weight) * champion_lgbm
    )
    baseline_metrics = ranking_mrr_three_slices(baseline_blend)
    _require_metrics_close(
        baseline_metrics,
        source_evaluation["setwise"]["fixed_blend"],
        "fixed Setwise blend",
    )
    del baseline_logits, baseline_setwise
    gc.collect()

    context_groups = {
        name: tuple(
            raw_gnn_indices[name] + offset * source_feature_count
            for offset in range(3)
        )
        for name in GNN_NAMES
    }
    all_context_indices = tuple(
        index
        for name in GNN_NAMES
        for index in context_groups[name]
    )
    candidate_permutations = np.argsort(
        np.random.default_rng(args.seed).random(EXPECTED_SHAPE[:2]),
        axis=1,
    ).astype(np.int16)
    frozen = {
        "status": "frozen_before_ablation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "validation_features": str(args.validation_features.resolve()),
        "validation_features_sha256": validation_sha_before,
        "validation_shape": list(val_features.shape),
        "setwise_model": str(args.setwise_model.resolve()),
        "setwise_model_sha256": _sha256(args.setwise_model),
        "setwise_weight": args.setwise_weight,
        "seed": args.seed,
        "permutation_protocol": "within_query_candidate_axis",
        "gnn_raw_indices": raw_gnn_indices,
        "gnn_context_indices": context_groups,
        "minimum_full_degradation": args.min_full_degradation,
        "cache_access": "read_only_memmap",
    }
    _write_json_atomic(frozen_path, frozen)

    variants: dict[str, dict[str, Any]] = {}
    for name in (*GNN_NAMES, "all"):
        columns = (
            all_context_indices
            if name == "all"
            else context_groups[name]
        )
        for mode in ("neutralize", "permute"):
            logits = _predict_variant(
                model,
                view,
                mean,
                std,
                feature_indices=feature_indices,
                batch_size=args.batch_size,
                ablation_columns=columns,
                mode=mode,
                candidate_permutations=candidate_permutations,
            )
            probabilities = _softmax(logits)
            blend = (
                args.setwise_weight * probabilities
                + (1.0 - args.setwise_weight) * champion_lgbm
            )
            metrics = ranking_mrr_three_slices(blend)
            degradation = {
                key: baseline_metrics[key] - metrics[key]
                for key in baseline_metrics
            }
            variants[f"{mode}_{name}"] = {
                "mode": mode,
                "features": list(GNN_NAMES if name == "all" else (name,)),
                "context_indices": list(columns),
                "metrics": metrics,
                "degradation": degradation,
            }
            del logits, probabilities, blend
            gc.collect()

    joint_names = ("neutralize_all", "permute_all")
    passing_joint_variants = [
        name
        for name in joint_names
        if (
            variants[name]["degradation"]["full"] + 1e-12
            >= args.min_full_degradation
            and all(
                variants[name]["degradation"][f"slice_{index}"] >= 0.0
                for index in range(3)
            )
        )
    ]
    perturbation_gate_passed = bool(passing_joint_variants)
    validation_sha_after = _sha256(args.validation_features)
    if validation_sha_after != validation_sha_before:
        raise RuntimeError("validation cache changed during ablation")

    report = {
        "status": "complete",
        "frozen_config": frozen,
        "baseline": {
            "setwise_expert": baseline_setwise_metrics,
            "lightgbm_expert": lgbm_metrics,
            "fixed_blend": baseline_metrics,
        },
        "variants": variants,
        "gate": {
            "minimum_full_degradation": args.min_full_degradation,
            "all_three_slices_non_improving": True,
            "passing_joint_variants": passing_joint_variants,
            "perturbation_gate_passed": perturbation_gate_passed,
            "no_gnn_retrain_authorized": perturbation_gate_passed,
        },
        "validation_cache_sha256_before": validation_sha_before,
        "validation_cache_sha256_after": validation_sha_after,
        "validation_cache_unchanged": True,
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _predict_variant(
    model: FusionMLP,
    features: SetwiseFeatureView,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    feature_indices: tuple[int, ...],
    batch_size: int,
    ablation_columns: tuple[int, ...] | None = None,
    mode: str | None = None,
    candidate_permutations: np.ndarray | None = None,
) -> np.ndarray:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        batch = np.asarray(features[start:end], dtype=np.float32)
        if ablation_columns is not None:
            if mode == "neutralize":
                batch = neutralize_feature_columns(
                    batch,
                    columns=ablation_columns,
                    neutral_values=mean[list(ablation_columns)],
                )
            elif mode == "permute" and candidate_permutations is not None:
                batch = permute_candidate_feature_columns(
                    batch,
                    columns=ablation_columns,
                    permutations=candidate_permutations[start:end],
                )
            else:
                raise ValueError("invalid feature ablation mode")
        if feature_indices != tuple(range(batch.shape[-1])):
            batch = batch[..., feature_indices]
        scores[start:end] = predict_logits(model, batch, mean, std)
    return scores


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


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


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
