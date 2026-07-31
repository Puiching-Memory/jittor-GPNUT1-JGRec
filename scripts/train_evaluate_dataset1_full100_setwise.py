from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.cuda import require_jittor_cuda
from jgrec.rankers.hybrid.full100_training import (
    passes_full100_gate,
    validate_joint_cache_reports,
)
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    build_fusion_from_state,
    fit_fusion_mlp_listwise_streaming,
)
from jgrec.rankers.hybrid.fusion_analysis import (
    inclusive_weight_grid,
    ranking_mrr_three_slices,
    select_setwise_model_blend_on_prefix,
)
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

EXPECTED_TRAIN_ROWS = 200_000
EXPECTED_VALIDATION_ROWS = 20_000
EXPECTED_CANDIDATES = 100
EXPECTED_FEATURES = 63
SELECTION_STOP = 13_334
TRAIN_SCALES = (100_000, 200_000)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train recent-100k/200k full-100 Dataset1 Setwise models and "
            "select one without reading the forward validation slice."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-start", type=float, default=0.05)
    parser.add_argument("--weight-stop", type=float, default=1.00)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--min-full-delta", type=float, default=0.002)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite Dataset1 Setwise experiment: {args.output_dir}"
        )
    if args.seed != 60:
        raise ValueError("the frozen Dataset1 experiment requires seed 60")
    if (
        args.epochs != 10
        or args.patience != 2
        or args.batch_size != 256
        or args.hidden_dim != 32
        or abs(args.learning_rate - 0.001) > 1e-12
    ):
        raise ValueError("Setwise training settings differ from the frozen protocol")
    weights = inclusive_weight_grid(
        args.weight_start,
        args.weight_stop,
        args.weight_step,
    )
    if weights != inclusive_weight_grid(0.05, 1.00, 0.05):
        raise ValueError("Setwise weight grid differs from the frozen protocol")
    require_jittor_cuda(jt)
    args.output_dir.mkdir(parents=True)

    frozen_path = args.output_dir / "frozen-config.json"
    progress_path = args.output_dir / "training-progress.json"
    selection_path = args.output_dir / "selection-report.json"
    evaluation_path = args.output_dir / "evaluation-report.json"
    started = time.time()

    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    joint_contract = validate_joint_cache_reports(
        train_report,
        validation_report,
    )
    if (
        train_report.get("dataset_name") != "dataset1"
        or validation_report.get("dataset_name") != "dataset1"
    ):
        raise ValueError("full-100 cache pair is not bound to dataset1")
    if (
        train_report.get("train_selection") != "recent"
        or int(train_report.get("requested_train_rows", -1))
        != EXPECTED_TRAIN_ROWS
    ):
        raise ValueError("Dataset1 training cache is not the frozen recent-200k cache")

    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    train_rows_path = Path(f"{args.train_cache_prefix}.train-row-indices.npy")
    validation_path = Path(f"{args.validation_cache_prefix}.val.npy")
    _require_report_artifact_hash(
        train_path,
        train_report,
        "features",
        "training features",
    )
    _require_report_artifact_hash(
        train_rows_path,
        train_report,
        "row_indices",
        "training row indices",
    )
    _require_report_artifact_hash(
        validation_path,
        validation_report,
        "features",
        "validation features",
    )
    train_sha_before = _sha256(train_path)
    validation_sha_before = _sha256(validation_path)

    train_features = np.load(train_path, mmap_mode="r", allow_pickle=False)
    validation_features = np.load(
        validation_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_train_shape = (
        EXPECTED_TRAIN_ROWS,
        EXPECTED_CANDIDATES,
        EXPECTED_FEATURES,
    )
    expected_validation_shape = (
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_CANDIDATES,
        EXPECTED_FEATURES,
    )
    if train_features.shape != expected_train_shape:
        raise ValueError(
            f"Dataset1 training feature shape mismatch: {train_features.shape}"
        )
    if validation_features.shape != expected_validation_shape:
        raise ValueError(
            "Dataset1 validation feature shape mismatch: "
            f"{validation_features.shape}"
        )
    train_row_indices = np.load(
        train_rows_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        train_row_indices.shape != (EXPECTED_TRAIN_ROWS,)
        or not np.all(np.diff(train_row_indices) == 1)
    ):
        raise ValueError(
            "recent-200k row indices must be one contiguous chronological window"
        )

    state = load_checkpoint_dataset(args.checkpoint, "dataset1")
    config = state["config"]
    feature_names = tuple(str(name) for name in state["feature_names"])
    if (
        feature_names != tuple(train_report["feature_names"])
        or feature_names != tuple(validation_report["feature_names"])
        or len(feature_names) != EXPECTED_FEATURES
    ):
        raise ValueError("Dataset1 checkpoint and cache feature schemas differ")
    if int(config.seed) != args.seed:
        raise ValueError("Dataset1 checkpoint seed differs from the frozen seed")

    frozen = {
        "status": "frozen_before_training",
        "dataset_name": "dataset1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "joint_cache_contract": joint_contract,
        "train_cache_report": str(args.train_cache_report.resolve()),
        "train_cache_report_sha256": _sha256(args.train_cache_report),
        "validation_cache_report": str(
            args.validation_cache_report.resolve()
        ),
        "validation_cache_report_sha256": _sha256(
            args.validation_cache_report
        ),
        "train_features": str(train_path.resolve()),
        "train_features_sha256": train_sha_before,
        "validation_features": str(validation_path.resolve()),
        "validation_features_sha256": validation_sha_before,
        "train_shape": list(train_features.shape),
        "validation_shape": list(validation_features.shape),
        "train_scales": list(TRAIN_SCALES),
        "recent_100k_view": [100_000, 200_000],
        "selection_rows": [0, SELECTION_STOP],
        "forward_rows": [SELECTION_STOP, EXPECTED_VALIDATION_ROWS],
        "selection_uses_forward_rows": False,
        "setwise": {
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "objective": "negative_log_softmax_of_candidate_zero",
            "context_transform": [
                "raw",
                "candidate_minus_row_mean",
                "candidate_minus_row_max",
            ],
        },
        "weight_grid": {
            "start": weights[0],
            "stop": weights[-1],
            "step": args.weight_step,
            "weights_tested": len(weights),
        },
        "tie_break": [
            "higher_selection_mrr",
            "higher_setwise_weight",
            "larger_training_scale",
        ],
        "baseline": "champion fixed MLP + LightGBM blend",
        "candidate_blend": "Setwise + champion LightGBM",
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "all_three_slices_non_decreasing": True,
        },
    }
    _write_json_atomic(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2), flush=True)

    baseline, champion_lightgbm = _champion_components(
        state,
        validation_features,
        batch_size=512,
    )
    validation_selection = SetwiseFeatureView(
        validation_features[:SELECTION_STOP]
    )
    validation_full = SetwiseFeatureView(validation_features)
    setwise_config = FusionConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=0.0,
        hidden_dim=args.hidden_dim,
        selection_metric="mrr",
        early_stop_patience=args.patience,
    )

    model_reports: dict[str, Any] = {}
    model_scores: dict[str, np.ndarray] = {}
    for train_rows in TRAIN_SCALES:
        model_name = f"recent_{train_rows // 1000}k"
        model_started = time.time()
        training_view = SetwiseFeatureView(train_features[-train_rows:])
        model, result, history = fit_fusion_mlp_listwise_streaming(
            training_view,
            validation_selection,
            setwise_config,
            np.random.default_rng(args.seed),
            verbose=True,
            feature_indices=tuple(range(training_view.shape[-1])),
            candidate_name=f"dataset1_setwise_{model_name}_full100",
        )
        model_path = args.output_dir / f"dataset1-setwise-{model_name}.npz"
        _save_setwise_model(
            model_path,
            result=result,
            hidden_dim=args.hidden_dim,
            source_feature_count=EXPECTED_FEATURES,
            training_rows=train_rows,
            seed=args.seed,
        )
        logits = _predict_streaming(
            model,
            validation_full,
            result.mean,
            result.std,
            feature_indices=result.feature_indices,
            batch_size=args.batch_size,
        )
        probabilities = _softmax(logits)
        prediction_path = (
            args.output_dir / f"validation-setwise-{model_name}.npy"
        )
        np.save(prediction_path, np.asarray(probabilities, dtype=np.float32))
        model_scores[model_name] = probabilities
        model_reports[model_name] = {
            "training_rows": train_rows,
            "training_view": [
                EXPECTED_TRAIN_ROWS - train_rows,
                EXPECTED_TRAIN_ROWS,
            ],
            "model": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "validation_prediction": str(prediction_path.resolve()),
            "validation_prediction_sha256": _sha256(prediction_path),
            "best_selection_ap": result.best_val_ap,
            "best_selection_mrr": result.best_val_mrr,
            "history": list(history),
            "elapsed_seconds": time.time() - model_started,
        }
        _write_json_atomic(
            progress_path,
            {
                "status": "training",
                "completed_models": list(model_reports),
                "models": model_reports,
                "elapsed_seconds": time.time() - started,
            },
        )
        del model, logits, probabilities, training_view
        gc.collect()

    selected = select_setwise_model_blend_on_prefix(
        model_scores,
        champion_lightgbm,
        selection_stop=SELECTION_STOP,
        primary_weights=weights,
        model_tie_break_order=("recent_100k", "recent_200k"),
    )
    selection_report = {
        "status": "locked_before_forward_evaluation",
        "selection_rows": [0, SELECTION_STOP],
        "forward_rows": [SELECTION_STOP, EXPECTED_VALIDATION_ROWS],
        "selection_uses_forward_rows": False,
        "model_name": selected.model_name,
        "training_rows": model_reports[selected.model_name]["training_rows"],
        "setwise_weight": selected.primary_weight,
        "selection_mrr": selected.selection_mrr,
        "models_tested": selected.models_tested,
        "weights_tested_per_model": selected.weights_tested,
        "model_sha256": model_reports[selected.model_name]["model_sha256"],
        "validation_prediction_sha256": model_reports[
            selected.model_name
        ]["validation_prediction_sha256"],
    }
    _write_json_atomic(selection_path, selection_report)
    print(json.dumps(selection_report, ensure_ascii=False, indent=2), flush=True)

    selected_scores = model_scores[selected.model_name]
    blended = (
        selected.primary_weight * selected_scores
        + (1.0 - selected.primary_weight) * champion_lightgbm
    )
    baseline_metrics = ranking_mrr_three_slices(baseline)
    expert_metrics = ranking_mrr_three_slices(selected_scores)
    candidate_metrics = ranking_mrr_three_slices(blended)
    metric_keys = ("full", "slice_0", "slice_1", "slice_2")
    deltas = {
        key: candidate_metrics[key] - baseline_metrics[key]
        for key in metric_keys
    }
    passed = passes_full100_gate(
        baseline_full_mrr=baseline_metrics["full"],
        candidate_full_mrr=candidate_metrics["full"],
        baseline_slice_mrrs=tuple(
            baseline_metrics[f"slice_{index}"] for index in range(3)
        ),
        candidate_slice_mrrs=tuple(
            candidate_metrics[f"slice_{index}"] for index in range(3)
        ),
        min_full_delta=args.min_full_delta,
    )
    train_sha_after = _sha256(train_path)
    validation_sha_after = _sha256(validation_path)
    if (
        train_sha_after != train_sha_before
        or validation_sha_after != validation_sha_before
    ):
        raise RuntimeError("source full-100 caches changed during Setwise training")

    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "package_authorized": passed,
        "package_generated": False,
        "winner": "setwise" if passed else None,
        "frozen_config": frozen,
        "selection_report": str(selection_path.resolve()),
        "selection_report_sha256": _sha256(selection_path),
        "selection": selection_report,
        "models": model_reports,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_vs_champion": deltas,
        "setwise": {
            "gate_passed": passed,
            "selected_weight": selected.primary_weight,
            "training_rows": model_reports[
                selected.model_name
            ]["training_rows"],
            "model_path": model_reports[selected.model_name]["model"],
            "model_sha256": model_reports[selected.model_name]["model_sha256"],
            "best_val_ap": model_reports[
                selected.model_name
            ]["best_selection_ap"],
            "best_val_mrr": model_reports[
                selected.model_name
            ]["best_selection_mrr"],
            "expert": expert_metrics,
            "fixed_blend": candidate_metrics,
            "full_delta": deltas["full"],
            "slice_deltas": [
                deltas[f"slice_{index}"] for index in range(3)
            ],
        },
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "full_delta_passed": bool(
                deltas["full"] + 1e-12 >= args.min_full_delta
            ),
            "all_three_slices_non_decreasing": bool(
                all(deltas[f"slice_{index}"] >= 0.0 for index in range(3))
            ),
            "forward_slice_non_decreasing": bool(deltas["slice_2"] >= 0.0),
        },
        "train_cache_sha256_before": train_sha_before,
        "train_cache_sha256_after": train_sha_after,
        "validation_cache_sha256_before": validation_sha_before,
        "validation_cache_sha256_after": validation_sha_after,
        "source_caches_unchanged": True,
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(evaluation_path, report)
    _write_json_atomic(
        progress_path,
        {
            "status": report["status"],
            "gate_passed": passed,
            "selection": selection_report,
            "delta_vs_champion": deltas,
            "elapsed_seconds": report["elapsed_seconds"],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 2


def _champion_components(
    state: dict[str, Any],
    validation_features: Any,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    fusion_result = state["fusion_result"]
    fusion_indices = tuple(
        int(index) for index in fusion_result.feature_indices
    )
    fusion_model = build_fusion_from_state(
        input_dim=len(fusion_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    mlp_logits = _predict_streaming(
        fusion_model,
        validation_features,
        fusion_result.mean,
        fusion_result.std,
        feature_indices=fusion_indices,
        batch_size=batch_size,
    )
    mlp = _softmax(mlp_logits)
    del mlp_logits, fusion_model
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError(
            "Dataset1 Setwise protocol requires the champion LightGBM expert"
        )
    lightgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, validation_features)
    )
    mlp_weight = float(lgbm_result.mlp_weight)
    baseline = mlp_weight * mlp + (1.0 - mlp_weight) * lightgbm
    return baseline, lightgbm


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


def _save_setwise_model(
    path: Path,
    *,
    result: Any,
    hidden_dim: int,
    source_feature_count: int,
    training_rows: int,
    seed: int,
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
        "training_rows": np.asarray([training_rows], dtype=np.int32),
        "training_seed": np.asarray([seed], dtype=np.int32),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _require_report_artifact_hash(
    path: Path,
    report: dict[str, Any],
    artifact_name: str,
    label: str,
) -> None:
    expected = str(report["artifacts"][artifact_name]["sha256"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
