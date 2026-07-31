from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset, set_model_state
from jgrec.rankers.hybrid.full100_training import (
    passes_full100_gate,
    validate_joint_cache_reports,
)
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    FusionMLP,
    fit_fusion_mlp_listwise_streaming,
    predict_logits,
)
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_three_slices
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView
from jgrec.rankers.hybrid.window_diversity import (
    blend_expert_subset,
    normalized_exponential_recency_weights,
    recent_window_view,
    select_uniform_subset_on_prefix,
)

EXPECTED_TRAIN_SHAPE = (200_000, 100, 63)
EXPECTED_VALIDATION_SHAPE = (20_000, 100, 63)
EXPERT_ORDER = (
    "recent50k",
    "recent100k",
    "recent200k",
    "recent200k_decay100k",
)
SELECTION_STOP = 13_334
SETWISE_WEIGHT = 0.80
SEED = 60
DECAY_HALF_LIFE_ROWS = 100_000


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train/select or forward-gate the frozen Dataset2 Setwise "
            "time-window diversity experiment."
        )
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    train = subparsers.add_parser("train-select")
    train.add_argument("--checkpoint", required=True, type=Path)
    train.add_argument("--train-cache-prefix", required=True, type=Path)
    train.add_argument("--train-cache-report", required=True, type=Path)
    train.add_argument("--validation-cache-prefix", required=True, type=Path)
    train.add_argument("--validation-cache-report", required=True, type=Path)
    train.add_argument("--champion-evaluation-report", required=True, type=Path)
    train.add_argument("--recent200k-model", required=True, type=Path)
    train.add_argument("--frozen-dataset1-csv", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--setwise-epochs", type=int, default=10)
    train.add_argument("--setwise-patience", type=int, default=2)
    train.add_argument("--setwise-batch-size", type=int, default=256)
    train.add_argument("--setwise-hidden-dim", type=int, default=32)
    train.add_argument("--setwise-learning-rate", type=float, default=0.001)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--output-dir", required=True, type=Path)
    gate.add_argument("--min-full-delta", type=float, default=0.001)

    args = parser.parse_args()
    if args.phase == "train-select":
        return _train_and_select(args)
    return _gate(args)


def _train_and_select(args: argparse.Namespace) -> int:
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite window experiment: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    artifacts_dir = args.output_dir / "artifacts"
    artifacts_dir.mkdir()

    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    champion_evaluation = _read_json(args.champion_evaluation_report)
    joint_contract = validate_joint_cache_reports(
        train_report,
        validation_report,
    )
    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    validation_path = Path(f"{args.validation_cache_prefix}.val.npy")
    train_features = np.load(train_path, mmap_mode="r", allow_pickle=False)
    validation_features = np.load(
        validation_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if train_features.shape != EXPECTED_TRAIN_SHAPE:
        raise ValueError(f"training feature shape mismatch: {train_features.shape}")
    if validation_features.shape != EXPECTED_VALIDATION_SHAPE:
        raise ValueError(
            f"validation feature shape mismatch: {validation_features.shape}"
        )
    if train_features.dtype != np.float32 or validation_features.dtype != np.float32:
        raise ValueError("window experiment requires float32 feature caches")

    train_sha = str(train_report["artifacts"]["features"]["sha256"])
    validation_sha = str(
        validation_report["artifacts"]["features"]["sha256"]
    )
    _require_hash(train_path, train_sha, "training features")
    _require_hash(validation_path, validation_sha, "validation features")

    source_frozen = champion_evaluation["frozen_config"]
    _require_hash(
        args.checkpoint,
        str(source_frozen["checkpoint_sha256"]),
        "champion checkpoint",
    )
    _require_hash(
        args.recent200k_model,
        str(champion_evaluation["setwise"]["model_sha256"]),
        "recent-200k Setwise model",
    )
    if float(champion_evaluation["setwise"]["selected_weight"]) != SETWISE_WEIGHT:
        raise ValueError("champion does not use the frozen Setwise weight 0.80")

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if feature_names != tuple(train_report["feature_names"]):
        raise ValueError("checkpoint and training cache feature schemas differ")
    if feature_names != tuple(validation_report["feature_names"]):
        raise ValueError("training and validation feature schemas differ")
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("champion checkpoint has no Dataset2 LightGBM expert")
    if tuple(int(index) for index in lgbm_result.feature_indices) != tuple(
        range(EXPECTED_TRAIN_SHAPE[-1])
    ):
        raise ValueError("window experiment requires all 63 LightGBM features")

    config = FusionConfig(
        epochs=args.setwise_epochs,
        batch_size=args.setwise_batch_size,
        lr=args.setwise_learning_rate,
        weight_decay=0.0,
        hidden_dim=args.setwise_hidden_dim,
        selection_metric="mrr",
        early_stop_patience=args.setwise_patience,
    )
    frozen = {
        "status": "frozen_before_training",
        "protocol_version": 1,
        "expert_order": list(EXPERT_ORDER),
        "seed": SEED,
        "selection_rows": [0, SELECTION_STOP],
        "forward_rows": [SELECTION_STOP, EXPECTED_VALIDATION_SHAPE[0]],
        "selection_uses_forward_rows": False,
        "expert_subset_policy": "all 15 non-empty subsets; uniform probability mean",
        "tie_break": "higher prefix MRR, then fewer experts, then frozen order",
        "outer_blend": {
            "setwise_expert_weight": SETWISE_WEIGHT,
            "champion_lightgbm_weight": 1.0 - SETWISE_WEIGHT,
            "weight_search": False,
        },
        "experts": {
            "recent50k": {
                "train_rows": 50_000,
                "row_weights": "uniform",
                "policy": "train",
            },
            "recent100k": {
                "train_rows": 100_000,
                "row_weights": "uniform",
                "policy": "train",
            },
            "recent200k": {
                "train_rows": 200_000,
                "row_weights": "uniform",
                "policy": "reuse champion",
            },
            "recent200k_decay100k": {
                "train_rows": 200_000,
                "row_weights": "exponential recency, normalized mean one",
                "half_life_rows": DECAY_HALF_LIFE_ROWS,
                "policy": "train",
            },
        },
        "setwise": {
            "epochs": config.epochs,
            "patience": config.early_stop_patience,
            "batch_size": config.batch_size,
            "hidden_dim": config.hidden_dim,
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "objective": "weighted negative log-softmax of candidate zero",
            "early_stopping_rows": [0, SELECTION_STOP],
            "context_transform_version": 1,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "champion_evaluation_report": str(
            args.champion_evaluation_report.resolve()
        ),
        "champion_evaluation_report_sha256": _sha256(
            args.champion_evaluation_report
        ),
        "recent200k_model": str(args.recent200k_model.resolve()),
        "recent200k_model_sha256": _sha256(args.recent200k_model),
        "train_features": str(train_path.resolve()),
        "train_features_sha256": train_sha,
        "validation_features": str(validation_path.resolve()),
        "validation_features_sha256": validation_sha,
        "joint_cache_contract": joint_contract,
        "train_shape": list(train_features.shape),
        "validation_shape": list(validation_features.shape),
        "feature_names": list(feature_names),
        "frozen_dataset1_csv": str(args.frozen_dataset1_csv.resolve()),
        "frozen_dataset1_csv_sha256": _sha256(args.frozen_dataset1_csv),
    }
    frozen_path = args.output_dir / "frozen-config.json"
    _write_json_atomic(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, sort_keys=True), flush=True)

    started = time.time()
    validation_prefix_view = SetwiseFeatureView(
        validation_features[:SELECTION_STOP]
    )
    validation_full_view = SetwiseFeatureView(validation_features)
    secondary_probabilities = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, validation_features)
    )
    secondary_path = artifacts_dir / "validation-probabilities-lightgbm.npy"
    np.save(secondary_path, secondary_probabilities.astype(np.float32))

    expert_probabilities: dict[str, np.ndarray] = {}
    expert_reports: dict[str, Any] = {}
    for expert_name in EXPERT_ORDER:
        expert_started = time.time()
        if expert_name == "recent200k":
            model, payload = _load_setwise_model(
                args.recent200k_model,
                expected_source_feature_count=EXPECTED_TRAIN_SHAPE[-1],
            )
            model_path = args.recent200k_model
            mean = payload["mean"]
            std = payload["std"]
            feature_indices = payload["feature_indices"]
            history: tuple[dict[str, float | int], ...] = ()
            best_prefix_mrr: float | None = None
            reused = True
            train_rows = 200_000
            weights_report: dict[str, Any] = {"policy": "uniform"}
        else:
            train_rows = {
                "recent50k": 50_000,
                "recent100k": 100_000,
                "recent200k_decay100k": 200_000,
            }[expert_name]
            raw_window = recent_window_view(train_features, train_rows)
            training_view = SetwiseFeatureView(raw_window)
            train_row_weights = None
            weights_report = {"policy": "uniform"}
            if expert_name == "recent200k_decay100k":
                train_row_weights = normalized_exponential_recency_weights(
                    row_count=train_rows,
                    half_life_rows=DECAY_HALF_LIFE_ROWS,
                )
                weights_report = {
                    "policy": "exponential_recency_normalized_mean_one",
                    "half_life_rows": DECAY_HALF_LIFE_ROWS,
                    "oldest": float(train_row_weights[0]),
                    "newest": float(train_row_weights[-1]),
                    "mean": float(
                        np.mean(train_row_weights, dtype=np.float64)
                    ),
                }
            model, result, history = fit_fusion_mlp_listwise_streaming(
                training_view,
                validation_prefix_view,
                config,
                np.random.default_rng(SEED),
                verbose=True,
                feature_indices=tuple(range(training_view.shape[-1])),
                candidate_name=f"dataset2_{expert_name}_seed{SEED}",
                train_row_weights=train_row_weights,
            )
            model_path = artifacts_dir / f"dataset2-setwise-{expert_name}.npz"
            _save_setwise_model(
                model_path,
                result=result,
                hidden_dim=config.hidden_dim,
                source_feature_count=EXPECTED_TRAIN_SHAPE[-1],
                seed=SEED,
                train_rows=train_rows,
                decay_half_life_rows=(
                    DECAY_HALF_LIFE_ROWS
                    if expert_name == "recent200k_decay100k"
                    else None
                ),
            )
            mean = np.asarray(result.mean, dtype=np.float32)
            std = np.asarray(result.std, dtype=np.float32)
            feature_indices = tuple(result.feature_indices)
            best_prefix_mrr = float(result.best_val_mrr)
            reused = False
            del training_view, raw_window, train_row_weights

        logits = _predict_streaming(
            model,
            validation_full_view,
            mean,
            std,
            feature_indices=feature_indices,
            batch_size=config.batch_size,
        )
        probabilities_path = (
            artifacts_dir / f"validation-probabilities-{expert_name}.npy"
        )
        np.save(probabilities_path, _softmax(logits).astype(np.float32))
        saved_probabilities = np.load(
            probabilities_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        expert_probabilities[expert_name] = saved_probabilities
        expert_reports[expert_name] = {
            "reused": reused,
            "train_rows": train_rows,
            "row_weights": weights_report,
            "model": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "validation_probabilities": str(probabilities_path.resolve()),
            "validation_probabilities_sha256": _sha256(probabilities_path),
            "best_prefix_mrr": best_prefix_mrr,
            "history": list(history),
            "elapsed_seconds": time.time() - expert_started,
        }
        print(
            f"[window-diversity] expert={expert_name} "
            f"elapsed={expert_reports[expert_name]['elapsed_seconds']:.1f}s",
            flush=True,
        )
        del model, logits
        gc.collect()

    saved_secondary = np.load(
        secondary_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    selection = select_uniform_subset_on_prefix(
        expert_probabilities,
        saved_secondary,
        selection_stop=SELECTION_STOP,
        expert_weight=SETWISE_WEIGHT,
        expert_order=EXPERT_ORDER,
    )
    baseline_candidate = next(
        candidate
        for candidate in selection.candidates
        if candidate.experts == ("recent200k",)
    )
    expected_baseline_prefix = float(
        champion_evaluation["setwise"]["selection_mrr"]
    )
    if (
        abs(baseline_candidate.selection_mrr - expected_baseline_prefix)
        > 1e-10
    ):
        raise ValueError(
            "recent-200k prefix MRR does not reproduce the champion: "
            f"actual={baseline_candidate.selection_mrr} "
            f"expected={expected_baseline_prefix}"
        )

    selection_report = {
        "status": "locked_before_forward_gate",
        "selection_rows": [0, SELECTION_STOP],
        "forward_rows": [SELECTION_STOP, EXPECTED_VALIDATION_SHAPE[0]],
        "selection_uses_forward_rows": False,
        "selected_experts": list(selection.selected_experts),
        "selection_mrr": selection.selection_mrr,
        "baseline_recent200k_selection_mrr": (
            baseline_candidate.selection_mrr
        ),
        "candidates": [
            asdict(candidate)
            for candidate in selection.candidates
        ],
        "experts": expert_reports,
        "secondary_probabilities": str(secondary_path.resolve()),
        "secondary_probabilities_sha256": _sha256(secondary_path),
        "frozen_config": str(frozen_path.resolve()),
        "frozen_config_sha256": _sha256(frozen_path),
        "elapsed_seconds": time.time() - started,
    }
    selection_path = args.output_dir / "selection-report.json"
    _write_json_atomic(selection_path, selection_report)
    selection_sha = _sha256(selection_path)
    _write_text_atomic(
        args.output_dir / "selection-report.sha256",
        f"{selection_sha}\n",
    )
    print(
        json.dumps(selection_report, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return 0


def _gate(args: argparse.Namespace) -> int:
    if abs(args.min_full_delta - 0.001) > 1e-12:
        raise ValueError("window experiment fixes minimum full delta at 0.001")
    report_path = args.output_dir / "evaluation-report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite gate report: {report_path}")
    frozen_path = args.output_dir / "frozen-config.json"
    selection_path = args.output_dir / "selection-report.json"
    selection_sha_path = args.output_dir / "selection-report.sha256"
    frozen = _read_json(frozen_path)
    selection = _read_json(selection_path)
    locked_sha = selection_sha_path.read_text(encoding="utf-8").strip()
    _require_hash(selection_path, locked_sha, "locked selection report")
    if selection.get("status") != "locked_before_forward_gate":
        raise ValueError("selection report is not locked for forward gating")
    if selection.get("selection_uses_forward_rows") is not False:
        raise ValueError("selection report did not isolate forward rows")
    if selection.get("selection_rows") != [0, SELECTION_STOP]:
        raise ValueError("selection report row boundary differs")
    if _sha256(frozen_path) != selection["frozen_config_sha256"]:
        raise ValueError("frozen config changed after selection")

    champion_evaluation = _read_json(
        Path(frozen["champion_evaluation_report"])
    )
    champion = {
        key: float(value)
        for key, value in champion_evaluation["setwise"]["fixed_blend"].items()
    }
    secondary_path = Path(selection["secondary_probabilities"])
    _require_hash(
        secondary_path,
        selection["secondary_probabilities_sha256"],
        "LightGBM probabilities",
    )
    secondary = np.load(secondary_path, mmap_mode="r", allow_pickle=False)
    selected_experts = tuple(str(name) for name in selection["selected_experts"])
    if (
        not selected_experts
        or any(name not in EXPERT_ORDER for name in selected_experts)
    ):
        raise ValueError("locked selection contains an unknown expert")

    expert_probabilities: dict[str, np.ndarray] = {}
    for expert_name in {*selected_experts, "recent200k"}:
        expert_report = selection["experts"][expert_name]
        probability_path = Path(expert_report["validation_probabilities"])
        _require_hash(
            probability_path,
            expert_report["validation_probabilities_sha256"],
            f"{expert_name} probabilities",
        )
        expert_probabilities[expert_name] = np.load(
            probability_path,
            mmap_mode="r",
            allow_pickle=False,
        )

    baseline_scores = blend_expert_subset(
        expert_probabilities,
        secondary,
        selected_experts=("recent200k",),
        expert_weight=SETWISE_WEIGHT,
    )
    reproduced_champion = ranking_mrr_three_slices(baseline_scores)
    _require_metrics_close(
        reproduced_champion,
        champion,
        "recent-200k champion",
    )
    selected_scores = blend_expert_subset(
        expert_probabilities,
        secondary,
        selected_experts=selected_experts,
        expert_weight=SETWISE_WEIGHT,
    )
    candidate = ranking_mrr_three_slices(selected_scores)
    metric_keys = ("full", "slice_0", "slice_1", "slice_2")
    deltas = {
        key: float(candidate[key] - champion[key])
        for key in metric_keys
    }
    passed = passes_full100_gate(
        baseline_full_mrr=champion["full"],
        candidate_full_mrr=candidate["full"],
        baseline_slice_mrrs=tuple(
            champion[f"slice_{index}"] for index in range(3)
        ),
        candidate_slice_mrrs=tuple(
            candidate[f"slice_{index}"] for index in range(3)
        ),
        min_full_delta=args.min_full_delta,
    )
    prediction_path = (
        args.output_dir / "artifacts" / "validation-selected-blend.npy"
    )
    np.save(prediction_path, selected_scores.astype(np.float32))
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "package_authorized": passed,
        "package_generated": False,
        "selection_report": str(selection_path.resolve()),
        "selection_report_sha256": locked_sha,
        "selected_experts": list(selected_experts),
        "selection_mrr": float(selection["selection_mrr"]),
        "champion": champion,
        "reproduced_champion": reproduced_champion,
        "candidate": candidate,
        "delta_vs_champion": deltas,
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "full_delta_passed": bool(
                deltas["full"] + 1e-12 >= args.min_full_delta
            ),
            "slice_0_non_decreasing": bool(deltas["slice_0"] >= 0.0),
            "slice_1_non_decreasing": bool(deltas["slice_1"] >= 0.0),
            "slice_2_non_decreasing": bool(deltas["slice_2"] >= 0.0),
            "all_three_slices_non_decreasing": bool(
                all(deltas[f"slice_{index}"] >= 0.0 for index in range(3))
            ),
        },
        "selected_prediction": str(prediction_path.resolve()),
        "selected_prediction_sha256": _sha256(prediction_path),
        "frozen_dataset1_csv": frozen["frozen_dataset1_csv"],
        "frozen_dataset1_csv_sha256": frozen[
            "frozen_dataset1_csv_sha256"
        ],
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 2


def _load_setwise_model(
    path: Path,
    *,
    expected_source_feature_count: int,
) -> tuple[FusionMLP, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    if int(payload["source_feature_count"][0]) != expected_source_feature_count:
        raise ValueError("Setwise source feature count differs")
    if int(payload["context_transform_version"][0]) != 1:
        raise ValueError("unsupported Setwise context transform")
    feature_indices = tuple(
        int(value) for value in payload["feature_indices"]
    )
    state = {
        key.removeprefix("state__"): np.asarray(value, dtype=np.float32)
        for key, value in payload.items()
        if key.startswith("state__")
    }
    model = FusionMLP(
        input_dim=len(feature_indices),
        hidden_dim=int(payload["hidden_dim"][0]),
    )
    set_model_state(model, state)
    return model, {
        "mean": np.asarray(payload["mean"], dtype=np.float32),
        "std": np.asarray(payload["std"], dtype=np.float32),
        "feature_indices": feature_indices,
    }


def _predict_streaming(
    model: FusionMLP,
    features: Any,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    feature_indices: tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        batch = np.asarray(features[start:end], dtype=np.float32)
        if feature_indices != tuple(range(batch.shape[-1])):
            batch = batch[..., feature_indices]
        scores[start:end] = predict_logits(model, batch, mean, std)
    return scores


def _save_setwise_model(
    path: Path,
    *,
    result: Any,
    hidden_dim: int,
    source_feature_count: int,
    seed: int,
    train_rows: int,
    decay_half_life_rows: int | None,
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
        "training_seed": np.asarray([seed], dtype=np.int32),
        "train_rows": np.asarray([train_rows], dtype=np.int32),
        "decay_half_life_rows": np.asarray(
            [-1 if decay_half_life_rows is None else decay_half_life_rows],
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _require_metrics_close(
    actual: dict[str, float],
    expected: dict[str, float],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if abs(float(actual[key]) - float(expected_value)) > 1e-10:
            raise ValueError(
                f"{label} metric mismatch for {key}: "
                f"actual={actual[key]} expected={expected_value}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: actual={actual} expected={expected}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    _write_text_atomic(path, f"{text}\n")


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
