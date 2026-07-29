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
    select_temporally_robust_candidate_on_prefix,
)

EXPECTED_TRAIN_SHAPE = (200_000, 100, 63)
EXPECTED_VALIDATION_SHAPE = (20_000, 100, 63)
FIRST_SLICE_STOP = 6_667
SELECTION_STOP = 13_334
SETWISE_WEIGHT = 0.80
SEED = 60
V1_TRANSFORM_VERSION = 1
V2_TRANSFORM_VERSION = 2
CANDIDATE_ORDER = (
    "v1_champion",
    "v2_relative",
    "v1_v2_uniform",
)
CANDIDATE_COMPLEXITY = {
    "v1_champion": 1,
    "v2_relative": 1,
    "v1_v2_uniform": 2,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train/select or forward-gate Dataset2 Setwise relative-context v2."
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
    train.add_argument("--v1-model", required=True, type=Path)
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
            f"refusing to overwrite relative-context experiment: "
            f"{args.output_dir}"
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
        raise ValueError("relative-context experiment requires float32 caches")

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
        args.v1_model,
        str(champion_evaluation["setwise"]["model_sha256"]),
        "v1 Setwise model",
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
        raise ValueError("relative-context experiment requires all 63 features")

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
        "seed": SEED,
        "source_feature_count": EXPECTED_TRAIN_SHAPE[-1],
        "v1_context_width": EXPECTED_TRAIN_SHAPE[-1] * 3,
        "v2_context_width": EXPECTED_TRAIN_SHAPE[-1] * 5,
        "v2_transform": {
            "version": V2_TRANSFORM_VERSION,
            "channels": [
                "raw",
                "raw_minus_row_mean",
                "raw_minus_row_max",
                "tie_neutral_ascending_percentile_midrank",
                "robust_row_zscore",
            ],
            "percentile_formula": (
                "average ascending tie rank / (candidate_count - 1)"
            ),
            "single_candidate_percentile": 0.5,
            "robust_zscore_formula": (
                "(value - row_median) / (1.4826 * row_MAD)"
            ),
            "zero_mad_policy": "return zero when MAD <= 1e-6",
            "clipping": False,
        },
        "selection_rows": [0, SELECTION_STOP],
        "visible_slices": [
            [0, FIRST_SLICE_STOP],
            [FIRST_SLICE_STOP, SELECTION_STOP],
        ],
        "forward_rows": [SELECTION_STOP, EXPECTED_VALIDATION_SHAPE[0]],
        "selection_uses_forward_rows": False,
        "candidate_order": list(CANDIDATE_ORDER),
        "candidate_complexity": CANDIDATE_COMPLEXITY,
        "candidate_policy": {
            "v1_champion": "0.80 * v1 + 0.20 * LightGBM",
            "v2_relative": "0.80 * v2 + 0.20 * LightGBM",
            "v1_v2_uniform": (
                "0.80 * mean(v1, v2) + 0.20 * LightGBM"
            ),
        },
        "selection_policy": (
            "require slice0 and slice1 >= v1 champion; maximize combined "
            "prefix MRR; tie-break by fewer Setwise models then frozen order"
        ),
        "setwise": {
            "epochs": config.epochs,
            "patience": config.early_stop_patience,
            "batch_size": config.batch_size,
            "hidden_dim": config.hidden_dim,
            "learning_rate": config.lr,
            "weight_decay": config.weight_decay,
            "objective": "negative log-softmax of candidate zero",
            "early_stopping_rows": [0, SELECTION_STOP],
            "outer_blend_weight": SETWISE_WEIGHT,
        },
        "gate": {
            "minimum_full_mrr_delta": 0.001,
            "all_three_slices_non_decreasing": True,
            "package_only_after_pass": True,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "champion_evaluation_report": str(
            args.champion_evaluation_report.resolve()
        ),
        "champion_evaluation_report_sha256": _sha256(
            args.champion_evaluation_report
        ),
        "v1_model": str(args.v1_model.resolve()),
        "v1_model_sha256": _sha256(args.v1_model),
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
    train_view = SetwiseFeatureView(
        train_features,
        transform_version=V2_TRANSFORM_VERSION,
    )
    validation_prefix_view = SetwiseFeatureView(
        validation_features[:SELECTION_STOP],
        transform_version=V2_TRANSFORM_VERSION,
    )
    v2_model, v2_result, history = fit_fusion_mlp_listwise_streaming(
        train_view,
        validation_prefix_view,
        config,
        np.random.default_rng(SEED),
        verbose=True,
        feature_indices=tuple(range(train_view.shape[-1])),
        candidate_name=f"dataset2_relative_context_v2_seed{SEED}",
    )
    v2_model_path = artifacts_dir / "dataset2-setwise-context-v2.npz"
    _save_setwise_model(
        v2_model_path,
        result=v2_result,
        hidden_dim=config.hidden_dim,
        source_feature_count=EXPECTED_TRAIN_SHAPE[-1],
        transform_version=V2_TRANSFORM_VERSION,
        seed=SEED,
    )
    del train_view, validation_prefix_view
    gc.collect()

    validation_v2_view = SetwiseFeatureView(
        validation_features,
        transform_version=V2_TRANSFORM_VERSION,
    )
    v2_logits = _predict_streaming(
        v2_model,
        validation_v2_view,
        np.asarray(v2_result.mean, dtype=np.float32),
        np.asarray(v2_result.std, dtype=np.float32),
        feature_indices=tuple(v2_result.feature_indices),
        batch_size=config.batch_size,
    )
    v2_probability_path = artifacts_dir / "validation-probabilities-v2.npy"
    np.save(v2_probability_path, _softmax(v2_logits).astype(np.float32))
    del validation_v2_view, v2_logits, v2_model
    gc.collect()

    v1_model, v1_payload = _load_setwise_model(
        args.v1_model,
        expected_source_feature_count=EXPECTED_TRAIN_SHAPE[-1],
        expected_transform_version=V1_TRANSFORM_VERSION,
    )
    validation_v1_view = SetwiseFeatureView(
        validation_features,
        transform_version=V1_TRANSFORM_VERSION,
    )
    v1_logits = _predict_streaming(
        v1_model,
        validation_v1_view,
        v1_payload["mean"],
        v1_payload["std"],
        feature_indices=v1_payload["feature_indices"],
        batch_size=config.batch_size,
    )
    v1_probability_path = artifacts_dir / "validation-probabilities-v1.npy"
    np.save(v1_probability_path, _softmax(v1_logits).astype(np.float32))
    del validation_v1_view, v1_logits, v1_model
    gc.collect()

    lgbm_probability_path = (
        artifacts_dir / "validation-probabilities-lightgbm.npy"
    )
    np.save(
        lgbm_probability_path,
        _softmax(
            predict_logits_lgbm(lgbm_result.model_text, validation_features)
        ).astype(np.float32),
    )
    v1_probability = np.load(
        v1_probability_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    v2_probability = np.load(
        v2_probability_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    lgbm_probability = np.load(
        lgbm_probability_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    candidates = _candidate_scores(
        v1_probability,
        v2_probability,
        lgbm_probability,
    )
    selection = select_temporally_robust_candidate_on_prefix(
        candidates,
        candidates["v1_champion"],
        first_slice_stop=FIRST_SLICE_STOP,
        selection_stop=SELECTION_STOP,
        candidate_complexity=CANDIDATE_COMPLEXITY,
        candidate_order=CANDIDATE_ORDER,
    )
    baseline = next(
        candidate
        for candidate in selection.candidates
        if candidate.name == "v1_champion"
    )
    _require_close(
        baseline.selection_mrr,
        float(champion_evaluation["setwise"]["selection_mrr"]),
        "v1 prefix MRR",
    )
    _require_close(
        baseline.slice_0_mrr,
        float(
            champion_evaluation["setwise"]["fixed_blend"]["slice_0"]
        ),
        "v1 slice0 MRR",
    )
    _require_close(
        baseline.slice_1_mrr,
        float(
            champion_evaluation["setwise"]["fixed_blend"]["slice_1"]
        ),
        "v1 slice1 MRR",
    )

    selection_report = {
        "status": "locked_before_forward_gate",
        "selection_rows": [0, SELECTION_STOP],
        "visible_slices": [
            [0, FIRST_SLICE_STOP],
            [FIRST_SLICE_STOP, SELECTION_STOP],
        ],
        "forward_rows": [SELECTION_STOP, EXPECTED_VALIDATION_SHAPE[0]],
        "selection_uses_forward_rows": False,
        "selected_candidate": selection.selected_name,
        "selection_mrr": selection.selection_mrr,
        "candidates": [
            asdict(candidate)
            for candidate in selection.candidates
        ],
        "v2": {
            "model": str(v2_model_path.resolve()),
            "model_sha256": _sha256(v2_model_path),
            "validation_probabilities": str(v2_probability_path.resolve()),
            "validation_probabilities_sha256": _sha256(v2_probability_path),
            "best_prefix_mrr": float(v2_result.best_val_mrr),
            "best_prefix_ap": float(v2_result.best_val_ap),
            "history": list(history),
        },
        "v1": {
            "model": str(args.v1_model.resolve()),
            "model_sha256": _sha256(args.v1_model),
            "validation_probabilities": str(v1_probability_path.resolve()),
            "validation_probabilities_sha256": _sha256(v1_probability_path),
        },
        "lightgbm": {
            "validation_probabilities": str(
                lgbm_probability_path.resolve()
            ),
            "validation_probabilities_sha256": _sha256(
                lgbm_probability_path
            ),
        },
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
        raise ValueError(
            "relative-context experiment fixes minimum full delta at 0.001"
        )
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
    if _sha256(frozen_path) != selection["frozen_config_sha256"]:
        raise ValueError("frozen config changed after selection")

    champion_evaluation = _read_json(
        Path(frozen["champion_evaluation_report"])
    )
    champion = {
        key: float(value)
        for key, value in champion_evaluation["setwise"]["fixed_blend"].items()
    }
    v1_path = Path(selection["v1"]["validation_probabilities"])
    v2_path = Path(selection["v2"]["validation_probabilities"])
    lgbm_path = Path(selection["lightgbm"]["validation_probabilities"])
    _require_hash(
        v1_path,
        selection["v1"]["validation_probabilities_sha256"],
        "v1 probabilities",
    )
    _require_hash(
        v2_path,
        selection["v2"]["validation_probabilities_sha256"],
        "v2 probabilities",
    )
    _require_hash(
        lgbm_path,
        selection["lightgbm"]["validation_probabilities_sha256"],
        "LightGBM probabilities",
    )
    candidates = _candidate_scores(
        np.load(v1_path, mmap_mode="r", allow_pickle=False),
        np.load(v2_path, mmap_mode="r", allow_pickle=False),
        np.load(lgbm_path, mmap_mode="r", allow_pickle=False),
    )
    reproduced_champion = ranking_mrr_three_slices(
        candidates["v1_champion"]
    )
    _require_metrics_close(
        reproduced_champion,
        champion,
        "v1 champion",
    )
    selected_name = str(selection["selected_candidate"])
    if selected_name not in CANDIDATE_ORDER:
        raise ValueError("locked selection contains an unknown candidate")
    selected_scores = candidates[selected_name]
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
        "selected_candidate": selected_name,
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


def _candidate_scores(
    v1_probability: np.ndarray,
    v2_probability: np.ndarray,
    lgbm_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    v1 = np.asarray(v1_probability, dtype=np.float64)
    v2 = np.asarray(v2_probability, dtype=np.float64)
    lgbm = np.asarray(lgbm_probability, dtype=np.float64)
    if v1.shape != v2.shape or v1.shape != lgbm.shape:
        raise ValueError("v1, v2, and LightGBM probabilities must align")
    return {
        "v1_champion": (
            SETWISE_WEIGHT * v1
            + (1.0 - SETWISE_WEIGHT) * lgbm
        ),
        "v2_relative": (
            SETWISE_WEIGHT * v2
            + (1.0 - SETWISE_WEIGHT) * lgbm
        ),
        "v1_v2_uniform": (
            SETWISE_WEIGHT * ((v1 + v2) / 2.0)
            + (1.0 - SETWISE_WEIGHT) * lgbm
        ),
    }


def _load_setwise_model(
    path: Path,
    *,
    expected_source_feature_count: int,
    expected_transform_version: int,
) -> tuple[FusionMLP, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    if int(payload["source_feature_count"][0]) != expected_source_feature_count:
        raise ValueError("Setwise source feature count differs")
    if (
        int(payload["context_transform_version"][0])
        != expected_transform_version
    ):
        raise ValueError("Setwise context transform version differs")
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
    transform_version: int,
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
        "context_transform_version": np.asarray(
            [transform_version],
            dtype=np.int32,
        ),
        "training_seed": np.asarray([seed], dtype=np.int32),
        "train_rows": np.asarray(
            [EXPECTED_TRAIN_SHAPE[0]],
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


def _require_close(
    actual: float,
    expected: float,
    label: str,
) -> None:
    if abs(actual - expected) > 1e-10:
        raise ValueError(
            f"{label} mismatch: actual={actual} expected={expected}"
        )


def _require_metrics_close(
    actual: dict[str, float],
    expected: dict[str, float],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        _require_close(
            float(actual[key]),
            float(expected_value),
            f"{label} {key}",
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
