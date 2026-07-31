from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
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
from jgrec.rankers.hybrid.fusion_analysis import (
    ranking_mrr_three_slices,
    uniform_rank_average,
)
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

EXPECTED_TRAIN_SHAPE = (200_000, 100, 63)
EXPECTED_VALIDATION_SHAPE = (20_000, 100, 63)
SEEDS = (17, 41, 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train Dataset2 Setwise seeds 17/41, reuse the contract-verified "
            "seed-60 champion, and evaluate their fixed uniform rank average."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--source-evaluation-report", required=True, type=Path)
    parser.add_argument("--seed60-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--setwise-weight", type=float, default=0.80)
    parser.add_argument("--min-full-delta", type=float, default=0.001)
    parser.add_argument("--setwise-epochs", type=int, default=10)
    parser.add_argument("--setwise-patience", type=int, default=2)
    parser.add_argument("--setwise-batch-size", type=int, default=256)
    parser.add_argument("--setwise-hidden-dim", type=int, default=32)
    parser.add_argument("--setwise-learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite three-seed experiment: {args.output_dir}"
        )
    if abs(args.setwise_weight - 0.80) > 1e-12:
        raise ValueError("three-seed experiment fixes the Setwise weight at 0.80")
    args.output_dir.mkdir(parents=True)
    frozen_path = args.output_dir / "frozen-config.json"
    report_path = args.output_dir / "evaluation-report.json"

    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    source_evaluation = _read_json(args.source_evaluation_report)
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
    train_feature_sha = train_report["artifacts"]["features"]["sha256"]
    validation_feature_sha = validation_report["artifacts"]["features"]["sha256"]
    _require_hash(train_path, train_feature_sha, "training features")
    _require_hash(validation_path, validation_feature_sha, "validation features")

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if feature_names != tuple(train_report["feature_names"]):
        raise ValueError("checkpoint and training cache feature schemas differ")
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("champion checkpoint has no Dataset2 LightGBM expert")
    feature_indices = tuple(int(index) for index in lgbm_result.feature_indices)
    if feature_indices != tuple(range(EXPECTED_TRAIN_SHAPE[-1])):
        raise ValueError("three-seed experiment requires all 63 features")

    source_frozen = source_evaluation["frozen_config"]
    _require_hash(
        args.checkpoint,
        source_frozen["checkpoint_sha256"],
        "champion checkpoint",
    )
    _require_hash(
        args.seed60_model,
        source_evaluation["setwise"]["model_sha256"],
        "seed-60 Setwise model",
    )

    setwise_config = FusionConfig(
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
        "seeds": list(SEEDS),
        "aggregation": (
            "uniform query-local rank-percentile average of each seed's "
            "0.80 Setwise + 0.20 champion LightGBM score"
        ),
        "weight_search": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "joint_cache_contract": joint_contract,
        "train_features": str(train_path.resolve()),
        "train_features_sha256": train_feature_sha,
        "validation_features": str(validation_path.resolve()),
        "validation_features_sha256": validation_feature_sha,
        "train_shape": list(train_features.shape),
        "validation_shape": list(validation_features.shape),
        "feature_names": list(feature_names),
        "setwise": {
            "epochs": setwise_config.epochs,
            "patience": setwise_config.early_stop_patience,
            "batch_size": setwise_config.batch_size,
            "hidden_dim": setwise_config.hidden_dim,
            "learning_rate": setwise_config.lr,
            "weight_decay": setwise_config.weight_decay,
            "objective": "negative_log_softmax_of_candidate_zero",
            "selection_metric": "full_candidate_mrr",
            "context_transform_version": 1,
            "blend_weight": args.setwise_weight,
        },
        "seed60": {
            "policy": "reuse contract-verified champion model",
            "model": str(args.seed60_model.resolve()),
            "model_sha256": _sha256(args.seed60_model),
        },
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "all_three_slices_non_decreasing": True,
            "package_only_after_pass": True,
        },
    }
    _write_json_atomic(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, sort_keys=True), flush=True)

    started = time.time()
    validation_view = SetwiseFeatureView(validation_features)
    champion_lgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, validation_features)
    )
    champion = {
        key: float(value)
        for key, value in source_evaluation["setwise"]["fixed_blend"].items()
    }

    seed_reports: dict[str, Any] = {}
    seed_blends: list[np.ndarray] = []
    for seed in SEEDS:
        seed_started = time.time()
        if seed == 60:
            model, payload = _load_setwise_model(
                args.seed60_model,
                expected_source_feature_count=EXPECTED_TRAIN_SHAPE[-1],
            )
            mean = payload["mean"]
            std = payload["std"]
            model_feature_indices = payload["feature_indices"]
            model_path = args.seed60_model
            history: tuple[dict[str, float | int], ...] = ()
            reused = True
        else:
            training_view = SetwiseFeatureView(train_features)
            model, result, history = fit_fusion_mlp_listwise_streaming(
                training_view,
                validation_view,
                setwise_config,
                np.random.default_rng(seed),
                verbose=True,
                feature_indices=tuple(range(training_view.shape[-1])),
                candidate_name=f"dataset2_setwise_seed{seed}",
            )
            del training_view
            model_path = args.output_dir / f"dataset2-setwise-seed{seed}.npz"
            _save_setwise_model(
                model_path,
                result=result,
                hidden_dim=args.setwise_hidden_dim,
                source_feature_count=EXPECTED_TRAIN_SHAPE[-1],
                seed=seed,
            )
            mean = np.asarray(result.mean, dtype=np.float32)
            std = np.asarray(result.std, dtype=np.float32)
            model_feature_indices = tuple(result.feature_indices)
            reused = False

        logits = _predict_streaming(
            model,
            validation_view,
            mean,
            std,
            feature_indices=model_feature_indices,
            batch_size=args.setwise_batch_size,
        )
        probabilities = _softmax(logits)
        blend = (
            args.setwise_weight * probabilities
            + (1.0 - args.setwise_weight) * champion_lgbm
        )
        expert_metrics = ranking_mrr_three_slices(probabilities)
        blend_metrics = ranking_mrr_three_slices(blend)
        if seed == 60:
            _require_metrics_close(
                expert_metrics,
                source_evaluation["setwise"]["expert"],
                "seed-60 Setwise expert",
            )
            _require_metrics_close(
                blend_metrics,
                champion,
                "seed-60 champion blend",
            )
        prediction_path = args.output_dir / f"validation-blend-seed{seed}.npy"
        np.save(prediction_path, np.asarray(blend, dtype=np.float32))
        seed_blends.append(blend)
        seed_reports[str(seed)] = {
            "reused": reused,
            "model": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "validation_prediction": str(prediction_path.resolve()),
            "validation_prediction_sha256": _sha256(prediction_path),
            "setwise_expert": expert_metrics,
            "fixed_blend": blend_metrics,
            "history": list(history),
            "elapsed_seconds": time.time() - seed_started,
        }
        print(
            f"[three-seed] seed={seed} blend_mrr={blend_metrics['full']:.8f}",
            flush=True,
        )
        del logits, probabilities, blend, model
        gc.collect()

    ensemble_scores = uniform_rank_average(tuple(seed_blends))
    ensemble_path = args.output_dir / "validation-uniform-rank-average.npy"
    np.save(ensemble_path, np.asarray(ensemble_scores, dtype=np.float32))
    ensemble = ranking_mrr_three_slices(ensemble_scores)
    slice_keys = ("slice_0", "slice_1", "slice_2")
    deltas = {
        key: ensemble[key] - champion[key]
        for key in ("full", *slice_keys)
    }
    passed = passes_full100_gate(
        baseline_full_mrr=champion["full"],
        candidate_full_mrr=ensemble["full"],
        baseline_slice_mrrs=tuple(champion[key] for key in slice_keys),
        candidate_slice_mrrs=tuple(ensemble[key] for key in slice_keys),
        min_full_delta=args.min_full_delta,
    )
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "package_authorized": passed,
        "package_generated": False,
        "frozen_config": frozen,
        "champion": champion,
        "seeds": seed_reports,
        "ensemble": ensemble,
        "delta_vs_champion": deltas,
        "ensemble_prediction": str(ensemble_path.resolve()),
        "ensemble_prediction_sha256": _sha256(ensemble_path),
        "gate": {
            "minimum_full_mrr_delta": args.min_full_delta,
            "full_delta_passed": bool(
                deltas["full"] + 1e-12 >= args.min_full_delta
            ),
            "all_three_slices_non_decreasing": bool(
                all(deltas[key] >= 0.0 for key in slice_keys)
            ),
        },
        "elapsed_seconds": time.time() - started,
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
    source_feature_count = int(payload["source_feature_count"][0])
    if source_feature_count != expected_source_feature_count:
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
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
