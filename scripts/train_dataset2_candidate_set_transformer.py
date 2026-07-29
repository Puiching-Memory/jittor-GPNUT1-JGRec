from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.rankers.hybrid.candidate_set_transformer import (
    CandidateSetTrainingConfig,
    CandidateSetTransformerConfig,
    compare_candidate_set_to_baseline,
    fit_candidate_set_transformer,
    predict_candidate_set_logits,
    save_candidate_set_checkpoint,
)
from jgrec.rankers.hybrid.full100_training import (
    validate_joint_cache_reports,
)

JITTOR_FEATURE_PREFIXES = (
    "gnn_",
    "gru_",
    "two_tower_",
    "source_profile_item2vec",
    "source_profile_recent_item2vec",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a pure-Jittor Dataset2 Candidate-Set "
            "Transformer. Champion scores are comparison-only."
        )
    )
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument(
        "--validation-cache-prefix",
        required=True,
        type=Path,
    )
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument(
        "--validation-cache-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--champion-validation-scores",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--feedforward-multiplier",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--relative-context",
        choices=("none", "mean_max"),
        default="mean_max",
    )
    parser.add_argument(
        "--pointwise-residual-dim",
        type=int,
        default=0,
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--min-full-delta", type=float, default=0.0002)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    args = parser.parse_args()

    if args.train_limit < 0 or args.validation_limit < 0:
        raise ValueError("cache row limits must be non-negative")
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite experiment: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    started = time.time()
    _configure_device(args.device)

    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    joint_contract = validate_joint_cache_reports(
        train_report,
        validation_report,
    )
    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    validation_path = Path(
        f"{args.validation_cache_prefix}.val.npy"
    )
    train_sha256 = _require_hash(
        train_path,
        train_report["artifacts"]["features"]["sha256"],
        "training feature cache",
    )
    validation_sha256 = _require_hash(
        validation_path,
        validation_report["artifacts"]["features"]["sha256"],
        "validation feature cache",
    )
    champion_sha256 = _require_hash(
        args.champion_validation_scores,
        _champion_hash_from_nearby_report(
            args.champion_validation_scores
        ),
        "champion validation scores",
    )

    train_features = np.load(
        train_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_features = np.load(
        validation_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    champion_scores = np.load(
        args.champion_validation_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    if train_features.ndim != 3 or validation_features.ndim != 3:
        raise ValueError("full-100 feature caches must be three-dimensional")
    if train_features.shape[1] != 100 or validation_features.shape[1] != 100:
        raise ValueError("Candidate-Set Transformer requires 100 candidates")
    if train_features.shape[-1] != validation_features.shape[-1]:
        raise ValueError("train and validation feature widths differ")
    if champion_scores.shape != validation_features.shape[:2]:
        raise ValueError(
            "champion validation scores do not align with validation cache"
        )

    feature_names = tuple(
        str(name) for name in train_report["feature_names"]
    )
    if feature_names != tuple(validation_report["feature_names"]):
        raise ValueError("train and validation feature schemas differ")
    if len(feature_names) != train_features.shape[-1]:
        raise ValueError("cache report does not describe every feature")
    feature_provenance = tuple(
        _feature_provenance(name) for name in feature_names
    )
    train_view = _tail_view(train_features, args.train_limit)
    validation_view = _tail_view(
        validation_features,
        args.validation_limit,
    )
    champion_view = _tail_view(
        champion_scores,
        args.validation_limit,
    )
    train_positives = np.zeros(train_view.shape[0], dtype=np.int32)
    validation_positives = np.zeros(
        validation_view.shape[0],
        dtype=np.int32,
    )
    model_config = CandidateSetTransformerConfig(
        input_dim=train_view.shape[-1],
        model_dim=args.model_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        feedforward_multiplier=args.feedforward_multiplier,
        relative_context=args.relative_context,
        pointwise_residual_dim=args.pointwise_residual_dim,
    )
    training_config = CandidateSetTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
    )
    frozen = {
        "status": "frozen_before_training",
        "protocol": "pure_jittor_candidate_set_transformer_v1",
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "champion_role": "comparison_only_no_blend",
        "joint_cache_contract": joint_contract,
        "train_cache": str(train_path.resolve()),
        "train_cache_sha256": train_sha256,
        "validation_cache": str(validation_path.resolve()),
        "validation_cache_sha256": validation_sha256,
        "champion_validation_scores": str(
            args.champion_validation_scores.resolve()
        ),
        "champion_validation_scores_sha256": champion_sha256,
        "train_shape": list(train_view.shape),
        "validation_shape": list(validation_view.shape),
        "feature_names": list(feature_names),
        "feature_provenance": list(feature_provenance),
        "model_config": {
            "input_dim": model_config.input_dim,
            "model_dim": model_config.model_dim,
            "heads": model_config.heads,
            "layers": model_config.layers,
            "dropout": model_config.dropout,
            "feedforward_multiplier": (
                model_config.feedforward_multiplier
            ),
            "relative_context": model_config.relative_context,
            "pointwise_residual_dim": (
                model_config.pointwise_residual_dim
            ),
        },
        "training_config": {
            "epochs": training_config.epochs,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "seed": training_config.seed,
            "early_stop_patience": (
                training_config.early_stop_patience
            ),
        },
        "device": args.device,
        "minimum_full_delta": args.min_full_delta,
    }
    frozen_path = args.output_dir / "frozen-config.json"
    _write_json_atomic(frozen_path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2), flush=True)

    model, result = fit_candidate_set_transformer(
        train_view,
        train_positives,
        validation_view,
        validation_positives,
        model_config=model_config,
        training_config=training_config,
        feature_names=feature_names,
        feature_provenance=feature_provenance,
        verbose=True,
    )
    model_path = args.output_dir / "candidate-set-transformer.npz"
    save_candidate_set_checkpoint(model_path, model, result)
    validation_logits = predict_candidate_set_logits(
        model,
        validation_view,
        mean=result.mean,
        std=result.std,
        batch_size=args.batch_size,
    )
    logits_path = args.output_dir / "validation-logits.npy"
    _save_array_atomic(logits_path, validation_logits)
    comparison = compare_candidate_set_to_baseline(
        validation_logits,
        champion_view,
        positive_indices=validation_positives,
    )
    deltas = comparison["delta_vs_baseline"]
    gate_passed = bool(
        deltas["full"] + 1e-12 >= args.min_full_delta
        and all(
            deltas[f"slice_{index}"] >= 0.0
            for index in range(3)
        )
    )
    report = {
        "status": "passed" if gate_passed else "rejected",
        "frozen_config": str(frozen_path.resolve()),
        "frozen_config_sha256": _sha256(frozen_path),
        "comparison": comparison,
        "best_val_mrr": result.best_val_mrr,
        "history": list(result.history),
        "model": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "validation_logits": str(logits_path.resolve()),
        "validation_logits_sha256": _sha256(logits_path),
        "checkpoint_provenance": {
            "trainable_frameworks": list(
                result.trainable_frameworks
            ),
            "non_jittor_trainable_models": list(
                result.non_jittor_trainable_models
            ),
        },
        "gate": {
            "passed": gate_passed,
            "minimum_full_delta": args.min_full_delta,
            "full_delta_passed": (
                deltas["full"] + 1e-12 >= args.min_full_delta
            ),
            "all_three_slices_non_decreasing": all(
                deltas[f"slice_{index}"] >= 0.0
                for index in range(3)
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.output_dir / "evaluation-report.json"
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if gate_passed else 2


def _feature_provenance(name: str) -> str:
    if name.startswith(JITTOR_FEATURE_PREFIXES):
        return "jittor"
    return "numpy_deterministic"


def _tail_view(values: Any, row_limit: int) -> Any:
    if row_limit == 0 or row_limit >= values.shape[0]:
        return values
    return values[-row_limit:]


def _configure_device(device: str) -> None:
    if device == "cuda":
        if not jt.has_cuda:
            raise RuntimeError("CUDA was requested but Jittor has no CUDA")
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0


def _champion_hash_from_nearby_report(path: Path) -> str:
    candidates = (
        path.parent.parent / "evaluation-report.json",
        path.parent / "evaluation-report.json",
    )
    resolved = path.resolve()
    for report_path in candidates:
        if not report_path.exists():
            continue
        report = _read_json(report_path)
        selected = report.get("selected_prediction")
        expected_hash = report.get("selected_prediction_sha256")
        if (
            selected is not None
            and expected_hash is not None
            and Path(selected).resolve() == resolved
        ):
            return str(expected_hash)
    raise ValueError(
        "champion scores lack a nearby evaluation report with a bound hash"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )
    return actual


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
