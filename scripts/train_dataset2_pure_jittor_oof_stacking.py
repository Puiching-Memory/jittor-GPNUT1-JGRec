from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.candidate_set_transformer import (
    CandidateSetFitResult,
    CandidateSetTrainingConfig,
    CandidateSetTransformerConfig,
    fit_candidate_set_transformer,
    fit_candidate_set_transformer_fixed,
    load_candidate_set_checkpoint,
    predict_candidate_set_logits,
    save_candidate_set_checkpoint,
)
from jgrec.rankers.hybrid.full100_training import (
    validate_joint_cache_reports,
)
from jgrec.rankers.hybrid.oof_models import (
    CandidateSetMLPConfig,
    CandidateSetMLPFitResult,
    CandidateSetMLPTrainingConfig,
    fit_candidate_set_mlp,
    load_candidate_set_mlp_checkpoint,
    predict_candidate_set_mlp_logits,
    save_candidate_set_mlp_checkpoint,
)
from jgrec.rankers.hybrid.oof_stacking import (
    OOFStackingFold,
    StableExpertLogitFeatureView,
    expanding_timestamp_oof_folds,
    oof_row_fold_assignments,
    stable_expert_logit_feature_names,
    stable_expert_logit_features,
    tie_neutral_mrr,
)

EXPERT_NAMES = ("cst_main", "cst_residual", "setwise_mlp")
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
            "Train rolling-origin pure-Jittor experts and an OOF stacking MLP."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("oof", "meta", "full", "all"),
        default="all",
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
    parser.add_argument("--warmup-rows", type=int, default=40_000)
    parser.add_argument("--fold-rows", type=int, default=40_000)
    parser.add_argument("--fold-count", type=int, default=4)
    parser.add_argument("--meta-train-fold-count", type=int, default=3)
    parser.add_argument(
        "--expert-validation-rows",
        type=int,
        default=8_000,
    )
    parser.add_argument("--expert-epochs", type=int, default=6)
    parser.add_argument("--expert-batch-size", type=int, default=256)
    parser.add_argument("--meta-epochs", type=int, default=12)
    parser.add_argument("--meta-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--minimum-full-delta", type=float, default=0.0002)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    args = parser.parse_args()

    started = time.time()
    _configure_device(args.device)
    context = _load_context(args)
    _freeze_protocol(args, context)
    phases = (
        ("oof", "meta", "full")
        if args.phase == "all"
        else (args.phase,)
    )
    for phase in phases:
        if phase == "oof":
            _run_oof(args, context)
        elif phase == "meta":
            _run_meta(args, context)
        else:
            _run_full(args, context)
    print(
        json.dumps(
            {
                "status": "complete",
                "phases": list(phases),
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    joint_contract = validate_joint_cache_reports(
        train_report,
        validation_report,
    )
    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    validation_path = Path(f"{args.validation_cache_prefix}.val.npy")
    time_path = Path(f"{args.train_cache_prefix}.train-time.npy")
    train_features = np.load(train_path, mmap_mode="r", allow_pickle=False)
    validation_features = np.load(
        validation_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    train_times = np.load(time_path, mmap_mode="r", allow_pickle=False)
    champion_scores = np.load(
        args.champion_validation_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        train_features.ndim != 3
        or validation_features.ndim != 3
        or train_features.shape[1] != 100
        or validation_features.shape[1] != 100
    ):
        raise ValueError("OOF stacking requires grouped full-100 caches")
    if train_features.shape[-1] != validation_features.shape[-1]:
        raise ValueError("OOF train/validation feature widths differ")
    if train_times.shape != (train_features.shape[0],):
        raise ValueError("OOF training timestamps do not align with rows")
    if champion_scores.shape != validation_features.shape[:2]:
        raise ValueError("champion validation scores do not align")
    feature_names = tuple(str(x) for x in train_report["feature_names"])
    if feature_names != tuple(validation_report["feature_names"]):
        raise ValueError("OOF train/validation feature schemas differ")
    if len(feature_names) != int(train_features.shape[-1]):
        raise ValueError("OOF feature report width differs")
    feature_provenance = tuple(
        _feature_provenance(name) for name in feature_names
    )
    folds = expanding_timestamp_oof_folds(
        train_times,
        warmup_rows=args.warmup_rows,
        fold_rows=args.fold_rows,
        fold_count=args.fold_count,
        meta_train_fold_count=args.meta_train_fold_count,
    )
    assignments = oof_row_fold_assignments(
        int(train_features.shape[0]),
        folds,
    )
    return {
        "train_report": train_report,
        "validation_report": validation_report,
        "joint_contract": joint_contract,
        "train_path": train_path,
        "validation_path": validation_path,
        "time_path": time_path,
        "train_features": train_features,
        "validation_features": validation_features,
        "train_times": train_times,
        "champion_scores": champion_scores,
        "feature_names": feature_names,
        "feature_provenance": feature_provenance,
        "folds": folds,
        "assignments": assignments,
    }


def _freeze_protocol(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "frozen-config.json"
    frozen = {
        "protocol": "dataset2_pure_jittor_oof_stacking_v1",
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "expert_names": list(EXPERT_NAMES),
        "train_cache": str(context["train_path"].resolve()),
        "validation_cache": str(context["validation_path"].resolve()),
        "train_time_cache": str(context["time_path"].resolve()),
        "train_cache_report_sha256": _sha256(args.train_cache_report),
        "validation_cache_report_sha256": _sha256(
            args.validation_cache_report
        ),
        "train_shape": list(context["train_features"].shape),
        "validation_shape": list(context["validation_features"].shape),
        "feature_names": list(context["feature_names"]),
        "feature_provenance": list(context["feature_provenance"]),
        "folds": [asdict(fold) for fold in context["folds"]],
        "warmup_rows": args.warmup_rows,
        "fold_rows": args.fold_rows,
        "fold_count": args.fold_count,
        "meta_train_fold_count": args.meta_train_fold_count,
        "expert_validation_rows": args.expert_validation_rows,
        "expert_epochs": args.expert_epochs,
        "expert_batch_size": args.expert_batch_size,
        "meta_epochs": args.meta_epochs,
        "meta_batch_size": args.meta_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "minimum_full_delta": args.minimum_full_delta,
        "device": args.device,
        "champion_role": "external_validation_comparison_only",
        "joint_cache_contract": context["joint_contract"],
    }
    frozen = json.loads(json.dumps(frozen, sort_keys=True))
    if path.exists():
        existing = _read_json(path)
        if existing != frozen:
            raise ValueError(
                "existing OOF stacking experiment has a different protocol"
            )
        return
    _write_json_atomic(path, frozen)
    _save_array_atomic(
        args.output_dir / "oof-row-fold-assignments.npy",
        context["assignments"],
    )


def _run_oof(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    features = context["train_features"]
    times = context["train_times"]
    folds: tuple[OOFStackingFold, ...] = context["folds"]
    oof_path = args.output_dir / "oof-expert-logits.npy"
    if oof_path.exists():
        oof_logits = np.load(oof_path, mmap_mode="r+", allow_pickle=False)
        expected = (len(EXPERT_NAMES), *features.shape[:2])
        if oof_logits.shape != expected:
            raise ValueError("existing OOF logits shape differs")
    else:
        oof_logits = np.lib.format.open_memmap(
            oof_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(EXPERT_NAMES), *features.shape[:2]),
        )
        oof_logits[:] = np.nan
        oof_logits.flush()

    fold_reports: list[dict[str, Any]] = []
    for expert_index, expert_name in enumerate(EXPERT_NAMES):
        for fold in folds:
            artifact_dir = (
                args.output_dir
                / "fold-experts"
                / expert_name
                / f"fold-{fold.index}"
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            logits_path = artifact_dir / "score-logits.npy"
            report_path = artifact_dir / "report.json"
            checkpoint_path = artifact_dir / "model.npz"
            if logits_path.exists() and report_path.exists():
                fold_logits = np.load(logits_path, allow_pickle=False)
                expected_shape = (
                    fold.score_rows[1] - fold.score_rows[0],
                    int(features.shape[1]),
                )
                if fold_logits.shape != expected_shape:
                    raise ValueError(
                        f"cached fold logits differ: {expert_name} "
                        f"fold={fold.index}"
                    )
                report = _read_json(report_path)
                if report["score_logits_sha256"] != _sha256(logits_path):
                    raise ValueError("cached OOF fold logits hash differs")
                oof_logits[
                    expert_index,
                    fold.score_rows[0] : fold.score_rows[1],
                ] = fold_logits
                oof_logits.flush()
                fold_reports.append(report)
                print(
                    f"[oof] reuse expert={expert_name} fold={fold.index}",
                    flush=True,
                )
                continue

            train_stop = fold.train_rows[1]
            split_target = max(
                1,
                train_stop - min(
                    args.expert_validation_rows,
                    max(1_000, train_stop // 5),
                ),
            )
            split = int(
                np.searchsorted(
                    times,
                    times[split_target],
                    side="left",
                )
            )
            if split <= 0 or split >= train_stop:
                raise ValueError("OOF internal expert split is invalid")
            if int(times[split - 1]) >= int(times[split]):
                raise RuntimeError("OOF internal expert split leaks time")
            train_positives = np.zeros(split, dtype=np.int32)
            validation_positives = np.zeros(
                train_stop - split,
                dtype=np.int32,
            )
            print(
                f"[oof] train expert={expert_name} fold={fold.index} "
                f"train=[0,{split}) validation=[{split},{train_stop}) "
                f"score={fold.score_rows}",
                flush=True,
            )
            model, result = _fit_fold_expert(
                expert_name,
                features[:split],
                train_positives,
                features[split:train_stop],
                validation_positives,
                args=args,
                context=context,
            )
            _save_expert_checkpoint(
                expert_name,
                checkpoint_path,
                model,
                result,
            )
            fold_logits = _predict_expert(
                expert_name,
                model,
                result,
                features[fold.score_rows[0] : fold.score_rows[1]],
                args.expert_batch_size,
            )
            _save_array_atomic(logits_path, fold_logits)
            oof_logits[
                expert_index,
                fold.score_rows[0] : fold.score_rows[1],
            ] = fold_logits
            oof_logits.flush()
            report = {
                "expert": expert_name,
                "fold": asdict(fold),
                "actual_train_rows": [0, split],
                "internal_validation_rows": [split, train_stop],
                "internal_train_time_max": int(times[split - 1]),
                "internal_validation_time_min": int(times[split]),
                "best_val_mrr": result.best_val_mrr,
                "best_epoch": _best_epoch(result.history),
                "history": list(result.history),
                "score_mrr": _mrr(fold_logits),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "score_logits": str(logits_path.resolve()),
                "score_logits_sha256": _sha256(logits_path),
                "trainable_frameworks": ["jittor"],
                "non_jittor_trainable_models": [],
            }
            _write_json_atomic(report_path, report)
            fold_reports.append(report)
            del model, result, fold_logits
            release_memory()

    score_start = folds[0].score_rows[0]
    if not np.all(np.isfinite(oof_logits[:, score_start:, :])):
        raise RuntimeError("OOF logits contain missing or non-finite rows")
    if np.any(np.isfinite(oof_logits[:, :score_start, :])):
        raise RuntimeError("OOF warmup rows were written unexpectedly")
    report = {
        "status": "complete",
        "protocol": "strict_expanding_timestamp_origin",
        "expert_names": list(EXPERT_NAMES),
        "fold_count": len(folds),
        "oof_rows": int(features.shape[0] - score_start),
        "warmup_rows": score_start,
        "fold_reports": fold_reports,
        "oof_logits": str(oof_path.resolve()),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(args.output_dir / "oof-report.json", report)


def _fit_fold_expert(
    expert_name: str,
    train_features: Any,
    train_positives: np.ndarray,
    validation_features: Any,
    validation_positives: np.ndarray,
    *,
    args: argparse.Namespace,
    context: dict[str, Any],
) -> tuple[Any, CandidateSetFitResult | CandidateSetMLPFitResult]:
    if expert_name == "setwise_mlp":
        return fit_candidate_set_mlp(
            train_features,
            train_positives,
            validation_features=validation_features,
            validation_positive_indices=validation_positives,
            model_config=CandidateSetMLPConfig(
                input_dim=int(train_features.shape[-1]),
                hidden_dim=128,
                dropout=0.05,
                relative_context="mean_max",
            ),
            training_config=CandidateSetMLPTrainingConfig(
                epochs=args.expert_epochs,
                batch_size=args.expert_batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                early_stop_patience=2,
            ),
            feature_names=context["feature_names"],
            feature_provenance=context["feature_provenance"],
            verbose=True,
        )
    return fit_candidate_set_transformer(
        train_features,
        train_positives,
        validation_features,
        validation_positives,
        model_config=_cst_config(
            expert_name,
            int(train_features.shape[-1]),
        ),
        training_config=CandidateSetTrainingConfig(
            epochs=args.expert_epochs,
            batch_size=args.expert_batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            early_stop_patience=2,
        ),
        feature_names=context["feature_names"],
        feature_provenance=context["feature_provenance"],
        verbose=True,
    )


def _run_meta(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    oof_report_path = args.output_dir / "oof-report.json"
    if not oof_report_path.exists():
        raise FileNotFoundError("OOF phase must complete before meta phase")
    folds: tuple[OOFStackingFold, ...] = context["folds"]
    oof_logits = np.load(
        args.output_dir / "oof-expert-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    score_start = folds[0].score_rows[0]
    meta_stop = folds[args.meta_train_fold_count - 1].score_rows[1]
    validation_start = folds[args.meta_train_fold_count].score_rows[0]
    if meta_stop != validation_start:
        raise RuntimeError("meta train/validation OOF rows are not contiguous")
    train_view = StableExpertLogitFeatureView(
        oof_logits,
        row_start=score_start,
        row_stop=meta_stop,
    )
    validation_view = StableExpertLogitFeatureView(
        oof_logits,
        row_start=validation_start,
        row_stop=int(oof_logits.shape[1]),
    )
    feature_names = stable_expert_logit_feature_names(EXPERT_NAMES)
    checkpoint_path = args.output_dir / "meta-stacking-mlp.npz"
    if checkpoint_path.exists():
        model, result = load_candidate_set_mlp_checkpoint(checkpoint_path)
    else:
        model, result = fit_candidate_set_mlp(
            train_view,
            np.zeros(train_view.shape[0], dtype=np.int32),
            validation_features=validation_view,
            validation_positive_indices=np.zeros(
                validation_view.shape[0],
                dtype=np.int32,
            ),
            model_config=CandidateSetMLPConfig(
                input_dim=train_view.shape[-1],
                hidden_dim=64,
                dropout=0.05,
                relative_context="none",
            ),
            training_config=CandidateSetMLPTrainingConfig(
                epochs=args.meta_epochs,
                batch_size=args.meta_batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                early_stop_patience=3,
            ),
            feature_names=feature_names,
            feature_provenance=tuple(
                "numpy_deterministic" for _ in feature_names
            ),
            verbose=True,
        )
        save_candidate_set_mlp_checkpoint(
            checkpoint_path,
            model,
            result,
        )
    meta_logits = predict_candidate_set_mlp_logits(
        model,
        validation_view,
        mean=result.mean,
        std=result.std,
        batch_size=args.meta_batch_size,
    )
    validation_expert_logits = np.asarray(
        oof_logits[:, validation_start:, :],
        dtype=np.float32,
    )
    consensus = _expert_consensus_percentile(
        validation_expert_logits,
        batch_size=args.meta_batch_size,
    )
    meta_percentile = _single_score_percentile(
        meta_logits,
        batch_size=args.meta_batch_size,
    )
    scan = []
    for meta_weight in np.linspace(0.0, 1.0, 21):
        scores = (
            float(meta_weight) * meta_percentile
            + (1.0 - float(meta_weight)) * consensus
        )
        scan.append(
            {
                "meta_weight": float(meta_weight),
                "metrics": _mrr_three_slices(scores),
            }
        )
    selected = max(
        scan,
        key=lambda row: (
            row["metrics"]["full"],
            min(
                row["metrics"][f"slice_{index}"]
                for index in range(3)
            ),
            row["meta_weight"],
        ),
    )
    selected_scores = (
        selected["meta_weight"] * meta_percentile
        + (1.0 - selected["meta_weight"]) * consensus
    ).astype(np.float32)
    _save_array_atomic(
        args.output_dir / "meta-validation-logits.npy",
        meta_logits,
    )
    _save_array_atomic(
        args.output_dir / "meta-validation-selected-scores.npy",
        selected_scores,
    )
    report = {
        "status": "complete",
        "meta_train_rows": [score_start, meta_stop],
        "meta_validation_rows": [
            validation_start,
            int(oof_logits.shape[1]),
        ],
        "feature_names": list(feature_names),
        "best_val_mrr": result.best_val_mrr,
        "history": list(result.history),
        "expert_metrics": {
            name: _mrr_three_slices(validation_expert_logits[index])
            for index, name in enumerate(EXPERT_NAMES)
        },
        "consensus_metrics": _mrr_three_slices(consensus),
        "raw_meta_metrics": _mrr_three_slices(meta_logits),
        "blend_scan": scan,
        "selected": selected,
        "selected_metrics": _mrr_three_slices(selected_scores),
        "meta_checkpoint": str(checkpoint_path.resolve()),
        "meta_checkpoint_sha256": _sha256(checkpoint_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(args.output_dir / "meta-report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def _run_full(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    meta_report_path = args.output_dir / "meta-report.json"
    if not meta_report_path.exists():
        raise FileNotFoundError("meta phase must complete before full phase")
    meta_report = _read_json(meta_report_path)
    oof_report = _read_json(args.output_dir / "oof-report.json")
    features = context["train_features"]
    validation_features = context["validation_features"]
    full_dir = args.output_dir / "full-experts"
    full_dir.mkdir(parents=True, exist_ok=True)
    validation_logits_path = (
        args.output_dir / "full-validation-expert-logits.npy"
    )
    if validation_logits_path.exists():
        validation_logits = np.load(
            validation_logits_path,
            mmap_mode="r+",
            allow_pickle=False,
        )
    else:
        validation_logits = np.lib.format.open_memmap(
            validation_logits_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                len(EXPERT_NAMES),
                *validation_features.shape[:2],
            ),
        )
        validation_logits[:] = np.nan
        validation_logits.flush()
    full_reports = []
    for expert_index, expert_name in enumerate(EXPERT_NAMES):
        checkpoint_path = full_dir / f"{expert_name}.npz"
        epochs = _selected_full_epochs(
            oof_report,
            expert_name,
            args.expert_epochs,
        )
        if checkpoint_path.exists():
            model, result = _load_expert_checkpoint(
                expert_name,
                checkpoint_path,
            )
        else:
            positives = np.zeros(features.shape[0], dtype=np.int32)
            if expert_name == "setwise_mlp":
                model, result = fit_candidate_set_mlp(
                    features,
                    positives,
                    model_config=CandidateSetMLPConfig(
                        input_dim=int(features.shape[-1]),
                        hidden_dim=128,
                        dropout=0.05,
                        relative_context="mean_max",
                    ),
                    training_config=CandidateSetMLPTrainingConfig(
                        epochs=epochs,
                        batch_size=args.expert_batch_size,
                        learning_rate=args.learning_rate,
                        seed=args.seed,
                    ),
                    feature_names=context["feature_names"],
                    feature_provenance=context["feature_provenance"],
                    verbose=True,
                )
            else:
                model, result = fit_candidate_set_transformer_fixed(
                    features,
                    positives,
                    model_config=_cst_config(
                        expert_name,
                        int(features.shape[-1]),
                    ),
                    training_config=CandidateSetTrainingConfig(
                        epochs=epochs,
                        batch_size=args.expert_batch_size,
                        learning_rate=args.learning_rate,
                        seed=args.seed,
                    ),
                    feature_names=context["feature_names"],
                    feature_provenance=context["feature_provenance"],
                    verbose=True,
                )
            _save_expert_checkpoint(
                expert_name,
                checkpoint_path,
                model,
                result,
            )
        expert_logits = _predict_expert(
            expert_name,
            model,
            result,
            validation_features,
            args.expert_batch_size,
        )
        validation_logits[expert_index] = expert_logits
        validation_logits.flush()
        full_reports.append(
            {
                "expert": expert_name,
                "epochs": epochs,
                "training_rows": result.training_rows,
                "selection_mode": result.selection_mode,
                "validation_metrics": _mrr_three_slices(expert_logits),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": _sha256(checkpoint_path),
            }
        )
        del model, result, expert_logits
        release_memory()
    if not np.all(np.isfinite(validation_logits)):
        raise RuntimeError("full expert validation logits are incomplete")

    meta_model, meta_result = load_candidate_set_mlp_checkpoint(
        args.output_dir / "meta-stacking-mlp.npz"
    )
    stable_view = StableExpertLogitFeatureView(validation_logits)
    raw_meta = predict_candidate_set_mlp_logits(
        meta_model,
        stable_view,
        mean=meta_result.mean,
        std=meta_result.std,
        batch_size=args.meta_batch_size,
    )
    consensus = _expert_consensus_percentile(
        validation_logits,
        batch_size=args.meta_batch_size,
    )
    meta_percentile = _single_score_percentile(
        raw_meta,
        batch_size=args.meta_batch_size,
    )
    meta_weight = float(meta_report["selected"]["meta_weight"])
    selected_scores = (
        meta_weight * meta_percentile
        + (1.0 - meta_weight) * consensus
    ).astype(np.float32)
    _save_array_atomic(
        args.output_dir / "full-validation-meta-logits.npy",
        raw_meta,
    )
    selected_path = args.output_dir / "full-validation-selected-scores.npy"
    _save_array_atomic(selected_path, selected_scores)
    comparison = _compare_tie_neutral(
        selected_scores,
        context["champion_scores"],
        positives=np.zeros(
            selected_scores.shape[0],
            dtype=np.int32,
        ),
    )
    deltas = comparison["delta_vs_baseline"]
    gate_passed = bool(
        deltas["full"] + 1e-12 >= args.minimum_full_delta
        and all(deltas[f"slice_{index}"] >= 0.0 for index in range(3))
    )
    report = {
        "status": "passed" if gate_passed else "rejected",
        "full_experts": full_reports,
        "meta_weight": meta_weight,
        "comparison": comparison,
        "gate": {
            "passed": gate_passed,
            "minimum_full_delta": args.minimum_full_delta,
            "all_three_slices_non_decreasing": all(
                deltas[f"slice_{index}"] >= 0.0 for index in range(3)
            ),
        },
        "selected_scores": str(selected_path.resolve()),
        "selected_scores_sha256": _sha256(selected_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(args.output_dir / "evaluation-report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def _cst_config(
    expert_name: str,
    input_dim: int,
) -> CandidateSetTransformerConfig:
    if expert_name not in {"cst_main", "cst_residual"}:
        raise ValueError(f"unsupported CST expert: {expert_name}")
    return CandidateSetTransformerConfig(
        input_dim=input_dim,
        model_dim=64,
        heads=4,
        layers=2,
        dropout=0.05,
        feedforward_multiplier=2,
        relative_context="mean_max",
        pointwise_residual_dim=(32 if expert_name == "cst_residual" else 0),
    )


def _save_expert_checkpoint(
    expert_name: str,
    path: Path,
    model: Any,
    result: Any,
) -> None:
    if expert_name == "setwise_mlp":
        save_candidate_set_mlp_checkpoint(path, model, result)
    else:
        save_candidate_set_checkpoint(path, model, result)


def _load_expert_checkpoint(
    expert_name: str,
    path: Path,
) -> tuple[Any, Any]:
    if expert_name == "setwise_mlp":
        return load_candidate_set_mlp_checkpoint(path)
    return load_candidate_set_checkpoint(path)


def _predict_expert(
    expert_name: str,
    model: Any,
    result: Any,
    features: Any,
    batch_size: int,
) -> np.ndarray:
    if expert_name == "setwise_mlp":
        return predict_candidate_set_mlp_logits(
            model,
            features,
            mean=result.mean,
            std=result.std,
            batch_size=batch_size,
        )
    return predict_candidate_set_logits(
        model,
        features,
        mean=result.mean,
        std=result.std,
        batch_size=batch_size,
    )


def _best_epoch(history: Any) -> int:
    rows = tuple(history)
    if not rows:
        raise ValueError("expert history is empty")
    return int(
        max(
            rows,
            key=lambda row: (
                float(row.get("val_mrr", -math.inf)),
                int(row["epoch"]),
            ),
        )["epoch"]
    )


def _selected_full_epochs(
    oof_report: dict[str, Any],
    expert_name: str,
    fallback: int,
) -> int:
    epochs = sorted(
        int(row["best_epoch"])
        for row in oof_report["fold_reports"]
        if row["expert"] == expert_name
    )
    if not epochs:
        return int(fallback)
    return max(1, int(np.median(np.asarray(epochs))))


def _expert_consensus_percentile(
    expert_logits: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    result = np.empty(expert_logits.shape[1:], dtype=np.float32)
    for start in range(0, int(expert_logits.shape[1]), batch_size):
        stop = min(start + batch_size, int(expert_logits.shape[1]))
        stable = stable_expert_logit_features(
            np.asarray(expert_logits[:, start:stop, :])
        )
        result[start:stop] = stable[..., -5]
    return result


def _single_score_percentile(
    scores: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    result = np.empty_like(scores, dtype=np.float32)
    for start in range(0, scores.shape[0], batch_size):
        stop = min(start + batch_size, scores.shape[0])
        result[start:stop] = stable_expert_logit_features(
            scores[None, start:stop, :]
        )[..., 0]
    return result


def _mrr(scores: np.ndarray) -> float:
    return tie_neutral_mrr(
        scores,
        np.zeros(scores.shape[0], dtype=np.int32),
    )


def _mrr_three_slices(scores: np.ndarray) -> dict[str, float]:
    boundaries = np.linspace(0, scores.shape[0], 4, dtype=np.int64)
    return {
        "full": _mrr(scores),
        **{
            f"slice_{index}": _mrr(
                scores[boundaries[index] : boundaries[index + 1]]
            )
            for index in range(3)
        },
    }


def _compare_tie_neutral(
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    *,
    positives: np.ndarray,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    positive_indices = np.asarray(positives, dtype=np.int32)
    if candidate.shape != baseline.shape:
        raise ValueError("OOF candidate and baseline score shapes differ")
    if positive_indices.shape != (candidate.shape[0],):
        raise ValueError("OOF comparison positive indices do not align")
    candidate_metrics = _mrr_three_slices(candidate)
    baseline_metrics = _mrr_three_slices(baseline)
    return {
        "protocol": "comparison_only_no_blend_tie_neutral",
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "delta_vs_baseline": {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in candidate_metrics
        },
    }


def _feature_provenance(name: str) -> str:
    if name.startswith(JITTOR_FEATURE_PREFIXES):
        return "jittor"
    return "numpy_deterministic"


def _configure_device(device: str) -> None:
    if device == "cuda":
        if not jt.has_cuda:
            raise RuntimeError("CUDA was requested but Jittor has no CUDA")
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
