from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.bounded_id_residual import (
    BoundedIDResidualConfig,
    BoundedIDResidualTrainingConfig,
    bounded_id_residual_audit,
    fit_bounded_id_residual_fixed,
    load_bounded_id_residual_checkpoint,
    predict_bounded_id_residual_logits,
    save_bounded_id_residual_checkpoint,
)
from jgrec.rankers.hybrid.candidate_set_transformer import (
    load_candidate_set_checkpoint,
    predict_candidate_set_logits,
)
from jgrec.rankers.hybrid.oof_stacking import tie_neutral_mrr
from jgrec.rankers.hybrid.source_conditioned_training import (
    load_source_conditioned_checkpoint,
    predict_source_conditioned_logits,
)
from jgrec.rankers.hybrid.source_sequence_cache import (
    SourceConditionedFold,
    SourceSequenceRows,
)

CAPS = (0.02, 0.05, 0.10)
SELECTION_MEAN_DELTA_MIN = 0.0002
GATE_MEAN_DELTA_MIN = 0.0002
ACTIVITY_DELTA_MIN = -0.001
EXTERNAL_DELTA_MIN = 0.0002


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train frozen-CST bounded candidate-ID residuals.",
    )
    parser.add_argument(
        "--phase",
        choices=("smoke", "selection", "gate", "external", "all"),
        default="all",
    )
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--sequence-cache-dir", required=True, type=Path)
    parser.add_argument("--base-result-dir", required=True, type=Path)
    parser.add_argument("--base-cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--validation-cache-prefix",
        type=Path,
        default=Path(
            "cache/supervised_features/"
            "dataset2_joint_recent200k_full100_val_seed60_20260725"
        ),
    )
    parser.add_argument("--champion-validation-scores", type=Path)
    parser.add_argument(
        "--full-base-checkpoint",
        type=Path,
        default=Path(
            "result/dataset2_pure_jittor_oof_stacking_20260726/"
            "full-experts/cst_main.npz"
        ),
    )
    parser.add_argument(
        "--full-base-validation-expert-logits",
        type=Path,
        default=Path(
            "result/dataset2_pure_jittor_oof_stacking_20260726/"
            "full-validation-expert-logits.npy"
        ),
    )
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    started = time.time()
    _configure_device(args.device, args.seed)
    context = _load_context(args)
    _freeze_protocol(args, context)
    if args.phase == "smoke":
        _run_smoke(args, context)
    elif args.phase == "selection":
        _run_selection(args, context)
    elif args.phase == "gate":
        _run_gate(args, context)
    elif args.phase == "external":
        _run_external(args, context)
    else:
        _run_selection(args, context)
        _run_gate(args, context)
        gate_path = args.output_dir / "gate-report.json"
        if gate_path.exists() and _read_json(gate_path)["passed"]:
            _run_external(args, context)
    print(
        json.dumps(
            {
                "status": "complete",
                "phase": args.phase,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.sequence_cache_dir / "fold-manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("trainable_frameworks") != ["jittor"]
        or manifest.get("non_jittor_trainable_models") != []
    ):
        raise ValueError("source sequence fold manifest is invalid")
    prefix = str(args.train_cache_prefix)
    features = np.load(
        f"{prefix}.train.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    candidates = np.load(
        f"{prefix}.train-candidates.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    times = np.load(
        f"{prefix}.train-time.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    dst = np.load(
        f"{prefix}.train-dst.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        features.shape
        != (
            int(manifest["train_rows"]),
            int(manifest["candidate_count"]),
            int(manifest["feature_count"]),
        )
        or candidates.shape != features.shape[:2]
        or times.shape != features.shape[:1]
        or dst.shape != features.shape[:1]
        or not np.array_equal(
            np.asarray(candidates[:, 0]),
            np.asarray(dst),
        )
    ):
        raise ValueError("bounded residual train cache contract differs")
    folds = tuple(
        SourceConditionedFold(
            index=int(row["index"]),
            train_rows=tuple(int(value) for value in row["train_rows"]),
            score_rows=tuple(int(value) for value in row["score_rows"]),
            role=str(row["role"]),
            train_time_max=int(row["train_time_max"]),
            score_time_min=int(row["score_time_min"]),
            score_time_max=int(row["score_time_max"]),
        )
        for row in manifest["folds"]
    )
    if len(folds) != 3:
        raise ValueError("bounded residual requires exactly three folds")
    num_items = int(
        max(
            int(manifest["num_items"]),
            int(np.max(candidates)),
        )
    )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "features": features,
        "candidates": candidates,
        "times": times,
        "folds": folds,
        "num_items": num_items,
    }


def _freeze_protocol(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.base_cache_dir.mkdir(parents=True, exist_ok=True)
    base_folds = []
    for fold in context["folds"]:
        directory = _base_fold_dir(args, fold.index)
        report_path = directory / "report.json"
        model_path = directory / "model.npz"
        score_path = directory / "score-logits.npy"
        base_folds.append(
            {
                "fold": asdict(fold),
                "report": str(report_path.resolve()),
                "report_sha256": _sha256(report_path),
                "model": str(model_path.resolve()),
                "model_sha256": _sha256(model_path),
                "score_logits": str(score_path.resolve()),
                "score_logits_sha256": _sha256(score_path),
            }
        )
    frozen = {
        "status": "frozen_before_training",
        "protocol": "dataset2_frozen_cst_bounded_id_residual_v2_absolute",
        "formula": (
            "base + cap * tanh("
            "raw_id_logits - row_mean(raw_id_logits))"
        ),
        "caps": list(CAPS),
        "model": {
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout,
            "num_items": context["num_items"],
            "trainable_branch": "candidate_id_embedding_and_linear_head",
            "base_gradients": False,
        },
        "training": asdict(_training_config(args)),
        "selection_folds": [0, 1],
        "gate_fold": 2,
        "selection_rule": {
            "each_fold_delta_min": 0.0,
            "two_fold_mean_delta_min": SELECTION_MEAN_DELTA_MIN,
            "tie_break": "higher mean delta then lower cap",
        },
        "gate_rule": {
            "fold2_delta_min": 0.0,
            "three_fold_mean_delta_min": GATE_MEAN_DELTA_MIN,
            "worst_activity_delta_min": ACTIVITY_DELTA_MIN,
        },
        "external_rule": {
            "read_only_after_gate": True,
            "full_delta_min": EXTERNAL_DELTA_MIN,
            "all_time_slice_deltas_non_negative": True,
        },
        "base_folds": base_folds,
        "fold_manifest": str(context["manifest_path"].resolve()),
        "fold_manifest_sha256": _sha256(context["manifest_path"]),
        "train_cache": str(args.train_cache_prefix.resolve()),
        "device": args.device,
        "positive_index": 0,
        "metric": "tie_neutral_mrr",
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    frozen = json.loads(json.dumps(frozen, sort_keys=True))
    path = args.output_dir / "frozen-config.json"
    if path.exists():
        if _read_json(path) != frozen:
            raise ValueError("existing bounded residual protocol differs")
        return
    _write_json_atomic(path, frozen)


def _run_smoke(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    report_path = args.output_dir / "smoke-report.json"
    if report_path.exists():
        print("[bounded-id] reuse smoke report", flush=True)
        return
    rows = min(1024, int(context["features"].shape[0]))
    base = np.asarray(
        context["features"][:rows, :, 0],
        dtype=np.float32,
    )
    candidates = context["candidates"][:rows]
    reports = []
    for cap in CAPS:
        model, result = fit_bounded_id_residual_fixed(
            base,
            candidates,
            np.zeros(rows, dtype=np.int32),
            model_config=_model_config(args, context, cap),
            training_config=BoundedIDResidualTrainingConfig(
                epochs=1,
                batch_size=min(args.batch_size, 128),
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
            ),
            verbose=True,
        )
        scores = predict_bounded_id_residual_logits(
            model,
            base,
            candidates,
            batch_size=args.predict_batch_size,
        )
        audit = bounded_id_residual_audit(base, scores, cap=cap)
        if not audit["passed"]:
            raise RuntimeError(f"smoke residual bound failed: {cap}")
        reports.append({"cap": cap, "audit": audit})
        del model, result, scores
        release_memory()
    _write_json_atomic(
        report_path,
        {
            "status": "passed",
            "rows": rows,
            "caps": reports,
        },
    )


def _run_selection(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    reports: dict[float, list[dict[str, Any]]] = {
        cap: [] for cap in CAPS
    }
    for cap in CAPS:
        for fold in context["folds"][:2]:
            reports[cap].append(_run_fold(args, context, cap, fold))
    rows = []
    eligible_caps = []
    for cap in CAPS:
        deltas = [
            float(report["delta_vs_frozen_base"]["full"])
            for report in reports[cap]
        ]
        mean_delta = float(np.mean(deltas))
        eligible = bool(
            min(deltas) >= 0.0
            and mean_delta >= SELECTION_MEAN_DELTA_MIN
        )
        if eligible:
            eligible_caps.append(cap)
        rows.append(
            {
                "cap": cap,
                "fold_mrrs": [
                    float(report["score_metrics"]["full"])
                    for report in reports[cap]
                ],
                "fold_deltas": deltas,
                "mean_delta": mean_delta,
                "eligible": eligible,
            }
        )
    selected = (
        max(
            eligible_caps,
            key=lambda cap: (
                next(
                    row["mean_delta"]
                    for row in rows
                    if row["cap"] == cap
                ),
                -cap,
            ),
        )
        if eligible_caps
        else None
    )
    lock = {
        "status": "locked_before_gate",
        "selected_cap": selected,
        "selection_passed": selected is not None,
        "gate_metrics_read": False,
        "rows": rows,
        "created_at_unix": time.time(),
    }
    lock_path = args.output_dir / "selection-lock.json"
    if lock_path.exists():
        existing = _read_json(lock_path)
        for value in (existing, lock):
            value.pop("created_at_unix", None)
        if existing != lock:
            raise ValueError("existing bounded residual selection lock differs")
    else:
        _write_json_atomic(lock_path, lock)
    print(json.dumps(lock, indent=2), flush=True)


def _run_gate(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    lock_path = args.output_dir / "selection-lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("selection must lock before gate")
    lock = _read_json(lock_path)
    if lock.get("gate_metrics_read") is not False:
        raise ValueError("selection lock did not hide gate metrics")
    fold = context["folds"][2]
    ordered_caps = list(CAPS)
    if lock["selected_cap"] is not None:
        selected = float(lock["selected_cap"])
        ordered_caps.remove(selected)
        ordered_caps.insert(0, selected)
    reports = {
        cap: _run_fold(args, context, cap, fold)
        for cap in ordered_caps
    }
    selected_cap = lock["selected_cap"]
    if selected_cap is None:
        gate = {
            "status": "rejected",
            "passed": False,
            "selected_cap": None,
            "reason": "no cap passed the frozen Fold0/1 selection rule",
            "diagnostic_fold2": {
                str(cap): reports[cap]["delta_vs_frozen_base"]
                for cap in CAPS
            },
        }
    else:
        selected_cap = float(selected_cap)
        selection_row = next(
            row for row in lock["rows"] if row["cap"] == selected_cap
        )
        selection_deltas = [
            float(value) for value in selection_row["fold_deltas"]
        ]
        report = reports[selected_cap]
        forward_delta = float(
            report["delta_vs_frozen_base"]["full"]
        )
        three_fold_mean = float(
            np.mean([*selection_deltas, forward_delta])
        )
        activity_deltas = {
            key: float(value)
            for key, value in report[
                "activity_delta_vs_frozen_base"
            ].items()
        }
        passed = bool(
            forward_delta >= 0.0
            and three_fold_mean >= GATE_MEAN_DELTA_MIN
            and min(activity_deltas.values()) >= ACTIVITY_DELTA_MIN
        )
        gate = {
            "status": "passed" if passed else "rejected",
            "passed": passed,
            "selected_cap": selected_cap,
            "selection_deltas": selection_deltas,
            "fold2_delta": forward_delta,
            "three_fold_mean_delta": three_fold_mean,
            "activity_deltas": activity_deltas,
            "thresholds": {
                "fold2_delta_min": 0.0,
                "three_fold_mean_delta_min": GATE_MEAN_DELTA_MIN,
                "activity_delta_min": ACTIVITY_DELTA_MIN,
            },
            "diagnostic_fold2": {
                str(cap): reports[cap]["delta_vs_frozen_base"]
                for cap in CAPS
            },
        }
    gate["selection_lock"] = str(lock_path.resolve())
    gate["selection_lock_sha256"] = _sha256(lock_path)
    path = args.output_dir / "gate-report.json"
    if path.exists() and _read_json(path) != gate:
        raise ValueError("existing bounded residual gate report differs")
    if not path.exists():
        _write_json_atomic(path, gate)
    print(json.dumps(gate, indent=2), flush=True)


def _run_external(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    gate_path = args.output_dir / "gate-report.json"
    if not gate_path.exists() or not _read_json(gate_path)["passed"]:
        raise RuntimeError("external evaluation forbidden before gate pass")
    if args.champion_validation_scores is None:
        raise ValueError("--champion-validation-scores is required")
    report_path = args.output_dir / "external-evaluation-report.json"
    if report_path.exists():
        print(json.dumps(_read_json(report_path), indent=2), flush=True)
        return
    cap = float(_read_json(gate_path)["selected_cap"])
    full_dir = args.output_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    train_base_path = args.base_cache_dir / "full-train-base-logits.npy"
    base_report_path = args.base_cache_dir / "full-base-report.json"
    full_model, full_result = load_candidate_set_checkpoint(
        args.full_base_checkpoint
    )
    if train_base_path.exists():
        train_base = np.load(
            train_base_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    else:
        train_base_values = predict_candidate_set_logits(
            full_model,
            context["features"],
            mean=full_result.mean,
            std=full_result.std,
            batch_size=args.predict_batch_size,
        )
        _save_array_atomic(train_base_path, train_base_values)
        del train_base_values
        train_base = np.load(
            train_base_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    expert_logits = np.load(
        args.full_base_validation_expert_logits,
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_base = expert_logits[0]
    validation_prefix = str(args.validation_cache_prefix)
    validation_features = np.load(
        f"{validation_prefix}.val.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_candidates = np.load(
        f"{validation_prefix}.val-candidates.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    replay = predict_candidate_set_logits(
        full_model,
        validation_features,
        mean=full_result.mean,
        std=full_result.std,
        batch_size=args.predict_batch_size,
    )
    max_replay_error = float(
        np.max(np.abs(replay - validation_base))
    )
    if max_replay_error > 2e-5:
        raise RuntimeError("full frozen CST validation replay differs")
    if not base_report_path.exists():
        _write_json_atomic(
            base_report_path,
            {
                "status": "complete",
                "checkpoint": str(args.full_base_checkpoint.resolve()),
                "checkpoint_sha256": _sha256(
                    args.full_base_checkpoint
                ),
                "train_logits": str(train_base_path.resolve()),
                "train_logits_sha256": _sha256(train_base_path),
                "validation_replay_max_absolute_error": max_replay_error,
            },
        )
    checkpoint_path = full_dir / "model.npz"
    if checkpoint_path.exists():
        model, result = load_bounded_id_residual_checkpoint(
            checkpoint_path
        )
    else:
        model, result = fit_bounded_id_residual_fixed(
            train_base,
            context["candidates"],
            np.zeros(train_base.shape[0], dtype=np.int32),
            model_config=_model_config(args, context, cap),
            training_config=_training_config(args),
            verbose=True,
        )
        save_bounded_id_residual_checkpoint(
            checkpoint_path,
            model,
            result,
        )
    candidate_scores = predict_bounded_id_residual_logits(
        model,
        validation_base,
        validation_candidates,
        batch_size=args.predict_batch_size,
    )
    audit = bounded_id_residual_audit(
        validation_base,
        candidate_scores,
        cap=cap,
    )
    if not audit["passed"]:
        raise RuntimeError("external bounded residual audit failed")
    champion = np.load(
        args.champion_validation_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    candidate_metrics = _score_metrics(
        candidate_scores,
        validation_features,
    )
    frozen_base_metrics = _score_metrics(
        validation_base,
        validation_features,
    )
    delta_vs_frozen_base = {
        key: float(candidate_metrics[key] - frozen_base_metrics[key])
        for key in candidate_metrics
    }
    champion_metrics = _score_metrics(champion, validation_features)
    deltas = {
        key: float(candidate_metrics[key] - champion_metrics[key])
        for key in candidate_metrics
    }
    passed = bool(
        deltas["full"] >= EXTERNAL_DELTA_MIN
        and all(
            deltas[f"time_slice_{index}"] >= 0.0
            for index in range(3)
        )
    )
    scores_path = full_dir / "validation-logits.npy"
    _save_array_atomic(scores_path, candidate_scores)
    report = {
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "selected_cap": cap,
        "candidate_metrics": candidate_metrics,
        "frozen_base_metrics": frozen_base_metrics,
        "delta_vs_frozen_base": delta_vs_frozen_base,
        "champion_metrics": champion_metrics,
        "delta_vs_champion": deltas,
        "bound_audit": audit,
        "base_validation_replay_max_absolute_error": max_replay_error,
        "model": str(checkpoint_path.resolve()),
        "model_sha256": _sha256(checkpoint_path),
        "validation_logits": str(scores_path.resolve()),
        "validation_logits_sha256": _sha256(scores_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "submission_generated": False,
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


def _run_fold(
    args: argparse.Namespace,
    context: dict[str, Any],
    cap: float,
    fold: SourceConditionedFold,
) -> dict[str, Any]:
    directory = (
        args.output_dir
        / "folds"
        / _cap_name(cap)
        / f"fold-{fold.index}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        report = _read_json(report_path)
        _verify_residual_report(report)
        print(
            f"[bounded-id] reuse cap={cap:.2f} fold={fold.index} "
            f"delta={report['delta_vs_frozen_base']['full']:+.6f}",
            flush=True,
        )
        return report
    train_base, score_base, base_report = _ensure_fold_base_logits(
        args,
        context,
        fold,
    )
    train_stop = int(fold.train_rows[1])
    score_start, score_stop = fold.score_rows
    checkpoint_path = directory / "model.npz"
    if checkpoint_path.exists():
        model, result = load_bounded_id_residual_checkpoint(
            checkpoint_path
        )
    else:
        print(
            f"[bounded-id] train cap={cap:.2f} fold={fold.index} "
            f"rows={train_stop}",
            flush=True,
        )
        model, result = fit_bounded_id_residual_fixed(
            train_base,
            context["candidates"][:train_stop],
            np.zeros(train_stop, dtype=np.int32),
            model_config=_model_config(args, context, cap),
            training_config=_training_config(args),
            verbose=True,
        )
        save_bounded_id_residual_checkpoint(
            checkpoint_path,
            model,
            result,
        )
    scores = predict_bounded_id_residual_logits(
        model,
        score_base,
        context["candidates"][score_start:score_stop],
        batch_size=args.predict_batch_size,
    )
    audit = bounded_id_residual_audit(score_base, scores, cap=cap)
    if not audit["passed"]:
        raise RuntimeError(
            f"bounded residual audit failed: cap={cap} fold={fold.index}"
        )
    features = context["features"][score_start:score_stop]
    metrics = _score_metrics(scores, features)
    base_metrics = _score_metrics(score_base, features)
    deltas = {
        key: float(metrics[key] - base_metrics[key])
        for key in metrics
    }
    scores_path = directory / "score-logits.npy"
    _save_array_atomic(scores_path, scores)
    report = {
        "status": "complete",
        "cap": cap,
        "fold": asdict(fold),
        "model_config": asdict(_model_config(args, context, cap)),
        "training_config": asdict(_training_config(args)),
        "training_history": list(result.history),
        "score_metrics": metrics,
        "frozen_base_metrics": base_metrics,
        "delta_vs_frozen_base": deltas,
        "activity_delta_vs_frozen_base": {
            key: deltas[key]
            for key in (
                "activity_q1",
                "activity_q2",
                "activity_q3",
                "activity_q4",
            )
        },
        "bound_audit": audit,
        "base_cache_report": str(base_report.resolve()),
        "base_cache_report_sha256": _sha256(base_report),
        "model": str(checkpoint_path.resolve()),
        "model_sha256": _sha256(checkpoint_path),
        "score_logits": str(scores_path.resolve()),
        "score_logits_sha256": _sha256(scores_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(report_path, report)
    print(
        f"[bounded-id] complete cap={cap:.2f} fold={fold.index} "
        f"mrr={metrics['full']:.6f} delta={deltas['full']:+.6f}",
        flush=True,
    )
    del model, result, scores
    release_memory()
    return report


def _ensure_fold_base_logits(
    args: argparse.Namespace,
    context: dict[str, Any],
    fold: SourceConditionedFold,
) -> tuple[np.ndarray, np.ndarray, Path]:
    cache_dir = args.base_cache_dir / f"fold-{fold.index}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = cache_dir / "train-base-logits.npy"
    report_path = cache_dir / "report.json"
    base_dir = _base_fold_dir(args, fold.index)
    model_path = base_dir / "model.npz"
    score_path = base_dir / "score-logits.npy"
    base_report_path = base_dir / "report.json"
    _verify_base_artifacts(model_path, score_path, base_report_path)
    score_base = np.load(score_path, mmap_mode="r", allow_pickle=False)
    if report_path.exists() and train_path.exists():
        report = _read_json(report_path)
        if (
            report["model_sha256"] != _sha256(model_path)
            or report["saved_score_logits_sha256"]
            != _sha256(score_path)
            or report["train_logits_sha256"] != _sha256(train_path)
        ):
            raise ValueError("cached frozen base artifacts differ")
        return (
            np.load(train_path, mmap_mode="r", allow_pickle=False),
            score_base,
            report_path,
        )
    train_stop = int(fold.train_rows[1])
    print(
        f"[bounded-id] cache frozen A logits fold={fold.index} "
        f"train_rows={train_stop}",
        flush=True,
    )
    model, result = load_source_conditioned_checkpoint(model_path)
    causal = _load_sequences(args.sequence_cache_dir, "train-causal")
    train_values = predict_source_conditioned_logits(
        model,
        context["features"][:train_stop],
        context["candidates"][:train_stop],
        _slice_sequences(causal, 0, train_stop),
        mean=result.mean,
        std=result.std,
        batch_size=args.predict_batch_size,
    )
    _save_array_atomic(train_path, train_values)
    score_start, score_stop = fold.score_rows
    score_sequences = _load_sequences(
        args.sequence_cache_dir,
        f"fold-{fold.index}-score-frozen",
    )
    replay = predict_source_conditioned_logits(
        model,
        context["features"][score_start:score_stop],
        context["candidates"][score_start:score_stop],
        score_sequences,
        mean=result.mean,
        std=result.std,
        batch_size=args.predict_batch_size,
    )
    max_replay_error = float(np.max(np.abs(replay - score_base)))
    if max_replay_error > 2e-5:
        raise RuntimeError(
            f"frozen A score replay differs: fold={fold.index} "
            f"error={max_replay_error}"
        )
    report = {
        "status": "complete",
        "fold": asdict(fold),
        "model": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "saved_score_logits": str(score_path.resolve()),
        "saved_score_logits_sha256": _sha256(score_path),
        "train_logits": str(train_path.resolve()),
        "train_logits_sha256": _sha256(train_path),
        "train_logits_shape": list(train_values.shape),
        "score_replay_max_absolute_error": max_replay_error,
        "base_report": str(base_report_path.resolve()),
        "base_report_sha256": _sha256(base_report_path),
        "base_gradients": False,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(report_path, report)
    del model, result, train_values, replay
    release_memory()
    return (
        np.load(train_path, mmap_mode="r", allow_pickle=False),
        score_base,
        report_path,
    )


def _model_config(
    args: argparse.Namespace,
    context: dict[str, Any],
    cap: float,
) -> BoundedIDResidualConfig:
    return BoundedIDResidualConfig(
        num_items=context["num_items"],
        embedding_dim=args.embedding_dim,
        cap=cap,
        dropout=args.dropout,
    )


def _training_config(
    args: argparse.Namespace,
) -> BoundedIDResidualTrainingConfig:
    return BoundedIDResidualTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


def _score_metrics(scores: Any, features: Any) -> dict[str, float]:
    values = np.asarray(scores)
    positives = np.zeros(values.shape[0], dtype=np.int32)
    result = {"full": tie_neutral_mrr(values, positives)}
    boundaries = np.linspace(0, values.shape[0], 4, dtype=np.int64)
    for index in range(3):
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        result[f"time_slice_{index}"] = tie_neutral_mrr(
            values[start:stop],
            positives[start:stop],
        )
    activity = np.asarray(features[:, 0, 6], dtype=np.float64)
    order = np.argsort(activity, kind="stable")
    boundaries = np.linspace(0, values.shape[0], 5, dtype=np.int64)
    for index in range(4):
        selected = order[
            boundaries[index] : boundaries[index + 1]
        ]
        result[f"activity_q{index + 1}"] = tie_neutral_mrr(
            values[selected],
            positives[selected],
        )
    return {key: float(value) for key, value in result.items()}


def _base_fold_dir(args: argparse.Namespace, fold: int) -> Path:
    return (
        args.base_result_dir
        / "folds"
        / "variant-A"
        / f"fold-{fold}"
    )


def _verify_base_artifacts(
    model_path: Path,
    score_path: Path,
    report_path: Path,
) -> None:
    report = _read_json(report_path)
    if (
        report.get("status") != "complete"
        or report.get("variant") != "A"
        or report.get("trainable_frameworks") != ["jittor"]
        or report.get("non_jittor_trainable_models") != []
        or report.get("model_sha256") != _sha256(model_path)
        or report.get("score_logits_sha256") != _sha256(score_path)
    ):
        raise ValueError("frozen A base artifact contract differs")


def _verify_residual_report(report: dict[str, Any]) -> None:
    if (
        report.get("status") != "complete"
        or report.get("trainable_frameworks") != ["jittor"]
        or report.get("non_jittor_trainable_models") != []
        or not report.get("bound_audit", {}).get("passed")
    ):
        raise ValueError("cached bounded residual report is invalid")
    for key in ("model", "score_logits", "base_cache_report"):
        path = Path(report[key])
        if not path.exists() or _sha256(path) != report[f"{key}_sha256"]:
            raise ValueError(f"cached bounded residual artifact differs: {path}")


def _load_sequences(directory: Path, prefix: str) -> SourceSequenceRows:
    return SourceSequenceRows(
        items=np.load(
            directory / f"{prefix}-items.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        time_buckets=np.load(
            directory / f"{prefix}-time-buckets.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        lengths=np.load(
            directory / f"{prefix}-lengths.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
    )


def _slice_sequences(
    rows: SourceSequenceRows,
    start: int,
    stop: int,
) -> SourceSequenceRows:
    return SourceSequenceRows(
        items=rows.items[start:stop],
        time_buckets=rows.time_buckets[start:stop],
        lengths=rows.lengths[start:stop],
    )


def _cap_name(cap: float) -> str:
    return f"cap-{round(cap * 100):03d}"


def _configure_device(device: str, seed: int) -> None:
    if device == "cuda":
        if not jt.has_cuda:
            raise RuntimeError("CUDA requested but Jittor has no CUDA")
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    jt.set_global_seed(seed)


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
