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
from jgrec.rankers.hybrid.bounded_source_decoder import (
    BoundedSourceDecoderConfig,
    BoundedSourceDecoderTrainingConfig,
    bounded_source_decoder_audit,
    fit_bounded_source_decoder_fixed,
    load_bounded_source_decoder_checkpoint,
    predict_bounded_source_decoder_logits,
    save_bounded_source_decoder_checkpoint,
)
from jgrec.rankers.hybrid.oof_stacking import tie_neutral_mrr
from jgrec.rankers.hybrid.source_sequence_cache import (
    SourceConditionedFold,
    SourceSequenceRows,
)

CAPS = (0.02, 0.05, 0.10)
SELECTION_MEAN_DELTA_MIN = 0.0001
GATE_MEAN_DELTA_MIN = 0.0001
ACTIVITY_DELTA_MIN = -0.0005
EXTERNAL_DELTA_MIN = 0.0002


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train frozen-CST bounded source-sequence decoders.",
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
    parser.add_argument(
        "--full-base-validation-expert-logits",
        type=Path,
        default=Path(
            "result/dataset2_pure_jittor_oof_stacking_20260726/"
            "full-validation-expert-logits.npy"
        ),
    )
    parser.add_argument("--champion-validation-scores", type=Path)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--support-tau", type=float, default=20.0)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
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
        gate = _read_json(args.output_dir / "gate-report.json")
        if gate["passed"]:
            _run_external(args, context)
        _finalize(args)
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
    sequences = _load_sequence_rows(args.sequence_cache_dir, "train-causal")
    train_rows = int(manifest["train_rows"])
    expected_scores = (train_rows, int(manifest["candidate_count"]))
    if (
        manifest.get("status") != "complete"
        or manifest.get("trainable_frameworks") != ["jittor"]
        or manifest.get("non_jittor_trainable_models") != []
        or features.shape[:2] != expected_scores
        or candidates.shape != expected_scores
        or times.shape != (train_rows,)
        or dst.shape != (train_rows,)
        or sequences.items.shape
        != (train_rows, int(manifest["max_length"]))
        or sequences.time_buckets.shape != sequences.items.shape
        or sequences.lengths.shape != (train_rows,)
        or not np.array_equal(
            np.asarray(candidates[:, 0]),
            np.asarray(dst),
        )
    ):
        raise ValueError("bounded source train cache contract differs")
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
        raise ValueError("bounded source decoder requires three folds")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "features": features,
        "candidates": candidates,
        "times": times,
        "dst": dst,
        "sequences": sequences,
        "folds": folds,
        "num_items": int(manifest["num_items"]),
    }


def _freeze_protocol(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_folds = []
    for fold in context["folds"]:
        base_report = (
            args.base_result_dir
            / "folds"
            / "variant-A"
            / f"fold-{fold.index}"
            / "report.json"
        )
        base_score = (
            args.base_result_dir
            / "folds"
            / "variant-A"
            / f"fold-{fold.index}"
            / "score-logits.npy"
        )
        train_base = (
            args.base_cache_dir
            / f"fold-{fold.index}"
            / "train-base-logits.npy"
        )
        base_folds.append(
            {
                "fold": asdict(fold),
                "base_report": str(base_report.resolve()),
                "base_report_sha256": _sha256(base_report),
                "score_base": str(base_score.resolve()),
                "score_base_sha256": _sha256(base_score),
                "train_base": str(train_base.resolve()),
                "train_base_sha256": _sha256(train_base),
            }
        )
    frozen = {
        "status": "frozen_before_training",
        "protocol": "dataset2_frozen_cst_bounded_source_decoder_v1",
        "formula": (
            "base + project_zero_mean_cap("
            "source_candidate_interaction * support_shrinkage)"
        ),
        "candidate_id_policy": (
            "candidate IDs are attention queries only; "
            "no standalone ID logit or hidden-state addition"
        ),
        "empty_history_policy": "exact frozen-base fallback",
        "caps": list(CAPS),
        "model": {
            "embedding_dim": args.embedding_dim,
            "heads": args.heads,
            "source_max_length": context["manifest"]["max_length"],
            "support_tau": args.support_tau,
            "dropout": args.dropout,
            "base_gradients": False,
        },
        "training": asdict(_training_config(args)),
        "selection_folds": [0, 1],
        "gate_fold": 2,
        "selection_rule": {
            "each_fold_delta_min": 0.0,
            "mean_delta_min": SELECTION_MEAN_DELTA_MIN,
            "tie_break": "higher mean delta then lower cap",
        },
        "gate_rule": {
            "fold2_delta_min": 0.0,
            "three_fold_mean_delta_min": GATE_MEAN_DELTA_MIN,
            "worst_activity_delta_min": ACTIVITY_DELTA_MIN,
        },
        "external_rule": {
            "read_only_after_gate": True,
            "champion_delta_min": EXTERNAL_DELTA_MIN,
            "all_time_slice_deltas_non_negative": True,
        },
        "base_folds": base_folds,
        "fold_manifest": str(context["manifest_path"].resolve()),
        "fold_manifest_sha256": _sha256(context["manifest_path"]),
        "full_train_base": str(
            (args.base_cache_dir / "full-train-base-logits.npy").resolve()
        ),
        "external_inputs_blinded_until_gate": {
            "validation_cache_prefix": str(
                args.validation_cache_prefix.resolve()
            ),
            "full_base_validation_expert_logits": str(
                args.full_base_validation_expert_logits.resolve()
            ),
            "champion_validation_scores": (
                str(args.champion_validation_scores.resolve())
                if args.champion_validation_scores is not None
                else None
            ),
        },
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    path = args.output_dir / "frozen-config.json"
    if path.exists() and _read_json(path) != frozen:
        raise ValueError("existing bounded source protocol differs")
    if not path.exists():
        _write_json_atomic(path, frozen)


def _run_smoke(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    report_path = args.output_dir / "smoke-report.json"
    if report_path.exists():
        print("[bounded-source] reuse smoke", flush=True)
        return
    rows = 512
    base = np.asarray(context["features"][:rows, :, 0], dtype=np.float32)
    candidates = context["candidates"][:rows]
    sequences = _slice_sequences(context["sequences"], 0, rows)
    support = _candidate_support(
        context["dst"][:rows],
        candidates,
        context["num_items"],
    )
    reports = []
    for cap in CAPS:
        model, result = fit_bounded_source_decoder_fixed(
            base,
            candidates,
            sequences,
            support,
            np.zeros(rows, dtype=np.int32),
            model_config=_model_config(args, context, cap),
            training_config=BoundedSourceDecoderTrainingConfig(
                epochs=1,
                batch_size=64,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
            ),
            verbose=False,
        )
        scores = predict_bounded_source_decoder_logits(
            model,
            base,
            candidates,
            sequences,
            support,
            batch_size=64,
        )
        audit = bounded_source_decoder_audit(
            base,
            scores,
            sequences.lengths,
            cap=cap,
        )
        if not audit["passed"]:
            raise RuntimeError(f"bounded source smoke audit failed: {cap}")
        checkpoint = args.output_dir / "smoke" / f"cap-{cap:.2f}.npz"
        save_bounded_source_decoder_checkpoint(
            checkpoint,
            model,
            result,
        )
        loaded, _ = load_bounded_source_decoder_checkpoint(checkpoint)
        replay = predict_bounded_source_decoder_logits(
            loaded,
            base,
            candidates,
            sequences,
            support,
            batch_size=64,
        )
        reports.append(
            {
                "cap": cap,
                "audit": audit,
                "checkpoint_max_replay_error": float(
                    np.max(np.abs(replay - scores))
                ),
            }
        )
        del model, loaded, scores, replay
        release_memory()
    _write_json_atomic(
        report_path,
        {"status": "passed", "rows": rows, "reports": reports},
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
    eligible = []
    for cap in CAPS:
        deltas = [
            float(report["delta_vs_frozen_base"]["full"])
            for report in reports[cap]
        ]
        mean_delta = float(np.mean(deltas))
        passed = bool(
            min(deltas) >= 0.0
            and mean_delta >= SELECTION_MEAN_DELTA_MIN
        )
        if passed:
            eligible.append(cap)
        rows.append(
            {
                "cap": cap,
                "fold_deltas": deltas,
                "mean_delta": mean_delta,
                "eligible": passed,
            }
        )
    selected = (
        max(
            eligible,
            key=lambda cap: (
                next(row["mean_delta"] for row in rows if row["cap"] == cap),
                -cap,
            ),
        )
        if eligible
        else None
    )
    lock = {
        "status": "locked_before_gate",
        "selected_cap": selected,
        "selection_passed": selected is not None,
        "gate_metrics_read": False,
        "rows": rows,
    }
    path = args.output_dir / "selection-lock.json"
    if path.exists() and _read_json(path) != lock:
        raise ValueError("existing bounded source selection lock differs")
    if not path.exists():
        _write_json_atomic(path, lock)
    print(json.dumps(lock, indent=2), flush=True)


def _run_gate(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    lock_path = args.output_dir / "selection-lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("selection must lock before Fold2")
    lock = _read_json(lock_path)
    selected = lock["selected_cap"]
    if selected is None:
        gate = {
            "status": "rejected",
            "passed": False,
            "selected_cap": None,
            "reason": "no cap passed Fold0/1 selection",
            "fold2_evaluated": False,
        }
    else:
        cap = float(selected)
        report = _run_fold(args, context, cap, context["folds"][2])
        selection_row = next(
            row for row in lock["rows"] if float(row["cap"]) == cap
        )
        selection_deltas = [
            float(value) for value in selection_row["fold_deltas"]
        ]
        fold2_delta = float(report["delta_vs_frozen_base"]["full"])
        mean_delta = float(np.mean([*selection_deltas, fold2_delta]))
        activity_deltas = {
            key: float(report["delta_vs_frozen_base"][key])
            for key in (
                "activity_q1",
                "activity_q2",
                "activity_q3",
                "activity_q4",
            )
        }
        passed = bool(
            fold2_delta >= 0.0
            and mean_delta >= GATE_MEAN_DELTA_MIN
            and min(activity_deltas.values()) >= ACTIVITY_DELTA_MIN
            and report["residual_audit"]["passed"]
        )
        gate = {
            "status": "passed" if passed else "rejected",
            "passed": passed,
            "selected_cap": cap,
            "selection_deltas": selection_deltas,
            "fold2_delta": fold2_delta,
            "three_fold_mean_delta": mean_delta,
            "activity_deltas": activity_deltas,
            "residual_audit": report["residual_audit"],
            "fold2_evaluated": True,
            "thresholds": {
                "fold2_delta_min": 0.0,
                "three_fold_mean_delta_min": GATE_MEAN_DELTA_MIN,
                "activity_delta_min": ACTIVITY_DELTA_MIN,
            },
        }
    gate["selection_lock"] = str(lock_path.resolve())
    gate["selection_lock_sha256"] = _sha256(lock_path)
    path = args.output_dir / "gate-report.json"
    if path.exists() and _read_json(path) != gate:
        raise ValueError("existing bounded source gate report differs")
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
    train_base = np.load(
        args.base_cache_dir / "full-train-base-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    train_support = _candidate_support(
        context["dst"],
        context["candidates"],
        context["num_items"],
    )
    directory = args.output_dir / "full"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.npz"
    if checkpoint.exists():
        model, result = load_bounded_source_decoder_checkpoint(checkpoint)
    else:
        model, result = fit_bounded_source_decoder_fixed(
            train_base,
            context["candidates"],
            context["sequences"],
            train_support,
            np.zeros(train_base.shape[0], dtype=np.int32),
            model_config=_model_config(args, context, cap),
            training_config=_training_config(args),
            verbose=True,
        )
        save_bounded_source_decoder_checkpoint(checkpoint, model, result)
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
    validation_sequences = _load_sequence_rows(
        args.sequence_cache_dir,
        "external-frozen",
    )
    validation_experts = np.load(
        args.full_base_validation_expert_logits,
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_base = np.asarray(validation_experts[0])
    validation_support = _candidate_support(
        context["dst"],
        validation_candidates,
        context["num_items"],
    )
    scores = predict_bounded_source_decoder_logits(
        model,
        validation_base,
        validation_candidates,
        validation_sequences,
        validation_support,
        batch_size=args.predict_batch_size,
    )
    audit = bounded_source_decoder_audit(
        validation_base,
        scores,
        validation_sequences.lengths,
        cap=cap,
    )
    if not audit["passed"]:
        raise RuntimeError("external bounded source audit failed")
    metrics = _score_metrics(scores, validation_features)
    base_metrics = _score_metrics(validation_base, validation_features)
    champion = np.load(
        args.champion_validation_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    champion_metrics = _score_metrics(champion, validation_features)
    delta_base = {
        key: float(metrics[key] - base_metrics[key]) for key in metrics
    }
    delta_champion = {
        key: float(metrics[key] - champion_metrics[key]) for key in metrics
    }
    passed = bool(
        delta_champion["full"] >= EXTERNAL_DELTA_MIN
        and all(
            delta_champion[f"time_slice_{index}"] >= 0.0
            for index in range(3)
        )
    )
    score_path = directory / "validation-scores.npy"
    _save_array_atomic(score_path, scores)
    report = {
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "selected_cap": cap,
        "candidate_metrics": metrics,
        "frozen_base_metrics": base_metrics,
        "champion_metrics": champion_metrics,
        "delta_vs_frozen_base": delta_base,
        "delta_vs_champion": delta_champion,
        "residual_audit": audit,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "validation_scores": str(score_path.resolve()),
        "validation_scores_sha256": _sha256(score_path),
        "training_history": list(result.history),
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
        args.output_dir / "folds" / f"cap-{cap:.2f}" / f"fold-{fold.index}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        report = _read_json(report_path)
        _verify_fold_report(report)
        print(
            f"[bounded-source] reuse cap={cap:.2f} fold={fold.index} "
            f"delta={report['delta_vs_frozen_base']['full']:+.6f}",
            flush=True,
        )
        return report
    train_stop = int(fold.train_rows[1])
    score_start, score_stop = fold.score_rows
    train_base = np.load(
        args.base_cache_dir
        / f"fold-{fold.index}"
        / "train-base-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    train_candidates = context["candidates"][:train_stop]
    train_sequences = _slice_sequences(context["sequences"], 0, train_stop)
    train_support = _candidate_support(
        context["dst"][:train_stop],
        train_candidates,
        context["num_items"],
    )
    checkpoint = directory / "model.npz"
    model, result = fit_bounded_source_decoder_fixed(
        train_base,
        train_candidates,
        train_sequences,
        train_support,
        np.zeros(train_stop, dtype=np.int32),
        model_config=_model_config(args, context, cap),
        training_config=_training_config(args),
        verbose=True,
    )
    save_bounded_source_decoder_checkpoint(checkpoint, model, result)
    score_base = np.load(
        args.base_result_dir
        / "folds"
        / "variant-A"
        / f"fold-{fold.index}"
        / "score-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    score_candidates = context["candidates"][score_start:score_stop]
    score_sequences = _load_sequence_rows(
        args.sequence_cache_dir,
        f"fold-{fold.index}-score-frozen",
    )
    score_support = _candidate_support(
        context["dst"][:train_stop],
        score_candidates,
        context["num_items"],
    )
    scores = predict_bounded_source_decoder_logits(
        model,
        score_base,
        score_candidates,
        score_sequences,
        score_support,
        batch_size=args.predict_batch_size,
    )
    audit = bounded_source_decoder_audit(
        score_base,
        scores,
        score_sequences.lengths,
        cap=cap,
    )
    if not audit["passed"]:
        raise RuntimeError(
            f"bounded source audit failed cap={cap} fold={fold.index}"
        )
    features = context["features"][score_start:score_stop]
    metrics = _score_metrics(scores, features)
    base_metrics = _score_metrics(score_base, features)
    deltas = {
        key: float(metrics[key] - base_metrics[key]) for key in metrics
    }
    score_path = directory / "score-logits.npy"
    _save_array_atomic(score_path, scores)
    report = {
        "status": "complete",
        "cap": cap,
        "fold": asdict(fold),
        "score_metrics": metrics,
        "frozen_base_metrics": base_metrics,
        "delta_vs_frozen_base": deltas,
        "residual_audit": audit,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "score_logits": str(score_path.resolve()),
        "score_logits_sha256": _sha256(score_path),
        "training_history": list(result.history),
        "support_history_rows": train_stop,
        "source_sequence_history_rule": (
            "train causal; score event_time < fold score origin"
        ),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(report_path, report)
    print(
        f"[bounded-source] complete cap={cap:.2f} fold={fold.index} "
        f"mrr={metrics['full']:.6f} delta={deltas['full']:+.6f}",
        flush=True,
    )
    del model, train_support, score_support, scores
    release_memory()
    return report


def _candidate_support(
    history_dst: Any,
    candidate_ids: Any,
    num_items: int,
) -> np.ndarray:
    history = np.asarray(history_dst, dtype=np.int64)
    candidates = np.asarray(candidate_ids)
    if (
        history.ndim != 1
        or candidates.ndim != 2
        or np.any(history < 0)
        or np.any(history > num_items)
        or np.any(candidates < 0)
        or np.any(candidates > num_items)
    ):
        raise ValueError("bounded source support IDs are invalid")
    counts = np.bincount(history, minlength=num_items + 1).astype(
        np.float32,
        copy=False,
    )
    return counts[candidates]


def _model_config(
    args: argparse.Namespace,
    context: dict[str, Any],
    cap: float,
) -> BoundedSourceDecoderConfig:
    return BoundedSourceDecoderConfig(
        num_items=context["num_items"],
        embedding_dim=args.embedding_dim,
        heads=args.heads,
        source_max_length=int(context["manifest"]["max_length"]),
        cap=cap,
        support_tau=args.support_tau,
        dropout=args.dropout,
    )


def _training_config(
    args: argparse.Namespace,
) -> BoundedSourceDecoderTrainingConfig:
    return BoundedSourceDecoderTrainingConfig(
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
        selected = order[boundaries[index] : boundaries[index + 1]]
        result[f"activity_q{index + 1}"] = tie_neutral_mrr(
            values[selected],
            positives[selected],
        )
    return {key: float(value) for key, value in result.items()}


def _load_sequence_rows(directory: Path, prefix: str) -> SourceSequenceRows:
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


def _verify_fold_report(report: dict[str, Any]) -> None:
    if (
        report.get("status") != "complete"
        or report.get("trainable_frameworks") != ["jittor"]
        or report.get("non_jittor_trainable_models") != []
        or not report.get("residual_audit", {}).get("passed")
    ):
        raise ValueError("cached bounded source fold report is invalid")
    for key in ("checkpoint", "score_logits"):
        path = Path(report[key])
        if not path.exists() or _sha256(path) != report[f"{key}_sha256"]:
            raise ValueError(f"bounded source artifact differs: {path}")


def _finalize(args: argparse.Namespace) -> None:
    selection = _read_json(args.output_dir / "selection-lock.json")
    gate = _read_json(args.output_dir / "gate-report.json")
    external_path = args.output_dir / "external-evaluation-report.json"
    external = _read_json(external_path) if external_path.exists() else None
    report = {
        "status": (
            "passed"
            if external is not None and external["passed"]
            else "rejected"
        ),
        "selection": selection,
        "gate": gate,
        "external_evaluated": external is not None,
        "external": external,
        "submission_generated": bool(
            external is not None
            and external.get("submission_generated", False)
        ),
        "online_champion_unchanged": not bool(
            external is not None and external["passed"]
        ),
        "overfit_control": {
            "base_frozen": True,
            "standalone_candidate_id_branch": False,
            "hard_residual_cap": True,
            "row_centered_residual": True,
            "frequency_shrinkage": True,
            "empty_history_exact_fallback": True,
        },
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(args.output_dir / "evaluation-report.json", report)


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
