from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.core.io import read_interactions
from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.oof_stacking import tie_neutral_mrr
from jgrec.rankers.hybrid.source_conditioned_cst import (
    SourceConditionedCSTConfig,
    abcd_model_config,
)
from jgrec.rankers.hybrid.source_conditioned_training import (
    SourceConditionedTrainingConfig,
    fit_source_conditioned_cst,
    fit_source_conditioned_cst_fixed,
    load_source_conditioned_checkpoint,
    predict_source_conditioned_logits,
    save_source_conditioned_checkpoint,
)
from jgrec.rankers.hybrid.source_sequence_cache import (
    SourceConditionedFold,
    SourceSequenceRows,
    build_causal_source_sequences,
)

VARIANTS = ("A", "B", "C", "D")
ACTIVITY_REGRESSION_LIMIT = -0.003
EXTERNAL_MINIMUM_DELTA = 0.0002


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run pure-Jittor Dataset2 source-conditioned CST A/B/C/D."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("smoke", "selection", "gate", "external", "all"),
        default="all",
    )
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--sequence-cache-dir", required=True, type=Path)
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
        "--validation-cache-report",
        type=Path,
        default=Path(
            "result/dataset2_joint_recent200k_full100_seed60_20260725/"
            "validation-cache-report.json"
        ),
    )
    parser.add_argument(
        "--champion-validation-scores",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/dataset2/train.csv"),
    )
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--candidate-layers", type=int, default=2)
    parser.add_argument("--source-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--internal-validation-rows", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=60)
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


def _load_context(
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest = _read_json(args.sequence_cache_dir / "fold-manifest.json")
    if (
        manifest.get("status") != "complete"
        or manifest.get("trainable_frameworks") != ["jittor"]
        or manifest.get("non_jittor_trainable_models") != []
    ):
        raise ValueError("source sequence cache manifest is invalid")
    base = str(args.train_cache_prefix)
    paths = {
        "features": Path(f"{base}.train.npy"),
        "candidates": Path(f"{base}.train-candidates.npy"),
        "src": Path(f"{base}.train-src.npy"),
        "time": Path(f"{base}.train-time.npy"),
        "dst": Path(f"{base}.train-dst.npy"),
    }
    features = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
    candidates = np.load(
        paths["candidates"],
        mmap_mode="r",
        allow_pickle=False,
    )
    sources = np.load(paths["src"], mmap_mode="r", allow_pickle=False)
    times = np.load(paths["time"], mmap_mode="r", allow_pickle=False)
    dst = np.load(paths["dst"], mmap_mode="r", allow_pickle=False)
    sequences = _load_sequence_rows(
        args.sequence_cache_dir,
        "train-causal",
    )
    if (
        features.shape != (
            manifest["train_rows"],
            manifest["candidate_count"],
            manifest["feature_count"],
        )
        or candidates.shape != features.shape[:2]
        or sources.shape != features.shape[:1]
        or times.shape != features.shape[:1]
        or dst.shape != features.shape[:1]
        or sequences.items.shape
        != (features.shape[0], manifest["max_length"])
        or not np.array_equal(np.asarray(candidates[:, 0]), np.asarray(dst))
    ):
        raise ValueError("source-conditioned ABCD cache contract differs")
    folds = tuple(
        SourceConditionedFold(
            index=int(row["index"]),
            train_rows=tuple(int(x) for x in row["train_rows"]),
            score_rows=tuple(int(x) for x in row["score_rows"]),
            role=str(row["role"]),
            train_time_max=int(row["train_time_max"]),
            score_time_min=int(row["score_time_min"]),
            score_time_max=int(row["score_time_max"]),
        )
        for row in manifest["folds"]
    )
    return {
        "manifest": manifest,
        "paths": paths,
        "features": features,
        "candidates": candidates,
        "sources": sources,
        "times": times,
        "dst": dst,
        "sequences": sequences,
        "folds": folds,
        "train_report": _read_json(args.train_cache_report),
    }


def _freeze_protocol(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "frozen-config.json"
    model_configs = {
        variant: asdict(_model_config(args, context, variant))
        for variant in VARIANTS
    }
    training_config = asdict(_training_config(args))
    frozen = {
        "status": "frozen_before_training",
        "protocol": "dataset2_source_conditioned_cst_abcd_v1",
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "variants": {
            "A": {
                "candidate_id": False,
                "source_sequence": False,
                "candidate_self_attention": True,
            },
            "B": {
                "candidate_id": True,
                "source_sequence": False,
                "candidate_self_attention": True,
            },
            "C": {
                "candidate_id": True,
                "source_sequence": True,
                "candidate_self_attention": False,
            },
            "D": {
                "candidate_id": True,
                "source_sequence": True,
                "candidate_self_attention": True,
            },
        },
        "model_configs": model_configs,
        "training_config": training_config,
        "internal_validation_rows": args.internal_validation_rows,
        "selection_folds": [0, 1],
        "gate_fold": 2,
        "selection_rule": (
            "highest two-fold mean among variants non-negative vs A "
            "on both selection folds"
        ),
        "gate_rule": {
            "fold2_delta_vs_A_min": 0.0,
            "three_fold_mean_delta_vs_A_min": 0.0,
            "worst_activity_quartile_delta_min": (
                ACTIVITY_REGRESSION_LIMIT
            ),
        },
        "external_rule": {
            "read_only_after_gate": True,
            "minimum_delta_vs_champion": EXTERNAL_MINIMUM_DELTA,
            "all_time_slices_non_negative": True,
        },
        "folds": [asdict(fold) for fold in context["folds"]],
        "sequence_manifest": str(
            (args.sequence_cache_dir / "fold-manifest.json").resolve()
        ),
        "sequence_manifest_sha256": _sha256(
            args.sequence_cache_dir / "fold-manifest.json"
        ),
        "train_cache_report": str(args.train_cache_report.resolve()),
        "train_cache_report_sha256": _sha256(args.train_cache_report),
        "device": args.device,
        "positive_index": 0,
        "metric": "tie_neutral_mrr",
    }
    frozen = json.loads(json.dumps(frozen, sort_keys=True))
    if path.exists():
        if _read_json(path) != frozen:
            raise ValueError("existing ABCD experiment protocol differs")
        return
    _write_json_atomic(path, frozen)


def _run_smoke(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    output = args.output_dir / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if report_path.exists():
        print("[smoke] reuse complete smoke", flush=True)
        return
    rows = min(768, int(context["features"].shape[0]))
    split = min(512, rows - 1)
    reports = []
    for variant in VARIANTS:
        print(f"[smoke] variant={variant}", flush=True)
        config = replace(
            _model_config(args, context, variant),
            model_dim=min(args.model_dim, 16),
            heads=(
                4 if min(args.model_dim, 16) % 4 == 0 else 2
            ),
            candidate_layers=1,
            source_layers=1,
        )
        training = replace(
            _training_config(args),
            epochs=1,
            batch_size=min(args.batch_size, 32),
            early_stop_patience=0,
        )
        model, result = fit_source_conditioned_cst(
            context["features"][:split],
            context["candidates"][:split],
            _slice_sequences(context["sequences"], 0, split),
            np.zeros(split, dtype=np.int32),
            context["features"][split:rows],
            context["candidates"][split:rows],
            _slice_sequences(context["sequences"], split, rows),
            np.zeros(rows - split, dtype=np.int32),
            model_config=config,
            training_config=training,
            verbose=True,
        )
        before = predict_source_conditioned_logits(
            model,
            context["features"][split:rows],
            context["candidates"][split:rows],
            _slice_sequences(context["sequences"], split, rows),
            mean=result.mean,
            std=result.std,
            batch_size=training.batch_size,
        )
        checkpoint = output / f"variant-{variant}.npz"
        save_source_conditioned_checkpoint(checkpoint, model, result)
        loaded_model, loaded_result = load_source_conditioned_checkpoint(
            checkpoint
        )
        after = predict_source_conditioned_logits(
            loaded_model,
            context["features"][split:rows],
            context["candidates"][split:rows],
            _slice_sequences(context["sequences"], split, rows),
            mean=loaded_result.mean,
            std=loaded_result.std,
            batch_size=training.batch_size,
        )
        max_reload_error = float(np.max(np.abs(before - after)))
        if max_reload_error > 2e-5:
            raise RuntimeError(
                f"smoke checkpoint reload differs: {variant}"
            )
        reports.append(
            {
                "variant": variant,
                "best_val_mrr": result.best_val_mrr,
                "max_reload_error": max_reload_error,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        )
        del model, loaded_model, result, loaded_result, before, after
        release_memory()
    _write_json_atomic(
        report_path,
        {
            "status": "passed",
            "device": args.device,
            "rows": rows,
            "variants": reports,
        },
    )


def _run_selection(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    reports: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in VARIANTS
    }
    for variant in VARIANTS:
        for fold in context["folds"][:2]:
            reports[variant].append(
                _run_fold(args, context, variant, fold)
            )
    lock_path = args.output_dir / "selection-lock.json"
    candidate = _selection_decision(reports)
    locked = {
        "status": "locked_before_gate",
        "selected_variant": candidate,
        "selection_folds": [0, 1],
        "gate_metrics_read": False,
        "variant_mrrs": {
            variant: [
                float(report["score_metrics"]["full"])
                for report in reports[variant]
            ]
            for variant in VARIANTS
        },
        "variant_mean_mrr": {
            variant: float(
                np.mean(
                    [
                        report["score_metrics"]["full"]
                        for report in reports[variant]
                    ]
                )
            )
            for variant in VARIANTS
        },
        "selection_deltas_vs_A": {
            variant: [
                float(
                    reports[variant][index]["score_metrics"]["full"]
                    - reports["A"][index]["score_metrics"]["full"]
                )
                for index in range(2)
            ]
            for variant in VARIANTS
        },
        "selection_rule": (
            "non-negative vs A on both folds, then mean MRR, "
            "then worst delta; conservative tie preference A/B/C/D"
        ),
        "created_at_unix": time.time(),
    }
    if lock_path.exists():
        existing = _read_json(lock_path)
        comparable_existing = {
            key: value
            for key, value in existing.items()
            if key != "created_at_unix"
        }
        comparable_locked = {
            key: value
            for key, value in locked.items()
            if key != "created_at_unix"
        }
        if comparable_existing != comparable_locked:
            raise ValueError("existing selection lock differs")
    else:
        _write_json_atomic(lock_path, locked)
    print(json.dumps(locked, indent=2), flush=True)


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
    reports = {
        variant: _run_fold(args, context, variant, fold)
        for variant in VARIANTS
    }
    selected = str(lock["selected_variant"])
    baseline = reports["A"]
    candidate = reports[selected]
    forward_delta = float(
        candidate["score_metrics"]["full"]
        - baseline["score_metrics"]["full"]
    )
    selection_deltas = [
        float(value)
        for value in lock["selection_deltas_vs_A"][selected]
    ]
    overall_mean_delta = float(
        np.mean([*selection_deltas, forward_delta])
    )
    activity_deltas = {
        key: float(
            candidate["score_metrics"][key]
            - baseline["score_metrics"][key]
        )
        for key in (
            "activity_q1",
            "activity_q2",
            "activity_q3",
            "activity_q4",
        )
    }
    passed = bool(
        selected != "A"
        and forward_delta >= 0.0
        and overall_mean_delta >= 0.0
        and min(activity_deltas.values())
        >= ACTIVITY_REGRESSION_LIMIT
    )
    report = {
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "selected_variant": selected,
        "selection_lock": str(lock_path.resolve()),
        "selection_lock_sha256": _sha256(lock_path),
        "fold": asdict(fold),
        "all_variant_gate_metrics": {
            variant: row["score_metrics"]
            for variant, row in reports.items()
        },
        "selection_deltas_vs_A": selection_deltas,
        "forward_delta_vs_A": forward_delta,
        "three_fold_mean_delta_vs_A": overall_mean_delta,
        "activity_deltas_vs_A": activity_deltas,
        "thresholds": {
            "forward_delta_min": 0.0,
            "overall_mean_delta_min": 0.0,
            "activity_delta_min": ACTIVITY_REGRESSION_LIMIT,
        },
        "reason": (
            "selected baseline A; no structural gain"
            if selected == "A"
            else (
                "all gate conditions passed"
                if passed
                else "one or more frozen gate conditions failed"
            )
        ),
    }
    report_path = args.output_dir / "gate-report.json"
    if report_path.exists() and _read_json(report_path) != report:
        raise ValueError("existing gate report differs")
    if not report_path.exists():
        _write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


def _run_external(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    gate_path = args.output_dir / "gate-report.json"
    if not gate_path.exists() or not _read_json(gate_path).get("passed"):
        raise RuntimeError("external evaluation is forbidden before gate pass")
    if args.champion_validation_scores is None:
        raise ValueError(
            "--champion-validation-scores is required after gate pass"
        )
    report_path = args.output_dir / "external-evaluation-report.json"
    if report_path.exists():
        print(json.dumps(_read_json(report_path), indent=2), flush=True)
        return
    selected = str(_read_json(gate_path)["selected_variant"])
    best_epochs = []
    for fold_index in (0, 1):
        fit_report = _read_json(
            args.output_dir
            / "folds"
            / f"variant-{selected}"
            / f"fold-{fold_index}"
            / "fit-selection.json"
        )
        best_epochs.append(int(fit_report["best_epoch"]))
    fixed_epochs = max(
        1,
        math.floor(float(np.median(best_epochs)) + 0.5),
    )
    model_path = args.output_dir / "full" / "model.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        model, result = load_source_conditioned_checkpoint(model_path)
    else:
        config = _model_config(args, context, selected)
        training = replace(
            _training_config(args),
            epochs=fixed_epochs,
            early_stop_patience=0,
        )
        rows = int(context["features"].shape[0])
        model, result = fit_source_conditioned_cst_fixed(
            context["features"],
            context["candidates"],
            context["sequences"],
            np.zeros(rows, dtype=np.int32),
            model_config=config,
            training_config=training,
            verbose=True,
        )
        save_source_conditioned_checkpoint(model_path, model, result)

    validation = _load_external_context(args, context)
    logits_path = args.output_dir / "full" / "validation-logits.npy"
    if logits_path.exists():
        logits = np.load(logits_path, allow_pickle=False)
    else:
        logits = predict_source_conditioned_logits(
            model,
            validation["features"],
            validation["candidates"],
            validation["sequences"],
            mean=result.mean,
            std=result.std,
            batch_size=args.batch_size,
        )
        _save_array_atomic(logits_path, logits)
    champion = np.load(
        args.champion_validation_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    if champion.shape != logits.shape:
        raise ValueError("champion external scores do not align")
    candidate_metrics = _score_metrics(
        logits,
        validation["features"],
    )
    champion_metrics = _score_metrics(
        champion,
        validation["features"],
    )
    deltas = {
        key: float(candidate_metrics[key] - champion_metrics[key])
        for key in candidate_metrics
    }
    passed = bool(
        deltas["full"] >= EXTERNAL_MINIMUM_DELTA
        and all(deltas[f"time_slice_{index}"] >= 0.0 for index in range(3))
    )
    report = {
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "selected_variant": selected,
        "fixed_epoch_rule": "rounded median of selection-fold best epochs",
        "selection_best_epochs": best_epochs,
        "fixed_epochs": fixed_epochs,
        "candidate_metrics": candidate_metrics,
        "champion_metrics": champion_metrics,
        "delta_vs_champion": deltas,
        "model": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "validation_logits": str(logits_path.resolve()),
        "validation_logits_sha256": _sha256(logits_path),
        "champion_validation_scores": str(
            args.champion_validation_scores.resolve()
        ),
        "champion_validation_scores_sha256": _sha256(
            args.champion_validation_scores
        ),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "submission_generated": False,
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


def _run_fold(
    args: argparse.Namespace,
    context: dict[str, Any],
    variant: str,
    fold: SourceConditionedFold,
) -> dict[str, Any]:
    artifact_dir = (
        args.output_dir
        / "folds"
        / f"variant-{variant}"
        / f"fold-{fold.index}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "report.json"
    if report_path.exists():
        report = _read_json(report_path)
        _verify_fold_report(report)
        print(
            f"[abcd] reuse variant={variant} fold={fold.index} "
            f"mrr={report['score_metrics']['full']:.6f}",
            flush=True,
        )
        return report
    train_stop = int(fold.train_rows[1])
    split = _internal_split(
        context["times"],
        train_stop,
        args.internal_validation_rows,
    )
    model_config = _model_config(args, context, variant)
    training_config = _training_config(args)
    selection_path = artifact_dir / "fit-selection.json"
    if selection_path.exists():
        fit_selection = _read_json(selection_path)
    else:
        print(
            f"[abcd] select epoch variant={variant} fold={fold.index} "
            f"train=[0,{split}) validation=[{split},{train_stop})",
            flush=True,
        )
        selection_model, selection_result = fit_source_conditioned_cst(
            context["features"][:split],
            context["candidates"][:split],
            _slice_sequences(context["sequences"], 0, split),
            np.zeros(split, dtype=np.int32),
            context["features"][split:train_stop],
            context["candidates"][split:train_stop],
            _slice_sequences(
                context["sequences"],
                split,
                train_stop,
            ),
            np.zeros(train_stop - split, dtype=np.int32),
            model_config=model_config,
            training_config=training_config,
            verbose=True,
        )
        fit_selection = {
            "variant": variant,
            "fold": fold.index,
            "internal_train_rows": [0, split],
            "internal_validation_rows": [split, train_stop],
            "internal_train_time_max": int(
                context["times"][split - 1]
            ),
            "internal_validation_time_min": int(
                context["times"][split]
            ),
            "best_epoch": selection_result.best_epoch,
            "best_val_mrr": selection_result.best_val_mrr,
            "history": list(selection_result.history),
        }
        _write_json_atomic(selection_path, fit_selection)
        del selection_model, selection_result
        release_memory()

    fixed_epochs = int(fit_selection["best_epoch"])
    checkpoint_path = artifact_dir / "model.npz"
    if checkpoint_path.exists():
        model, result = load_source_conditioned_checkpoint(checkpoint_path)
    else:
        print(
            f"[abcd] fixed train variant={variant} fold={fold.index} "
            f"rows=[0,{train_stop}) epochs={fixed_epochs}",
            flush=True,
        )
        fixed_training = replace(
            training_config,
            epochs=fixed_epochs,
            early_stop_patience=0,
        )
        model, result = fit_source_conditioned_cst_fixed(
            context["features"][:train_stop],
            context["candidates"][:train_stop],
            _slice_sequences(context["sequences"], 0, train_stop),
            np.zeros(train_stop, dtype=np.int32),
            model_config=model_config,
            training_config=fixed_training,
            verbose=True,
        )
        save_source_conditioned_checkpoint(
            checkpoint_path,
            model,
            result,
        )

    score_start, score_stop = fold.score_rows
    score_sequences = _load_sequence_rows(
        args.sequence_cache_dir,
        f"fold-{fold.index}-score-frozen",
    )
    logits_path = artifact_dir / "score-logits.npy"
    if logits_path.exists():
        logits = np.load(logits_path, allow_pickle=False)
    else:
        print(
            f"[abcd] score variant={variant} fold={fold.index} "
            f"rows=[{score_start},{score_stop})",
            flush=True,
        )
        logits = predict_source_conditioned_logits(
            model,
            context["features"][score_start:score_stop],
            context["candidates"][score_start:score_stop],
            score_sequences,
            mean=result.mean,
            std=result.std,
            batch_size=args.batch_size,
        )
        _save_array_atomic(logits_path, logits)
    metrics = _score_metrics(
        logits,
        context["features"][score_start:score_stop],
    )
    report = {
        "status": "complete",
        "variant": variant,
        "fold": asdict(fold),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "fit_selection": str(selection_path.resolve()),
        "fit_selection_sha256": _sha256(selection_path),
        "fixed_epochs": fixed_epochs,
        "fixed_training_history": list(result.history),
        "score_metrics": metrics,
        "exact_positive_tie_rows": _positive_tie_rows(logits),
        "model": str(checkpoint_path.resolve()),
        "model_sha256": _sha256(checkpoint_path),
        "score_logits": str(logits_path.resolve()),
        "score_logits_sha256": _sha256(logits_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(report_path, report)
    print(
        f"[abcd] complete variant={variant} fold={fold.index} "
        f"mrr={metrics['full']:.6f}",
        flush=True,
    )
    del model, result, logits
    release_memory()
    return report


def _load_external_context(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> dict[str, Any]:
    validation_report = _read_json(args.validation_cache_report)
    base = str(args.validation_cache_prefix)
    paths = {
        "features": Path(f"{base}.val.npy"),
        "candidates": Path(f"{base}.val-candidates.npy"),
        "src": Path(f"{base}.val-src.npy"),
        "time": Path(f"{base}.val-time.npy"),
        "dst": Path(f"{base}.val-dst.npy"),
    }
    features = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
    candidates = np.load(
        paths["candidates"],
        mmap_mode="r",
        allow_pickle=False,
    )
    sources = np.load(paths["src"], mmap_mode="r", allow_pickle=False)
    times = np.load(paths["time"], mmap_mode="r", allow_pickle=False)
    dst = np.load(paths["dst"], mmap_mode="r", allow_pickle=False)
    if (
        candidates.shape != features.shape[:2]
        or sources.shape != features.shape[:1]
        or times.shape != features.shape[:1]
        or dst.shape != features.shape[:1]
        or not np.array_equal(np.asarray(candidates[:, 0]), np.asarray(dst))
    ):
        raise ValueError("external validation sidecars do not align")
    cache_prefix = "external-frozen"
    items_path = args.sequence_cache_dir / f"{cache_prefix}-items.npy"
    if items_path.exists():
        sequences = _load_sequence_rows(
            args.sequence_cache_dir,
            cache_prefix,
        )
    else:
        train_end = int(context["train_report"]["split"]["train_end"])
        interactions = read_interactions(args.train_csv).sort_by_time()
        interactions = interactions[:train_end]
        sequences = build_causal_source_sequences(
            interactions,
            query_src=sources,
            query_time=times,
            max_length=int(context["manifest"]["max_length"]),
        )
        _save_sequence_rows(
            args.sequence_cache_dir,
            cache_prefix,
            sequences,
        )
        _write_json_atomic(
            args.sequence_cache_dir / "external-frozen-report.json",
            {
                "status": "complete",
                "history_rows": train_end,
                "strict_history_rule": (
                    "row < train_end and event_time < query_time"
                ),
                "validation_cache_report": str(
                    args.validation_cache_report.resolve()
                ),
                "validation_cache_report_sha256": _sha256(
                    args.validation_cache_report
                ),
            },
        )
    if sequences.items.shape[0] != features.shape[0]:
        raise ValueError("external source sequences do not align")
    return {
        "validation_report": validation_report,
        "features": features,
        "candidates": candidates,
        "sources": sources,
        "times": times,
        "dst": dst,
        "sequences": sequences,
    }


def _selection_decision(
    reports: dict[str, list[dict[str, Any]]],
) -> str:
    baseline = np.asarray(
        [
            report["score_metrics"]["full"]
            for report in reports["A"]
        ],
        dtype=np.float64,
    )
    preference = {"A": 3, "B": 2, "C": 1, "D": 0}
    eligible: list[tuple[str, float, float]] = []
    for variant in VARIANTS:
        metrics = np.asarray(
            [
                report["score_metrics"]["full"]
                for report in reports[variant]
            ],
            dtype=np.float64,
        )
        deltas = metrics - baseline
        if np.all(deltas >= 0.0):
            eligible.append(
                (
                    variant,
                    float(metrics.mean()),
                    float(deltas.min()),
                )
            )
    return max(
        eligible,
        key=lambda row: (
            row[1],
            row[2],
            preference[row[0]],
        ),
    )[0]


def _model_config(
    args: argparse.Namespace,
    context: dict[str, Any],
    variant: str,
) -> SourceConditionedCSTConfig:
    return abcd_model_config(
        variant,
        input_dim=int(context["features"].shape[-1]),
        num_items=int(context["manifest"]["num_items"]),
        model_dim=args.model_dim,
        heads=args.heads,
        candidate_layers=args.candidate_layers,
        source_layers=args.source_layers,
        source_max_length=int(context["manifest"]["max_length"]),
        dropout=args.dropout,
        feedforward_multiplier=2,
        relative_context="mean_max",
    )


def _training_config(
    args: argparse.Namespace,
) -> SourceConditionedTrainingConfig:
    return SourceConditionedTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        early_stop_patience=args.early_stop_patience,
    )


def _internal_split(
    times: np.ndarray,
    train_stop: int,
    requested_rows: int,
) -> int:
    if requested_rows <= 0 or requested_rows >= train_stop:
        raise ValueError("internal validation rows are invalid")
    target = train_stop - requested_rows
    split = int(
        np.searchsorted(
            times,
            times[target],
            side="left",
        )
    )
    if (
        split <= 0
        or split >= train_stop
        or int(times[split - 1]) >= int(times[split])
    ):
        raise ValueError("internal validation split leaks timestamp")
    return split


def _score_metrics(
    scores: np.ndarray,
    features: Any,
) -> dict[str, float]:
    values = np.asarray(scores)
    positives = np.zeros(values.shape[0], dtype=np.int32)
    result = {
        "full": tie_neutral_mrr(values, positives),
    }
    boundaries = np.linspace(
        0,
        values.shape[0],
        4,
        dtype=np.int64,
    )
    for index in range(3):
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        result[f"time_slice_{index}"] = tie_neutral_mrr(
            values[start:stop],
            positives[start:stop],
        )
    activity = np.asarray(features[:, 0, 6], dtype=np.float64)
    order = np.argsort(activity, kind="stable")
    activity_boundaries = np.linspace(
        0,
        values.shape[0],
        5,
        dtype=np.int64,
    )
    for index in range(4):
        selected = order[
            activity_boundaries[index] : activity_boundaries[index + 1]
        ]
        result[f"activity_q{index + 1}"] = tie_neutral_mrr(
            values[selected],
            positives[selected],
        )
    return {key: float(value) for key, value in result.items()}


def _positive_tie_rows(scores: np.ndarray) -> int:
    values = np.asarray(scores)
    positive = values[:, :1]
    return int(np.sum(np.any(values[:, 1:] == positive, axis=1)))


def _load_sequence_rows(
    directory: Path,
    prefix: str,
) -> SourceSequenceRows:
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


def _save_sequence_rows(
    directory: Path,
    prefix: str,
    rows: SourceSequenceRows,
) -> None:
    _save_array_atomic(
        directory / f"{prefix}-items.npy",
        np.asarray(rows.items, dtype=np.int32),
    )
    _save_array_atomic(
        directory / f"{prefix}-time-buckets.npy",
        np.asarray(rows.time_buckets, dtype=np.int32),
    )
    _save_array_atomic(
        directory / f"{prefix}-lengths.npy",
        np.asarray(rows.lengths, dtype=np.int32),
    )


def _verify_fold_report(report: dict[str, Any]) -> None:
    if (
        report.get("status") != "complete"
        or report.get("trainable_frameworks") != ["jittor"]
        or report.get("non_jittor_trainable_models") != []
    ):
        raise ValueError("cached fold report is invalid")
    for key in ("model", "score_logits"):
        path = Path(report[key])
        if not path.exists() or _sha256(path) != report[f"{key}_sha256"]:
            raise ValueError(f"cached fold artifact differs: {path}")


def _configure_device(device: str) -> None:
    if device == "cuda":
        if not jt.has_cuda:
            raise RuntimeError("CUDA requested but Jittor has no CUDA")
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    jt.set_global_seed(60)


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
