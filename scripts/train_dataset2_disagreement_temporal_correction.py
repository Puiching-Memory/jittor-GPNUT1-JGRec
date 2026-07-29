from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.confidence_routed_topk_id import (
    ConfidenceRouterConfig,
    ConfidenceRouterTrainingConfig,
    SparseRoutingConfig,
    correction_improvement_labels,
    fit_confidence_router,
    hard_confidence_route,
    load_confidence_router_checkpoint,
    save_confidence_router_checkpoint,
)
from jgrec.rankers.hybrid.disagreement_temporal_correction import (
    CorrectionSignal,
    hybrid_consensus_signal,
    oof_disagreement_signal,
    proposal_router_features,
    score_multiset_correction_audit,
    strict_temporal_support_signal,
    topk_score_multiset_proposal,
)
from jgrec.rankers.hybrid.oof_stacking import tie_neutral_mrr
from jgrec.rankers.hybrid.source_sequence_cache import SourceConditionedFold

TOP_K = 10
MAXIMUM_ROUTE_FRACTION = 0.05
ROUTER_HOLDOUT_ROWS = 20_000
RECENT_SUPPORT_ROWS = 20_000
SELECTION_MEAN_DELTA_MIN = 0.0001
GATE_MEAN_DELTA_MIN = 0.0001
ACTIVITY_DELTA_MIN = -0.0005
EXTERNAL_DELTA_MIN = 0.0002


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    signal: str
    top_k: int = TOP_K
    maximum_route_fraction: float = MAXIMUM_ROUTE_FRACTION


CANDIDATES = (
    CandidateConfig("oof-disagreement-top10-route05", "oof-disagreement"),
    CandidateConfig(
        "strict-temporal-support-top10-route05",
        "strict-temporal-support",
    ),
    CandidateConfig("hybrid-consensus-top10-route05", "hybrid-consensus"),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run sparse Dataset2 corrections from OOF disagreement and "
            "strict temporal support."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=("smoke", "selection", "gate", "external", "all"),
        default="all",
    )
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--sequence-cache-dir", required=True, type=Path)
    parser.add_argument("--base-result-dir", required=True, type=Path)
    parser.add_argument("--oof-expert-logits", required=True, type=Path)
    parser.add_argument(
        "--full-validation-expert-logits",
        required=True,
        type=Path,
    )
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
    parser.add_argument("--router-hidden-dim", type=int, default=16)
    parser.add_argument("--router-dropout", type=float, default=0.05)
    parser.add_argument("--router-epochs", type=int, default=8)
    parser.add_argument("--router-learning-rate", type=float, default=0.001)
    parser.add_argument("--router-weight-decay", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--predict-batch-size", type=int, default=1024)
    parser.add_argument("--minimum-route-probability", type=float, default=0.5)
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
    src = np.load(
        f"{prefix}.train-src.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    dst = np.load(
        f"{prefix}.train-dst.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    times = np.load(
        f"{prefix}.train-time.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    oof = np.load(
        args.oof_expert_logits,
        mmap_mode="r",
        allow_pickle=False,
    )
    train_rows = int(manifest["train_rows"])
    candidate_count = int(manifest["candidate_count"])
    expected_scores = (train_rows, candidate_count)
    if (
        manifest.get("status") != "complete"
        or manifest.get("trainable_frameworks") != ["jittor"]
        or manifest.get("non_jittor_trainable_models") != []
        or features.shape[:2] != expected_scores
        or candidates.shape != expected_scores
        or src.shape != (train_rows,)
        or dst.shape != (train_rows,)
        or times.shape != (train_rows,)
        or oof.ndim != 3
        or oof.shape[0] < 2
        or oof.shape[1:] != expected_scores
        or not np.array_equal(
            np.asarray(candidates[:, 0]),
            np.asarray(dst),
        )
        or np.any(np.diff(np.asarray(times, dtype=np.int64)) < 0)
    ):
        raise ValueError("signal correction train cache contract differs")
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
        raise ValueError("signal correction requires three folds")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "features": features,
        "candidates": candidates,
        "src": src,
        "dst": dst,
        "times": times,
        "oof": oof,
        "folds": folds,
    }


def _freeze_protocol(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "status": "frozen_before_training",
        "protocol": "dataset2_oof_temporal_signal_correction_v1",
        "candidates": [asdict(config) for config in CANDIDATES],
        "router_holdout_rows": ROUTER_HOLDOUT_ROWS,
        "recent_support_rows": RECENT_SUPPORT_ROWS,
        "minimum_route_probability": args.minimum_route_probability,
        "selection_rule": {
            "each_fold_delta_min": 0.0,
            "mean_delta_min": SELECTION_MEAN_DELTA_MIN,
        },
        "gate_rule": {
            "fold2_delta_min": 0.0,
            "three_fold_mean_delta_min": GATE_MEAN_DELTA_MIN,
            "activity_delta_min": ACTIVITY_DELTA_MIN,
        },
        "external_rule": {
            "champion_delta_min": EXTERNAL_DELTA_MIN,
            "each_time_slice_delta_min": 0.0,
        },
        "router": {
            "hidden_dim": args.router_hidden_dim,
            "dropout": args.router_dropout,
            "epochs": args.router_epochs,
            "learning_rate": args.router_learning_rate,
            "weight_decay": args.router_weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "train_cache_prefix": str(args.train_cache_prefix.resolve()),
        "fold_manifest": str(context["manifest_path"].resolve()),
        "fold_manifest_sha256": _sha256(context["manifest_path"]),
        "oof_expert_logits": str(args.oof_expert_logits.resolve()),
        "oof_expert_logits_sha256": _sha256(args.oof_expert_logits),
        "external_inputs_blinded_until_gate": {
            "validation_cache_prefix": str(
                args.validation_cache_prefix.resolve()
            ),
            "full_validation_expert_logits": str(
                args.full_validation_expert_logits.resolve()
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
    if path.exists():
        if _read_json(path) != frozen:
            raise ValueError("existing signal correction protocol differs")
        return
    _write_json_atomic(path, frozen)


def _run_smoke(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    path = args.output_dir / "smoke-report.json"
    if path.exists():
        print("[signal-correction] reuse smoke", flush=True)
        return
    history_stop = 61_024
    holdout = slice(history_stop, history_stop + 512)
    score = slice(history_stop + 512, history_stop + 1024)
    reports = []
    for config in CANDIDATES:
        holdout_signal = _build_signal(
            config.signal,
            context,
            expert_logits=context["oof"][:, holdout],
            history_stop=history_stop,
            query_slice=holdout,
            origin_time=int(context["times"][holdout.start]),
        )
        holdout_base = np.asarray(context["oof"][0, holdout])
        holdout_proposal = topk_score_multiset_proposal(
            holdout_base,
            holdout_signal.candidate_scores,
            top_k=config.top_k,
        )
        labels, _ = correction_improvement_labels(
            holdout_base,
            holdout_proposal,
            np.zeros(holdout_base.shape[0], dtype=np.int32),
        )
        features, names = proposal_router_features(
            holdout_base,
            holdout_proposal,
            holdout_signal,
            top_k=config.top_k,
        )
        router, result = fit_confidence_router(
            features,
            labels,
            model_config=ConfidenceRouterConfig(
                input_dim=features.shape[1],
                hidden_dim=8,
                dropout=0.0,
            ),
            training_config=ConfidenceRouterTrainingConfig(
                epochs=1,
                batch_size=128,
                learning_rate=args.router_learning_rate,
                weight_decay=args.router_weight_decay,
                seed=args.seed,
            ),
            verbose=False,
        )
        score_signal = _build_signal(
            config.signal,
            context,
            expert_logits=context["oof"][:, score],
            history_stop=holdout.stop,
            query_slice=score,
            origin_time=int(context["times"][score.start]),
        )
        score_base = np.asarray(context["oof"][0, score])
        proposal = topk_score_multiset_proposal(
            score_base,
            score_signal.candidate_scores,
            top_k=config.top_k,
        )
        score_features, score_names = proposal_router_features(
            score_base,
            proposal,
            score_signal,
            top_k=config.top_k,
        )
        probabilities = result.predict(
            router,
            score_features,
            batch_size=args.predict_batch_size,
        )
        routed = hard_confidence_route(
            score_base,
            proposal,
            probabilities,
            config=SparseRoutingConfig(
                config.maximum_route_fraction,
                args.minimum_route_probability,
            ),
        )
        audit = score_multiset_correction_audit(
            score_base,
            proposal,
            routed.scores,
            routed.route_mask,
            top_k=config.top_k,
            maximum_route_fraction=config.maximum_route_fraction,
        )
        if names != score_names or not audit["passed"]:
            raise RuntimeError("signal correction smoke contract failed")
        reports.append(
            {
                "candidate": asdict(config),
                "feature_names": list(names),
                "router_positive_rows": int(np.sum(labels)),
                "audit": audit,
            }
        )
    _write_json_atomic(
        path,
        {
            "status": "passed",
            "history_stop": history_stop,
            "holdout_rows": [holdout.start, holdout.stop],
            "score_rows": [score.start, score.stop],
            "reports": reports,
        },
    )


def _run_selection(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    reports: dict[str, list[dict[str, Any]]] = {
        config.name: [] for config in CANDIDATES
    }
    for config in CANDIDATES:
        for fold in context["folds"][:2]:
            reports[config.name].append(
                _run_candidate_fold(args, context, config, fold)
            )
    rows = []
    eligible = []
    for config in CANDIDATES:
        deltas = [
            float(report["delta_vs_frozen_base"]["full"])
            for report in reports[config.name]
        ]
        mean_delta = float(np.mean(deltas))
        passed = bool(
            min(deltas) >= 0.0
            and mean_delta >= SELECTION_MEAN_DELTA_MIN
        )
        if passed:
            eligible.append(config)
        rows.append(
            {
                **asdict(config),
                "fold_deltas": deltas,
                "mean_delta": mean_delta,
                "route_fractions": [
                    report["multiset_audit"]["route_fraction"]
                    for report in reports[config.name]
                ],
                "eligible": passed,
            }
        )
    selected = (
        max(
            eligible,
            key=lambda config: next(
                row["mean_delta"]
                for row in rows
                if row["name"] == config.name
            ),
        ).name
        if eligible
        else None
    )
    lock = {
        "status": "locked_before_gate",
        "selected_candidate": selected,
        "selection_passed": selected is not None,
        "gate_metrics_read": False,
        "rows": rows,
    }
    path = args.output_dir / "selection-lock.json"
    if path.exists() and _read_json(path) != lock:
        raise ValueError("existing signal correction selection lock differs")
    if not path.exists():
        _write_json_atomic(path, lock)
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
        raise ValueError("selection lock exposed gate metrics")
    selected_name = lock["selected_candidate"]
    if selected_name is None:
        gate = {
            "status": "rejected",
            "passed": False,
            "selected_candidate": None,
            "reason": "no candidate passed Fold0/1 selection",
            "fold2_evaluated": False,
            "external_evaluation_allowed": False,
        }
    else:
        selected = _candidate_by_name(str(selected_name))
        report = _run_candidate_fold(
            args,
            context,
            selected,
            context["folds"][2],
        )
        selection_row = next(
            row for row in lock["rows"] if row["name"] == selected_name
        )
        selection_deltas = [
            float(value) for value in selection_row["fold_deltas"]
        ]
        fold2_delta = float(report["delta_vs_frozen_base"]["full"])
        three_fold_mean = float(
            np.mean([*selection_deltas, fold2_delta])
        )
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
            and three_fold_mean >= GATE_MEAN_DELTA_MIN
            and min(activity_deltas.values()) >= ACTIVITY_DELTA_MIN
            and report["multiset_audit"]["passed"]
        )
        gate = {
            "status": "passed" if passed else "rejected",
            "passed": passed,
            "selected_candidate": selected_name,
            "selection_deltas": selection_deltas,
            "fold2_delta": fold2_delta,
            "three_fold_mean_delta": three_fold_mean,
            "activity_deltas": activity_deltas,
            "multiset_audit": report["multiset_audit"],
            "fold2_evaluated": True,
            "external_evaluation_allowed": passed,
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
        raise ValueError("existing signal correction gate report differs")
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

    selected = _candidate_by_name(
        str(_read_json(gate_path)["selected_candidate"])
    )
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
    validation_src = np.load(
        f"{validation_prefix}.val-src.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_times = np.load(
        f"{validation_prefix}.val-time.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_experts = np.load(
        args.full_validation_expert_logits,
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_base = np.asarray(validation_experts[0])
    if (
        validation_experts.ndim != 3
        or validation_experts.shape[0] != context["oof"].shape[0]
        or validation_experts.shape[1:] != validation_candidates.shape
        or validation_src.shape != validation_candidates.shape[:1]
        or validation_times.shape != validation_candidates.shape[:1]
        or validation_features.shape[:2] != validation_candidates.shape
    ):
        raise ValueError("external signal correction inputs differ")
    model_artifacts = _ensure_signal_pipeline(
        args,
        context,
        selected,
        fold_name="full",
        train_stop=context["candidates"].shape[0],
        score_base=validation_base,
        score_experts=validation_experts,
        score_candidates=validation_candidates,
        score_src=validation_src,
        score_origin_time=int(validation_times[0]),
    )
    routed = hard_confidence_route(
        validation_base,
        model_artifacts["proposal"],
        model_artifacts["probabilities"],
        config=SparseRoutingConfig(
            selected.maximum_route_fraction,
            args.minimum_route_probability,
        ),
    )
    audit = score_multiset_correction_audit(
        validation_base,
        model_artifacts["proposal"],
        routed.scores,
        routed.route_mask,
        top_k=selected.top_k,
        maximum_route_fraction=selected.maximum_route_fraction,
    )
    if not audit["passed"]:
        raise RuntimeError("external score-multiset correction audit failed")
    metrics = _score_metrics(routed.scores, validation_features)
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
    score_path = args.output_dir / "full" / "validation-scores.npy"
    route_path = args.output_dir / "full" / "route-mask.npy"
    _save_array_atomic(score_path, routed.scores)
    _save_array_atomic(route_path, routed.route_mask)
    report = {
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "selected_candidate": asdict(selected),
        "candidate_metrics": metrics,
        "frozen_base_metrics": base_metrics,
        "champion_metrics": champion_metrics,
        "delta_vs_frozen_base": delta_base,
        "delta_vs_champion": delta_champion,
        "multiset_audit": audit,
        "model_pipeline_report": str(
            model_artifacts["report_path"].resolve()
        ),
        "model_pipeline_report_sha256": _sha256(
            model_artifacts["report_path"]
        ),
        "validation_scores": str(score_path.resolve()),
        "validation_scores_sha256": _sha256(score_path),
        "route_mask": str(route_path.resolve()),
        "route_mask_sha256": _sha256(route_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "submission_generated": False,
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2), flush=True)


def _run_candidate_fold(
    args: argparse.Namespace,
    context: dict[str, Any],
    config: CandidateConfig,
    fold: SourceConditionedFold,
) -> dict[str, Any]:
    directory = (
        args.output_dir / "folds" / config.name / f"fold-{fold.index}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        report = _read_json(report_path)
        _verify_candidate_report(report)
        print(
            f"[signal-correction] reuse {config.name} fold={fold.index} "
            f"delta={report['delta_vs_frozen_base']['full']:+.6f}",
            flush=True,
        )
        return report
    train_stop = int(fold.train_rows[1])
    score_start, score_stop = fold.score_rows
    score_base = np.load(
        args.base_result_dir
        / "folds"
        / "variant-A"
        / f"fold-{fold.index}"
        / "score-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    artifacts = _ensure_signal_pipeline(
        args,
        context,
        config,
        fold_name=f"fold-{fold.index}",
        train_stop=train_stop,
        score_base=score_base,
        score_experts=context["oof"][:, score_start:score_stop],
        score_candidates=context["candidates"][score_start:score_stop],
        score_src=context["src"][score_start:score_stop],
        score_origin_time=int(context["times"][score_start]),
    )
    routed = hard_confidence_route(
        score_base,
        artifacts["proposal"],
        artifacts["probabilities"],
        config=SparseRoutingConfig(
            config.maximum_route_fraction,
            args.minimum_route_probability,
        ),
    )
    audit = score_multiset_correction_audit(
        score_base,
        artifacts["proposal"],
        routed.scores,
        routed.route_mask,
        top_k=config.top_k,
        maximum_route_fraction=config.maximum_route_fraction,
    )
    if not audit["passed"]:
        raise RuntimeError(
            f"score-multiset audit failed: {config.name} fold={fold.index}"
        )
    score_features = context["features"][score_start:score_stop]
    metrics = _score_metrics(routed.scores, score_features)
    base_metrics = _score_metrics(score_base, score_features)
    proposal_metrics = _score_metrics(artifacts["proposal"], score_features)
    deltas = {
        key: float(metrics[key] - base_metrics[key]) for key in metrics
    }
    labels, rewards = correction_improvement_labels(
        score_base,
        artifacts["proposal"],
        np.zeros(score_base.shape[0], dtype=np.int32),
    )
    routed_rows = routed.route_mask
    score_path = directory / "scores.npy"
    route_path = directory / "route-mask.npy"
    _save_array_atomic(score_path, routed.scores)
    _save_array_atomic(route_path, routed.route_mask)
    report = {
        "status": "complete",
        "candidate": asdict(config),
        "fold": asdict(fold),
        "score_metrics": metrics,
        "frozen_base_metrics": base_metrics,
        "ungated_proposal_metrics": proposal_metrics,
        "delta_vs_frozen_base": deltas,
        "multiset_audit": audit,
        "routed_row_outcomes": {
            "improved": int(np.sum(labels[routed_rows] > 0.0)),
            "harmed": int(np.sum(rewards[routed_rows] < 0.0)),
            "neutral": int(np.sum(rewards[routed_rows] == 0.0)),
            "mean_reward": (
                float(np.mean(rewards[routed_rows]))
                if np.any(routed_rows)
                else 0.0
            ),
        },
        "model_pipeline_report": str(artifacts["report_path"].resolve()),
        "model_pipeline_report_sha256": _sha256(artifacts["report_path"]),
        "scores": str(score_path.resolve()),
        "scores_sha256": _sha256(score_path),
        "route_mask": str(route_path.resolve()),
        "route_mask_sha256": _sha256(route_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(report_path, report)
    print(
        f"[signal-correction] complete {config.name} fold={fold.index} "
        f"mrr={metrics['full']:.6f} delta={deltas['full']:+.6f} "
        f"route={audit['route_fraction']:.3%}",
        flush=True,
    )
    return report


def _ensure_signal_pipeline(
    args: argparse.Namespace,
    context: dict[str, Any],
    config: CandidateConfig,
    *,
    fold_name: str,
    train_stop: int,
    score_base: Any,
    score_experts: Any,
    score_candidates: Any,
    score_src: Any,
    score_origin_time: int,
) -> dict[str, Any]:
    directory = args.output_dir / "models" / config.name / fold_name
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    proposal_path = directory / "score-proposal.npy"
    probability_path = directory / "score-route-probabilities.npy"
    if report_path.exists():
        report = _read_json(report_path)
        for key, path in (
            ("proposal_sha256", proposal_path),
            ("probabilities_sha256", probability_path),
        ):
            if report[key] != _sha256(path):
                raise ValueError("cached signal correction artifact differs")
        return {
            "proposal": np.load(
                proposal_path,
                mmap_mode="r",
                allow_pickle=False,
            ),
            "probabilities": np.load(
                probability_path,
                mmap_mode="r",
                allow_pickle=False,
            ),
            "report_path": report_path,
        }
    split = _router_split(
        np.asarray(context["times"][:train_stop]),
        requested_rows=ROUTER_HOLDOUT_ROWS,
    )
    print(
        f"[signal-correction] build {config.name} {fold_name} "
        f"prefix={split} holdout={train_stop - split}",
        flush=True,
    )
    holdout_slice = slice(split, train_stop)
    holdout_base = np.asarray(context["oof"][0, holdout_slice])
    holdout_signal = _build_signal(
        config.signal,
        context,
        expert_logits=context["oof"][:, holdout_slice],
        history_stop=split,
        query_slice=holdout_slice,
        origin_time=int(context["times"][split]),
    )
    holdout_proposal = topk_score_multiset_proposal(
        holdout_base,
        holdout_signal.candidate_scores,
        top_k=config.top_k,
    )
    labels, rewards = correction_improvement_labels(
        holdout_base,
        holdout_proposal,
        np.zeros(train_stop - split, dtype=np.int32),
    )
    router_features, feature_names = proposal_router_features(
        holdout_base,
        holdout_proposal,
        holdout_signal,
        top_k=config.top_k,
    )
    router_path = directory / "router.npz"
    if router_path.exists():
        router, router_result = load_confidence_router_checkpoint(router_path)
    else:
        router, router_result = fit_confidence_router(
            router_features,
            labels,
            model_config=ConfidenceRouterConfig(
                input_dim=router_features.shape[1],
                hidden_dim=args.router_hidden_dim,
                dropout=args.router_dropout,
            ),
            training_config=ConfidenceRouterTrainingConfig(
                epochs=args.router_epochs,
                batch_size=args.batch_size,
                learning_rate=args.router_learning_rate,
                weight_decay=args.router_weight_decay,
                seed=args.seed,
            ),
            verbose=True,
        )
        save_confidence_router_checkpoint(router_path, router, router_result)

    score_signal = _build_signal_from_arrays(
        config.signal,
        context,
        expert_logits=score_experts,
        history_stop=train_stop,
        query_src=score_src,
        candidate_ids=score_candidates,
        origin_time=score_origin_time,
    )
    proposal = topk_score_multiset_proposal(
        score_base,
        score_signal.candidate_scores,
        top_k=config.top_k,
    )
    score_features, score_feature_names = proposal_router_features(
        score_base,
        proposal,
        score_signal,
        top_k=config.top_k,
    )
    if score_feature_names != feature_names:
        raise RuntimeError("router signal feature contract differs")
    probabilities = router_result.predict(
        router,
        score_features,
        batch_size=args.predict_batch_size,
    )
    _save_array_atomic(proposal_path, proposal)
    _save_array_atomic(probability_path, probabilities)
    holdout_base_mrr = tie_neutral_mrr(
        holdout_base,
        np.zeros(holdout_base.shape[0], dtype=np.int32),
    )
    holdout_proposal_mrr = tie_neutral_mrr(
        holdout_proposal,
        np.zeros(holdout_base.shape[0], dtype=np.int32),
    )
    report = {
        "status": "complete",
        "candidate": asdict(config),
        "router_split": [split, train_stop],
        "router_split_time": {
            "history_time_max": int(context["times"][split - 1]),
            "holdout_time_min": int(context["times"][split]),
            "strict": bool(
                int(context["times"][split - 1])
                < int(context["times"][split])
            ),
        },
        "score_origin_time": int(score_origin_time),
        "score_history_time_max": int(context["times"][train_stop - 1]),
        "score_history_strict": bool(
            int(context["times"][train_stop - 1])
            < int(score_origin_time)
        ),
        "oof_supervision_rows": [split, train_stop],
        "router_labels": {
            "positive_rows": int(np.sum(labels)),
            "negative_reward_rows": int(np.sum(rewards < 0.0)),
            "neutral_rows": int(np.sum(rewards == 0.0)),
            "positive_fraction": float(np.mean(labels)),
        },
        "holdout_proposal_delta": float(
            holdout_proposal_mrr - holdout_base_mrr
        ),
        "feature_names": list(feature_names),
        "router": str(router_path.resolve()),
        "router_sha256": _sha256(router_path),
        "proposal": str(proposal_path.resolve()),
        "proposal_sha256": _sha256(proposal_path),
        "probabilities": str(probability_path.resolve()),
        "probabilities_sha256": _sha256(probability_path),
        "probability_summary": {
            "minimum": float(np.min(probabilities)),
            "mean": float(np.mean(probabilities)),
            "maximum": float(np.max(probabilities)),
            "above_minimum_fraction": float(
                np.mean(
                    probabilities >= args.minimum_route_probability
                )
            ),
        },
        "router_training_history": list(router_result.history),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(report_path, report)
    del router
    release_memory()
    return {
        "proposal": proposal,
        "probabilities": probabilities,
        "report_path": report_path,
    }


def _build_signal(
    signal_name: str,
    context: dict[str, Any],
    *,
    expert_logits: Any,
    history_stop: int,
    query_slice: slice,
    origin_time: int,
) -> CorrectionSignal:
    return _build_signal_from_arrays(
        signal_name,
        context,
        expert_logits=expert_logits,
        history_stop=history_stop,
        query_src=context["src"][query_slice],
        candidate_ids=context["candidates"][query_slice],
        origin_time=origin_time,
    )


def _build_signal_from_arrays(
    signal_name: str,
    context: dict[str, Any],
    *,
    expert_logits: Any,
    history_stop: int,
    query_src: Any,
    candidate_ids: Any,
    origin_time: int,
) -> CorrectionSignal:
    oof_signal = None
    temporal_signal = None
    if signal_name in ("oof-disagreement", "hybrid-consensus"):
        oof_signal = oof_disagreement_signal(expert_logits)
    if signal_name in ("strict-temporal-support", "hybrid-consensus"):
        temporal_signal = strict_temporal_support_signal(
            context["src"][:history_stop],
            context["dst"][:history_stop],
            context["times"][:history_stop],
            query_src,
            candidate_ids,
            origin_time=origin_time,
            recent_rows=RECENT_SUPPORT_ROWS,
        )
    if signal_name == "oof-disagreement":
        assert oof_signal is not None
        return oof_signal
    if signal_name == "strict-temporal-support":
        assert temporal_signal is not None
        return temporal_signal
    if signal_name == "hybrid-consensus":
        assert oof_signal is not None and temporal_signal is not None
        return hybrid_consensus_signal(oof_signal, temporal_signal)
    raise ValueError(f"unknown correction signal: {signal_name}")


def _router_split(times: np.ndarray, *, requested_rows: int) -> int:
    if times.ndim != 1 or requested_rows <= 0 or requested_rows >= len(times):
        raise ValueError("router holdout size is invalid")
    target = len(times) - requested_rows
    split = int(np.searchsorted(times, times[target], side="left"))
    if (
        split <= 0
        or split >= len(times)
        or int(times[split - 1]) >= int(times[split])
    ):
        raise ValueError("router time split leaks timestamp")
    return split


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


def _candidate_by_name(name: str) -> CandidateConfig:
    for config in CANDIDATES:
        if config.name == name:
            return config
    raise ValueError(f"unknown candidate config: {name}")


def _verify_candidate_report(report: dict[str, Any]) -> None:
    if (
        report.get("status") != "complete"
        or report.get("trainable_frameworks") != ["jittor"]
        or report.get("non_jittor_trainable_models") != []
        or not report.get("multiset_audit", {}).get("passed")
    ):
        raise ValueError("cached signal correction report is invalid")
    for key in ("scores", "route_mask", "model_pipeline_report"):
        path = Path(report[key])
        if not path.exists() or _sha256(path) != report[f"{key}_sha256"]:
            raise ValueError(f"cached signal correction artifact differs: {path}")


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
