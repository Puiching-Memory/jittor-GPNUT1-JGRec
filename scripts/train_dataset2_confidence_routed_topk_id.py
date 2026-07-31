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
from jgrec.rankers.hybrid.candidate_set_transformer import (
    load_candidate_set_checkpoint,
    predict_candidate_set_logits,
)
from jgrec.rankers.hybrid.confidence_routed_topk_id import (
    ROUTER_FEATURE_NAMES,
    ConfidenceRouterConfig,
    ConfidenceRouterTrainingConfig,
    SparseRoutingConfig,
    TopKIDCorrectionConfig,
    TopKIDCorrectionTrainingConfig,
    confidence_router_features,
    correction_improvement_labels,
    fit_confidence_router,
    fit_topk_id_correction_fixed,
    hard_confidence_route,
    load_confidence_router_checkpoint,
    load_topk_id_correction_checkpoint,
    predict_topk_id_correction,
    save_confidence_router_checkpoint,
    save_topk_id_correction_checkpoint,
    sparse_correction_audit,
    topk_mask_from_scores,
)
from jgrec.rankers.hybrid.oof_stacking import tie_neutral_mrr
from jgrec.rankers.hybrid.source_sequence_cache import SourceConditionedFold

CAP = 0.10
ROUTER_HOLDOUT_ROWS = 20_000
SELECTION_MEAN_DELTA_MIN = 0.0001
GATE_MEAN_DELTA_MIN = 0.0001
ACTIVITY_DELTA_MIN = -0.0005
EXTERNAL_DELTA_MIN = 0.0002


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    top_k: int
    maximum_route_fraction: float


CANDIDATES = (
    CandidateConfig("top5-route05", 5, 0.05),
    CandidateConfig("top10-route05", 10, 0.05),
    CandidateConfig("top10-route10", 10, 0.10),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run confidence-routed sparse top-k ID corrections.",
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
    parser.add_argument("--correction-dropout", type=float, default=0.10)
    parser.add_argument("--correction-epochs", type=int, default=3)
    parser.add_argument("--correction-learning-rate", type=float, default=0.01)
    parser.add_argument("--correction-weight-decay", type=float, default=0.001)
    parser.add_argument("--router-hidden-dim", type=int, default=16)
    parser.add_argument("--router-dropout", type=float, default=0.05)
    parser.add_argument("--router-epochs", type=int, default=8)
    parser.add_argument("--router-learning-rate", type=float, default=0.001)
    parser.add_argument("--router-weight-decay", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--predict-batch-size", type=int, default=512)
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
        manifest.get("status") != "complete"
        or manifest.get("trainable_frameworks") != ["jittor"]
        or manifest.get("non_jittor_trainable_models") != []
        or features.shape
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
        raise ValueError("confidence-routed train cache contract differs")
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
        raise ValueError("confidence routing requires three folds")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "features": features,
        "candidates": candidates,
        "times": times,
        "dst": dst,
        "folds": folds,
        "num_items": int(
            max(
                int(manifest["num_items"]),
                int(np.max(candidates)),
            )
        ),
    }


def _freeze_protocol(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_artifacts = []
    for fold in context["folds"]:
        base_report = args.base_cache_dir / f"fold-{fold.index}" / "report.json"
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
        base_artifacts.append(
            {
                "fold": asdict(fold),
                "base_report": str(base_report.resolve()),
                "base_report_sha256": _sha256(base_report),
                "train_base": str(train_base.resolve()),
                "train_base_sha256": _sha256(train_base),
                "score_base": str(base_score.resolve()),
                "score_base_sha256": _sha256(base_score),
            }
        )
    frozen = {
        "status": "frozen_before_training",
        "protocol": "dataset2_confidence_routed_topk_id_v1",
        "candidates": [asdict(value) for value in CANDIDATES],
        "absolute_cap": CAP,
        "router_holdout_rows": ROUTER_HOLDOUT_ROWS,
        "minimum_route_probability": args.minimum_route_probability,
        "correction_model": asdict(
            _correction_model_config(args, context)
        ),
        "correction_training": asdict(_correction_training_config(args)),
        "router_model": asdict(_router_model_config(args)),
        "router_training": asdict(_router_training_config(args)),
        "router_features": list(ROUTER_FEATURE_NAMES),
        "router_feature_contract": "label_free_at_inference",
        "router_label": "proposal reciprocal-rank strictly improves base",
        "selection_folds": [0, 1],
        "gate_fold": 2,
        "selection_rule": {
            "each_fold_delta_min": 0.0,
            "two_fold_mean_delta_min": SELECTION_MEAN_DELTA_MIN,
        },
        "gate_rule": {
            "fold2_delta_min": 0.0,
            "three_fold_mean_delta_min": GATE_MEAN_DELTA_MIN,
            "worst_activity_delta_min": ACTIVITY_DELTA_MIN,
            "sparsity_audit_required": True,
        },
        "external_rule": {
            "read_only_after_gate": True,
            "full_delta_vs_champion_min": EXTERNAL_DELTA_MIN,
            "all_time_slices_non_negative": True,
        },
        "base_artifacts": base_artifacts,
        "manifest": str(context["manifest_path"].resolve()),
        "manifest_sha256": _sha256(context["manifest_path"]),
        "positive_index": 0,
        "metric": "tie_neutral_mrr",
        "device": args.device,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    frozen = json.loads(json.dumps(frozen, sort_keys=True))
    path = args.output_dir / "frozen-config.json"
    if path.exists():
        if _read_json(path) != frozen:
            raise ValueError("existing confidence routing protocol differs")
        return
    _write_json_atomic(path, frozen)


def _run_smoke(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> None:
    path = args.output_dir / "smoke-report.json"
    if path.exists():
        print("[confidence-topk] reuse smoke", flush=True)
        return
    fold = context["folds"][0]
    base = np.load(
        args.base_cache_dir / "fold-0" / "train-base-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    rows = min(1536, int(base.shape[0]))
    split = 1024
    top_k = 10
    mask = topk_mask_from_scores(base[:rows], top_k=top_k)
    prefix_model, _ = fit_topk_id_correction_fixed(
        base[:split],
        context["candidates"][:split],
        mask[:split],
        np.zeros(split, dtype=np.int32),
        model_config=_correction_model_config(args, context),
        training_config=TopKIDCorrectionTrainingConfig(
            epochs=1,
            batch_size=128,
            learning_rate=args.correction_learning_rate,
            weight_decay=args.correction_weight_decay,
            seed=args.seed,
        ),
        verbose=True,
    )
    proposal = predict_topk_id_correction(
        prefix_model,
        base[split:rows],
        context["candidates"][split:rows],
        mask[split:rows],
        batch_size=args.predict_batch_size,
    )
    labels, rewards = correction_improvement_labels(
        base[split:rows],
        proposal,
        np.zeros(rows - split, dtype=np.int32),
    )
    if len(np.unique(labels)) < 2:
        labels = np.arange(labels.shape[0], dtype=np.int32) % 2
    support = _item_support(
        context["dst"][:split],
        context["num_items"],
    )
    features, names = confidence_router_features(
        base[split:rows],
        proposal,
        context["candidates"][split:rows],
        support,
        mask[split:rows],
    )
    router, router_result = fit_confidence_router(
        features,
        labels,
        model_config=_router_model_config(args),
        training_config=ConfidenceRouterTrainingConfig(
            epochs=1,
            batch_size=128,
            learning_rate=args.router_learning_rate,
            weight_decay=args.router_weight_decay,
            seed=args.seed,
        ),
        verbose=True,
    )
    probabilities = router_result.predict(
        router,
        features,
        batch_size=args.predict_batch_size,
    )
    routed = hard_confidence_route(
        base[split:rows],
        proposal,
        probabilities,
        config=SparseRoutingConfig(0.10, 0.5),
    )
    audit = sparse_correction_audit(
        base[split:rows],
        proposal,
        routed.scores,
        mask[split:rows],
        routed.route_mask,
        cap=CAP,
        maximum_route_fraction=0.10,
    )
    if names != ROUTER_FEATURE_NAMES or not audit["passed"]:
        raise RuntimeError("confidence routing smoke contract failed")
    _write_json_atomic(
        path,
        {
            "status": "passed",
            "fold": fold.index,
            "rows": rows,
            "router_positive_rows": int(labels.sum()),
            "reward_nonzero_rows": int(np.sum(rewards != 0.0)),
            "audit": audit,
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
                    report["sparsity_audit"]["route_fraction"]
                    for report in reports[config.name]
                ],
                "eligible": passed,
            }
        )
    selected = (
        max(
            eligible,
            key=lambda config: (
                next(
                    row["mean_delta"]
                    for row in rows
                    if row["name"] == config.name
                ),
                -config.maximum_route_fraction,
                -config.top_k,
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
        "created_at_unix": time.time(),
    }
    path = args.output_dir / "selection-lock.json"
    if path.exists():
        existing = _read_json(path)
        for value in (existing, lock):
            value.pop("created_at_unix", None)
        if existing != lock:
            raise ValueError("existing confidence selection lock differs")
    else:
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
    ordered = list(CANDIDATES)
    if selected_name is not None:
        selected = _candidate_by_name(str(selected_name))
        ordered.remove(selected)
        ordered.insert(0, selected)
    fold = context["folds"][2]
    reports = {
        config.name: _run_candidate_fold(
            args,
            context,
            config,
            fold,
        )
        for config in ordered
    }
    if selected_name is None:
        gate = {
            "status": "rejected",
            "passed": False,
            "selected_candidate": None,
            "reason": "no candidate passed Fold0/1 selection",
        }
    else:
        selected_report = reports[str(selected_name)]
        selection_row = next(
            row
            for row in lock["rows"]
            if row["name"] == selected_name
        )
        selection_deltas = [
            float(value) for value in selection_row["fold_deltas"]
        ]
        forward_delta = float(
            selected_report["delta_vs_frozen_base"]["full"]
        )
        three_fold_mean = float(
            np.mean([*selection_deltas, forward_delta])
        )
        activity_deltas = {
            key: float(
                selected_report["delta_vs_frozen_base"][key]
            )
            for key in (
                "activity_q1",
                "activity_q2",
                "activity_q3",
                "activity_q4",
            )
        }
        passed = bool(
            forward_delta >= 0.0
            and three_fold_mean >= GATE_MEAN_DELTA_MIN
            and min(activity_deltas.values()) >= ACTIVITY_DELTA_MIN
            and selected_report["sparsity_audit"]["passed"]
        )
        gate = {
            "status": "passed" if passed else "rejected",
            "passed": passed,
            "selected_candidate": selected_name,
            "selection_deltas": selection_deltas,
            "fold2_delta": forward_delta,
            "three_fold_mean_delta": three_fold_mean,
            "activity_deltas": activity_deltas,
            "sparsity_audit": selected_report["sparsity_audit"],
            "thresholds": {
                "fold2_delta_min": 0.0,
                "three_fold_mean_delta_min": GATE_MEAN_DELTA_MIN,
                "activity_delta_min": ACTIVITY_DELTA_MIN,
            },
        }
    gate["diagnostics"] = {
        name: {
            "delta": report["delta_vs_frozen_base"],
            "sparsity_audit": report["sparsity_audit"],
        }
        for name, report in reports.items()
    }
    gate["selection_lock"] = str(lock_path.resolve())
    gate["selection_lock_sha256"] = _sha256(lock_path)
    path = args.output_dir / "gate-report.json"
    if path.exists() and _read_json(path) != gate:
        raise ValueError("existing confidence gate report differs")
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
    train_base_path = args.base_cache_dir / "full-train-base-logits.npy"
    full_base_model, full_base_result = load_candidate_set_checkpoint(
        args.full_base_checkpoint
    )
    if train_base_path.exists():
        train_base = np.load(
            train_base_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    else:
        values = predict_candidate_set_logits(
            full_base_model,
            context["features"],
            mean=full_base_result.mean,
            std=full_base_result.std,
            batch_size=args.predict_batch_size,
        )
        _save_array_atomic(train_base_path, values)
        train_base = np.load(
            train_base_path,
            mmap_mode="r",
            allow_pickle=False,
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
    expert_logits = np.load(
        args.full_base_validation_expert_logits,
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_base = expert_logits[0]
    replay = predict_candidate_set_logits(
        full_base_model,
        validation_features,
        mean=full_base_result.mean,
        std=full_base_result.std,
        batch_size=args.predict_batch_size,
    )
    replay_error = float(np.max(np.abs(replay - validation_base)))
    if replay_error > 2e-5:
        raise RuntimeError("full frozen CST replay differs")
    model_artifacts = _ensure_model_pipeline(
        args,
        context,
        top_k=selected.top_k,
        fold_name="full",
        train_base=train_base,
        train_candidates=context["candidates"],
        train_dst=context["dst"],
        train_times=context["times"],
        score_base=validation_base,
        score_candidates=validation_candidates,
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
    score_mask = topk_mask_from_scores(
        validation_base,
        top_k=selected.top_k,
    )
    audit = sparse_correction_audit(
        validation_base,
        model_artifacts["proposal"],
        routed.scores,
        score_mask,
        routed.route_mask,
        cap=CAP,
        maximum_route_fraction=selected.maximum_route_fraction,
    )
    if not audit["passed"]:
        raise RuntimeError("external sparse correction audit failed")
    candidate_metrics = _score_metrics(routed.scores, validation_features)
    base_metrics = _score_metrics(validation_base, validation_features)
    champion = np.load(
        args.champion_validation_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    champion_metrics = _score_metrics(champion, validation_features)
    delta_base = {
        key: float(candidate_metrics[key] - base_metrics[key])
        for key in candidate_metrics
    }
    delta_champion = {
        key: float(candidate_metrics[key] - champion_metrics[key])
        for key in candidate_metrics
    }
    passed = bool(
        delta_champion["full"] >= EXTERNAL_DELTA_MIN
        and all(
            delta_champion[f"time_slice_{index}"] >= 0.0
            for index in range(3)
        )
    )
    scores_path = args.output_dir / "full" / "validation-scores.npy"
    route_path = args.output_dir / "full" / "route-mask.npy"
    _save_array_atomic(scores_path, routed.scores)
    _save_array_atomic(route_path, routed.route_mask)
    report = {
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "selected_candidate": asdict(selected),
        "candidate_metrics": candidate_metrics,
        "frozen_base_metrics": base_metrics,
        "champion_metrics": champion_metrics,
        "delta_vs_frozen_base": delta_base,
        "delta_vs_champion": delta_champion,
        "sparsity_audit": audit,
        "base_replay_max_absolute_error": replay_error,
        "model_pipeline_report": str(
            model_artifacts["report_path"].resolve()
        ),
        "model_pipeline_report_sha256": _sha256(
            model_artifacts["report_path"]
        ),
        "validation_scores": str(scores_path.resolve()),
        "validation_scores_sha256": _sha256(scores_path),
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
        args.output_dir
        / "folds"
        / config.name
        / f"fold-{fold.index}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    if report_path.exists():
        report = _read_json(report_path)
        _verify_candidate_report(report)
        print(
            f"[confidence-topk] reuse {config.name} fold={fold.index} "
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
    score_base = np.load(
        args.base_result_dir
        / "folds"
        / "variant-A"
        / f"fold-{fold.index}"
        / "score-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    model_artifacts = _ensure_model_pipeline(
        args,
        context,
        top_k=config.top_k,
        fold_name=f"fold-{fold.index}",
        train_base=train_base,
        train_candidates=context["candidates"][:train_stop],
        train_dst=context["dst"][:train_stop],
        train_times=context["times"][:train_stop],
        score_base=score_base,
        score_candidates=context["candidates"][score_start:score_stop],
    )
    routed = hard_confidence_route(
        score_base,
        model_artifacts["proposal"],
        model_artifacts["probabilities"],
        config=SparseRoutingConfig(
            config.maximum_route_fraction,
            args.minimum_route_probability,
        ),
    )
    score_mask = topk_mask_from_scores(
        score_base,
        top_k=config.top_k,
    )
    audit = sparse_correction_audit(
        score_base,
        model_artifacts["proposal"],
        routed.scores,
        score_mask,
        routed.route_mask,
        cap=CAP,
        maximum_route_fraction=config.maximum_route_fraction,
    )
    if not audit["passed"]:
        raise RuntimeError(
            f"sparse correction audit failed: {config.name} "
            f"fold={fold.index}"
        )
    features = context["features"][score_start:score_stop]
    metrics = _score_metrics(routed.scores, features)
    base_metrics = _score_metrics(score_base, features)
    proposal_metrics = _score_metrics(
        model_artifacts["proposal"],
        features,
    )
    deltas = {
        key: float(metrics[key] - base_metrics[key])
        for key in metrics
    }
    labels, rewards = correction_improvement_labels(
        score_base,
        model_artifacts["proposal"],
        np.zeros(score_base.shape[0], dtype=np.int32),
    )
    routed_rows = routed.route_mask
    scores_path = directory / "scores.npy"
    route_path = directory / "route-mask.npy"
    _save_array_atomic(scores_path, routed.scores)
    _save_array_atomic(route_path, routed.route_mask)
    report = {
        "status": "complete",
        "candidate": asdict(config),
        "fold": asdict(fold),
        "score_metrics": metrics,
        "frozen_base_metrics": base_metrics,
        "ungated_proposal_metrics": proposal_metrics,
        "delta_vs_frozen_base": deltas,
        "sparsity_audit": audit,
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
        "model_pipeline_report": str(
            model_artifacts["report_path"].resolve()
        ),
        "model_pipeline_report_sha256": _sha256(
            model_artifacts["report_path"]
        ),
        "scores": str(scores_path.resolve()),
        "scores_sha256": _sha256(scores_path),
        "route_mask": str(route_path.resolve()),
        "route_mask_sha256": _sha256(route_path),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(report_path, report)
    print(
        f"[confidence-topk] complete {config.name} fold={fold.index} "
        f"mrr={metrics['full']:.6f} delta={deltas['full']:+.6f} "
        f"route={audit['route_fraction']:.3%}",
        flush=True,
    )
    return report


def _ensure_model_pipeline(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    top_k: int,
    fold_name: str,
    train_base: Any,
    train_candidates: Any,
    train_dst: Any,
    train_times: Any,
    score_base: Any,
    score_candidates: Any,
) -> dict[str, Any]:
    directory = args.output_dir / "models" / f"top-{top_k}" / fold_name
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
                raise ValueError("cached confidence model artifact differs")
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
        np.asarray(train_times),
        requested_rows=ROUTER_HOLDOUT_ROWS,
    )
    print(
        f"[confidence-topk] build model top_k={top_k} "
        f"{fold_name} prefix={split} holdout={len(train_times) - split}",
        flush=True,
    )
    train_mask = topk_mask_from_scores(train_base, top_k=top_k)
    prefix_path = directory / "prefix-correction.npz"
    if prefix_path.exists():
        prefix_model, prefix_result = (
            load_topk_id_correction_checkpoint(prefix_path)
        )
    else:
        prefix_model, prefix_result = fit_topk_id_correction_fixed(
            train_base[:split],
            train_candidates[:split],
            train_mask[:split],
            np.zeros(split, dtype=np.int32),
            model_config=_correction_model_config(args, context),
            training_config=_correction_training_config(args),
            verbose=True,
        )
        save_topk_id_correction_checkpoint(
            prefix_path,
            prefix_model,
            prefix_result,
        )
    holdout_proposal = predict_topk_id_correction(
        prefix_model,
        train_base[split:],
        train_candidates[split:],
        train_mask[split:],
        batch_size=args.predict_batch_size,
    )
    labels, rewards = correction_improvement_labels(
        train_base[split:],
        holdout_proposal,
        np.zeros(len(train_times) - split, dtype=np.int32),
    )
    support_prefix = _item_support(
        train_dst[:split],
        context["num_items"],
    )
    router_features, feature_names = confidence_router_features(
        train_base[split:],
        holdout_proposal,
        train_candidates[split:],
        support_prefix,
        train_mask[split:],
    )
    if feature_names != ROUTER_FEATURE_NAMES:
        raise RuntimeError("router feature contract differs")
    router_path = directory / "router.npz"
    if router_path.exists():
        router, router_result = load_confidence_router_checkpoint(
            router_path
        )
    else:
        router, router_result = fit_confidence_router(
            router_features,
            labels,
            model_config=_router_model_config(args),
            training_config=_router_training_config(args),
            verbose=True,
        )
        save_confidence_router_checkpoint(
            router_path,
            router,
            router_result,
        )
    full_path = directory / "full-correction.npz"
    if full_path.exists():
        full_model, full_result = load_topk_id_correction_checkpoint(
            full_path
        )
    else:
        full_model, full_result = fit_topk_id_correction_fixed(
            train_base,
            train_candidates,
            train_mask,
            np.zeros(len(train_times), dtype=np.int32),
            model_config=_correction_model_config(args, context),
            training_config=_correction_training_config(args),
            verbose=True,
        )
        save_topk_id_correction_checkpoint(
            full_path,
            full_model,
            full_result,
        )
    score_mask = topk_mask_from_scores(score_base, top_k=top_k)
    proposal = predict_topk_id_correction(
        full_model,
        score_base,
        score_candidates,
        score_mask,
        batch_size=args.predict_batch_size,
    )
    support_full = _item_support(train_dst, context["num_items"])
    score_features, score_feature_names = confidence_router_features(
        score_base,
        proposal,
        score_candidates,
        support_full,
        score_mask,
    )
    if score_feature_names != ROUTER_FEATURE_NAMES:
        raise RuntimeError("score router feature contract differs")
    probabilities = router_result.predict(
        router,
        score_features,
        batch_size=args.predict_batch_size,
    )
    _save_array_atomic(proposal_path, proposal)
    _save_array_atomic(probability_path, probabilities)
    base_holdout_mrr = tie_neutral_mrr(
        np.asarray(train_base[split:]),
        np.zeros(len(train_times) - split, dtype=np.int32),
    )
    proposal_holdout_mrr = tie_neutral_mrr(
        holdout_proposal,
        np.zeros(len(train_times) - split, dtype=np.int32),
    )
    report = {
        "status": "complete",
        "top_k": top_k,
        "router_split": [split, len(train_times)],
        "router_split_time": {
            "prefix_time_max": int(train_times[split - 1]),
            "holdout_time_min": int(train_times[split]),
            "strict": bool(
                int(train_times[split - 1]) < int(train_times[split])
            ),
        },
        "router_labels": {
            "positive_rows": int(labels.sum()),
            "negative_reward_rows": int(np.sum(rewards < 0.0)),
            "neutral_rows": int(np.sum(rewards == 0.0)),
            "positive_fraction": float(labels.mean()),
        },
        "holdout_proposal_delta": float(
            proposal_holdout_mrr - base_holdout_mrr
        ),
        "prefix_correction": str(prefix_path.resolve()),
        "prefix_correction_sha256": _sha256(prefix_path),
        "router": str(router_path.resolve()),
        "router_sha256": _sha256(router_path),
        "full_correction": str(full_path.resolve()),
        "full_correction_sha256": _sha256(full_path),
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
                    probabilities
                    >= args.minimum_route_probability
                )
            ),
        },
        "prefix_training_history": list(prefix_result.history),
        "router_training_history": list(router_result.history),
        "full_training_history": list(full_result.history),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(report_path, report)
    del prefix_model, router, full_model
    release_memory()
    return {
        "proposal": proposal,
        "probabilities": probabilities,
        "report_path": report_path,
    }


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


def _item_support(dst: Any, num_items: int) -> np.ndarray:
    values = np.asarray(dst, dtype=np.int64)
    if (
        values.ndim != 1
        or np.any(values < 0)
        or np.any(values > num_items)
    ):
        raise ValueError("item support destinations are invalid")
    return np.bincount(values, minlength=num_items + 1).astype(
        np.int64,
        copy=False,
    )


def _correction_model_config(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> TopKIDCorrectionConfig:
    return TopKIDCorrectionConfig(
        num_items=context["num_items"],
        embedding_dim=args.embedding_dim,
        cap=CAP,
        dropout=args.correction_dropout,
    )


def _correction_training_config(
    args: argparse.Namespace,
) -> TopKIDCorrectionTrainingConfig:
    return TopKIDCorrectionTrainingConfig(
        epochs=args.correction_epochs,
        batch_size=args.batch_size,
        learning_rate=args.correction_learning_rate,
        weight_decay=args.correction_weight_decay,
        seed=args.seed,
    )


def _router_model_config(
    args: argparse.Namespace,
) -> ConfidenceRouterConfig:
    return ConfidenceRouterConfig(
        input_dim=len(ROUTER_FEATURE_NAMES),
        hidden_dim=args.router_hidden_dim,
        dropout=args.router_dropout,
    )


def _router_training_config(
    args: argparse.Namespace,
) -> ConfidenceRouterTrainingConfig:
    return ConfidenceRouterTrainingConfig(
        epochs=args.router_epochs,
        batch_size=args.batch_size,
        learning_rate=args.router_learning_rate,
        weight_decay=args.router_weight_decay,
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
        or not report.get("sparsity_audit", {}).get("passed")
    ):
        raise ValueError("cached confidence candidate report is invalid")
    for key in ("scores", "route_mask", "model_pipeline_report"):
        path = Path(report[key])
        if not path.exists() or _sha256(path) != report[f"{key}_sha256"]:
            raise ValueError(f"cached confidence artifact differs: {path}")


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
