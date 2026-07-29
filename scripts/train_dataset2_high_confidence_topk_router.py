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
from jgrec.rankers.hybrid.high_confidence_topk_router import (
    ROUTER_FEATURE_NAMES,
    BoundedTopKAlternative,
    ResidualAdvantageRouterConfig,
    ResidualAdvantageRouterTrainingConfig,
    audit_bounded_topk_route,
    bounded_topk_alternative,
    fit_residual_advantage_router,
    hard_high_confidence_route,
    load_residual_advantage_router_checkpoint,
    predict_residual_advantages,
    route_reward_targets,
    router_candidate_support_features,
    router_summary_features,
    save_residual_advantage_router_checkpoint,
    timestamp_router_split,
)

TOP_K_VALUES = (5, 10, 20)
SWITCH_CAPS = (0.01, 0.02)
ROUTE_FRACTIONS = (0.005, 0.01, 0.02, 0.03, 0.05)
MAXIMUM_ROUTE_FRACTION = 0.05
SELECTION_MIN_DELTA = 0.0
GATE_MIN_DELTA = 0.0
GATE_WORST_SLICE_MIN = -0.0001


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a pure-Jittor high-confidence router over bounded "
            "short/medium/long OOF residuals."
        ),
    )
    parser.add_argument("--oof-dir", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--predict-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--nonzero-weight", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    started = time.time()
    _configure_device(args.device, args.seed)
    if args.output_dir.exists():
        evaluation_path = args.output_dir / "evaluation-report.json"
        if evaluation_path.exists():
            print(json.dumps(_read_json(evaluation_path), indent=2), flush=True)
            return 0
        raise FileExistsError(f"router output is incomplete: {args.output_dir}")
    build_dir = args.output_dir.with_name(f"{args.output_dir.name}.building")
    if build_dir.exists():
        raise FileExistsError(f"stale router build exists: {build_dir}")
    build_dir.mkdir(parents=True)

    context = _load_context(args)
    _write_frozen_protocol(args, context, build_dir)
    variant_rows = [
        _train_variant(
            args,
            context,
            build_dir,
            top_k=top_k,
            cap=cap,
        )
        for top_k in TOP_K_VALUES
        for cap in SWITCH_CAPS
    ]
    lock = _select_variant(args, context, build_dir, variant_rows)
    gate = _run_gate(args, context, build_dir, lock)
    evaluation = {
        "status": "accepted" if gate["passed"] else "rejected",
        "passed": bool(gate["passed"]),
        "protocol": "default_short_high_confidence_bounded_topk_router_v1",
        "selection": lock,
        "gate": gate,
        "online_champion_changed": False,
        "submission_generated": False,
        "external_evaluated": False,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
        "elapsed_seconds": float(time.time() - started),
    }
    _write_json_atomic(build_dir / "evaluation-report.json", evaluation)
    os.replace(build_dir, args.output_dir)
    print(json.dumps(evaluation, indent=2), flush=True)
    return 0


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.oof_dir / "manifest.json"
    audit_path = args.oof_dir / "audit.json"
    manifest = _read_json(manifest_path)
    audit = _read_json(audit_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("horizon_axis") != ["short", "medium", "long"]
        or manifest.get("trainable_frameworks") != ["jittor"]
        or manifest.get("non_jittor_trainable_models") != []
        or not audit.get("passed")
    ):
        raise ValueError("multi-horizon OOF artifact is not eligible")
    arrays = {
        "residuals": np.load(
            args.oof_dir / "residuals.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        "corrected": np.load(
            args.oof_dir / "corrected-logits.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        "valid": np.load(
            args.oof_dir / "valid-mask.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
        "gaps": np.load(
            args.oof_dir / "gap-days.npy",
            mmap_mode="r",
            allow_pickle=False,
        ),
    }
    expected = (3, 200_000, 100)
    if (
        arrays["residuals"].shape != expected
        or arrays["corrected"].shape != expected
        or arrays["valid"].shape != expected[:2]
        or arrays["gaps"].shape != expected[:2]
    ):
        raise ValueError("multi-horizon OOF array shapes differ")
    common = np.flatnonzero(np.all(arrays["valid"], axis=0))
    if (
        common.size <= 10
        or not np.array_equal(
            common,
            np.arange(common[0], common[-1] + 1),
        )
    ):
        raise ValueError("common multi-horizon coverage is not contiguous")
    start, stop = int(common[0]), int(common[-1]) + 1
    time_path = Path(f"{args.train_cache_prefix}.train-time.npy")
    feature_path = Path(f"{args.train_cache_prefix}.train.npy")
    times = np.load(time_path, mmap_mode="r", allow_pickle=False)
    candidate_features = np.load(
        feature_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    cache_report = _read_json(args.train_cache_report)
    candidate_feature_names = tuple(cache_report.get("feature_names", []))
    if (
        times.shape != (expected[1],)
        or candidate_features.shape != (
            expected[1],
            expected[2],
            len(candidate_feature_names),
        )
        or not candidate_feature_names
    ):
        raise ValueError("router timestamp sidecar does not align")
    split = timestamp_router_split(times[start:stop])
    return {
        **arrays,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "audit_path": audit_path,
        "time_path": time_path,
        "feature_path": feature_path,
        "cache_report_path": args.train_cache_report,
        "candidate_features": candidate_features,
        "candidate_feature_names": candidate_feature_names,
        "times": times,
        "common_start": start,
        "common_stop": stop,
        "split": split,
    }


def _write_frozen_protocol(
    args: argparse.Namespace,
    context: dict[str, Any],
    build_dir: Path,
) -> None:
    protocol = {
        "status": "frozen_before_training",
        "protocol": "default_short_high_confidence_bounded_topk_router_v1",
        "default_path": "short corrected logits",
        "alternative_path": (
            "short corrected + project_topk("
            "alternative_residual - short_residual)"
        ),
        "top_k_values": list(TOP_K_VALUES),
        "switch_caps": list(SWITCH_CAPS),
        "route_fractions": list(ROUTE_FRACTIONS),
        "maximum_route_fraction": MAXIMUM_ROUTE_FRACTION,
        "selection_min_delta": SELECTION_MIN_DELTA,
        "gate_min_delta": GATE_MIN_DELTA,
        "gate_worst_slice_min": GATE_WORST_SLICE_MIN,
        "temporal_split": asdict(context["split"]),
        "common_rows": [
            context["common_start"],
            context["common_stop"],
        ],
        "summary_feature_names": list(ROUTER_FEATURE_NAMES),
        "candidate_feature_names": list(
            context["candidate_feature_names"]
        ),
        "model": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "candidate_id_features": False,
            "positive_column_features": False,
            "candidate_permutation_invariant": True,
        },
        "training": asdict(_training_config(args)),
        "selection_rule": (
            "maximize selection delta, then worst slice, then lower "
            "coverage/cap/top_k"
        ),
        "gate_rule": {
            "delta_min": GATE_MIN_DELTA,
            "worst_time_slice_min": GATE_WORST_SLICE_MIN,
            "maximum_route_fraction": MAXIMUM_ROUTE_FRACTION,
        },
        "oof_manifest": str(context["manifest_path"].resolve()),
        "oof_manifest_sha256": _sha256(context["manifest_path"]),
        "oof_audit": str(context["audit_path"].resolve()),
        "oof_audit_sha256": _sha256(context["audit_path"]),
        "time_sidecar": str(context["time_path"].resolve()),
        "time_sidecar_sha256": _sha256(context["time_path"]),
        "feature_cache": str(context["feature_path"].resolve()),
        "feature_cache_report": str(
            context["cache_report_path"].resolve()
        ),
        "feature_cache_report_sha256": _sha256(
            context["cache_report_path"]
        ),
        "gate_policy": (
            "gate targets and selected gate metrics are computed only "
            "after selection-lock.json is persisted"
        ),
        "external_inputs_read": False,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(build_dir / "frozen-config.json", protocol)


def _train_variant(
    args: argparse.Namespace,
    context: dict[str, Any],
    build_dir: Path,
    *,
    top_k: int,
    cap: float,
) -> dict[str, Any]:
    label = _variant_label(top_k, cap)
    directory = build_dir / "variants" / label
    directory.mkdir(parents=True)
    arrays = _variant_arrays(context, top_k=top_k, cap=cap)
    split = context["split"]
    train = slice(*split.train_rows)
    selection = slice(*split.selection_rows)
    selection_stop = int(split.selection_rows[1])
    rewards_until_selection = route_reward_targets(
        arrays["default"][:selection_stop],
        (
            arrays["medium"].scores[:selection_stop],
            arrays["long"].scores[:selection_stop],
        ),
        np.zeros(selection_stop, dtype=np.int32),
    )
    oracle = {
        "train": _oracle_metrics(
            rewards_until_selection[train],
            maximum_fraction=MAXIMUM_ROUTE_FRACTION,
        ),
        "selection": _oracle_metrics(
            rewards_until_selection[selection],
            maximum_fraction=MAXIMUM_ROUTE_FRACTION,
        ),
        "gate": "not_read_before_selection_lock",
    }
    model, result = fit_residual_advantage_router(
        arrays["features"][train],
        rewards_until_selection[train],
        model_config=ResidualAdvantageRouterConfig(
            input_dim=arrays["features"].shape[1],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ),
        training_config=_training_config(args),
        feature_names=arrays["feature_names"],
        verbose=False,
    )
    checkpoint = directory / "model.npz"
    save_residual_advantage_router_checkpoint(checkpoint, model, result)
    predictions = predict_residual_advantages(
        model,
        arrays["features"][selection],
        mean=result.mean,
        std=result.std,
        reward_scale=result.training_config.reward_scale,
        batch_size=args.predict_batch_size,
    )
    loaded, loaded_result = load_residual_advantage_router_checkpoint(
        checkpoint
    )
    replay = predict_residual_advantages(
        loaded,
        arrays["features"][selection],
        mean=loaded_result.mean,
        std=loaded_result.std,
        reward_scale=loaded_result.training_config.reward_scale,
        batch_size=args.predict_batch_size,
    )
    replay_error = float(np.max(np.abs(replay - predictions)))
    if replay_error > 1e-6:
        raise RuntimeError(f"{label} checkpoint replay failed: {replay_error}")

    scans = _selection_scans(
        arrays["default"][selection],
        (
            arrays["medium"].scores[selection],
            arrays["long"].scores[selection],
        ),
        (
            _slice_alternative(arrays["medium"], selection),
            _slice_alternative(arrays["long"], selection),
        ),
        predictions,
        cap=cap,
    )
    best_scan = max(
        scans,
        key=lambda row: (
            row["metrics"]["delta_mrr"],
            row["metrics"]["worst_time_slice_delta"],
            -row["metrics"]["route_fraction"],
        ),
    )
    report = {
        "status": "complete",
        "variant": label,
        "top_k": top_k,
        "switch_cap": cap,
        "oracle": oracle,
        "training_rows": result.training_rows,
        "training_history": list(result.history),
        "nonzero_training_targets": result.nonzero_targets,
        "checkpoint": str(
            (
                args.output_dir
                / "variants"
                / label
                / "model.npz"
            ).resolve()
        ),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_replay_error": replay_error,
        "selection_scans": scans,
        "best_selection": best_scan,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(directory / "report.json", report)
    print(
        f"[router] {label} "
        f"oracle_sel={oracle['selection']['delta_mrr']:+.6f} "
        f"actual_sel={best_scan['metrics']['delta_mrr']:+.6f} "
        f"coverage={best_scan['metrics']['route_fraction']:.4%}",
        flush=True,
    )
    del model, loaded, predictions, replay, arrays
    release_memory()
    return report


def _selection_scans(
    default: np.ndarray,
    alternatives: tuple[np.ndarray, np.ndarray],
    alternative_records: tuple[Any, Any],
    predictions: np.ndarray,
    *,
    cap: float,
) -> list[dict[str, Any]]:
    rows = np.arange(predictions.shape[0])
    best = np.argmax(predictions, axis=1)
    best_advantage = predictions[rows, best]
    other = predictions[rows, 1 - best]
    confidence = best_advantage - np.maximum(other, 0.0)
    eligible_confidence = np.sort(
        confidence[(best_advantage > 0.0) & (confidence > 0.0)]
    )[::-1]
    scans = []
    for fraction in ROUTE_FRACTIONS:
        quota = math.floor(default.shape[0] * fraction)
        if quota > 0 and eligible_confidence.size:
            threshold = float(
                eligible_confidence[
                    min(quota, eligible_confidence.size) - 1
                ]
            )
        else:
            threshold = 1e9
        routed = hard_high_confidence_route(
            default,
            alternatives,
            predictions,
            minimum_confidence=threshold,
            maximum_route_fraction=fraction,
        )
        audit = audit_bounded_topk_route(
            default,
            alternative_records,
            routed,
            cap=cap,
            maximum_route_fraction=fraction,
        )
        if not audit["passed"]:
            raise RuntimeError("selection bounded route audit failed")
        scans.append(
            {
                "maximum_route_fraction": fraction,
                "minimum_confidence": threshold,
                "metrics": _route_metrics(
                    default,
                    routed.scores,
                    routed.route_index,
                ),
                "audit": audit,
            }
        )
    return scans


def _select_variant(
    args: argparse.Namespace,
    context: dict[str, Any],
    build_dir: Path,
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = []
    for report in reports:
        row = report["best_selection"]
        metrics = row["metrics"]
        if (
            metrics["delta_mrr"] > SELECTION_MIN_DELTA
            and metrics["route_fraction"] <= MAXIMUM_ROUTE_FRACTION
        ):
            candidates.append((report, row))
    if not candidates:
        raise RuntimeError("no router variant passed selection")
    report, scan = max(
        candidates,
        key=lambda value: (
            value[1]["metrics"]["delta_mrr"],
            value[1]["metrics"]["worst_time_slice_delta"],
            -value[1]["metrics"]["route_fraction"],
            -value[0]["switch_cap"],
            -value[0]["top_k"],
        ),
    )
    lock = {
        "status": "locked_before_gate",
        "selected_variant": report["variant"],
        "top_k": report["top_k"],
        "switch_cap": report["switch_cap"],
        "maximum_route_fraction": scan["maximum_route_fraction"],
        "minimum_confidence": scan["minimum_confidence"],
        "selection_metrics": scan["metrics"],
        "selection_audit": scan["audit"],
        "selection_rule": (
            "maximum positive delta, then worst slice, then lower "
            "coverage/cap/top_k"
        ),
        "checkpoint": report["checkpoint"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "variant_reports": [
            {
                "variant": row["variant"],
                "best_selection": row["best_selection"],
            }
            for row in reports
        ],
        "gate_metrics_read": False,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(build_dir / "selection-lock.json", lock)
    print(
        f"[router] locked {lock['selected_variant']} "
        f"delta={lock['selection_metrics']['delta_mrr']:+.6f} "
        f"coverage={lock['selection_metrics']['route_fraction']:.4%}",
        flush=True,
    )
    return lock


def _run_gate(
    args: argparse.Namespace,
    context: dict[str, Any],
    build_dir: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    persisted = _read_json(build_dir / "selection-lock.json")
    if persisted != lock or persisted.get("gate_metrics_read") is not False:
        raise RuntimeError("selection lock was not frozen before gate")
    arrays = _variant_arrays(
        context,
        top_k=int(lock["top_k"]),
        cap=float(lock["switch_cap"]),
    )
    gate = slice(*context["split"].gate_rows)
    checkpoint = (
        build_dir
        / "variants"
        / lock["selected_variant"]
        / "model.npz"
    )
    model, result = load_residual_advantage_router_checkpoint(checkpoint)
    predictions = predict_residual_advantages(
        model,
        arrays["features"][gate],
        mean=result.mean,
        std=result.std,
        reward_scale=result.training_config.reward_scale,
        batch_size=args.predict_batch_size,
    )
    routed = hard_high_confidence_route(
        arrays["default"][gate],
        (
            arrays["medium"].scores[gate],
            arrays["long"].scores[gate],
        ),
        predictions,
        minimum_confidence=float(lock["minimum_confidence"]),
        maximum_route_fraction=float(lock["maximum_route_fraction"]),
    )
    audit = audit_bounded_topk_route(
        arrays["default"][gate],
        (
            _slice_alternative(arrays["medium"], gate),
            _slice_alternative(arrays["long"], gate),
        ),
        routed,
        cap=float(lock["switch_cap"]),
        maximum_route_fraction=float(lock["maximum_route_fraction"]),
    )
    metrics = _route_metrics(
        arrays["default"][gate],
        routed.scores,
        routed.route_index,
    )
    rewards = route_reward_targets(
        arrays["default"][gate],
        (
            arrays["medium"].scores[gate],
            arrays["long"].scores[gate],
        ),
        np.zeros(predictions.shape[0], dtype=np.int32),
    )
    oracle = _oracle_metrics(
        rewards,
        maximum_fraction=MAXIMUM_ROUTE_FRACTION,
    )
    passed = bool(
        audit["passed"]
        and metrics["delta_mrr"] >= GATE_MIN_DELTA
        and metrics["route_fraction"] <= MAXIMUM_ROUTE_FRACTION
        and metrics["worst_time_slice_delta"] >= GATE_WORST_SLICE_MIN
    )
    score_path = build_dir / "gate-scores.npy"
    route_path = build_dir / "gate-route-index.npy"
    prediction_path = build_dir / "gate-predicted-advantages.npy"
    _save_array_atomic(score_path, routed.scores)
    _save_array_atomic(route_path, routed.route_index)
    _save_array_atomic(prediction_path, predictions)
    report = {
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "selected_variant": lock["selected_variant"],
        "gate_rows": list(context["split"].gate_rows),
        "global_gate_rows": [
            context["common_start"] + context["split"].gate_rows[0],
            context["common_start"] + context["split"].gate_rows[1],
        ],
        "metrics": metrics,
        "oracle": oracle,
        "audit": audit,
        "checkpoint_replay_source": str(Path(lock["checkpoint"]).resolve()),
        "gate_scores": str(
            (args.output_dir / "gate-scores.npy").resolve()
        ),
        "gate_scores_sha256": _sha256(score_path),
        "gate_route_index": str(
            (args.output_dir / "gate-route-index.npy").resolve()
        ),
        "gate_route_index_sha256": _sha256(route_path),
        "gate_predicted_advantages": str(
            (args.output_dir / "gate-predicted-advantages.npy").resolve()
        ),
        "gate_predicted_advantages_sha256": _sha256(prediction_path),
        "external_evaluated": False,
        "submission_generated": False,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(build_dir / "gate-report.json", report)
    print(
        f"[router] gate status={report['status']} "
        f"delta={metrics['delta_mrr']:+.6f} "
        f"coverage={metrics['route_fraction']:.4%}",
        flush=True,
    )
    del model, arrays, predictions, routed
    release_memory()
    return report


def _variant_arrays(
    context: dict[str, Any],
    *,
    top_k: int,
    cap: float,
) -> dict[str, Any]:
    start = context["common_start"]
    stop = context["common_stop"]
    default = np.asarray(context["corrected"][0, start:stop])
    residuals = np.asarray(context["residuals"][:, start:stop])
    gaps = np.asarray(context["gaps"][:, start:stop])
    medium = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[1],
        top_k=top_k,
        cap=cap,
    )
    long = bounded_topk_alternative(
        default,
        residuals[0],
        residuals[2],
        top_k=top_k,
        cap=cap,
    )
    summary_features, summary_names = router_summary_features(
        default,
        residuals,
        gaps,
        (medium.scores, long.scores),
    )
    candidate_features, candidate_names = (
        router_candidate_support_features(
            context["candidate_features"][start:stop],
            context["candidate_feature_names"],
            default,
            (medium.scores, long.scores),
        )
    )
    if summary_names != ROUTER_FEATURE_NAMES:
        raise RuntimeError("router feature contract changed")
    features = np.concatenate(
        (summary_features, candidate_features),
        axis=1,
    ).astype(np.float32, copy=False)
    feature_names = summary_names + candidate_names
    return {
        "default": default,
        "residuals": residuals,
        "gaps": gaps,
        "medium": medium,
        "long": long,
        "features": features,
        "feature_names": feature_names,
    }


def _oracle_metrics(
    rewards: np.ndarray,
    *,
    maximum_fraction: float,
) -> dict[str, Any]:
    values = np.asarray(rewards, dtype=np.float64)
    best_route = np.argmax(values, axis=1)
    best_reward = values[np.arange(values.shape[0]), best_route]
    positive = np.flatnonzero(best_reward > 0.0)
    quota = math.floor(values.shape[0] * maximum_fraction)
    selected = positive[
        np.argsort(-best_reward[positive], kind="stable")
    ][:quota]
    return {
        "rows": int(values.shape[0]),
        "delta_mrr": float(np.sum(best_reward[selected]) / values.shape[0]),
        "route_fraction": float(selected.size / values.shape[0]),
        "positive_opportunity_fraction": float(
            np.mean(best_reward > 0.0)
        ),
        "medium_rows": int(np.sum(best_route[selected] == 0)),
        "long_rows": int(np.sum(best_route[selected] == 1)),
        "all_route_mean_reward": float(np.mean(values)),
    }


def _route_metrics(
    default_scores: np.ndarray,
    routed_scores: np.ndarray,
    route_index: np.ndarray,
) -> dict[str, Any]:
    default_rr = _row_rr(default_scores)
    routed_rr = _row_rr(routed_scores)
    reward = routed_rr - default_rr
    boundaries = np.linspace(0, reward.size, 4, dtype=np.int64)
    slice_deltas = [
        float(np.mean(reward[boundaries[index] : boundaries[index + 1]]))
        for index in range(3)
    ]
    return {
        "rows": int(reward.size),
        "default_short_mrr": float(np.mean(default_rr)),
        "routed_mrr": float(np.mean(routed_rr)),
        "delta_mrr": float(np.mean(reward)),
        "time_slice_deltas": slice_deltas,
        "worst_time_slice_delta": float(min(slice_deltas)),
        "route_fraction": float(np.mean(route_index > 0)),
        "routed_rows": int(np.sum(route_index > 0)),
        "medium_rows": int(np.sum(route_index == 1)),
        "long_rows": int(np.sum(route_index == 2)),
        "gain_rows": int(np.sum(reward > 0.0)),
        "loss_rows": int(np.sum(reward < 0.0)),
        "unchanged_rows": int(np.sum(reward == 0.0)),
    }


def _row_rr(scores: np.ndarray) -> np.ndarray:
    positive = scores[:, :1]
    greater = np.sum(scores > positive, axis=1)
    equal = np.sum(scores == positive, axis=1)
    return 1.0 / (greater + (equal + 1.0) / 2.0)


def _training_config(
    args: argparse.Namespace,
) -> ResidualAdvantageRouterTrainingConfig:
    return ResidualAdvantageRouterTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        reward_scale=args.reward_scale,
        nonzero_weight=args.nonzero_weight,
        seed=args.seed,
    )


def _variant_label(top_k: int, cap: float) -> str:
    return f"topk-{top_k}-cap-{cap:.2f}"


def _slice_alternative(
    alternative: BoundedTopKAlternative,
    rows: slice,
) -> BoundedTopKAlternative:
    return BoundedTopKAlternative(
        scores=alternative.scores[rows],
        delta=alternative.delta[rows],
        topk_mask=alternative.topk_mask[rows],
        top_k=alternative.top_k,
        cap=alternative.cap,
    )


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
