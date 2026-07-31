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
    audit_bounded_topk_route,
    bounded_topk_alternative,
    timestamp_router_split,
)
from jgrec.rankers.hybrid.oof_utility_router import (
    OOFUtilityRouterConfig,
    OOFUtilityRouterTrainingConfig,
    UtilityPrediction,
    action_utility_features,
    fit_oof_utility_router,
    load_oof_utility_router_checkpoint,
    predict_oof_utility,
    route_by_expected_utility,
    save_oof_utility_router_checkpoint,
)

TOP_K = 10
SWITCH_CAP = 0.02
ROUTE_FRACTIONS = (0.0025, 0.005, 0.008, 0.01)
CHANGE_PROBABILITY_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
MAXIMUM_ROUTE_FRACTION = 0.01
MINIMUM_UTILITY = 0.0
MINIMUM_ROUTED_CHANGE_FRACTION = 0.12
SELECTION_MIN_DELTA = 0.0
GATE_MIN_DELTA = 0.0
SELECTION_WORST_SLICE_MIN = 0.0
GATE_WORST_SLICE_MIN = 0.0
HARD_NEGATIVE_FRACTION = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the pure-Jittor Dataset2 OOF Utility Router v3 over "
            "frozen bounded short/medium/long decoder actions."
        )
    )
    parser.add_argument("--oof-dir", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--predict-batch-size", type=int, default=2048)
    parser.add_argument("--feature-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--change-positive-weight", type=float, default=8.0)
    parser.add_argument("--loss-direction-weight", type=float, default=3.0)
    parser.add_argument("--magnitude-weight", type=float, default=0.5)
    parser.add_argument("--hard-negative-weight", type=float, default=4.0)
    parser.add_argument("--regret-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    started = time.time()
    _configure_device(args.device, args.seed)
    if args.output_dir.exists():
        report_path = args.output_dir / "evaluation-report.json"
        if report_path.exists():
            print(json.dumps(_read_json(report_path), indent=2), flush=True)
            return 0
        raise FileExistsError(
            f"utility router output is incomplete: {args.output_dir}"
        )
    build_dir = args.output_dir.with_name(
        f"{args.output_dir.name}.building"
    )
    if build_dir.exists():
        raise FileExistsError(
            f"stale utility router build exists: {build_dir}"
        )
    build_dir.mkdir(parents=True)

    context = _load_context(args)
    _write_frozen_protocol(args, context, build_dir)
    training = _build_training_data(args, context)
    selection = _build_common_span(
        args,
        context,
        global_start=context["selection_start"],
        global_stop=context["gate_start"],
    )
    model, result, mining = _fit_with_hard_negative_mining(
        args,
        training,
    )
    checkpoint = build_dir / "model.npz"
    save_oof_utility_router_checkpoint(checkpoint, model, result)
    selection_predictions = _predict_actions(
        model,
        result,
        selection["features"],
        batch_size=args.predict_batch_size,
    )
    replay_model, replay_result = load_oof_utility_router_checkpoint(
        checkpoint
    )
    replay = _predict_actions(
        replay_model,
        replay_result,
        selection["features"],
        batch_size=args.predict_batch_size,
    )
    replay_error = _prediction_error(selection_predictions, replay)
    if replay_error > 1e-7:
        raise RuntimeError(
            f"utility router checkpoint replay failed: {replay_error}"
        )
    del replay_model, replay_result, replay
    release_memory()

    scans = _selection_scans(selection, selection_predictions)
    lock = _lock_selection(
        build_dir,
        checkpoint,
        scans,
        training,
        mining,
        replay_error,
    )
    if lock is None:
        evaluation = {
            "status": "rejected",
            "passed": False,
            "protocol": "dataset2_oof_utility_router_v3",
            "selection": {
                "status": "no_candidate_passed",
                "gate_metrics_read": False,
                "scans": scans,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "training": training["report"],
                "hard_negative_mining": mining,
            },
            "gate": "not_read",
            "phase2": {
                "status": "blocked_v3_failed",
                "started": False,
            },
            "online_champion_changed": False,
            "submission_generated": False,
            "external_evaluated": False,
            "trainable_frameworks": ["jittor"],
            "non_jittor_trainable_models": [],
            "elapsed_seconds": float(time.time() - started),
        }
        _write_json_atomic(
            build_dir / "evaluation-report.json",
            evaluation,
        )
        os.replace(build_dir, args.output_dir)
        print(json.dumps(evaluation, indent=2), flush=True)
        return 0

    del selection, selection_predictions, training, model
    release_memory()
    gate = _run_gate(args, context, build_dir, lock)
    passed = bool(gate["passed"])
    evaluation = {
        "status": "accepted" if passed else "rejected",
        "passed": passed,
        "protocol": "dataset2_oof_utility_router_v3",
        "selection": lock,
        "gate": gate,
        "phase2": {
            "status": "eligible_not_started" if passed
            else "blocked_v3_failed",
            "started": False,
        },
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
    action_ranges: list[tuple[int, int]] = []
    for action_axis in (1, 2):
        valid = np.flatnonzero(
            arrays["valid"][0] & arrays["valid"][action_axis]
        )
        if (
            valid.size <= 10
            or not np.array_equal(
                valid,
                np.arange(valid[0], valid[-1] + 1),
            )
        ):
            raise ValueError("action OOF coverage is not contiguous")
        action_ranges.append((int(valid[0]), int(valid[-1]) + 1))
    common_start = max(start for start, _stop in action_ranges)
    common_stop = min(stop for _start, stop in action_ranges)
    if common_start >= common_stop:
        raise ValueError("common action coverage is empty")

    time_path = Path(f"{args.train_cache_prefix}.train-time.npy")
    feature_path = Path(f"{args.train_cache_prefix}.train.npy")
    times = np.load(time_path, mmap_mode="r", allow_pickle=False)
    candidate_features = np.load(
        feature_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    cache_report = _read_json(args.train_cache_report)
    feature_names = tuple(cache_report.get("feature_names", []))
    if (
        times.shape != (expected[1],)
        or candidate_features.shape
        != (expected[1], expected[2], len(feature_names))
        or not feature_names
    ):
        raise ValueError("utility router sidecars do not align")
    split = timestamp_router_split(times[common_start:common_stop])
    selection_start = common_start + split.selection_rows[0]
    gate_start = common_start + split.gate_rows[0]
    return {
        **arrays,
        "manifest_path": manifest_path,
        "audit_path": audit_path,
        "time_path": time_path,
        "feature_path": feature_path,
        "cache_report_path": args.train_cache_report,
        "candidate_features": candidate_features,
        "candidate_feature_names": feature_names,
        "times": times,
        "action_ranges": tuple(action_ranges),
        "common_start": common_start,
        "common_stop": common_stop,
        "split": split,
        "selection_start": selection_start,
        "gate_start": gate_start,
    }


def _write_frozen_protocol(
    args: argparse.Namespace,
    context: dict[str, Any],
    build_dir: Path,
) -> None:
    protocol = {
        "status": "frozen_before_training",
        "protocol": "dataset2_oof_utility_router_v3",
        "default_path": "short corrected logits",
        "alternative_path": (
            "short corrected + bounded_topk("
            "action_residual - short_residual)"
        ),
        "top_k": TOP_K,
        "switch_cap": SWITCH_CAP,
        "route_fractions": list(ROUTE_FRACTIONS),
        "change_probability_thresholds": list(
            CHANGE_PROBABILITY_THRESHOLDS
        ),
        "minimum_utility": MINIMUM_UTILITY,
        "maximum_route_fraction": MAXIMUM_ROUTE_FRACTION,
        "minimum_routed_change_fraction": (
            MINIMUM_ROUTED_CHANGE_FRACTION
        ),
        "selection_min_delta": SELECTION_MIN_DELTA,
        "gate_min_delta": GATE_MIN_DELTA,
        "selection_worst_slice_min": SELECTION_WORST_SLICE_MIN,
        "gate_worst_slice_min": GATE_WORST_SLICE_MIN,
        "hard_negative_fraction": HARD_NEGATIVE_FRACTION,
        "temporal_split": asdict(context["split"]),
        "global_selection_rows": [
            context["selection_start"],
            context["gate_start"],
        ],
        "global_gate_rows": [
            context["gate_start"],
            context["common_stop"],
        ],
        "action_training_ranges": {
            "medium": [
                context["action_ranges"][0][0],
                context["selection_start"],
            ],
            "long": [
                context["action_ranges"][1][0],
                context["selection_start"],
            ],
        },
        "gate_disclosure": (
            "diagnostic replay: this interval was observed by prior "
            "experiments and is not claimed as statistically unseen"
        ),
        "model": {
            "type": "gain_nochange_loss_hurdle_mlp",
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "outputs": [
                "change_logit",
                "gain_given_change_logit",
                "gain_magnitude",
                "loss_magnitude",
            ],
            "experts_frozen": True,
            "candidate_id_features": False,
            "positive_column_features": False,
            "candidate_permutation_invariant": True,
        },
        "training": asdict(_training_config(args, epochs=args.epochs)),
        "warmup_epochs": args.warmup_epochs,
        "selection_rule": (
            "both consecutive selection slices positive, no negative time "
            "slice, routed RR-change fraction >= 12%, coverage <= 1%; "
            "maximize delta then worst slice then change hit rate then "
            "lower coverage and higher change threshold"
        ),
        "gate_policy": (
            "selection-lock.json is persisted before gate features, "
            "rewards, metrics, or oracle are computed"
        ),
        "phase2_policy": (
            "change-only LambdaMRR is blocked unless v3 gate passes every "
            "frozen criterion"
        ),
        "oof_manifest": str(context["manifest_path"].resolve()),
        "oof_manifest_sha256": _sha256(context["manifest_path"]),
        "oof_audit": str(context["audit_path"].resolve()),
        "oof_audit_sha256": _sha256(context["audit_path"]),
        "feature_cache": str(context["feature_path"].resolve()),
        "feature_cache_report": str(
            context["cache_report_path"].resolve()
        ),
        "feature_cache_report_sha256": _sha256(
            context["cache_report_path"]
        ),
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(build_dir / "frozen-config.json", protocol)


def _build_training_data(
    args: argparse.Namespace,
    context: dict[str, Any],
) -> dict[str, Any]:
    action_rows = []
    feature_names: tuple[str, ...] | None = None
    for action_index, (valid_start, _valid_stop) in enumerate(
        context["action_ranges"]
    ):
        span = _build_action_span(
            args,
            context,
            action_index=action_index,
            global_start=valid_start,
            global_stop=context["selection_start"],
        )
        if feature_names is None:
            feature_names = span["feature_names"]
        elif span["feature_names"] != feature_names:
            raise RuntimeError("utility action feature contracts differ")
        action_rows.append(span)
        print(
            f"[utility-v3] action={action_index + 1} "
            f"training_rows={span['features'].shape[0]} "
            f"gain={np.sum(span['rewards'] > 0.0)} "
            f"loss={np.sum(span['rewards'] < 0.0)}",
            flush=True,
        )
    assert feature_names is not None
    features = np.concatenate(
        [span["features"] for span in action_rows],
        axis=0,
    ).astype(np.float32, copy=False)
    rewards = np.concatenate(
        [span["rewards"] for span in action_rows],
        axis=0,
    ).astype(np.float32, copy=False)
    report = {
        "rows": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "medium_rows": int(action_rows[0]["features"].shape[0]),
        "long_rows": int(action_rows[1]["features"].shape[0]),
        "gain_rows": int(np.sum(rewards > 0.0)),
        "no_change_rows": int(np.sum(rewards == 0.0)),
        "loss_rows": int(np.sum(rewards < 0.0)),
        "maximum_global_row_exclusive": context["selection_start"],
    }
    return {
        "features": features,
        "rewards": rewards,
        "feature_names": feature_names,
        "report": report,
    }


def _build_common_span(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    global_start: int,
    global_stop: int,
) -> dict[str, Any]:
    spans = tuple(
        _build_action_span(
            args,
            context,
            action_index=action_index,
            global_start=global_start,
            global_stop=global_stop,
        )
        for action_index in (0, 1)
    )
    if (
        spans[0]["feature_names"] != spans[1]["feature_names"]
        or not np.array_equal(spans[0]["default"], spans[1]["default"])
    ):
        raise RuntimeError("common action spans do not align")
    features = np.stack(
        (spans[0]["features"], spans[1]["features"]),
        axis=1,
    )
    rewards = np.column_stack(
        (spans[0]["rewards"], spans[1]["rewards"])
    ).astype(np.float32, copy=False)
    return {
        "global_start": global_start,
        "global_stop": global_stop,
        "times": np.asarray(context["times"][global_start:global_stop]),
        "default": spans[0]["default"],
        "alternatives": (
            spans[0]["alternative"],
            spans[1]["alternative"],
        ),
        "features": features,
        "feature_names": spans[0]["feature_names"],
        "rewards": rewards,
        "available": np.ones(rewards.shape, dtype=bool),
    }


def _build_action_span(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    action_index: int,
    global_start: int,
    global_stop: int,
) -> dict[str, Any]:
    action_axis = action_index + 1
    if (
        action_index not in (0, 1)
        or global_start < context["action_ranges"][action_index][0]
        or global_stop > context["action_ranges"][action_index][1]
        or global_start >= global_stop
        or not np.all(
            context["valid"][
                (0, action_axis),
                global_start:global_stop,
            ]
        )
    ):
        raise ValueError("utility action span is unavailable")
    default = np.asarray(
        context["corrected"][0, global_start:global_stop],
        dtype=np.float32,
    )
    short_residual = np.asarray(
        context["residuals"][0, global_start:global_stop],
        dtype=np.float32,
    )
    action_residual = np.asarray(
        context["residuals"][action_axis, global_start:global_stop],
        dtype=np.float32,
    )
    alternative = bounded_topk_alternative(
        default,
        short_residual,
        action_residual,
        top_k=TOP_K,
        cap=SWITCH_CAP,
    )
    short_gap = np.asarray(
        context["gaps"][0, global_start:global_stop],
        dtype=np.float32,
    )
    action_gap = np.asarray(
        context["gaps"][action_axis, global_start:global_stop],
        dtype=np.float32,
    )
    features, feature_names = _chunked_action_features(
        args,
        context,
        global_start=global_start,
        default=default,
        short_residual=short_residual,
        action_residual=action_residual,
        short_gap=short_gap,
        action_gap=action_gap,
        alternative_scores=alternative.scores,
        action_index=action_index,
    )
    rewards = _row_rr(alternative.scores) - _row_rr(default)
    return {
        "default": default,
        "alternative": alternative,
        "features": features,
        "feature_names": feature_names,
        "rewards": np.asarray(rewards, dtype=np.float32),
    }


def _chunked_action_features(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    global_start: int,
    default: np.ndarray,
    short_residual: np.ndarray,
    action_residual: np.ndarray,
    short_gap: np.ndarray,
    action_gap: np.ndarray,
    alternative_scores: np.ndarray,
    action_index: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    chunks = []
    feature_names: tuple[str, ...] | None = None
    for start in range(0, default.shape[0], args.feature_batch_size):
        stop = min(start + args.feature_batch_size, default.shape[0])
        chunk, names = action_utility_features(
            default[start:stop],
            short_residual[start:stop],
            action_residual[start:stop],
            short_gap[start:stop],
            action_gap[start:stop],
            alternative_scores[start:stop],
            context["candidate_features"][
                global_start + start : global_start + stop
            ],
            context["candidate_feature_names"],
            action_index=action_index,
        )
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("utility feature names changed between chunks")
        chunks.append(chunk)
    assert feature_names is not None
    return np.concatenate(chunks, axis=0), feature_names


def _fit_with_hard_negative_mining(
    args: argparse.Namespace,
    training: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    model_config = OOFUtilityRouterConfig(
        input_dim=training["features"].shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    warmup_config = _training_config(args, epochs=args.warmup_epochs)
    warmup, warmup_result = fit_oof_utility_router(
        training["features"],
        training["rewards"],
        model_config=model_config,
        training_config=warmup_config,
        feature_names=training["feature_names"],
        verbose=True,
    )
    warmup_predictions = predict_oof_utility(
        warmup,
        training["features"],
        result=warmup_result,
        batch_size=args.predict_batch_size,
    )
    non_gain = training["rewards"] <= 0.0
    non_gain_indices = np.flatnonzero(non_gain)
    quota = max(
        1,
        math.floor(training["rewards"].size * HARD_NEGATIVE_FRACTION),
    )
    ranked = non_gain_indices[
        np.argsort(
            -warmup_predictions.expected_utility[non_gain_indices],
            kind="stable",
        )
    ]
    hard = np.zeros(training["rewards"].shape, dtype=bool)
    hard[ranked[:quota]] = True
    hard[training["rewards"] < 0.0] = True
    mining = {
        "warmup_epochs": args.warmup_epochs,
        "candidate_fraction": HARD_NEGATIVE_FRACTION,
        "candidate_quota": quota,
        "hard_negative_rows": int(np.sum(hard)),
        "hard_no_change_rows": int(
            np.sum(hard & (training["rewards"] == 0.0))
        ),
        "hard_loss_rows": int(
            np.sum(hard & (training["rewards"] < 0.0))
        ),
        "maximum_warmup_utility": float(
            np.max(warmup_predictions.expected_utility)
        ),
    }
    del warmup, warmup_result, warmup_predictions
    release_memory()
    model, result = fit_oof_utility_router(
        training["features"],
        training["rewards"],
        model_config=model_config,
        training_config=_training_config(args, epochs=args.epochs),
        feature_names=training["feature_names"],
        hard_negative_mask=hard,
        verbose=True,
    )
    return model, result, mining


def _predict_actions(
    model: Any,
    result: Any,
    features: np.ndarray,
    *,
    batch_size: int,
) -> UtilityPrediction:
    if features.ndim != 3 or features.shape[1] != 2:
        raise ValueError("utility action prediction features must be [N,2,F]")
    flat = features.reshape(-1, features.shape[2])
    prediction = predict_oof_utility(
        model,
        flat,
        result=result,
        batch_size=batch_size,
    )
    shape = (features.shape[0], 2)
    return UtilityPrediction(
        change_probability=prediction.change_probability.reshape(shape),
        gain_probability_given_change=(
            prediction.gain_probability_given_change.reshape(shape)
        ),
        expected_gain=prediction.expected_gain.reshape(shape),
        expected_loss=prediction.expected_loss.reshape(shape),
        expected_utility=prediction.expected_utility.reshape(shape),
    )


def _selection_scans(
    selection: dict[str, Any],
    prediction: UtilityPrediction,
) -> list[dict[str, Any]]:
    scans = []
    for fraction in ROUTE_FRACTIONS:
        for threshold in CHANGE_PROBABILITY_THRESHOLDS:
            routed = route_by_expected_utility(
                selection["default"],
                tuple(
                    alternative.scores
                    for alternative in selection["alternatives"]
                ),
                prediction.expected_utility,
                prediction.change_probability,
                available=selection["available"],
                minimum_utility=MINIMUM_UTILITY,
                minimum_change_probability=threshold,
                maximum_route_fraction=fraction,
            )
            audit = audit_bounded_topk_route(
                selection["default"],
                selection["alternatives"],
                routed,
                cap=SWITCH_CAP,
                maximum_route_fraction=fraction,
            )
            if not audit["passed"]:
                raise RuntimeError("utility selection safety audit failed")
            metrics = _route_metrics(
                selection["default"],
                routed.scores,
                routed.route_index,
                slice_count=2,
            )
            scans.append(
                {
                    "maximum_route_fraction": fraction,
                    "minimum_change_probability": threshold,
                    "minimum_utility": MINIMUM_UTILITY,
                    "metrics": metrics,
                    "audit": audit,
                }
            )
    return scans


def _lock_selection(
    build_dir: Path,
    checkpoint: Path,
    scans: list[dict[str, Any]],
    training: dict[str, Any],
    mining: dict[str, Any],
    replay_error: float,
) -> dict[str, Any] | None:
    candidates = [
        scan
        for scan in scans
        if (
            scan["metrics"]["delta_mrr"] > SELECTION_MIN_DELTA
            and scan["metrics"]["worst_time_slice_delta"]
            >= SELECTION_WORST_SLICE_MIN
            and scan["metrics"]["routed_change_fraction"]
            >= MINIMUM_ROUTED_CHANGE_FRACTION
            and scan["metrics"]["route_fraction"]
            <= MAXIMUM_ROUTE_FRACTION
        )
    ]
    if not candidates:
        return None
    selected = max(
        candidates,
        key=lambda row: (
            row["metrics"]["delta_mrr"],
            row["metrics"]["worst_time_slice_delta"],
            row["metrics"]["routed_change_fraction"],
            -row["metrics"]["route_fraction"],
            row["minimum_change_probability"],
        ),
    )
    lock = {
        "status": "locked_before_gate",
        "top_k": TOP_K,
        "switch_cap": SWITCH_CAP,
        "maximum_route_fraction": selected["maximum_route_fraction"],
        "minimum_change_probability": selected[
            "minimum_change_probability"
        ],
        "minimum_utility": selected["minimum_utility"],
        "selection_metrics": selected["metrics"],
        "selection_audit": selected["audit"],
        "selection_rule": (
            "positive on both consecutive slices, >=12% routed change "
            "fraction, <=1% coverage; maximize aggregate delta"
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_replay_error": replay_error,
        "training": training["report"],
        "hard_negative_mining": mining,
        "scans": scans,
        "gate_metrics_read": False,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(build_dir / "selection-lock.json", lock)
    print(
        "[utility-v3] locked "
        f"delta={selected['metrics']['delta_mrr']:+.8f} "
        f"coverage={selected['metrics']['route_fraction']:.4%} "
        "change_hit="
        f"{selected['metrics']['routed_change_fraction']:.2%}",
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
        raise RuntimeError("utility selection lock was not frozen before gate")
    gate = _build_common_span(
        args,
        context,
        global_start=context["gate_start"],
        global_stop=context["common_stop"],
    )
    model, result = load_oof_utility_router_checkpoint(
        build_dir / "model.npz"
    )
    prediction = _predict_actions(
        model,
        result,
        gate["features"],
        batch_size=args.predict_batch_size,
    )
    routed = route_by_expected_utility(
        gate["default"],
        tuple(
            alternative.scores for alternative in gate["alternatives"]
        ),
        prediction.expected_utility,
        prediction.change_probability,
        available=gate["available"],
        minimum_utility=float(lock["minimum_utility"]),
        minimum_change_probability=float(
            lock["minimum_change_probability"]
        ),
        maximum_route_fraction=float(lock["maximum_route_fraction"]),
    )
    audit = audit_bounded_topk_route(
        gate["default"],
        gate["alternatives"],
        routed,
        cap=SWITCH_CAP,
        maximum_route_fraction=float(lock["maximum_route_fraction"]),
    )
    metrics = _route_metrics(
        gate["default"],
        routed.scores,
        routed.route_index,
        slice_count=3,
    )
    oracle = _oracle_metrics(
        gate["rewards"],
        maximum_fraction=MAXIMUM_ROUTE_FRACTION,
    )
    passed = bool(
        audit["passed"]
        and metrics["delta_mrr"] > GATE_MIN_DELTA
        and metrics["worst_time_slice_delta"] >= GATE_WORST_SLICE_MIN
        and metrics["route_fraction"] <= MAXIMUM_ROUTE_FRACTION
        and metrics["routed_change_fraction"]
        >= MINIMUM_ROUTED_CHANGE_FRACTION
    )
    score_path = build_dir / "gate-scores.npy"
    route_path = build_dir / "gate-route-index.npy"
    utility_path = build_dir / "gate-expected-utility.npy"
    change_path = build_dir / "gate-change-probability.npy"
    _save_array_atomic(score_path, routed.scores)
    _save_array_atomic(route_path, routed.route_index)
    _save_array_atomic(utility_path, prediction.expected_utility)
    _save_array_atomic(change_path, prediction.change_probability)
    report = {
        "status": "passed" if passed else "rejected",
        "passed": passed,
        "gate_disclosure": (
            "diagnostic replay of a previously observed interval"
        ),
        "global_gate_rows": [
            context["gate_start"],
            context["common_stop"],
        ],
        "metrics": metrics,
        "oracle": oracle,
        "audit": audit,
        "checkpoint_replay_source": str(
            (build_dir / "model.npz").resolve()
        ),
        "gate_scores_sha256": _sha256(score_path),
        "gate_route_index_sha256": _sha256(route_path),
        "gate_expected_utility_sha256": _sha256(utility_path),
        "gate_change_probability_sha256": _sha256(change_path),
        "external_evaluated": False,
        "submission_generated": False,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(build_dir / "gate-report.json", report)
    print(
        f"[utility-v3] gate status={report['status']} "
        f"delta={metrics['delta_mrr']:+.8f} "
        f"coverage={metrics['route_fraction']:.4%} "
        f"change_hit={metrics['routed_change_fraction']:.2%}",
        flush=True,
    )
    return report


def _route_metrics(
    default_scores: np.ndarray,
    routed_scores: np.ndarray,
    route_index: np.ndarray,
    *,
    slice_count: int,
) -> dict[str, Any]:
    default_rr = _row_rr(default_scores)
    routed_rr = _row_rr(routed_scores)
    reward = routed_rr - default_rr
    boundaries = np.linspace(
        0,
        reward.size,
        slice_count + 1,
        dtype=np.int64,
    )
    slice_deltas = [
        float(np.mean(reward[boundaries[index] : boundaries[index + 1]]))
        for index in range(slice_count)
    ]
    routed_rows = int(np.sum(route_index > 0))
    changed_rows = int(np.sum(reward != 0.0))
    return {
        "rows": int(reward.size),
        "default_short_mrr": float(np.mean(default_rr)),
        "routed_mrr": float(np.mean(routed_rr)),
        "delta_mrr": float(np.mean(reward)),
        "time_slice_deltas": slice_deltas,
        "worst_time_slice_delta": float(min(slice_deltas)),
        "route_fraction": float(np.mean(route_index > 0)),
        "routed_rows": routed_rows,
        "medium_rows": int(np.sum(route_index == 1)),
        "long_rows": int(np.sum(route_index == 2)),
        "gain_rows": int(np.sum(reward > 0.0)),
        "loss_rows": int(np.sum(reward < 0.0)),
        "unchanged_rows": int(np.sum(reward == 0.0)),
        "routed_changed_rows": changed_rows,
        "routed_change_fraction": (
            float(changed_rows / routed_rows) if routed_rows else 0.0
        ),
    }


def _oracle_metrics(
    rewards: np.ndarray,
    *,
    maximum_fraction: float,
) -> dict[str, Any]:
    values = np.asarray(rewards, dtype=np.float64)
    best_action = np.argmax(values, axis=1)
    best_reward = values[np.arange(values.shape[0]), best_action]
    positive = np.flatnonzero(best_reward > 0.0)
    quota = math.floor(values.shape[0] * maximum_fraction)
    selected = positive[
        np.argsort(-best_reward[positive], kind="stable")
    ][:quota]
    return {
        "rows": int(values.shape[0]),
        "delta_mrr": float(np.sum(best_reward[selected]) / values.shape[0]),
        "route_fraction": float(selected.size / values.shape[0]),
        "positive_opportunity_fraction": float(np.mean(best_reward > 0.0)),
        "medium_rows": int(np.sum(best_action[selected] == 0)),
        "long_rows": int(np.sum(best_action[selected] == 1)),
    }


def _row_rr(scores: np.ndarray) -> np.ndarray:
    positive = scores[:, :1]
    greater = np.sum(scores > positive, axis=1)
    equal = np.sum(scores == positive, axis=1)
    return 1.0 / (greater + (equal + 1.0) / 2.0)


def _prediction_error(
    left: UtilityPrediction,
    right: UtilityPrediction,
) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    getattr(left, field) - getattr(right, field)
                )
            )
        )
        for field in (
            "change_probability",
            "gain_probability_given_change",
            "expected_gain",
            "expected_loss",
            "expected_utility",
        )
    )


def _training_config(
    args: argparse.Namespace,
    *,
    epochs: int,
) -> OOFUtilityRouterTrainingConfig:
    return OOFUtilityRouterTrainingConfig(
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        reward_scale=args.reward_scale,
        change_positive_weight=args.change_positive_weight,
        loss_direction_weight=args.loss_direction_weight,
        magnitude_weight=args.magnitude_weight,
        hard_negative_weight=args.hard_negative_weight,
        regret_weight=args.regret_weight,
        seed=args.seed,
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
