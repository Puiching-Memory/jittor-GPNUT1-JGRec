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
    audit_bounded_topk_route,
    bounded_topk_alternative,
    hard_high_confidence_route,
    route_reward_targets,
    router_candidate_support_features,
    router_summary_features,
    timestamp_router_split,
)
from jgrec.rankers.hybrid.joint_oof_lambdamrr import (
    JointOOFLambdaMRRConfig,
    JointOOFLambdaMRRTrainingConfig,
    bounded_joint_topk_alternatives,
    fit_joint_oof_lambdamrr,
    load_joint_oof_lambdamrr_checkpoint,
    predict_joint_oof_lambdamrr,
    save_joint_oof_lambdamrr_checkpoint,
)

TOP_K_VALUES = (10, 20)
SWITCH_CAPS = (0.01, 0.02)
RANK_LOSS_WEIGHTS = (0.03, 0.10, 0.30)
ROUTE_FRACTIONS = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)
MAXIMUM_ROUTE_FRACTION = 0.05
SELECTION_MIN_DELTA = 0.0
SELECTION_WORST_SLICE_MIN = -0.00005
GATE_MIN_DELTA = 0.0
GATE_WORST_SLICE_MIN = -0.00005

JOINT_CANDIDATE_FEATURE_NAMES = (
    "default_score",
    "default_minus_row_mean",
    "default_minus_row_max",
    "default_percentile",
    "short_residual",
    "medium_residual",
    "long_residual",
    "medium_minus_short_residual",
    "long_minus_short_residual",
    "medium_base_delta",
    "long_base_delta",
)


class JointCandidateFeatureView:
    """Candidate-equivariant features evaluated lazily from mmap inputs."""

    def __init__(
        self,
        raw_features: Any,
        raw_feature_names: tuple[str, ...],
        default_scores: Any,
        residuals: Any,
        base_alternative_scores: tuple[Any, Any],
    ) -> None:
        self._raw = raw_features
        self._default = np.asarray(default_scores)
        self._residuals = np.asarray(residuals)
        self._bases = tuple(
            np.asarray(value) for value in base_alternative_scores
        )
        raw_shape = tuple(int(value) for value in raw_features.shape)
        names = tuple(str(value) for value in raw_feature_names)
        if (
            len(raw_shape) != 3
            or raw_shape[:2] != self._default.shape
            or self._residuals.shape != (3, *self._default.shape)
            or len(self._bases) != 2
            or any(value.shape != self._default.shape for value in self._bases)
            or len(names) != raw_shape[2]
            or len(set(names)) != len(names)
        ):
            raise ValueError("joint candidate feature inputs do not align")
        self.feature_names = names + JOINT_CANDIDATE_FEATURE_NAMES
        self.shape = (
            raw_shape[0],
            raw_shape[1],
            len(self.feature_names),
        )

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            row_key = key[0]
            trailing = key[1:]
        else:
            row_key = key
            trailing = ()
        raw = np.asarray(self._raw[row_key], dtype=np.float32)
        default = np.asarray(self._default[row_key], dtype=np.float32)
        residuals = np.asarray(
            self._residuals[:, row_key],
            dtype=np.float32,
        )
        bases = tuple(
            np.asarray(value[row_key], dtype=np.float32)
            for value in self._bases
        )
        squeeze = raw.ndim == 2
        if squeeze:
            raw = raw[None, ...]
            default = default[None, ...]
            residuals = residuals[:, None, ...]
            bases = tuple(value[None, ...] for value in bases)
        row_mean = default.mean(axis=1, keepdims=True)
        row_max = default.max(axis=1, keepdims=True)
        order = np.argsort(default, axis=1, kind="stable")
        ranks = np.empty(default.shape, dtype=np.float32)
        np.put_along_axis(
            ranks,
            order,
            np.arange(default.shape[1], dtype=np.float32)[None, :],
            axis=1,
        )
        percentile = ranks / float(max(default.shape[1] - 1, 1))
        extra = np.stack(
            (
                default,
                default - row_mean,
                default - row_max,
                percentile,
                residuals[0],
                residuals[1],
                residuals[2],
                residuals[1] - residuals[0],
                residuals[2] - residuals[0],
                bases[0] - default,
                bases[1] - default,
            ),
            axis=2,
        ).astype(np.float32, copy=False)
        output = np.concatenate((raw, extra), axis=2).astype(
            np.float32,
            copy=False,
        )
        if squeeze:
            output = output[0]
        if trailing:
            return output[(slice(None), *trailing)]
        return output

    def slice_rows(self, rows: slice) -> JointCandidateFeatureView:
        return JointCandidateFeatureView(
            self._raw[rows],
            self.feature_names[: self._raw.shape[2]],
            self._default[rows],
            self._residuals[:, rows],
            tuple(value[rows] for value in self._bases),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly train a pure-Jittor OOF horizon router and bounded "
            "top-k LambdaMRR residual head."
        )
    )
    parser.add_argument("--oof-dir", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--nonzero-weight", type=float, default=16.0)
    parser.add_argument("--route-loss-weight", type=float, default=1.0)
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
            f"joint output is incomplete: {args.output_dir}"
        )
    build_dir = args.output_dir.with_name(
        f"{args.output_dir.name}.building"
    )
    if build_dir.exists():
        raise FileExistsError(f"stale joint build exists: {build_dir}")
    build_dir.mkdir(parents=True)

    context = _load_context(args)
    _write_frozen_protocol(args, context, build_dir)
    reports = [
        _train_variant(
            args,
            context,
            build_dir,
            top_k=top_k,
            cap=cap,
            rank_loss_weight=rank_loss_weight,
        )
        for top_k in TOP_K_VALUES
        for cap in SWITCH_CAPS
        for rank_loss_weight in RANK_LOSS_WEIGHTS
    ]
    lock = _select_variant(build_dir, reports)
    if lock is None:
        evaluation = {
            "status": "no_selection_candidate",
            "passed": False,
            "protocol": "joint_oof_router_topk_lambdamrr_v1",
            "selection": {
                "status": "no_eligible_candidate",
                "gate_metrics_read": False,
                "variant_reports": reports,
            },
            "gate": "not_read",
            "online_champion_changed": False,
            "submission_generated": False,
            "external_evaluated": False,
            "trainable_frameworks": ["jittor"],
            "non_jittor_trainable_models": [],
            "elapsed_seconds": float(time.time() - started),
        }
    else:
        gate = _run_gate(args, context, build_dir, lock)
        evaluation = {
            "status": "accepted" if gate["passed"] else "rejected",
            "passed": bool(gate["passed"]),
            "protocol": "joint_oof_router_topk_lambdamrr_v1",
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
    residuals = np.load(
        args.oof_dir / "residuals.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    corrected = np.load(
        args.oof_dir / "corrected-logits.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    valid = np.load(
        args.oof_dir / "valid-mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    gaps = np.load(
        args.oof_dir / "gap-days.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    expected = (3, 200_000, 100)
    if (
        residuals.shape != expected
        or corrected.shape != expected
        or valid.shape != expected[:2]
        or gaps.shape != expected[:2]
    ):
        raise ValueError("multi-horizon OOF shapes differ")
    common = np.flatnonzero(np.all(valid, axis=0))
    if (
        common.size <= 10
        or not np.array_equal(
            common,
            np.arange(common[0], common[-1] + 1),
        )
    ):
        raise ValueError("common OOF coverage is not contiguous")
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
    feature_names = tuple(cache_report.get("feature_names", []))
    if (
        times.shape != (expected[1],)
        or candidate_features.shape
        != (expected[1], expected[2], len(feature_names))
        or not feature_names
    ):
        raise ValueError("joint feature cache does not align")
    return {
        "manifest_path": manifest_path,
        "audit_path": audit_path,
        "residuals": residuals,
        "corrected": corrected,
        "gaps": gaps,
        "time_path": time_path,
        "feature_path": feature_path,
        "cache_report_path": args.train_cache_report,
        "candidate_features": candidate_features,
        "candidate_feature_names": feature_names,
        "common_start": start,
        "common_stop": stop,
        "split": timestamp_router_split(times[start:stop]),
    }


def _write_frozen_protocol(
    args: argparse.Namespace,
    context: dict[str, Any],
    build_dir: Path,
) -> None:
    protocol = {
        "status": "frozen_before_training",
        "protocol": "joint_oof_router_topk_lambdamrr_v1",
        "default_path": "short corrected logits",
        "top_k_values": list(TOP_K_VALUES),
        "switch_caps": list(SWITCH_CAPS),
        "rank_loss_weights": list(RANK_LOSS_WEIGHTS),
        "route_fractions": list(ROUTE_FRACTIONS),
        "maximum_route_fraction": MAXIMUM_ROUTE_FRACTION,
        "selection_min_delta": SELECTION_MIN_DELTA,
        "selection_worst_slice_min": SELECTION_WORST_SLICE_MIN,
        "gate_min_delta": GATE_MIN_DELTA,
        "gate_worst_slice_min": GATE_WORST_SLICE_MIN,
        "temporal_split": asdict(context["split"]),
        "common_rows": [
            context["common_start"],
            context["common_stop"],
        ],
        "model": {
            "shared_row_trunk": True,
            "route_heads": 2,
            "route_specific_candidate_residual_heads": 2,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "candidate_id_features": False,
            "positive_column_features": False,
            "candidate_permutation_equivariant": True,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "reward_scale": args.reward_scale,
            "nonzero_weight": args.nonzero_weight,
            "route_loss_weight": args.route_loss_weight,
            "seed": args.seed,
        },
        "selection_rule": (
            "positive delta and stable time slices; maximize delta, then "
            "worst slice, then lower coverage/cap/top_k/lambda weight"
        ),
        "gate_policy": (
            "selection-lock.json is persisted before gate labels or "
            "metrics are computed"
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
    summary, summary_names = router_summary_features(
        default,
        residuals,
        gaps,
        (medium.scores, long.scores),
    )
    support, support_names = router_candidate_support_features(
        context["candidate_features"][start:stop],
        context["candidate_feature_names"],
        default,
        (medium.scores, long.scores),
    )
    if summary_names != ROUTER_FEATURE_NAMES:
        raise RuntimeError("joint router summary contract changed")
    row_features = np.concatenate((summary, support), axis=1).astype(
        np.float32,
        copy=False,
    )
    candidate_view = JointCandidateFeatureView(
        context["candidate_features"][start:stop],
        context["candidate_feature_names"],
        default,
        residuals,
        (medium.scores, long.scores),
    )
    return {
        "default": default,
        "residuals": residuals,
        "medium": medium,
        "long": long,
        "row_features": row_features,
        "row_feature_names": summary_names + support_names,
        "candidate_features": candidate_view,
    }


def _train_variant(
    args: argparse.Namespace,
    context: dict[str, Any],
    build_dir: Path,
    *,
    top_k: int,
    cap: float,
    rank_loss_weight: float,
) -> dict[str, Any]:
    label = _variant_label(top_k, cap, rank_loss_weight)
    directory = build_dir / "variants" / label
    directory.mkdir(parents=True)
    arrays = _variant_arrays(context, top_k=top_k, cap=cap)
    split = context["split"]
    train = slice(*split.train_rows)
    selection = slice(*split.selection_rows)
    selection_stop = int(split.selection_rows[1])
    rewards = route_reward_targets(
        arrays["default"][:selection_stop],
        (
            arrays["medium"].scores[:selection_stop],
            arrays["long"].scores[:selection_stop],
        ),
        np.zeros(selection_stop, dtype=np.int32),
    )
    training_config = _training_config(
        args,
        rank_loss_weight=rank_loss_weight,
    )
    model, result = fit_joint_oof_lambdamrr(
        arrays["row_features"][train],
        arrays["candidate_features"].slice_rows(train),
        arrays["default"][train],
        (
            arrays["medium"].scores[train],
            arrays["long"].scores[train],
        ),
        rewards[train],
        top_k=top_k,
        cap=cap,
        model_config=JointOOFLambdaMRRConfig(
            row_input_dim=arrays["row_features"].shape[1],
            candidate_input_dim=arrays["candidate_features"].shape[2],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ),
        training_config=training_config,
        row_feature_names=arrays["row_feature_names"],
        candidate_feature_names=arrays[
            "candidate_features"
        ].feature_names,
        verbose=True,
    )
    checkpoint = directory / "model.npz"
    save_joint_oof_lambdamrr_checkpoint(checkpoint, model, result)
    predictions, residual_predictions = predict_joint_oof_lambdamrr(
        model,
        arrays["row_features"][selection],
        arrays["candidate_features"].slice_rows(selection),
        row_mean=result.row_mean,
        row_std=result.row_std,
        candidate_mean=result.candidate_mean,
        candidate_std=result.candidate_std,
        reward_scale=result.training_config.reward_scale,
        batch_size=args.predict_batch_size,
    )
    alternatives = bounded_joint_topk_alternatives(
        arrays["default"][selection],
        (
            arrays["medium"].scores[selection],
            arrays["long"].scores[selection],
        ),
        residual_predictions,
        top_k=top_k,
        cap=cap,
    )
    loaded, loaded_result = load_joint_oof_lambdamrr_checkpoint(
        checkpoint
    )
    replay_predictions, replay_residuals = predict_joint_oof_lambdamrr(
        loaded,
        arrays["row_features"][selection],
        arrays["candidate_features"].slice_rows(selection),
        row_mean=loaded_result.row_mean,
        row_std=loaded_result.row_std,
        candidate_mean=loaded_result.candidate_mean,
        candidate_std=loaded_result.candidate_std,
        reward_scale=loaded_result.training_config.reward_scale,
        batch_size=args.predict_batch_size,
    )
    replay_error = max(
        float(np.max(np.abs(replay_predictions - predictions))),
        float(np.max(np.abs(replay_residuals - residual_predictions))),
    )
    if replay_error > 1e-6:
        raise RuntimeError(f"{label} checkpoint replay failed")
    scans = _selection_scans(
        arrays["default"][selection],
        alternatives,
        predictions,
        cap=cap,
    )
    best = max(
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
        "rank_loss_weight": rank_loss_weight,
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
        "selection_oracle": _oracle_metrics(
            route_reward_targets(
                arrays["default"][selection],
                (alternatives[0].scores, alternatives[1].scores),
                np.zeros(predictions.shape[0], dtype=np.int32),
            ),
            maximum_fraction=MAXIMUM_ROUTE_FRACTION,
        ),
        "selection_scans": scans,
        "best_selection": best,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(directory / "report.json", report)
    print(
        f"[joint] {label} "
        f"delta={best['metrics']['delta_mrr']:+.6f} "
        f"worst={best['metrics']['worst_time_slice_delta']:+.6f} "
        f"coverage={best['metrics']['route_fraction']:.4%}",
        flush=True,
    )
    del (
        model,
        loaded,
        predictions,
        replay_predictions,
        residual_predictions,
        replay_residuals,
        alternatives,
        arrays,
    )
    release_memory()
    return report


def _selection_scans(
    default: np.ndarray,
    alternatives: tuple[BoundedTopKAlternative, BoundedTopKAlternative],
    predictions: np.ndarray,
    *,
    cap: float,
) -> list[dict[str, Any]]:
    rows = np.arange(predictions.shape[0])
    best = np.argmax(predictions, axis=1)
    best_advantage = predictions[rows, best]
    other = predictions[rows, 1 - best]
    confidence = best_advantage - np.maximum(other, 0.0)
    eligible = np.sort(
        confidence[(best_advantage > 0.0) & (confidence > 0.0)]
    )[::-1]
    scans = []
    for fraction in ROUTE_FRACTIONS:
        quota = math.floor(default.shape[0] * fraction)
        threshold = (
            float(eligible[min(quota, eligible.size) - 1])
            if quota > 0 and eligible.size
            else 1e9
        )
        routed = hard_high_confidence_route(
            default,
            (alternatives[0].scores, alternatives[1].scores),
            predictions,
            minimum_confidence=threshold,
            maximum_route_fraction=fraction,
        )
        audit = audit_bounded_topk_route(
            default,
            alternatives,
            routed,
            cap=cap,
            maximum_route_fraction=fraction,
        )
        if not audit["passed"]:
            raise RuntimeError("joint selection safety audit failed")
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
    build_dir: Path,
    reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = []
    for report in reports:
        best = report["best_selection"]
        metrics = best["metrics"]
        if (
            metrics["delta_mrr"] > SELECTION_MIN_DELTA
            and metrics["worst_time_slice_delta"]
            >= SELECTION_WORST_SLICE_MIN
            and metrics["route_fraction"] <= MAXIMUM_ROUTE_FRACTION
        ):
            candidates.append((report, best))
    if not candidates:
        return None
    report, scan = max(
        candidates,
        key=lambda value: (
            value[1]["metrics"]["delta_mrr"],
            value[1]["metrics"]["worst_time_slice_delta"],
            -value[1]["metrics"]["route_fraction"],
            -value[0]["switch_cap"],
            -value[0]["top_k"],
            -value[0]["rank_loss_weight"],
        ),
    )
    lock = {
        "status": "locked_before_gate",
        "selected_variant": report["variant"],
        "top_k": report["top_k"],
        "switch_cap": report["switch_cap"],
        "rank_loss_weight": report["rank_loss_weight"],
        "maximum_route_fraction": scan["maximum_route_fraction"],
        "minimum_confidence": scan["minimum_confidence"],
        "selection_metrics": scan["metrics"],
        "selection_audit": scan["audit"],
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
        f"[joint] locked {lock['selected_variant']} "
        f"delta={lock['selection_metrics']['delta_mrr']:+.6f}",
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
        raise RuntimeError("joint selection was not locked before gate")
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
    model, result = load_joint_oof_lambdamrr_checkpoint(checkpoint)
    predictions, residual_predictions = predict_joint_oof_lambdamrr(
        model,
        arrays["row_features"][gate],
        arrays["candidate_features"].slice_rows(gate),
        row_mean=result.row_mean,
        row_std=result.row_std,
        candidate_mean=result.candidate_mean,
        candidate_std=result.candidate_std,
        reward_scale=result.training_config.reward_scale,
        batch_size=args.predict_batch_size,
    )
    alternatives = bounded_joint_topk_alternatives(
        arrays["default"][gate],
        (
            arrays["medium"].scores[gate],
            arrays["long"].scores[gate],
        ),
        residual_predictions,
        top_k=int(lock["top_k"]),
        cap=float(lock["switch_cap"]),
    )
    routed = hard_high_confidence_route(
        arrays["default"][gate],
        (alternatives[0].scores, alternatives[1].scores),
        predictions,
        minimum_confidence=float(lock["minimum_confidence"]),
        maximum_route_fraction=float(lock["maximum_route_fraction"]),
    )
    audit = audit_bounded_topk_route(
        arrays["default"][gate],
        alternatives,
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
        (alternatives[0].scores, alternatives[1].scores),
        np.zeros(predictions.shape[0], dtype=np.int32),
    )
    passed = bool(
        audit["passed"]
        and metrics["delta_mrr"] > GATE_MIN_DELTA
        and metrics["worst_time_slice_delta"] >= GATE_WORST_SLICE_MIN
        and metrics["route_fraction"] <= MAXIMUM_ROUTE_FRACTION
    )
    score_path = build_dir / "gate-scores.npy"
    route_path = build_dir / "gate-route-index.npy"
    prediction_path = build_dir / "gate-predicted-advantages.npy"
    residual_path = build_dir / "gate-candidate-residuals.npy"
    _save_array_atomic(score_path, routed.scores)
    _save_array_atomic(route_path, routed.route_index)
    _save_array_atomic(prediction_path, predictions)
    _save_array_atomic(residual_path, residual_predictions)
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
        "oracle": _oracle_metrics(
            rewards,
            maximum_fraction=MAXIMUM_ROUTE_FRACTION,
        ),
        "audit": audit,
        "checkpoint_replay_source": str(
            (
                args.output_dir
                / "variants"
                / lock["selected_variant"]
                / "model.npz"
            ).resolve()
        ),
        "gate_scores_sha256": _sha256(score_path),
        "gate_route_index_sha256": _sha256(route_path),
        "gate_predicted_advantages_sha256": _sha256(prediction_path),
        "gate_candidate_residuals_sha256": _sha256(residual_path),
        "external_evaluated": False,
        "submission_generated": False,
        "trainable_frameworks": ["jittor"],
        "non_jittor_trainable_models": [],
    }
    _write_json_atomic(build_dir / "gate-report.json", report)
    print(
        f"[joint] gate status={report['status']} "
        f"delta={metrics['delta_mrr']:+.6f} "
        f"worst={metrics['worst_time_slice_delta']:+.6f}",
        flush=True,
    )
    return report


def _route_metrics(
    default_scores: np.ndarray,
    routed_scores: np.ndarray,
    route_index: np.ndarray,
) -> dict[str, Any]:
    default_rr = _row_rr(default_scores)
    routed_rr = _row_rr(routed_scores)
    reward = routed_rr - default_rr
    boundaries = np.linspace(0, reward.size, 4, dtype=np.int64)
    slices = [
        float(np.mean(reward[boundaries[index] : boundaries[index + 1]]))
        for index in range(3)
    ]
    return {
        "rows": int(reward.size),
        "default_short_mrr": float(np.mean(default_rr)),
        "routed_mrr": float(np.mean(routed_rr)),
        "delta_mrr": float(np.mean(reward)),
        "time_slice_deltas": slices,
        "worst_time_slice_delta": float(min(slices)),
        "route_fraction": float(np.mean(route_index > 0)),
        "routed_rows": int(np.sum(route_index > 0)),
        "medium_rows": int(np.sum(route_index == 1)),
        "long_rows": int(np.sum(route_index == 2)),
        "gain_rows": int(np.sum(reward > 0.0)),
        "loss_rows": int(np.sum(reward < 0.0)),
        "unchanged_rows": int(np.sum(reward == 0.0)),
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
        "positive_opportunity_fraction": float(np.mean(best_reward > 0.0)),
        "medium_rows": int(np.sum(best_route[selected] == 0)),
        "long_rows": int(np.sum(best_route[selected] == 1)),
    }


def _row_rr(scores: np.ndarray) -> np.ndarray:
    positive = scores[:, :1]
    greater = np.sum(scores > positive, axis=1)
    equal = np.sum(scores == positive, axis=1)
    return 1.0 / (greater + (equal + 1.0) / 2.0)


def _training_config(
    args: argparse.Namespace,
    *,
    rank_loss_weight: float,
) -> JointOOFLambdaMRRTrainingConfig:
    return JointOOFLambdaMRRTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        reward_scale=args.reward_scale,
        nonzero_weight=args.nonzero_weight,
        route_loss_weight=args.route_loss_weight,
        rank_loss_weight=rank_loss_weight,
        seed=args.seed,
    )


def _variant_label(
    top_k: int,
    cap: float,
    rank_loss_weight: float,
) -> str:
    return (
        f"topk-{top_k}-cap-{cap:.2f}-lambda-{rank_loss_weight:.2f}"
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
