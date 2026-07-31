from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import set_model_state
from jgrec.rankers.hybrid.fusion import FusionMLP
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_three_slices
from jgrec.rankers.hybrid.multi_expert_gate import (
    MultiExpertGateConfig,
    fit_multi_expert_gate,
    multi_expert_score_descriptors,
    predict_multi_expert_gate,
    route_multi_expert,
    select_multi_expert_config_on_forward_slice,
)
from jgrec.rankers.hybrid.multi_interest_gate import (
    MULTI_INTEREST_GATE_DESCRIPTOR_NAMES,
    confidence_gate_descriptors,
    predict_confidence_gate,
    reciprocal_ranks,
    route_query_experts,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

EXPECTED_BASE_SHAPE = (20_000, 100, 63)
EXPECTED_PROXY_SHAPE = (20_000, 100, 9)
TRAIN_ROWS = (0, 6_667)
SELECTION_ROWS = (6_667, 13_334)
FORWARD_ROWS = (13_334, 20_000)
DESCRIPTOR_EXPERT_ORDER = (
    "current_gate",
    "v1_champion",
    "multi_interest",
    "window_ensemble",
)
ALTERNATIVE_EXPERT_ORDER = (
    "v1_champion",
    "multi_interest",
    "window_ensemble",
)
CONFIG_GRID = tuple(
    MultiExpertGateConfig(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        minimum_predicted_lift=minimum_predicted_lift,
    )
    for max_depth in (1, 2, 3)
    for min_samples_leaf in (250, 500, 1_000)
    for minimum_predicted_lift in (0.0025, 0.005, 0.01)
)


class _AugmentedFeatures:
    def __init__(self, base: Any, proxy: Any) -> None:
        if base.shape[:2] != proxy.shape[:2]:
            raise ValueError("base and proxy query shapes differ")
        self._base = base
        self._proxy = proxy
        self.shape = (
            int(base.shape[0]),
            int(base.shape[1]),
            int(base.shape[2] + proxy.shape[2]),
        )

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(self._base[key], dtype=np.float32),
                np.asarray(self._proxy[key], dtype=np.float32),
            ),
            axis=-1,
            dtype=np.float32,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("train-select")
    select.add_argument("--validation-features", required=True, type=Path)
    select.add_argument("--validation-cache-report", required=True, type=Path)
    select.add_argument("--validation-proxy", required=True, type=Path)
    select.add_argument("--proxy-report", required=True, type=Path)
    select.add_argument("--multi-interest-model", required=True, type=Path)
    select.add_argument("--current-gate-model", required=True, type=Path)
    select.add_argument("--current-gate-report", required=True, type=Path)
    select.add_argument("--window-selection-report", required=True, type=Path)
    select.add_argument("--window-evaluation-report", required=True, type=Path)
    select.add_argument(
        "--window-artifacts-dir",
        required=True,
        type=Path,
    )
    select.add_argument("--frozen-dataset1-csv", required=True, type=Path)
    select.add_argument("--output-dir", required=True, type=Path)
    select.add_argument("--batch-size", type=int, default=256)
    select.add_argument("--setwise-weight", type=float, default=0.80)
    select.add_argument("--minimum-selection-delta", type=float, default=0.001)
    select.add_argument("--maximum-coverage", type=float, default=0.25)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--output-dir", required=True, type=Path)
    gate.add_argument("--minimum-forward-delta", type=float, default=0.001)
    gate.add_argument("--maximum-coverage", type=float, default=0.25)

    args = parser.parse_args()
    if args.command == "train-select":
        return _train_select(args)
    return _gate(args)


def _train_select(args: argparse.Namespace) -> int:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    artifacts_dir = args.output_dir / "artifacts"
    artifacts_dir.mkdir()
    started = time.time()

    validation_report = _read_json(args.validation_cache_report)
    proxy_report = _read_json(args.proxy_report)
    current_gate_report = _read_json(args.current_gate_report)
    window_selection = _read_json(args.window_selection_report)
    window_evaluation = _read_json(args.window_evaluation_report)
    feature_names = tuple(
        str(name) for name in validation_report["feature_names"]
    )
    if len(feature_names) != EXPECTED_BASE_SHAPE[-1]:
        raise ValueError("validation feature schema differs")

    probability_paths = {
        "lightgbm": (
            args.window_artifacts_dir
            / "validation-probabilities-lightgbm.npy"
        ),
        "recent100k": (
            args.window_artifacts_dir
            / "validation-probabilities-recent100k.npy"
        ),
        "recent200k": (
            args.window_artifacts_dir
            / "validation-probabilities-recent200k.npy"
        ),
        "recent200k_decay100k": (
            args.window_artifacts_dir
            / "validation-probabilities-recent200k_decay100k.npy"
        ),
    }
    input_paths = {
        "validation_features": args.validation_features,
        "validation_cache_report": args.validation_cache_report,
        "validation_proxy": args.validation_proxy,
        "proxy_report": args.proxy_report,
        "multi_interest_model": args.multi_interest_model,
        "current_gate_model": args.current_gate_model,
        "current_gate_report": args.current_gate_report,
        "window_selection_report": args.window_selection_report,
        "window_evaluation_report": args.window_evaluation_report,
        "frozen_dataset1_csv": args.frozen_dataset1_csv,
        **{
            f"window_{name}": path
            for name, path in probability_paths.items()
        },
    }
    input_hashes = {
        name: _sha256(path) for name, path in input_paths.items()
    }
    _require_expected_hashes(
        args,
        validation_report,
        proxy_report,
        current_gate_report,
        window_selection,
        input_hashes,
    )

    dummy_scores = {
        name: np.ones((1, 2), dtype=np.float32)
        for name in DESCRIPTOR_EXPERT_ORDER
    }
    _, descriptor_names = multi_expert_score_descriptors(
        dummy_scores,
        expert_order=DESCRIPTOR_EXPERT_ORDER,
    )
    frozen = {
        "status": "frozen_before_scoring",
        "protocol_version": 2,
        "baseline": "current_multi_interest_confidence_gate",
        "online_baseline_score": 1.3521011401636023,
        "descriptor_expert_order": list(DESCRIPTOR_EXPERT_ORDER),
        "alternative_expert_order": list(ALTERNATIVE_EXPERT_ORDER),
        "descriptor_names": list(descriptor_names),
        "descriptor_count": len(descriptor_names),
        "descriptors_are_score_only": True,
        "labels_in_descriptors": False,
        "window_policy": (
            "0.80 * mean(recent100k, recent200k, "
            "recent200k_decay100k) + 0.20 * LightGBM"
        ),
        "train_rows": list(TRAIN_ROWS),
        "selection_rows": list(SELECTION_ROWS),
        "forward_rows": list(FORWARD_ROWS),
        "selection_uses_forward_rows": False,
        "config_grid": [asdict(config) for config in CONFIG_GRID],
        "selection_gate": {
            "minimum_mrr_delta": args.minimum_selection_delta,
            "maximum_coverage": args.maximum_coverage,
        },
        "forward_gate": {
            "minimum_mrr_delta": 0.001,
            "maximum_coverage": 0.25,
        },
        "tie_break": (
            "higher selection delta, then lower coverage, "
            "then frozen config order; expert ties use frozen order"
        ),
        "input_paths": {
            name: str(path.resolve()) for name, path in input_paths.items()
        },
        "input_sha256": input_hashes,
        "expected_metrics": {
            "v1_champion": proxy_report["baseline"],
            "multi_interest": proxy_report["candidate"],
            "window_ensemble": window_evaluation["candidate"],
            "current_gate_delta_vs_v1": current_gate_report[
                "production_validation_delta"
            ],
            "current_gate_coverage": current_gate_report[
                "production_validation_coverage"
            ],
        },
        "frozen_dataset1_csv_sha256": input_hashes[
            "frozen_dataset1_csv"
        ],
    }
    frozen_path = args.output_dir / "frozen-config.json"
    _write_json_atomic(frozen_path, frozen)
    frozen_sha = _sha256(frozen_path)
    print(json.dumps(frozen, ensure_ascii=False, sort_keys=True), flush=True)

    base_features = np.load(
        args.validation_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    proxy_features = np.load(
        args.validation_proxy,
        mmap_mode="r",
        allow_pickle=False,
    )
    if base_features.shape != EXPECTED_BASE_SHAPE:
        raise ValueError(f"unexpected validation features: {base_features.shape}")
    if proxy_features.shape != EXPECTED_PROXY_SHAPE:
        raise ValueError(f"unexpected validation proxy: {proxy_features.shape}")

    saved_probabilities = {
        name: np.load(path, mmap_mode="r", allow_pickle=False)
        for name, path in probability_paths.items()
    }
    if any(
        values.shape != EXPECTED_BASE_SHAPE[:2]
        for values in saved_probabilities.values()
    ):
        raise ValueError("window expert probabilities differ in shape")
    setwise_weight = float(args.setwise_weight)
    lightgbm = np.asarray(saved_probabilities["lightgbm"])
    v1 = (
        setwise_weight * np.asarray(saved_probabilities["recent200k"])
        + (1.0 - setwise_weight) * lightgbm
    )
    window = (
        setwise_weight
        * np.mean(
            np.stack(
                (
                    saved_probabilities["recent100k"],
                    saved_probabilities["recent200k"],
                    saved_probabilities["recent200k_decay100k"],
                ),
                axis=0,
            ),
            axis=0,
            dtype=np.float64,
        )
        + (1.0 - setwise_weight) * lightgbm
    )

    jt.flags.use_cuda = 1
    multi_interest_setwise = _load_setwise_probabilities(
        args.multi_interest_model,
        _AugmentedFeatures(base_features, proxy_features),
        args.batch_size,
    )
    multi_interest = (
        setwise_weight * multi_interest_setwise
        + (1.0 - setwise_weight) * lightgbm
    )
    with args.current_gate_model.open("rb") as handle:
        current_gate_model = pickle.load(handle)
    current_descriptors = confidence_gate_descriptors(
        base_features,
        feature_names,
        proxy_features,
        v1,
        multi_interest,
    )
    current_use_candidate, _ = predict_confidence_gate(
        current_gate_model,
        current_descriptors,
        descriptor_names=MULTI_INTEREST_GATE_DESCRIPTOR_NAMES,
    )
    current_gate = route_query_experts(
        v1,
        multi_interest,
        current_use_candidate,
    )
    expert_scores = {
        "current_gate": current_gate,
        "v1_champion": v1,
        "multi_interest": multi_interest,
        "window_ensemble": window,
    }
    metrics = {
        name: ranking_mrr_three_slices(scores)
        for name, scores in expert_scores.items()
    }
    _require_metrics_close(
        metrics["v1_champion"],
        proxy_report["baseline"],
        "v1 champion",
    )
    _require_metrics_close(
        metrics["multi_interest"],
        proxy_report["candidate"],
        "multi-interest",
    )
    _require_metrics_close(
        metrics["window_ensemble"],
        window_evaluation["candidate"],
        "window ensemble",
    )
    current_delta = _metric_delta(
        metrics["current_gate"],
        metrics["v1_champion"],
    )
    _require_metrics_close(
        current_delta,
        current_gate_report["production_validation_delta"],
        "current confidence gate delta",
    )
    actual_current_coverage = float(np.mean(current_use_candidate))
    if not np.isclose(
        actual_current_coverage,
        float(current_gate_report["production_validation_coverage"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("current confidence gate coverage differs")

    descriptors, actual_descriptor_names = (
        multi_expert_score_descriptors(
            expert_scores,
            expert_order=DESCRIPTOR_EXPERT_ORDER,
        )
    )
    if actual_descriptor_names != descriptor_names:
        raise RuntimeError("multi-expert descriptor schema changed")
    artifact_paths: dict[str, Path] = {}
    for name, scores in expert_scores.items():
        path = artifacts_dir / f"validation-scores-{name}.npy"
        np.save(path, np.asarray(scores, dtype=np.float32))
        artifact_paths[f"scores_{name}"] = path
    descriptors_path = artifacts_dir / "validation-descriptors.npy"
    np.save(descriptors_path, descriptors)
    artifact_paths["descriptors"] = descriptors_path

    selection, trials = _select_on_slice_one(
        descriptors,
        expert_scores,
        descriptor_names=descriptor_names,
        minimum_delta=args.minimum_selection_delta,
        maximum_coverage=args.maximum_coverage,
    )
    artifact_hashes = {
        name: _sha256(path) for name, path in artifact_paths.items()
    }
    selection_report: dict[str, Any] = {
        "status": (
            "locked_before_forward_gate"
            if selection is not None
            else "no_eligible_candidate"
        ),
        "frozen_config": str(frozen_path.resolve()),
        "frozen_config_sha256": frozen_sha,
        "train_rows": list(TRAIN_ROWS),
        "selection_rows": list(SELECTION_ROWS),
        "forward_rows": list(FORWARD_ROWS),
        "selection_uses_forward_rows": False,
        "baseline": "current_gate",
        "alternative_expert_order": list(ALTERNATIVE_EXPERT_ORDER),
        "baseline_selection_mrr": trials[0]["fallback_mrr"],
        "trials": trials,
        "artifact_paths": {
            name: str(path.resolve())
            for name, path in artifact_paths.items()
        },
        "artifact_sha256": artifact_hashes,
        "elapsed_seconds": time.time() - started,
    }
    if selection is not None:
        selected_model = selection["model"]
        model_path = artifacts_dir / "selection-router.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(
                selected_model,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        selection_report["selected"] = {
            key: value
            for key, value in selection.items()
            if key != "model"
        }
        selection_report["selection_router"] = str(model_path.resolve())
        selection_report["selection_router_sha256"] = _sha256(model_path)
        selection_report["used_descriptor_names"] = (
            _used_descriptor_names(selected_model)
        )

    selection_path = args.output_dir / "selection-report.json"
    _write_json_atomic(selection_path, selection_report)
    selection_sha = _sha256(selection_path)
    _write_text_atomic(
        args.output_dir / "selection-report.sha256",
        f"{selection_sha}\n",
    )
    print(
        json.dumps(selection_report, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return 0 if selection is not None else 2


def _gate(args: argparse.Namespace) -> int:
    selection_path = args.output_dir / "selection-report.json"
    selection_sha_path = args.output_dir / "selection-report.sha256"
    frozen_path = args.output_dir / "frozen-config.json"
    selection = _read_json(selection_path)
    frozen = _read_json(frozen_path)
    expected_selection_sha = selection_sha_path.read_text(
        encoding="utf-8"
    ).strip()
    if _sha256(selection_path) != expected_selection_sha:
        raise ValueError("selection report SHA-256 lock differs")
    if selection["status"] != "locked_before_forward_gate":
        raise RuntimeError("no eligible locked candidate to gate")
    if selection["selection_uses_forward_rows"] is not False:
        raise RuntimeError("selection report used forward rows")
    if selection["forward_rows"] != list(FORWARD_ROWS):
        raise RuntimeError("forward row contract differs")
    if _sha256(frozen_path) != selection["frozen_config_sha256"]:
        raise ValueError("frozen config SHA-256 differs")
    for name, path_text in selection["artifact_paths"].items():
        path = Path(path_text)
        expected = selection["artifact_sha256"][name]
        if _sha256(path) != expected:
            raise ValueError(f"artifact SHA-256 differs: {name}")
    for name, path_text in frozen["input_paths"].items():
        path = Path(path_text)
        if _sha256(path) != frozen["input_sha256"][name]:
            raise ValueError(f"frozen input SHA-256 differs: {name}")

    descriptor_names = tuple(frozen["descriptor_names"])
    descriptors = np.load(
        selection["artifact_paths"]["descriptors"],
        mmap_mode="r",
        allow_pickle=False,
    )
    scores = {
        name: np.load(
            selection["artifact_paths"][f"scores_{name}"],
            mmap_mode="r",
            allow_pickle=False,
        )
        for name in DESCRIPTOR_EXPERT_ORDER
    }
    fallback = scores["current_gate"]
    alternatives = {
        name: scores[name] for name in ALTERNATIVE_EXPERT_ORDER
    }
    config = MultiExpertGateConfig(**selection["selected"]["config"])
    train_slice = slice(TRAIN_ROWS[0], SELECTION_ROWS[1])
    forward_slice = slice(*FORWARD_ROWS)
    fallback_rr = reciprocal_ranks(fallback[train_slice])
    rewards = np.column_stack(
        [
            reciprocal_ranks(alternatives[name][train_slice])
            - fallback_rr
            for name in ALTERNATIVE_EXPERT_ORDER
        ]
    )
    final_model = fit_multi_expert_gate(
        descriptors[train_slice],
        rewards,
        config,
        descriptor_names=descriptor_names,
        expert_order=ALTERNATIVE_EXPERT_ORDER,
        seed=60,
    )
    predicted_lifts = predict_multi_expert_gate(
        final_model,
        descriptors[forward_slice],
        descriptor_names=descriptor_names,
    )
    routed_forward = route_multi_expert(
        fallback[forward_slice],
        {
            name: values[forward_slice]
            for name, values in alternatives.items()
        },
        predicted_lifts,
        expert_order=ALTERNATIVE_EXPERT_ORDER,
        minimum_predicted_lift=config.minimum_predicted_lift,
    )
    forward_fallback_mrr = float(
        np.mean(reciprocal_ranks(fallback[forward_slice]))
    )
    forward_candidate_mrr = float(
        np.mean(reciprocal_ranks(routed_forward.scores))
    )
    forward_delta = forward_candidate_mrr - forward_fallback_mrr
    coverage = float(np.mean(routed_forward.use_alternative))
    fallback_exact = bool(
        np.array_equal(
            routed_forward.scores[~routed_forward.use_alternative],
            fallback[forward_slice][~routed_forward.use_alternative],
        )
    )
    gate_passed = bool(
        forward_delta + 1e-12 >= args.minimum_forward_delta
        and coverage <= args.maximum_coverage + 1e-12
        and fallback_exact
    )

    all_predicted_lifts = predict_multi_expert_gate(
        final_model,
        descriptors,
        descriptor_names=descriptor_names,
    )
    routed_all = route_multi_expert(
        fallback,
        alternatives,
        all_predicted_lifts,
        expert_order=ALTERNATIVE_EXPERT_ORDER,
        minimum_predicted_lift=config.minimum_predicted_lift,
    )
    final_model_path = args.output_dir / "artifacts" / "final-router.pkl"
    with final_model_path.open("wb") as handle:
        pickle.dump(final_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    prediction_path = (
        args.output_dir / "artifacts" / "validation-routed-v2.npy"
    )
    np.save(prediction_path, routed_all.scores.astype(np.float32))
    baseline_metrics = ranking_mrr_three_slices(fallback)
    candidate_metrics = ranking_mrr_three_slices(routed_all.scores)
    counts = {
        name: int(np.sum(routed_forward.selected_experts == name))
        for name in ("current_gate", *ALTERNATIVE_EXPERT_ORDER)
    }
    report = {
        "status": "accepted" if gate_passed else "rejected",
        "gate_passed": gate_passed,
        "package_authorized": gate_passed,
        "package_generated": False,
        "selection_report": str(selection_path.resolve()),
        "selection_report_sha256": expected_selection_sha,
        "selected_config": asdict(config),
        "forward_rows": list(FORWARD_ROWS),
        "forward": {
            "fallback_mrr": forward_fallback_mrr,
            "candidate_mrr": forward_candidate_mrr,
            "delta": forward_delta,
            "minimum_delta": args.minimum_forward_delta,
            "coverage": coverage,
            "maximum_coverage": args.maximum_coverage,
            "fallback_exact": fallback_exact,
            "selected_expert_counts": counts,
        },
        "diagnostic_full_metrics_after_prefix_refit": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": _metric_delta(candidate_metrics, baseline_metrics),
        },
        "used_descriptor_names": _used_descriptor_names(final_model),
        "final_router": str(final_model_path.resolve()),
        "final_router_sha256": _sha256(final_model_path),
        "validation_prediction": str(prediction_path.resolve()),
        "validation_prediction_sha256": _sha256(prediction_path),
        "frozen_dataset1_csv_sha256": frozen[
            "frozen_dataset1_csv_sha256"
        ],
    }
    _write_json_atomic(args.output_dir / "evaluation-report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if gate_passed else 2


def _select_on_slice_one(
    descriptors: np.ndarray,
    expert_scores: dict[str, np.ndarray],
    *,
    descriptor_names: tuple[str, ...],
    minimum_delta: float,
    maximum_coverage: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    fallback = expert_scores["current_gate"]
    alternatives = {
        name: expert_scores[name] for name in ALTERNATIVE_EXPERT_ORDER
    }
    train_slice = slice(*TRAIN_ROWS)
    selection_slice = slice(*SELECTION_ROWS)
    fallback_train_rr = reciprocal_ranks(fallback[train_slice])
    rewards = np.column_stack(
        [
            reciprocal_ranks(alternatives[name][train_slice])
            - fallback_train_rr
            for name in ALTERNATIVE_EXPERT_ORDER
        ]
    )
    fallback_selection_mrr = float(
        np.mean(reciprocal_ranks(fallback[selection_slice]))
    )
    trials: list[dict[str, Any]] = []
    models = []
    for config_index, config in enumerate(CONFIG_GRID):
        model = fit_multi_expert_gate(
            descriptors[train_slice],
            rewards,
            config,
            descriptor_names=descriptor_names,
            expert_order=ALTERNATIVE_EXPERT_ORDER,
            seed=60,
        )
        predicted = predict_multi_expert_gate(
            model,
            descriptors[selection_slice],
            descriptor_names=descriptor_names,
        )
        routed = route_multi_expert(
            fallback[selection_slice],
            {
                name: values[selection_slice]
                for name, values in alternatives.items()
            },
            predicted,
            expert_order=ALTERNATIVE_EXPERT_ORDER,
            minimum_predicted_lift=config.minimum_predicted_lift,
        )
        selection_mrr = float(np.mean(reciprocal_ranks(routed.scores)))
        delta = selection_mrr - fallback_selection_mrr
        coverage = float(np.mean(routed.use_alternative))
        eligible = bool(
            delta + 1e-12 >= minimum_delta
            and coverage <= maximum_coverage + 1e-12
        )
        counts = {
            name: int(np.sum(routed.selected_experts == name))
            for name in ("current_gate", *ALTERNATIVE_EXPERT_ORDER)
        }
        trials.append(
            {
                "config_index": config_index,
                "config": asdict(config),
                "fallback_mrr": fallback_selection_mrr,
                "selection_mrr": selection_mrr,
                "delta": delta,
                "coverage": coverage,
                "eligible": eligible,
                "selected_expert_counts": counts,
            }
        )
        models.append(model)

    eligible_indices = [
        index for index, trial in enumerate(trials) if trial["eligible"]
    ]
    selected: dict[str, Any] | None = None
    if eligible_indices:
        selected_index = max(
            eligible_indices,
            key=lambda index: (
                trials[index]["delta"],
                -trials[index]["coverage"],
                -trials[index]["config_index"],
            ),
        )
        selected = {
            **trials[selected_index],
            "model": models[selected_index],
        }

    core_selection = select_multi_expert_config_on_forward_slice(
        descriptors,
        fallback,
        alternatives,
        configs=CONFIG_GRID,
        descriptor_names=descriptor_names,
        expert_order=ALTERNATIVE_EXPERT_ORDER,
        train_rows=TRAIN_ROWS,
        selection_rows=SELECTION_ROWS,
        minimum_selection_delta=minimum_delta,
        maximum_coverage=maximum_coverage,
        seed=60,
    )
    if (selected is None) != (core_selection is None):
        raise RuntimeError("forward selector and report selection differ")
    if selected is not None and (
        selected["config"] != asdict(core_selection.model.config)
        or not np.isclose(
            selected["delta"],
            core_selection.delta,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError("forward selector and report winner differ")
    return selected, trials


def _load_setwise_probabilities(
    path: Path,
    features: Any,
    batch_size: int,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    indices = tuple(int(value) for value in payload["feature_indices"])
    model = FusionMLP(
        input_dim=len(indices),
        hidden_dim=int(payload["hidden_dim"][0]),
    )
    set_model_state(
        model,
        {
            key.removeprefix("state__"): np.asarray(
                value,
                dtype=np.float32,
            )
            for key, value in payload.items()
            if key.startswith("state__")
        },
    )
    view = SetwiseFeatureView(features)
    logits = np.empty(features.shape[:2], dtype=np.float32)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    with jt.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            values = np.asarray(view[start:end], dtype=np.float32)
            normalized = (
                (values[..., indices] - mean) / std
            ).astype(np.float32, copy=False)
            logits[start:end] = np.asarray(
                model(jt.array(normalized, dtype=jt.float32)).numpy(),
                dtype=np.float32,
            )
    return _softmax(logits)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _used_descriptor_names(model: Any) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for expert_name, model_bytes in zip(
        model.expert_order,
        model.model_bytes,
        strict=True,
    ):
        estimator = pickle.loads(model_bytes)
        used_indices = sorted(
            {
                int(index)
                for index in estimator.tree_.feature
                if int(index) >= 0
            }
        )
        output[expert_name] = [
            model.descriptor_names[index] for index in used_indices
        ]
    return output


def _require_expected_hashes(
    args: argparse.Namespace,
    validation_report: dict[str, Any],
    proxy_report: dict[str, Any],
    current_gate_report: dict[str, Any],
    window_selection: dict[str, Any],
    input_hashes: dict[str, str],
) -> None:
    expected = {
        "validation_features": validation_report["artifacts"]["features"][
            "sha256"
        ],
        "validation_proxy": proxy_report["artifacts"][
            "validation_proxy_sha256"
        ],
        "multi_interest_model": proxy_report["artifacts"]["model_sha256"],
        "current_gate_model": current_gate_report[
            "confidence_gate_sha256"
        ],
        "window_lightgbm": window_selection[
            "secondary_probabilities_sha256"
        ],
        "window_recent100k": window_selection["experts"]["recent100k"][
            "validation_probabilities_sha256"
        ],
        "window_recent200k": window_selection["experts"]["recent200k"][
            "validation_probabilities_sha256"
        ],
        "window_recent200k_decay100k": window_selection["experts"][
            "recent200k_decay100k"
        ]["validation_probabilities_sha256"],
        "frozen_dataset1_csv": current_gate_report["dataset1_sha256"],
    }
    for name, expected_sha in expected.items():
        if input_hashes[name] != expected_sha:
            raise ValueError(
                f"{name} SHA-256 differs: "
                f"{input_hashes[name]} != {expected_sha}"
            )
    if not 0.0 < args.setwise_weight < 1.0:
        raise ValueError("setwise weight must be between zero and one")


def _metric_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in ("full", "slice_0", "slice_1", "slice_2")
    }


def _require_metrics_close(
    actual: dict[str, float],
    expected: dict[str, float],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if not np.isclose(
            actual[key],
            expected_value,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"{label} {key} differs: "
                f"{actual[key]} != {expected_value}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
