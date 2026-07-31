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

import numpy as np

from jgrec.rankers.hybrid.multi_expert_gate import (
    MultiExpertGateConfig,
    expert_top1_feature_deltas,
    fit_multi_expert_gate,
    multi_expert_score_descriptors,
    predict_multi_expert_gate,
    route_multi_expert,
    select_multi_expert_config_on_forward_slice,
)
from jgrec.rankers.hybrid.multi_interest_gate import reciprocal_ranks
from jgrec.rankers.hybrid.multi_interest_proxy import (
    MULTI_INTEREST_FEATURE_NAMES,
)

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
RAW_TOP1_FEATURE_NAMES = (
    "dst_popularity",
    "dst_recency",
    "candidate_test_freq",
    "candidate_unseen_test_freq",
    "candidate_dst_pop_row_rank",
    "candidate_dst_recency_row_rank",
    "candidate_test_freq_row_rank",
    "target_pop_w020",
    "target_recency_w020",
    "source_profile_cosine_sum",
    "source_profile_recent_cosine_sum",
    "source_profile_recent_item2vec_cosine",
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
EXPECTED_DATASET1_SHA256 = (
    "6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("train-select")
    select.add_argument("--source-output-dir", required=True, type=Path)
    select.add_argument("--validation-features", required=True, type=Path)
    select.add_argument("--validation-cache-report", required=True, type=Path)
    select.add_argument("--validation-proxy", required=True, type=Path)
    select.add_argument("--proxy-report", required=True, type=Path)
    select.add_argument("--frozen-dataset1-csv", required=True, type=Path)
    select.add_argument("--output-dir", required=True, type=Path)
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

    source_frozen_path = args.source_output_dir / "frozen-config.json"
    source_selection_path = (
        args.source_output_dir / "selection-report.json"
    )
    source_selection_sha_path = (
        args.source_output_dir / "selection-report.sha256"
    )
    source_frozen = _read_json(source_frozen_path)
    source_selection = _read_json(source_selection_path)
    _validate_source_lock(
        source_frozen_path,
        source_selection_path,
        source_selection_sha_path,
        source_selection,
    )
    validation_report = _read_json(args.validation_cache_report)
    proxy_report = _read_json(args.proxy_report)
    base_feature_names = tuple(
        str(name) for name in validation_report["feature_names"]
    )
    if (
        len(base_feature_names) != EXPECTED_BASE_SHAPE[-1]
        or len(set(base_feature_names)) != len(base_feature_names)
    ):
        raise ValueError("validation feature schema differs")
    missing_raw = [
        name for name in RAW_TOP1_FEATURE_NAMES
        if name not in base_feature_names
    ]
    if missing_raw:
        raise ValueError(f"raw top1 features are missing: {missing_raw}")
    if tuple(MULTI_INTEREST_FEATURE_NAMES) != tuple(
        proxy_report["frozen_config"]["proxy_feature_names"]
    ):
        raise ValueError("multi-interest proxy feature schema differs")

    score_paths = {
        name: Path(
            source_selection["artifact_paths"][f"scores_{name}"]
        )
        for name in DESCRIPTOR_EXPERT_ORDER
    }
    input_paths = {
        "source_frozen_config": source_frozen_path,
        "source_selection_report": source_selection_path,
        "source_selection_sha256": source_selection_sha_path,
        "validation_features": args.validation_features,
        "validation_cache_report": args.validation_cache_report,
        "validation_proxy": args.validation_proxy,
        "proxy_report": args.proxy_report,
        "frozen_dataset1_csv": args.frozen_dataset1_csv,
        **{
            f"scores_{name}": path for name, path in score_paths.items()
        },
    }
    input_hashes = {
        name: _sha256(path) for name, path in input_paths.items()
    }
    _validate_inputs(
        source_frozen,
        source_selection,
        validation_report,
        proxy_report,
        input_hashes,
    )

    score_names, raw_delta_names, proxy_delta_names = _descriptor_schema()
    descriptor_names = (
        *score_names,
        *raw_delta_names,
        *proxy_delta_names,
    )
    if len(score_names) != 74 or len(descriptor_names) != 137:
        raise RuntimeError("unexpected multi-expert v3 descriptor count")
    frozen = {
        "status": "frozen_before_descriptor_materialization",
        "protocol_version": 3,
        "baseline": "current_multi_interest_confidence_gate",
        "online_baseline_score": 1.3521011401636023,
        "source_protocol": "score-only multi-expert confidence v2 r3",
        "source_selection_status": source_selection["status"],
        "descriptor_expert_order": list(DESCRIPTOR_EXPERT_ORDER),
        "alternative_expert_order": list(ALTERNATIVE_EXPERT_ORDER),
        "raw_top1_feature_names": list(RAW_TOP1_FEATURE_NAMES),
        "multi_interest_top1_feature_names": list(
            MULTI_INTEREST_FEATURE_NAMES
        ),
        "descriptor_groups": {
            "score_only": list(score_names),
            "raw_top1_delta": list(raw_delta_names),
            "multi_interest_top1_delta": list(proxy_delta_names),
        },
        "descriptor_names": list(descriptor_names),
        "descriptor_count": len(descriptor_names),
        "descriptors_are_score_only": False,
        "top1_delta_definition": (
            "tie-neutral mean(candidate feature on alternative top1 set) "
            "- mean(candidate feature on current_gate top1 set)"
        ),
        "labels_in_descriptors": False,
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
            "higher selection delta, then lower coverage, then frozen "
            "config order; top1 sets include all exact score ties; "
            "expert ties use frozen expert order"
        ),
        "input_paths": {
            name: str(path.resolve()) for name, path in input_paths.items()
        },
        "input_sha256": input_hashes,
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
        raise ValueError(
            f"unexpected validation features: {base_features.shape}"
        )
    if proxy_features.shape != EXPECTED_PROXY_SHAPE:
        raise ValueError(
            f"unexpected validation proxy: {proxy_features.shape}"
        )
    expert_scores = {
        name: np.load(path, mmap_mode="r", allow_pickle=False)
        for name, path in score_paths.items()
    }
    if any(
        scores.shape != EXPECTED_BASE_SHAPE[:2]
        for scores in expert_scores.values()
    ):
        raise ValueError("source expert score shapes differ")

    score_descriptors, actual_score_names = (
        multi_expert_score_descriptors(
            expert_scores,
            expert_order=DESCRIPTOR_EXPERT_ORDER,
        )
    )
    raw_deltas, actual_raw_names = expert_top1_feature_deltas(
        expert_scores,
        base_features,
        candidate_feature_names=base_feature_names,
        selected_feature_names=RAW_TOP1_FEATURE_NAMES,
        fallback_expert="current_gate",
        alternative_order=ALTERNATIVE_EXPERT_ORDER,
    )
    proxy_deltas, actual_proxy_names = expert_top1_feature_deltas(
        expert_scores,
        proxy_features,
        candidate_feature_names=tuple(MULTI_INTEREST_FEATURE_NAMES),
        selected_feature_names=tuple(MULTI_INTEREST_FEATURE_NAMES),
        fallback_expert="current_gate",
        alternative_order=ALTERNATIVE_EXPERT_ORDER,
    )
    if (
        actual_score_names != score_names
        or actual_raw_names != raw_delta_names
        or actual_proxy_names != proxy_delta_names
    ):
        raise RuntimeError("multi-expert v3 descriptor schema changed")
    descriptors = np.concatenate(
        (score_descriptors, raw_deltas, proxy_deltas),
        axis=1,
        dtype=np.float32,
    )
    if descriptors.shape != (EXPECTED_BASE_SHAPE[0], len(descriptor_names)):
        raise RuntimeError("multi-expert v3 descriptor shape differs")
    descriptor_path = artifacts_dir / "validation-descriptors-v3.npy"
    np.save(descriptor_path, descriptors)

    selection, trials = _select_on_slice_one(
        descriptors,
        expert_scores,
        descriptor_names=descriptor_names,
        minimum_delta=args.minimum_selection_delta,
        maximum_coverage=args.maximum_coverage,
    )
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
            "descriptors": str(descriptor_path.resolve()),
        },
        "artifact_sha256": {
            "descriptors": _sha256(descriptor_path),
        },
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
            key: value for key, value in selection.items() if key != "model"
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
        if _sha256(Path(path_text)) != selection["artifact_sha256"][name]:
            raise ValueError(f"artifact SHA-256 differs: {name}")
    for name, path_text in frozen["input_paths"].items():
        if _sha256(Path(path_text)) != frozen["input_sha256"][name]:
            raise ValueError(f"frozen input SHA-256 differs: {name}")

    descriptor_names = tuple(frozen["descriptor_names"])
    descriptors = np.load(
        selection["artifact_paths"]["descriptors"],
        mmap_mode="r",
        allow_pickle=False,
    )
    scores = {
        name: np.load(
            frozen["input_paths"][f"scores_{name}"],
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
    fallback_train_rr = reciprocal_ranks(fallback[train_slice])
    rewards = np.column_stack(
        [
            reciprocal_ranks(alternatives[name][train_slice])
            - fallback_train_rr
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
        args.output_dir / "artifacts" / "validation-routed-v3.npy"
    )
    np.save(prediction_path, routed_all.scores.astype(np.float32))
    baseline_metrics = _three_slice_metrics(fallback)
    candidate_metrics = _three_slice_metrics(routed_all.scores)
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


def _descriptor_schema(
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    dummy_scores = {
        name: np.ones((1, 2), dtype=np.float32)
        for name in DESCRIPTOR_EXPERT_ORDER
    }
    _, score_names = multi_expert_score_descriptors(
        dummy_scores,
        expert_order=DESCRIPTOR_EXPERT_ORDER,
    )
    _, raw_names = expert_top1_feature_deltas(
        dummy_scores,
        np.ones(
            (1, 2, len(RAW_TOP1_FEATURE_NAMES)),
            dtype=np.float32,
        ),
        candidate_feature_names=RAW_TOP1_FEATURE_NAMES,
        selected_feature_names=RAW_TOP1_FEATURE_NAMES,
        fallback_expert="current_gate",
        alternative_order=ALTERNATIVE_EXPERT_ORDER,
    )
    _, proxy_names = expert_top1_feature_deltas(
        dummy_scores,
        np.ones(
            (1, 2, len(MULTI_INTEREST_FEATURE_NAMES)),
            dtype=np.float32,
        ),
        candidate_feature_names=tuple(MULTI_INTEREST_FEATURE_NAMES),
        selected_feature_names=tuple(MULTI_INTEREST_FEATURE_NAMES),
        fallback_expert="current_gate",
        alternative_order=ALTERNATIVE_EXPERT_ORDER,
    )
    return score_names, raw_names, proxy_names


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


def _validate_source_lock(
    source_frozen_path: Path,
    source_selection_path: Path,
    source_selection_sha_path: Path,
    source_selection: dict[str, Any],
) -> None:
    expected_sha = source_selection_sha_path.read_text(
        encoding="utf-8"
    ).strip()
    if _sha256(source_selection_path) != expected_sha:
        raise ValueError("source selection report SHA-256 lock differs")
    if _sha256(source_frozen_path) != source_selection[
        "frozen_config_sha256"
    ]:
        raise ValueError("source frozen config SHA-256 differs")
    if source_selection["selection_uses_forward_rows"] is not False:
        raise RuntimeError("source selection report used forward rows")
    for name, path_text in source_selection["artifact_paths"].items():
        if _sha256(Path(path_text)) != source_selection[
            "artifact_sha256"
        ][name]:
            raise ValueError(f"source artifact SHA-256 differs: {name}")


def _validate_inputs(
    source_frozen: dict[str, Any],
    source_selection: dict[str, Any],
    validation_report: dict[str, Any],
    proxy_report: dict[str, Any],
    input_hashes: dict[str, str],
) -> None:
    expected = {
        "validation_features": validation_report["artifacts"]["features"][
            "sha256"
        ],
        "validation_proxy": proxy_report["artifacts"][
            "validation_proxy_sha256"
        ],
        "frozen_dataset1_csv": EXPECTED_DATASET1_SHA256,
        **{
            f"scores_{name}": source_selection["artifact_sha256"][
                f"scores_{name}"
            ]
            for name in DESCRIPTOR_EXPERT_ORDER
        },
    }
    for name, expected_sha in expected.items():
        if input_hashes[name] != expected_sha:
            raise ValueError(
                f"{name} SHA-256 differs: "
                f"{input_hashes[name]} != {expected_sha}"
            )
    source_expected = source_frozen["input_sha256"]
    for name in (
        "validation_features",
        "validation_proxy",
        "frozen_dataset1_csv",
    ):
        if input_hashes[name] != source_expected[name]:
            raise ValueError(f"{name} differs from source v2 input")


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


def _three_slice_metrics(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores)
    return {
        "full": float(np.mean(reciprocal_ranks(values))),
        "slice_0": float(
            np.mean(reciprocal_ranks(values[slice(*TRAIN_ROWS)]))
        ),
        "slice_1": float(
            np.mean(reciprocal_ranks(values[slice(*SELECTION_ROWS)]))
        ),
        "slice_2": float(
            np.mean(reciprocal_ranks(values[slice(*FORWARD_ROWS)]))
        ),
    }


def _metric_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in ("full", "slice_0", "slice_1", "slice_2")
    }


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
