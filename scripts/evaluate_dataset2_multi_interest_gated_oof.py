from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset, set_model_state
from jgrec.rankers.hybrid.fusion import FusionMLP
from jgrec.rankers.hybrid.fusion_analysis import ranking_mrr_three_slices
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.multi_interest_gate import (
    MULTI_INTEREST_GATE_DESCRIPTOR_NAMES,
    ConfidenceGateConfig,
    blocked_temporal_oof_gate,
    confidence_gate_descriptors,
    passes_stability_gate,
    reciprocal_ranks,
    select_stable_high_confidence_trial,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

EXPECTED_BASE_SHAPE = (20_000, 100, 63)
EXPECTED_PROXY_SHAPE = (20_000, 100, 9)


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
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-features", required=True, type=Path)
    parser.add_argument("--validation-proxy", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--proxy-report", required=True, type=Path)
    parser.add_argument("--champion-setwise-model", required=True, type=Path)
    parser.add_argument("--candidate-setwise-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--setwise-weight", type=float, default=0.80)
    parser.add_argument("--minimum-full-delta", type=float, default=0.002)
    parser.add_argument(
        "--minimum-confidence-threshold",
        type=float,
        default=0.005,
    )
    parser.add_argument("--maximum-gate-coverage", type=float, default=0.35)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    jt.flags.use_cuda = 1

    validation_report = _read_json(args.validation_cache_report)
    proxy_report = _read_json(args.proxy_report)
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
    _require_hash(
        args.validation_features,
        validation_report["artifacts"]["features"]["sha256"],
        "validation features",
    )
    _require_hash(
        args.validation_proxy,
        proxy_report["artifacts"]["validation_proxy_sha256"],
        "validation proxy",
    )

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if tuple(validation_report["feature_names"]) != feature_names:
        raise ValueError("checkpoint and validation feature schemas differ")
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("checkpoint has no Dataset2 LightGBM")

    frozen = {
        "status": "frozen_before_prediction",
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_features": str(args.validation_features.resolve()),
        "validation_features_sha256": _sha256(args.validation_features),
        "validation_proxy": str(args.validation_proxy.resolve()),
        "validation_proxy_sha256": _sha256(args.validation_proxy),
        "champion_setwise_model": str(args.champion_setwise_model.resolve()),
        "champion_setwise_sha256": _sha256(args.champion_setwise_model),
        "candidate_setwise_model": str(args.candidate_setwise_model.resolve()),
        "candidate_setwise_sha256": _sha256(args.candidate_setwise_model),
        "setwise_weight": args.setwise_weight,
        "fold_count": 3,
        "minimum_full_delta": args.minimum_full_delta,
        "minimum_confidence_threshold": args.minimum_confidence_threshold,
        "maximum_gate_coverage": args.maximum_gate_coverage,
        "all_folds_non_decreasing": True,
        "all_three_slices_non_decreasing": True,
        "gate_grid": {
            "max_depth": [1, 2, 3],
            "min_samples_leaf": [500, 1000, 2000],
            "minimum_predicted_lift": [0.0, 0.0025, 0.005, 0.01],
        },
        "descriptor_names": list(MULTI_INTEREST_GATE_DESCRIPTOR_NAMES),
        "labels_in_descriptor_schema": False,
        "query_level_routing": True,
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)

    lgbm = _softmax(
        predict_logits_lgbm(lgbm_result.model_text, base_features)
    )
    champion_setwise = _load_setwise_probabilities(
        args.champion_setwise_model,
        base_features,
        args.batch_size,
    )
    candidate_setwise = _load_setwise_probabilities(
        args.candidate_setwise_model,
        _AugmentedFeatures(base_features, proxy_features),
        args.batch_size,
    )
    champion = (
        args.setwise_weight * champion_setwise
        + (1.0 - args.setwise_weight) * lgbm
    )
    candidate = (
        args.setwise_weight * candidate_setwise
        + (1.0 - args.setwise_weight) * lgbm
    )
    champion_metrics = ranking_mrr_three_slices(champion)
    candidate_metrics = ranking_mrr_three_slices(candidate)
    _require_metrics_close(
        champion_metrics,
        proxy_report["baseline"],
        "champion",
    )
    _require_metrics_close(
        candidate_metrics,
        proxy_report["candidate"],
        "multi-interest",
    )

    descriptors = confidence_gate_descriptors(
        base_features,
        feature_names,
        proxy_features,
        champion,
        candidate,
    )
    champion_rr = reciprocal_ranks(champion)
    candidate_rr = reciprocal_ranks(candidate)
    rr_delta = candidate_rr - champion_rr
    diagnosis = _diagnose(
        descriptors,
        champion_rr,
        candidate_rr,
    )

    trials: list[dict[str, Any]] = []
    trial_results: list[Any] = []
    for max_depth in frozen["gate_grid"]["max_depth"]:
        for min_samples_leaf in frozen["gate_grid"]["min_samples_leaf"]:
            for minimum_predicted_lift in frozen["gate_grid"][
                "minimum_predicted_lift"
            ]:
                config = ConfidenceGateConfig(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    minimum_predicted_lift=minimum_predicted_lift,
                )
                result = blocked_temporal_oof_gate(
                    descriptors,
                    champion,
                    candidate,
                    config,
                    fold_count=3,
                    seed=60,
                )
                passed = passes_stability_gate(
                    full_delta=result.full_delta,
                    fold_deltas=result.fold_deltas,
                    slice_deltas=result.slice_deltas,
                    minimum_full_delta=args.minimum_full_delta,
                )
                trials.append(
                    {
                        "config": {
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "minimum_predicted_lift": minimum_predicted_lift,
                        },
                        "champion_mrr": result.champion_mrr,
                        "candidate_mrr": result.candidate_mrr,
                        "gated_mrr": result.gated_mrr,
                        "full_delta": result.full_delta,
                        "fold_deltas": list(result.fold_deltas),
                        "slice_deltas": list(result.slice_deltas),
                        "fold_coverage": list(result.fold_coverage),
                        "coverage": float(result.use_candidate.mean()),
                        "passed": passed,
                    }
                )
                trial_results.append(result)

    stable_indices = [
        index for index, trial in enumerate(trials) if trial["passed"]
    ]
    selected_index = select_stable_high_confidence_trial(
        trials,
        minimum_predicted_lift=args.minimum_confidence_threshold,
        maximum_coverage=args.maximum_gate_coverage,
    )
    gate_passed = selected_index is not None
    if selected_index is None:
        selected_index = max(
            range(len(trials)),
            key=lambda index: (
                min(trials[index]["fold_deltas"]),
                trials[index]["full_delta"],
                -trials[index]["coverage"],
            ),
        )
    selected = trials[selected_index]
    selected_result = trial_results[selected_index]
    np.savez_compressed(
        args.output_dir / "paired-query-diagnostics.npz",
        champion_rr=champion_rr.astype(np.float32),
        candidate_rr=candidate_rr.astype(np.float32),
        rr_delta=rr_delta.astype(np.float32),
        descriptors=descriptors,
        selected_use_candidate=selected_result.use_candidate,
        selected_predicted_lift=selected_result.predicted_lift.astype(
            np.float32
        ),
        descriptor_names=np.asarray(
            MULTI_INTEREST_GATE_DESCRIPTOR_NAMES,
            dtype="U64",
        ),
    )
    report = {
        "status": "passed" if gate_passed else "stopped",
        "gate_passed": gate_passed,
        "package_authorized": gate_passed,
        "package_generated": False,
        "frozen_config": frozen,
        "champion": champion_metrics,
        "multi_interest": candidate_metrics,
        "paired_diagnosis": diagnosis,
        "selected": selected,
        "stable_trial_count": len(stable_indices),
        "high_confidence_stable_trial_count": sum(
            1
            for trial in trials
            if trial["passed"]
            and trial["config"]["minimum_predicted_lift"]
            >= args.minimum_confidence_threshold
            and trial["coverage"] <= args.maximum_gate_coverage
        ),
        "trial_count": len(trials),
        "trials": trials,
        "decision": (
            "continue_to_production_gate"
            if gate_passed
            else "stop_multi_interest_direction"
        ),
        "artifacts": {
            "paired_query_diagnostics_sha256": _sha256(
                args.output_dir / "paired-query-diagnostics.npz"
            )
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "gated-oof-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _diagnose(
    descriptors: np.ndarray,
    champion_rr: np.ndarray,
    candidate_rr: np.ndarray,
) -> dict[str, Any]:
    delta = candidate_rr - champion_rr
    index = {
        name: position
        for position, name in enumerate(MULTI_INTEREST_GATE_DESCRIPTOR_NAMES)
    }
    source_activity = descriptors[:, index["source_activity"]]
    prior_strength = descriptors[:, index["prior_strength_max"]]
    source_quantiles = np.quantile(source_activity, (0.25, 0.5, 0.75))
    prior_quantiles = np.quantile(prior_strength, (0.25, 0.5, 0.75))
    masks = {
        "repeat_edge_present": (
            descriptors[:, index["repeat_candidate_fraction"]] > 0.0
        ),
        "new_edges_only": (
            descriptors[:, index["repeat_candidate_fraction"]] <= 0.0
        ),
        "cold_target_present": (
            descriptors[:, index["target_unseen_fraction"]] > 0.0
        ),
        "all_targets_seen": (
            descriptors[:, index["target_unseen_fraction"]] <= 0.0
        ),
        "recent_memory_hit": (
            descriptors[:, index["memory_recent_hit_fraction"]] > 0.0
        ),
        "no_recent_memory_hit": (
            descriptors[:, index["memory_recent_hit_fraction"]] <= 0.0
        ),
        "short_memory_stronger": (
            descriptors[:, index["memory_short_minus_long"]] > 0.0
        ),
        "long_memory_not_weaker": (
            descriptors[:, index["memory_short_minus_long"]] <= 0.0
        ),
    }
    for name, values, quantiles in (
        ("source_activity", source_activity, source_quantiles),
        ("prior_strength", prior_strength, prior_quantiles),
    ):
        lower, middle, upper = quantiles
        masks[f"{name}_q1"] = values <= lower
        masks[f"{name}_q2"] = (values > lower) & (values <= middle)
        masks[f"{name}_q3"] = (values > middle) & (values <= upper)
        masks[f"{name}_q4"] = values > upper
    return {
        "rows": int(delta.size),
        "candidate_better_rows": int(np.sum(delta > 0.0)),
        "candidate_worse_rows": int(np.sum(delta < 0.0)),
        "tie_rows": int(np.sum(delta == 0.0)),
        "candidate_better_rate": float(np.mean(delta > 0.0)),
        "candidate_worse_rate": float(np.mean(delta < 0.0)),
        "mean_rr_delta": float(delta.mean()),
        "oracle_query_gate_delta": float(np.maximum(delta, 0.0).mean()),
        "segments": {
            name: _segment_metrics(mask, champion_rr, candidate_rr)
            for name, mask in masks.items()
        },
    }


def _segment_metrics(
    mask: np.ndarray,
    champion_rr: np.ndarray,
    candidate_rr: np.ndarray,
) -> dict[str, Any]:
    rows = int(mask.sum())
    if rows == 0:
        return {"rows": 0}
    delta = candidate_rr[mask] - champion_rr[mask]
    return {
        "rows": rows,
        "coverage": float(mask.mean()),
        "champion_mrr": float(champion_rr[mask].mean()),
        "candidate_mrr": float(candidate_rr[mask].mean()),
        "delta": float(delta.mean()),
        "better_rate": float(np.mean(delta > 0.0)),
        "worse_rate": float(np.mean(delta < 0.0)),
    }


def _load_setwise_probabilities(
    path: Path,
    features: Any,
    batch_size: int,
) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    indices = tuple(int(value) for value in payload["feature_indices"])
    model = FusionMLP(
        input_dim=len(indices),
        hidden_dim=int(payload["hidden_dim"][0]),
    )
    set_model_state(
        model,
        {
            key.removeprefix("state__"): np.asarray(
                payload[key],
                dtype=np.float32,
            )
            for key in payload.files
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


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 differs: {actual} != {expected}")


def _require_metrics_close(
    actual: dict[str, float],
    expected: dict[str, float],
    label: str,
) -> None:
    for key, expected_value in expected.items():
        if abs(float(actual[key]) - float(expected_value)) > 1e-10:
            raise ValueError(
                f"{label} {key} differs: {actual[key]} != {expected_value}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
