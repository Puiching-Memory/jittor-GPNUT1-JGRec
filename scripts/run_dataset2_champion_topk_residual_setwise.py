from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import set_model_state
from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.champion_residual import (
    champion_hard_negative_indices,
    lambda_mrr_pair_weights,
    route_champion_topk_residual,
)
from jgrec.rankers.hybrid.fusion import FusionMLP
from jgrec.rankers.hybrid.multi_interest_gate import reciprocal_ranks
from jgrec.rankers.hybrid.multi_interest_proxy import (
    MULTI_INTEREST_FEATURE_NAMES,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

EXPECTED_BASE_SHAPE = (20_000, 100, 63)
EXPECTED_PROXY_SHAPE = (20_000, 100, 9)
TRAIN_ROWS = (0, 6_667)
SELECTION_ROWS = (6_667, 13_334)
FORWARD_ROWS = (13_334, 20_000)
TOP_K_GRID = (10, 20)
SWITCH_GAIN_GRID = (0.05, 0.10, 0.20, 0.40)
SEED = 60
EXPECTED_SELECTION_BASELINE_MRR = 0.5510080326704802
EXPECTED_DATASET1_SHA256 = (
    "6760fd966fde3a8de4693d06f27ec0f2458e49c44e2ea48fc79e1393434e6e2b"
)


@dataclass(frozen=True)
class ResidualTrainingConfig:
    epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    hidden_dim: int = 32
    context_transform_version: int = 1


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
    if tuple(MULTI_INTEREST_FEATURE_NAMES) != tuple(
        proxy_report["frozen_config"]["proxy_feature_names"]
    ):
        raise ValueError("multi-interest proxy feature schema differs")

    champion_path = Path(
        source_selection["artifact_paths"]["scores_current_gate"]
    )
    input_paths = {
        "source_frozen_config": source_frozen_path,
        "source_selection_report": source_selection_path,
        "source_selection_sha256": source_selection_sha_path,
        "validation_features": args.validation_features,
        "validation_cache_report": args.validation_cache_report,
        "validation_proxy": args.validation_proxy,
        "proxy_report": args.proxy_report,
        "champion_scores": champion_path,
        "frozen_dataset1_csv": args.frozen_dataset1_csv,
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
    training_config = ResidualTrainingConfig()
    frozen = {
        "status": "frozen_before_training",
        "protocol_version": 1,
        "experiment": "champion_topk_residual_setwise",
        "baseline": "current_multi_interest_confidence_gate",
        "online_baseline_score": 1.3521011401636023,
        "positive_candidate_column": 0,
        "top_k_grid": list(TOP_K_GRID),
        "switch_gain_grid": list(SWITCH_GAIN_GRID),
        "training_config": asdict(training_config),
        "source_feature_count": (
            EXPECTED_BASE_SHAPE[-1] + EXPECTED_PROXY_SHAPE[-1]
        ),
        "setwise_feature_count": (
            (
                EXPECTED_BASE_SHAPE[-1]
                + EXPECTED_PROXY_SHAPE[-1]
            )
            * 3
        ),
        "setwise_context": "raw + raw-row_mean + raw-row_max",
        "training_group": (
            "positive candidate 0 plus champion top-k negatives"
        ),
        "loss": (
            "static champion-rank delta-MRR weighted pairwise softplus "
            "on champion log-score plus learned residual"
        ),
        "inference": (
            "rerank only champion top-k; reuse the champion top-k score "
            "multiset; exact champion fallback unless top1 switch gain "
            "reaches threshold"
        ),
        "train_rows": list(TRAIN_ROWS),
        "selection_rows": list(SELECTION_ROWS),
        "forward_rows": list(FORWARD_ROWS),
        "selection_uses_forward_rows": False,
        "selection_gate": {
            "minimum_mrr_delta": args.minimum_selection_delta,
            "maximum_coverage": args.maximum_coverage,
        },
        "forward_gate": {
            "minimum_mrr_delta": 0.001,
            "maximum_coverage": 0.25,
        },
        "tie_break": (
            "higher slice1 delta, then lower coverage, then frozen "
            "top-k/threshold order"
        ),
        "input_paths": {
            name: str(path.resolve()) for name, path in input_paths.items()
        },
        "input_sha256": input_hashes,
        "base_feature_names": list(base_feature_names),
        "proxy_feature_names": list(MULTI_INTEREST_FEATURE_NAMES),
        "frozen_dataset1_csv_sha256": input_hashes[
            "frozen_dataset1_csv"
        ],
    }
    frozen_path = args.output_dir / "frozen-config.json"
    _write_json_atomic(frozen_path, frozen)
    frozen_sha = _sha256(frozen_path)
    print(json.dumps(frozen, ensure_ascii=False, sort_keys=True), flush=True)

    base_features, proxy_features, champion_scores = _load_inputs(
        args.validation_features,
        args.validation_proxy,
        champion_path,
    )
    baseline_selection_mrr = float(
        np.mean(
            reciprocal_ranks(
                champion_scores[slice(*SELECTION_ROWS)]
            )
        )
    )
    if not np.isclose(
        baseline_selection_mrr,
        EXPECTED_SELECTION_BASELINE_MRR,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "current champion slice1 MRR differs: "
            f"{baseline_selection_mrr}"
        )
    feature_view = SetwiseFeatureView(
        _AugmentedFeatures(base_features, proxy_features),
        transform_version=training_config.context_transform_version,
    )
    jt.flags.use_cuda = 1

    model_reports: dict[str, Any] = {}
    residual_scores_by_top_k: dict[int, np.ndarray] = {}
    artifact_paths: dict[str, Path] = {}
    for top_k in TOP_K_GRID:
        model_started = time.time()
        model, mean, std, history = _fit_residual_model(
            feature_view,
            champion_scores,
            train_rows=TRAIN_ROWS,
            top_k=top_k,
            config=training_config,
            seed=SEED,
        )
        model_path = artifacts_dir / f"residual-top{top_k}.npz"
        _save_residual_model(
            model_path,
            model=model,
            mean=mean,
            std=std,
            top_k=top_k,
            config=training_config,
            train_rows=TRAIN_ROWS,
            seed=SEED,
        )
        del model
        gc.collect()
        release_memory()
        reloaded, loaded_mean, loaded_std, loaded_top_k = (
            _load_residual_model(model_path)
        )
        if loaded_top_k != top_k:
            raise RuntimeError("residual model top-k round-trip differs")
        residual_scores = _predict_residual_scores(
            reloaded,
            feature_view,
            loaded_mean,
            loaded_std,
            batch_size=training_config.batch_size,
        )
        residual_path = (
            artifacts_dir / f"validation-residual-top{top_k}.npy"
        )
        np.save(residual_path, residual_scores)
        residual_scores_by_top_k[top_k] = residual_scores
        artifact_paths[f"model_top{top_k}"] = model_path
        artifact_paths[f"residual_top{top_k}"] = residual_path
        model_reports[f"top{top_k}"] = {
            "top_k": top_k,
            "model": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "validation_residual": str(residual_path.resolve()),
            "validation_residual_sha256": _sha256(residual_path),
            "train_rows": list(TRAIN_ROWS),
            "history": history,
            "elapsed_seconds": time.time() - model_started,
        }
        print(
            f"[champion-residual] top_k={top_k} "
            f"loss={history[-1]['loss']:.6f} "
            f"elapsed={model_reports[f'top{top_k}']['elapsed_seconds']:.1f}s",
            flush=True,
        )
        del reloaded
        gc.collect()
        release_memory()

    selected, trials = _select_on_slice_one(
        champion_scores,
        residual_scores_by_top_k,
        minimum_delta=args.minimum_selection_delta,
        maximum_coverage=args.maximum_coverage,
    )
    selection_report: dict[str, Any] = {
        "status": (
            "locked_before_forward_gate"
            if selected is not None
            else "no_eligible_candidate"
        ),
        "frozen_config": str(frozen_path.resolve()),
        "frozen_config_sha256": frozen_sha,
        "train_rows": list(TRAIN_ROWS),
        "selection_rows": list(SELECTION_ROWS),
        "forward_rows": list(FORWARD_ROWS),
        "selection_uses_forward_rows": False,
        "baseline": "current_gate",
        "baseline_selection_mrr": baseline_selection_mrr,
        "models": model_reports,
        "trials": trials,
        "artifact_paths": {
            name: str(path.resolve())
            for name, path in artifact_paths.items()
        },
        "artifact_sha256": {
            name: _sha256(path) for name, path in artifact_paths.items()
        },
        "elapsed_seconds": time.time() - started,
    }
    if selected is not None:
        selection_report["selected"] = selected
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
    return 0 if selected is not None else 2


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
        raise RuntimeError("no eligible locked residual candidate to gate")
    if selection["selection_uses_forward_rows"] is not False:
        raise RuntimeError("selection report used forward rows")
    if selection["forward_rows"] != list(FORWARD_ROWS):
        raise RuntimeError("forward row contract differs")
    if _sha256(frozen_path) != selection["frozen_config_sha256"]:
        raise ValueError("frozen config SHA-256 differs")
    for name, path_text in selection["artifact_paths"].items():
        if _sha256(Path(path_text)) != selection["artifact_sha256"][name]:
            raise ValueError(f"selection artifact SHA-256 differs: {name}")
    for name, path_text in frozen["input_paths"].items():
        if _sha256(Path(path_text)) != frozen["input_sha256"][name]:
            raise ValueError(f"frozen input SHA-256 differs: {name}")

    training_config = ResidualTrainingConfig(
        **frozen["training_config"]
    )
    base_features, proxy_features, champion_scores = _load_inputs(
        Path(frozen["input_paths"]["validation_features"]),
        Path(frozen["input_paths"]["validation_proxy"]),
        Path(frozen["input_paths"]["champion_scores"]),
    )
    feature_view = SetwiseFeatureView(
        _AugmentedFeatures(base_features, proxy_features),
        transform_version=training_config.context_transform_version,
    )
    selected = selection["selected"]
    top_k = int(selected["top_k"])
    threshold = float(selected["minimum_switch_gain"])
    jt.flags.use_cuda = 1
    model, mean, std, history = _fit_residual_model(
        feature_view,
        champion_scores,
        train_rows=(TRAIN_ROWS[0], SELECTION_ROWS[1]),
        top_k=top_k,
        config=training_config,
        seed=SEED,
    )
    final_model_path = args.output_dir / "artifacts" / "final-prefix-model.npz"
    _save_residual_model(
        final_model_path,
        model=model,
        mean=mean,
        std=std,
        top_k=top_k,
        config=training_config,
        train_rows=(TRAIN_ROWS[0], SELECTION_ROWS[1]),
        seed=SEED,
    )
    residual_scores = _predict_residual_scores(
        model,
        feature_view,
        mean,
        std,
        batch_size=training_config.batch_size,
    )
    residual_path = (
        args.output_dir / "artifacts" / "final-prefix-validation-residual.npy"
    )
    np.save(residual_path, residual_scores)

    forward_slice = slice(*FORWARD_ROWS)
    forward_routed = route_champion_topk_residual(
        champion_scores[forward_slice],
        residual_scores[forward_slice],
        top_k=top_k,
        minimum_switch_gain=threshold,
    )
    forward_champion = champion_scores[forward_slice]
    fallback_mrr = float(
        np.mean(reciprocal_ranks(forward_champion))
    )
    candidate_mrr = float(
        np.mean(reciprocal_ranks(forward_routed.scores))
    )
    delta = candidate_mrr - fallback_mrr
    coverage = float(np.mean(forward_routed.use_residual))
    safety = _routing_safety(
        forward_champion,
        forward_routed.scores,
        forward_routed.use_residual,
        top_k=top_k,
    )
    gate_passed = bool(
        delta + 1e-12 >= args.minimum_forward_delta
        and coverage <= args.maximum_coverage + 1e-12
        and all(safety.values())
    )

    routed_all = route_champion_topk_residual(
        champion_scores,
        residual_scores,
        top_k=top_k,
        minimum_switch_gain=threshold,
    )
    routed_path = (
        args.output_dir / "artifacts" / "validation-routed-residual.npy"
    )
    np.save(routed_path, routed_all.scores)
    counts = {
        "current_gate": int(
            np.sum(~forward_routed.use_residual)
        ),
        f"residual_top{top_k}": int(
            np.sum(forward_routed.use_residual)
        ),
    }
    baseline_metrics = _three_slice_metrics(champion_scores)
    candidate_metrics = _three_slice_metrics(routed_all.scores)
    report = {
        "status": "accepted" if gate_passed else "rejected",
        "gate_passed": gate_passed,
        "package_authorized": gate_passed,
        "package_generated": False,
        "selection_report": str(selection_path.resolve()),
        "selection_report_sha256": expected_selection_sha,
        "selected": selected,
        "prefix_train_rows": [TRAIN_ROWS[0], SELECTION_ROWS[1]],
        "training_history": history,
        "forward_rows": list(FORWARD_ROWS),
        "forward": {
            "fallback_mrr": fallback_mrr,
            "candidate_mrr": candidate_mrr,
            "delta": delta,
            "minimum_delta": args.minimum_forward_delta,
            "coverage": coverage,
            "maximum_coverage": args.maximum_coverage,
            "safety": safety,
            "selected_expert_counts": counts,
        },
        "diagnostic_full_metrics_after_prefix_refit": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": _metric_delta(candidate_metrics, baseline_metrics),
        },
        "final_prefix_model": str(final_model_path.resolve()),
        "final_prefix_model_sha256": _sha256(final_model_path),
        "validation_residual": str(residual_path.resolve()),
        "validation_residual_sha256": _sha256(residual_path),
        "validation_routed": str(routed_path.resolve()),
        "validation_routed_sha256": _sha256(routed_path),
        "frozen_dataset1_csv_sha256": frozen[
            "frozen_dataset1_csv_sha256"
        ],
    }
    _write_json_atomic(args.output_dir / "evaluation-report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if gate_passed else 2


def _fit_residual_model(
    feature_view: Any,
    champion_scores: np.ndarray,
    *,
    train_rows: tuple[int, int],
    top_k: int,
    config: ResidualTrainingConfig,
    seed: int,
) -> tuple[FusionMLP, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_start, train_stop = train_rows
    if not 0 <= train_start < train_stop <= feature_view.shape[0]:
        raise ValueError("residual training rows are invalid")
    train_scores = np.asarray(
        champion_scores[train_start:train_stop],
        dtype=np.float32,
    )
    hard_negatives = champion_hard_negative_indices(
        train_scores,
        top_k=top_k,
    )
    group_indices = np.concatenate(
        (
            np.zeros((train_scores.shape[0], 1), dtype=np.int64),
            hard_negatives,
        ),
        axis=1,
    )
    pair_weights = lambda_mrr_pair_weights(
        train_scores,
        hard_negatives,
    )
    mean, std = _group_feature_normalizer(
        feature_view,
        group_indices,
        row_offset=train_start,
        batch_size=config.batch_size,
    )
    model = FusionMLP(
        input_dim=feature_view.shape[-1],
        hidden_dim=config.hidden_dim,
    )
    _initialize_model(model, seed=seed)
    optimizer = jt.nn.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(train_scores.shape[0])
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        batch_count = 0
        for start in range(0, len(order), config.batch_size):
            relative_rows = order[start : start + config.batch_size]
            global_rows = relative_rows + train_start
            batch_context = np.asarray(
                feature_view[global_rows],
                dtype=np.float32,
            )
            batch_groups = group_indices[relative_rows]
            selected = _gather_candidates(batch_context, batch_groups)
            normalized = ((selected - mean) / std).astype(
                np.float32,
                copy=False,
            )
            group_scores = _gather_candidates_2d(
                train_scores[relative_rows],
                batch_groups,
            )
            base_logits = np.log(
                np.maximum(group_scores, np.float32(1e-12))
            ).astype(np.float32, copy=False)
            weights = pair_weights[relative_rows]
            denominator = float(np.sum(weights, dtype=np.float64))
            if denominator <= 0.0:
                continue

            residual = model(jt.array(normalized, dtype=jt.float32))
            base = jt.array(base_logits, dtype=jt.float32)
            margins = (
                base[:, :1] + residual[:, :1]
                - base[:, 1:] - residual[:, 1:]
            )
            weight_var = jt.array(weights, dtype=jt.float32)
            loss = (
                jt.nn.softplus(-margins) * weight_var
            ).sum() / denominator
            optimizer.step(loss)
            loss_value = float(loss.item())
            weighted_loss_sum += loss_value * denominator
            weight_sum += denominator
            batch_count += 1
            del (
                batch_context,
                selected,
                normalized,
                group_scores,
                base_logits,
                residual,
                base,
                margins,
                weight_var,
                loss,
            )
            if batch_count % 16 == 0:
                release_memory()
        epoch_loss = weighted_loss_sum / max(weight_sum, 1e-12)
        if not math.isfinite(epoch_loss):
            raise FloatingPointError(
                f"non-finite residual loss at epoch {epoch}"
            )
        history.append(
            {
                "epoch": epoch,
                "loss": float(epoch_loss),
                "batches": batch_count,
                "pair_weight_sum": float(weight_sum),
            }
        )
        print(
            f"[champion-residual] top_k={top_k} epoch={epoch} "
            f"loss={epoch_loss:.6f} batches={batch_count}",
            flush=True,
        )
        release_memory()
    return model, mean, std, history


def _group_feature_normalizer(
    feature_view: Any,
    group_indices: np.ndarray,
    *,
    row_offset: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_sum: np.ndarray | None = None
    feature_sq_sum: np.ndarray | None = None
    count = 0
    for start in range(0, group_indices.shape[0], batch_size):
        end = min(start + batch_size, group_indices.shape[0])
        context = np.asarray(
            feature_view[row_offset + start : row_offset + end],
            dtype=np.float32,
        )
        selected = _gather_candidates(
            context,
            group_indices[start:end],
        ).astype(np.float64, copy=False)
        flat = selected.reshape((-1, selected.shape[-1]))
        batch_sum = np.sum(flat, axis=0)
        batch_sq_sum = np.sum(flat * flat, axis=0)
        if feature_sum is None:
            feature_sum = batch_sum
            feature_sq_sum = batch_sq_sum
        else:
            feature_sum += batch_sum
            feature_sq_sum += batch_sq_sum
        count += flat.shape[0]
        del context, selected, flat, batch_sum, batch_sq_sum
    if count <= 0 or feature_sum is None or feature_sq_sum is None:
        raise ValueError("residual feature normalizer received no rows")
    mean64 = feature_sum / count
    variance64 = np.maximum(
        feature_sq_sum / count - mean64 * mean64,
        0.0,
    )
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance64).astype(np.float32)
    std[std < np.float32(1e-6)] = 1.0
    return mean, std


def _predict_residual_scores(
    model: FusionMLP,
    feature_view: Any,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(feature_view.shape[:2], dtype=np.float32)
    with jt.no_grad():
        for start in range(0, feature_view.shape[0], batch_size):
            end = min(start + batch_size, feature_view.shape[0])
            context = np.asarray(
                feature_view[start:end],
                dtype=np.float32,
            )
            normalized = ((context - mean) / std).astype(
                np.float32,
                copy=False,
            )
            residual = model(jt.array(normalized, dtype=jt.float32))
            output[start:end] = np.asarray(
                residual.numpy(),
                dtype=np.float32,
            )
            del context, normalized, residual
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("residual prediction is non-finite")
    return output


def _select_on_slice_one(
    champion_scores: np.ndarray,
    residual_scores_by_top_k: dict[int, np.ndarray],
    *,
    minimum_delta: float,
    maximum_coverage: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    selection_slice = slice(*SELECTION_ROWS)
    champion_selection = champion_scores[selection_slice]
    baseline_mrr = float(
        np.mean(reciprocal_ranks(champion_selection))
    )
    trials: list[dict[str, Any]] = []
    for config_index, (top_k, threshold) in enumerate(
        (top_k, threshold)
        for top_k in TOP_K_GRID
        for threshold in SWITCH_GAIN_GRID
    ):
        routed = route_champion_topk_residual(
            champion_selection,
            residual_scores_by_top_k[top_k][selection_slice],
            top_k=top_k,
            minimum_switch_gain=threshold,
        )
        candidate_mrr = float(
            np.mean(reciprocal_ranks(routed.scores))
        )
        delta = candidate_mrr - baseline_mrr
        coverage = float(np.mean(routed.use_residual))
        safety = _routing_safety(
            champion_selection,
            routed.scores,
            routed.use_residual,
            top_k=top_k,
        )
        eligible = bool(
            delta + 1e-12 >= minimum_delta
            and coverage <= maximum_coverage + 1e-12
            and all(safety.values())
        )
        trials.append(
            {
                "config_index": config_index,
                "top_k": top_k,
                "minimum_switch_gain": threshold,
                "fallback_mrr": baseline_mrr,
                "candidate_mrr": candidate_mrr,
                "delta": delta,
                "coverage": coverage,
                "routed_rows": int(np.sum(routed.use_residual)),
                "safety": safety,
                "eligible": eligible,
            }
        )
    eligible = [trial for trial in trials if trial["eligible"]]
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda trial: (
                trial["delta"],
                -trial["coverage"],
                -trial["config_index"],
            ),
        )
    return selected, trials


def _routing_safety(
    champion_scores: np.ndarray,
    routed_scores: np.ndarray,
    use_residual: np.ndarray,
    *,
    top_k: int,
) -> dict[str, bool]:
    champion = np.asarray(champion_scores)
    routed = np.asarray(routed_scores)
    use = np.asarray(use_residual, dtype=bool)
    fallback_exact = bool(
        np.array_equal(routed[~use], champion[~use])
    )
    score_multiset_preserved = bool(
        np.array_equal(
            np.sort(routed, axis=1),
            np.sort(champion, axis=1),
        )
    )
    top_indices = np.argsort(
        -champion,
        axis=1,
        kind="stable",
    )[:, :top_k]
    top_mask = np.zeros(champion.shape, dtype=bool)
    rows = np.arange(champion.shape[0])[:, None]
    top_mask[rows, top_indices] = True
    outside_topk_exact = bool(
        np.array_equal(routed[~top_mask], champion[~top_mask])
    )
    return {
        "fallback_exact": fallback_exact,
        "score_multiset_preserved": score_multiset_preserved,
        "outside_topk_exact": outside_topk_exact,
    }


def _load_inputs(
    base_path: Path,
    proxy_path: Path,
    champion_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.load(base_path, mmap_mode="r", allow_pickle=False)
    proxy = np.load(proxy_path, mmap_mode="r", allow_pickle=False)
    champion = np.load(
        champion_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if base.shape != EXPECTED_BASE_SHAPE:
        raise ValueError(f"unexpected validation features: {base.shape}")
    if proxy.shape != EXPECTED_PROXY_SHAPE:
        raise ValueError(f"unexpected validation proxy: {proxy.shape}")
    if champion.shape != EXPECTED_BASE_SHAPE[:2]:
        raise ValueError(f"unexpected champion scores: {champion.shape}")
    if not np.all(np.isfinite(champion)):
        raise ValueError("champion scores must be finite")
    return base, proxy, champion


def _gather_candidates(
    values: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    rows = np.arange(values.shape[0])[:, None]
    return values[rows, candidate_indices]


def _gather_candidates_2d(
    values: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    rows = np.arange(values.shape[0])[:, None]
    return values[rows, candidate_indices]


def _initialize_model(model: FusionMLP, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}
    for key, value in model.state_dict().items():
        shape = tuple(int(dimension) for dimension in value.shape)
        if key.endswith(".weight") and len(shape) >= 2:
            fan_in = int(shape[1] * np.prod(shape[2:], dtype=np.int64))
            bound = math.sqrt(1.0 / max(fan_in, 1))
            state[key] = rng.uniform(
                -bound,
                bound,
                size=shape,
            ).astype(np.float32)
        elif key.endswith(".bias"):
            state[key] = np.zeros(shape, dtype=np.float32)
        else:
            state[key] = np.asarray(
                value.numpy(),
                dtype=np.float32,
            ).copy()
    set_model_state(model, state)


def _save_residual_model(
    path: Path,
    *,
    model: FusionMLP,
    mean: np.ndarray,
    std: np.ndarray,
    top_k: int,
    config: ResidualTrainingConfig,
    train_rows: tuple[int, int],
    seed: int,
) -> None:
    payload = {
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "top_k": np.asarray([top_k], dtype=np.int32),
        "hidden_dim": np.asarray([config.hidden_dim], dtype=np.int32),
        "source_feature_count": np.asarray([72], dtype=np.int32),
        "setwise_feature_count": np.asarray([216], dtype=np.int32),
        "context_transform_version": np.asarray(
            [config.context_transform_version],
            dtype=np.int32,
        ),
        "train_rows": np.asarray(train_rows, dtype=np.int32),
        "training_seed": np.asarray([seed], dtype=np.int32),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(
                value.numpy(),
                dtype=np.float32,
            )
            for key, value in model.state_dict().items()
        }
    )
    np.savez_compressed(path, **payload)


def _load_residual_model(
    path: Path,
) -> tuple[FusionMLP, np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as source:
        payload = {
            key: np.asarray(source[key]) for key in source.files
        }
    if (
        int(payload["source_feature_count"][0]) != 72
        or int(payload["setwise_feature_count"][0]) != 216
        or int(payload["context_transform_version"][0]) != 1
    ):
        raise ValueError("residual model feature contract differs")
    model = FusionMLP(
        input_dim=216,
        hidden_dim=int(payload["hidden_dim"][0]),
    )
    set_model_state(
        model,
        {
            key.removeprefix("state__"): value
            for key, value in payload.items()
            if key.startswith("state__")
        },
    )
    return (
        model,
        np.asarray(payload["mean"], dtype=np.float32),
        np.asarray(payload["std"], dtype=np.float32),
        int(payload["top_k"][0]),
    )


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
        "champion_scores": source_selection["artifact_sha256"][
            "scores_current_gate"
        ],
        "frozen_dataset1_csv": EXPECTED_DATASET1_SHA256,
    }
    for name, expected_sha in expected.items():
        if input_hashes[name] != expected_sha:
            raise ValueError(
                f"{name} SHA-256 differs: "
                f"{input_hashes[name]} != {expected_sha}"
            )
    for name in (
        "validation_features",
        "validation_proxy",
        "frozen_dataset1_csv",
    ):
        if input_hashes[name] != source_frozen["input_sha256"][name]:
            raise ValueError(f"{name} differs from source champion input")


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
