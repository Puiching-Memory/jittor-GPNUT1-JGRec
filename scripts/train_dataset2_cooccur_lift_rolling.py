from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_interactions
from jgrec.listwise_mlp_exact_blend import (
    build_rolling_selection_manifest,
    materialize_fold_candidates,
)
from jgrec.rankers.hybrid.cooccur_lift import (
    COOCCUR_LIFT_FEATURE_NAMES,
    CooccurLiftAugmentedView,
    load_frozen_cooccur_lift_config,
    training_seed,
)
from jgrec.rankers.hybrid.cooccur_lift_native import (
    materialize_compact_cooccur_lift,
)
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    FusionResult,
    fit_fusion_mlp_listwise_fixed,
    predict_logits,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView
from jgrec.setwise_prob_seed_bag import (
    load_verified_source_baseline,
    validate_source_rolling_manifest,
)

EXPECTED_FEATURE_SHAPE = (200_000, 100, 63)
EXPECTED_MATRIX_SHAPE = EXPECTED_FEATURE_SHAPE[:2]
EXPECTED_LIFT_SHAPE = (*EXPECTED_MATRIX_SHAPE, 2)
CHAMPION_SETWISE_WEIGHT = 0.80


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the frozen causal cooccurrence-lift columns, train "
            "one 65-column Setwise auxiliary per exact-rolling fold, and "
            "write selector-ready score matrices plus diagnostic-only Q1 metrics."
        )
    )
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--short-none-scores", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--source-rolling-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    config = load_frozen_cooccur_lift_config(args.frozen_config)
    source_manifest = _read_json(args.source_rolling_manifest)
    source_folds = validate_source_rolling_manifest(source_manifest, config)

    args.output_dir.mkdir(parents=True)
    frozen_copy = args.output_dir / "precommitted-config.json"
    shutil.copyfile(args.frozen_config, frozen_copy)
    if _sha256(frozen_copy) != _sha256(args.frozen_config):
        raise RuntimeError("frozen config copy differs from source")
    started = time.time()

    paths = _cache_paths(args.train_cache_prefix)
    cache_report = _read_json(args.train_cache_report)
    features = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
    candidates = np.load(
        paths["candidates"],
        mmap_mode="r",
        allow_pickle=False,
    )
    destinations = np.load(
        paths["destinations"],
        mmap_mode="r",
        allow_pickle=False,
    )
    sources = np.load(paths["sources"], mmap_mode="r", allow_pickle=False)
    event_time = np.load(paths["times"], mmap_mode="r", allow_pickle=False)
    short_none = np.load(
        args.short_none_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    _validate_training_assets(
        features=features,
        candidates=candidates,
        destinations=destinations,
        sources=sources,
        event_time=event_time,
        short_none=short_none,
    )

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if feature_names != tuple(cache_report["feature_names"]):
        raise ValueError("checkpoint and cache feature schemas differ")
    gnn_column = feature_names.index("gnn_short")
    setwise_result = state.get("setwise_fusion_result")
    setwise_state = state.get("setwise_fusion_state")
    lgbm_result = state.get("lgbm_result")
    if setwise_result is None or setwise_state is None:
        raise ValueError("champion checkpoint has no Setwise head")
    if lgbm_result is None:
        raise ValueError("champion checkpoint has no LightGBM expert")
    if abs(float(lgbm_result.mlp_weight) - CHAMPION_SETWISE_WEIGHT) > 1e-12:
        raise ValueError("champion Setwise weight differs from 0.80")
    setwise_hidden_dim = int(state["setwise_hidden_dim"])
    del state
    gc.collect()

    interactions = read_interactions(args.train_csv).sort_by_time()
    if len(interactions) < EXPECTED_MATRIX_SHAPE[0]:
        raise ValueError("train.csv has fewer rows than the 200k cache")
    time_span = int(interactions.time[-1]) - int(interactions.time[0])
    short_window = float(time_span) * config.short_window_ratio
    if short_window <= 0:
        raise ValueError("realized short window must be positive")
    materialization_contract = {
        "status": "materializing_before_fold_training",
        "integration_id": config.integration_id,
        "precommitted_config": str(frozen_copy.resolve()),
        "precommitted_config_sha256": _sha256(frozen_copy),
        "train_csv": str(args.train_csv.resolve()),
        "train_csv_sha256": _sha256(args.train_csv),
        "train_time_min": int(interactions.time[0]),
        "train_time_max": int(interactions.time[-1]),
        "train_time_span": time_span,
        "short_window_ratio": config.short_window_ratio,
        "realized_short_window": short_window,
        "history_limit": config.history_limit,
        "index_build": {
            "build_transitions": False,
            "build_cooccurs": True,
            "cooccur_history_limit": config.cooccur_history_limit,
            "future_only_transition_cooccur": False,
            "cooccur_time_decay_ratio": 0.0,
            "cooccur_time_decay_score_reused": False,
            "storage_backend": (
                "TemporalInteractionIndex-exact causal streaming "
                "dense-triangular uint8 plus exact overflow"
            ),
        },
        "lift_shape": list(EXPECTED_LIFT_SHAPE),
        "lift_dtype": "float32",
        "positive_popularity_shape": [EXPECTED_MATRIX_SHAPE[0]],
        "positive_popularity_dtype": "int32",
        "candidate_metrics_read": False,
        "external_scores_read": False,
    }
    _write_json(
        args.output_dir / "materialization-contract.json",
        materialization_contract,
    )
    print(
        json.dumps(materialization_contract, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    lift_path = args.output_dir / "lift-features.npy"
    positive_popularity_path = args.output_dir / "positive-dst-causal-popularity.npy"
    materialization_started = time.time()
    native_contract = materialize_compact_cooccur_lift(
        interactions=interactions,
        sources=sources,
        candidates=candidates,
        destinations=destinations,
        event_time=event_time,
        short_window=short_window,
        lift_path=lift_path,
        positive_popularity_path=positive_popularity_path,
        progress_path=args.output_dir / "materialization-progress.json",
        work_dir=args.output_dir,
    )
    materialization_contract.update(
        {
            "status": "materialized_before_fold_training",
            "native_backend": native_contract,
            "materialization_elapsed_seconds": (
                time.time() - materialization_started
            ),
        }
    )
    _write_json(
        args.output_dir / "materialization-contract.json",
        materialization_contract,
    )
    lift_features = np.load(lift_path, mmap_mode="r", allow_pickle=False)
    positive_popularity = np.load(
        positive_popularity_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if lift_features.shape != EXPECTED_LIFT_SHAPE:
        raise ValueError(f"unexpected lift feature shape: {lift_features.shape}")

    run_contract = {
        "status": "training_without_model_or_weight_selection",
        "integration_id": config.integration_id,
        "precommitted_config": str(frozen_copy.resolve()),
        "precommitted_config_sha256": _sha256(frozen_copy),
        "source_rolling_manifest": str(args.source_rolling_manifest.resolve()),
        "source_rolling_manifest_sha256": _sha256(args.source_rolling_manifest),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_csv": str(args.train_csv.resolve()),
        "train_csv_sha256": materialization_contract["train_csv_sha256"],
        "train_cache": str(paths["features"].resolve()),
        "train_cache_sha256": _sha256(paths["features"]),
        "candidate_sidecar_sha256": _sha256(paths["candidates"]),
        "destination_sidecar_sha256": _sha256(paths["destinations"]),
        "source_sidecar_sha256": _sha256(paths["sources"]),
        "time_sidecar_sha256": _sha256(paths["times"]),
        "short_none_scores_sha256": _sha256(args.short_none_scores),
        "lift_features": str(lift_path.resolve()),
        "lift_features_sha256": _sha256(lift_path),
        "lift_features_shape": list(lift_features.shape),
        "lift_features_dtype": str(lift_features.dtype),
        "positive_dst_causal_popularity": str(positive_popularity_path.resolve()),
        "positive_dst_causal_popularity_sha256": _sha256(positive_popularity_path),
        "realized_short_window": short_window,
        "folds": [
            {
                "fold_id": fold_id,
                "train_rows": list(train_rows),
                "score_rows": list(score_rows),
                "training_seed": training_seed(config, fold_index),
            }
            for fold_index, (fold_id, train_rows, score_rows) in enumerate(config.folds)
        ],
        "weights": list(config.weights),
        "seed_salt": config.seed_salt,
        "setwise_epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "early_stop_patience": config.early_stop_patience,
        "selection_metric": config.selection_metric,
        "gnn_short_column": gnn_column,
        "base_feature_count": 63,
        "augmented_feature_count": 65,
        "setwise_context_feature_count": config.context_feature_count,
        "setwise_feature_indices": list(config.context_feature_indices),
        "setwise_hidden_dim": setwise_hidden_dim,
        "candidate_formula": ("candidate_w = (1 - w) * fold_champion_probability + w * auxiliary_probability"),
        "q1_diagnostic_only": True,
        "q1_can_select_weight_or_model": False,
        "global_ranking_metrics_computed_by_producer": False,
        "external_scores_read": False,
        "materialization_elapsed_seconds": (time.time() - materialization_started),
        "native_materializer": native_contract,
    }
    _write_json(args.output_dir / "run-contract.json", run_contract)
    print(json.dumps(run_contract, ensure_ascii=False, sort_keys=True), flush=True)

    jt.flags.use_cuda = 1
    fold_entries: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    q1_fold_reports: list[dict[str, Any]] = []
    pooled_q1 = {
        format(weight, "g"): {
            "baseline_reciprocal_rank_sum": 0.0,
            "candidate_reciprocal_rank_sum": 0.0,
            "query_count": 0,
        }
        for weight in config.weights
    }
    for fold_index, (
        fold_id,
        train_rows,
        score_rows,
    ) in enumerate(config.folds):
        fold_started = time.time()
        train_start, train_stop = train_rows
        score_start, score_stop = score_rows
        train_augmented = CooccurLiftAugmentedView(
            features[train_start:train_stop],
            short_none_scores=short_none[train_start:train_stop],
            gnn_short_column=gnn_column,
            lift_features=lift_features[train_start:train_stop],
        )
        score_augmented = CooccurLiftAugmentedView(
            features[score_start:score_stop],
            short_none_scores=short_none[score_start:score_stop],
            gnn_short_column=gnn_column,
            lift_features=lift_features[score_start:score_stop],
        )
        train_view = SetwiseFeatureView(train_augmented, transform_version=1)
        score_view = SetwiseFeatureView(score_augmented, transform_version=1)
        if train_view.shape[-1] != config.context_feature_count:
            raise ValueError("Setwise context view does not have 195 columns")

        seed = training_seed(config, fold_index)
        jt.set_global_seed(seed)
        setwise_config = FusionConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            hidden_dim=setwise_hidden_dim,
            selection_metric=config.selection_metric,
            early_stop_patience=config.early_stop_patience,
        )
        model, result, losses = fit_fusion_mlp_listwise_fixed(
            train_view,
            score_view,
            setwise_config,
            np.random.default_rng(seed),
            verbose=False,
            feature_indices=config.context_feature_indices,
            candidate_name=f"{config.integration_id}_{fold_id}",
        )
        auxiliary_probabilities = _predict_probabilities(
            model,
            score_view,
            result,
            batch_size=config.batch_size,
        )
        model_dir = args.output_dir / "models" / fold_id
        model_dir.mkdir(parents=True)
        model_path = model_dir / f"cooccur-lift-setwise-seed{seed}.npz"
        _save_fusion_result(
            model_path,
            result=result,
            hidden_dim=setwise_hidden_dim,
            source_feature_count=65,
            transform_version=1,
            seed=seed,
            salt=config.seed_salt,
        )
        auxiliary_path = args.output_dir / "auxiliary-probabilities" / fold_id / f"cooccur-lift-setwise-seed{seed}.npy"
        auxiliary_path.parent.mkdir(parents=True)
        np.save(
            auxiliary_path,
            np.asarray(auxiliary_probabilities, dtype=np.float32),
            allow_pickle=False,
        )
        del model, result, train_view, score_view
        _release_jittor()

        source_fold = source_folds[fold_index]
        baseline_probabilities = load_verified_source_baseline(
            source_fold,
            candidates[score_start:score_stop],
        )
        entry = materialize_fold_candidates(
            output_dir=args.output_dir / "scores" / fold_id,
            fold_id=fold_id,
            integration_id=config.integration_id,
            train_time_max=int(event_time[train_stop - 1]),
            score_time_min=int(event_time[score_start]),
            score_time_max=int(event_time[score_stop - 1]),
            baseline_scores=baseline_probabilities,
            auxiliary_scores=auxiliary_probabilities,
            candidate_ids=candidates[score_start:score_stop],
            weights=config.weights,
        )
        entry["train_rows"] = list(train_rows)
        entry["score_rows"] = list(score_rows)
        fold_entries.append(entry)

        q1_report = _q1_fold_diagnostic(
            fold_id=fold_id,
            positive_popularity=np.asarray(
                positive_popularity[score_start:score_stop],
                dtype=np.int64,
            ),
            baseline_scores=baseline_probabilities,
            candidate_descriptors=entry["candidates"],
            weights=config.weights,
        )
        q1_fold_reports.append(q1_report)
        _update_pooled_q1(pooled_q1, q1_report)
        fold_report = {
            "fold_id": fold_id,
            "train_rows": list(train_rows),
            "score_rows": list(score_rows),
            "training_seed": seed,
            "setwise_epochs": config.epochs,
            "losses": list(losses),
            "model": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "auxiliary_probabilities": str(auxiliary_path.resolve()),
            "auxiliary_probabilities_sha256": _sha256(auxiliary_path),
            "source_baseline_sha256": source_fold["baseline"]["sha256"],
            "score_artifacts": entry,
            "q1_diagnostic": q1_report,
            "global_ranking_metrics_computed_by_producer": False,
            "elapsed_seconds": time.time() - fold_started,
        }
        fold_reports.append(fold_report)
        _write_json(
            args.output_dir / "rolling-progress.json",
            {
                "status": "training",
                "completed_folds": len(fold_reports),
                "total_folds": len(config.folds),
                "folds": fold_reports,
                "external_scores_read": False,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(
            json.dumps(
                {
                    "fold_id": fold_id,
                    "completed_folds": len(fold_reports),
                    "total_folds": len(config.folds),
                    "elapsed_seconds": fold_report["elapsed_seconds"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        del (
            train_augmented,
            score_augmented,
            auxiliary_probabilities,
            baseline_probabilities,
        )
        _release_jittor()

    manifest_path = args.output_dir / "rolling-manifest.json"
    build_rolling_selection_manifest(
        integration_id=config.integration_id,
        fold_entries=fold_entries,
        output_path=manifest_path,
    )
    q1_report = {
        "schema_version": 1,
        "integration_id": config.integration_id,
        "status": "diagnostic_only_not_selection_input",
        "segment": (
            "per-fold stable lower quartile of causal positive-destination "
            "popularity; ties broken by chronological cache row order"
        ),
        "selection_authorized": False,
        "formula_or_window_change_authorized": False,
        "folds": q1_fold_reports,
        "pooled": _finalize_pooled_q1(pooled_q1),
        "external_scores_read": False,
    }
    q1_path = args.output_dir / "dst-pop-q1-diagnostic.json"
    _write_json(q1_path, q1_report)
    report = {
        "status": "complete",
        "integration_id": config.integration_id,
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "rolling_manifest": str(manifest_path.resolve()),
        "rolling_manifest_sha256": _sha256(manifest_path),
        "dst_pop_q1_diagnostic": str(q1_path.resolve()),
        "dst_pop_q1_diagnostic_sha256": _sha256(q1_path),
        "global_ranking_metrics_computed_by_producer": False,
        "external_scores_read": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "rolling-training-report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "integration_id": report["integration_id"],
                "fold_count": report["fold_count"],
                "rolling_manifest": report["rolling_manifest"],
                "rolling_manifest_sha256": report["rolling_manifest_sha256"],
                "dst_pop_q1_diagnostic": report["dst_pop_q1_diagnostic"],
                "external_scores_read": False,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _cache_paths(prefix: Path) -> dict[str, Path]:
    return {
        "features": Path(f"{prefix}.train.npy"),
        "candidates": Path(f"{prefix}.train-candidates.npy"),
        "destinations": Path(f"{prefix}.train-dst.npy"),
        "sources": Path(f"{prefix}.train-src.npy"),
        "times": Path(f"{prefix}.train-time.npy"),
    }


def _validate_training_assets(
    *,
    features: np.ndarray,
    candidates: np.ndarray,
    destinations: np.ndarray,
    sources: np.ndarray,
    event_time: np.ndarray,
    short_none: np.ndarray,
) -> None:
    if features.shape != EXPECTED_FEATURE_SHAPE:
        raise ValueError(f"unexpected train feature shape: {features.shape}")
    if candidates.shape != EXPECTED_MATRIX_SHAPE:
        raise ValueError(f"unexpected candidate shape: {candidates.shape}")
    if short_none.shape != EXPECTED_MATRIX_SHAPE:
        raise ValueError(f"unexpected short_none score shape: {short_none.shape}")
    for label, values in (
        ("destinations", destinations),
        ("sources", sources),
        ("event_time", event_time),
    ):
        if values.shape != (EXPECTED_MATRIX_SHAPE[0],):
            raise ValueError(f"unexpected {label} shape: {values.shape}")
    if not np.array_equal(candidates[:, 0], destinations):
        raise ValueError("candidate zero differs from destination sidecar")
    if not np.all(event_time[1:] >= event_time[:-1]):
        raise ValueError("training cache must be chronological")


def _q1_fold_diagnostic(
    *,
    fold_id: str,
    positive_popularity: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_descriptors: dict[str, dict[str, Any]],
    weights: tuple[float, ...],
) -> dict[str, Any]:
    count = len(positive_popularity)
    q1_count = (count + 3) // 4
    order = np.argsort(positive_popularity, kind="stable")
    mask = np.zeros(count, dtype=bool)
    mask[order[:q1_count]] = True
    baseline_rr = _reciprocal_ranks(baseline_scores)[mask]
    weight_reports: dict[str, dict[str, Any]] = {}
    for weight in weights:
        key = format(weight, "g")
        candidate = np.load(
            candidate_descriptors[key]["path"],
            mmap_mode="r",
            allow_pickle=False,
        )
        candidate_rr = _reciprocal_ranks(candidate)[mask]
        candidate_mrr = float(candidate_rr.mean())
        baseline_mrr = float(baseline_rr.mean())
        weight_reports[key] = {
            "weight": float(weight),
            "baseline_mrr": baseline_mrr,
            "candidate_mrr": candidate_mrr,
            "delta_candidate_minus_baseline": candidate_mrr - baseline_mrr,
            "baseline_reciprocal_rank_sum": float(baseline_rr.sum()),
            "candidate_reciprocal_rank_sum": float(candidate_rr.sum()),
            "query_count": q1_count,
        }
    return {
        "fold_id": fold_id,
        "query_count": q1_count,
        "fold_query_count": count,
        "q1_max_causal_popularity": int(positive_popularity[mask].max()),
        "tie_break": "stable chronological cache row order",
        "weights": weight_reports,
    }


def _update_pooled_q1(
    pooled: dict[str, dict[str, float | int]],
    fold_report: dict[str, Any],
) -> None:
    for key, report in fold_report["weights"].items():
        target = pooled[key]
        target["baseline_reciprocal_rank_sum"] = float(target["baseline_reciprocal_rank_sum"]) + float(
            report["baseline_reciprocal_rank_sum"]
        )
        target["candidate_reciprocal_rank_sum"] = float(target["candidate_reciprocal_rank_sum"]) + float(
            report["candidate_reciprocal_rank_sum"]
        )
        target["query_count"] = int(target["query_count"]) + int(report["query_count"])


def _finalize_pooled_q1(
    pooled: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for key, report in pooled.items():
        count = int(report["query_count"])
        baseline_mrr = float(report["baseline_reciprocal_rank_sum"]) / count
        candidate_mrr = float(report["candidate_reciprocal_rank_sum"]) / count
        output[key] = {
            "weight": float(key),
            "query_count": count,
            "baseline_mrr": baseline_mrr,
            "candidate_mrr": candidate_mrr,
            "delta_candidate_minus_baseline": candidate_mrr - baseline_mrr,
        }
    return output


def _reciprocal_ranks(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores)
    ranks = 1 + np.sum(values[:, 1:] > values[:, :1], axis=1)
    return 1.0 / ranks


def _predict_probabilities(
    model: Any,
    features: Any,
    result: FusionResult,
    *,
    batch_size: int,
) -> np.ndarray:
    probabilities = np.empty(features.shape[:2], dtype=np.float64)
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        selected = np.asarray(features[start:stop], dtype=np.float32)
        if result.feature_indices != tuple(range(selected.shape[-1])):
            selected = selected[..., result.feature_indices]
        logits = predict_logits(
            model,
            selected,
            result.mean,
            result.std,
        )
        probabilities[start:stop] = _softmax(logits)
    return probabilities


def _save_fusion_result(
    path: Path,
    *,
    result: FusionResult,
    hidden_dim: int,
    source_feature_count: int,
    transform_version: int,
    seed: int,
    salt: int,
) -> None:
    payload = {
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
        "feature_indices": np.asarray(
            result.feature_indices,
            dtype=np.int32,
        ),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
        "source_feature_count": np.asarray(
            [source_feature_count],
            dtype=np.int32,
        ),
        "context_transform_version": np.asarray(
            [transform_version],
            dtype=np.int32,
        ),
        "training_seed": np.asarray([seed], dtype=np.int64),
        "seed_salt": np.asarray([salt], dtype=np.int64),
        "feature_names": np.asarray(COOCCUR_LIFT_FEATURE_NAMES),
    }
    payload.update({f"state__{key}": np.asarray(value, dtype=np.float32) for key, value in result.state.items()})
    np.savez_compressed(path, **payload)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _release_jittor() -> None:
    gc.collect()
    jt.sync_all()
    jt.clean()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
