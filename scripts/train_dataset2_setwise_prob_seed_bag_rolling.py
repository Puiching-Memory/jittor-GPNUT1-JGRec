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
from jgrec.listwise_mlp_exact_blend import (
    build_rolling_selection_manifest,
    materialize_fold_candidates,
)
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    FusionResult,
    fit_fusion_mlp_listwise_fixed,
    predict_logits,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView
from jgrec.setwise_prob_seed_bag import (
    load_frozen_seed_bag_config,
    load_verified_source_baseline,
    mean_seed_probabilities,
    training_seeds,
    validate_source_rolling_manifest,
)

EXPECTED_FEATURE_SHAPE = (200_000, 100, 63)
EXPECTED_MATRIX_SHAPE = EXPECTED_FEATURE_SHAPE[:2]
CHAMPION_SETWISE_WEIGHT = 0.80


class _ColumnOverlay:
    def __init__(
        self,
        source: Any,
        replacement: np.ndarray,
        column: int,
    ) -> None:
        if replacement.shape != source.shape[:2]:
            raise ValueError("replacement scores must match cache rows")
        self._source = source
        self._replacement = replacement
        self._column = int(column)
        self.shape = tuple(int(value) for value in source.shape)
        self.ndim = 3
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key: Any) -> np.ndarray:
        values = np.array(self._source[key], dtype=np.float32, copy=True)
        values[..., self._column] = self._replacement[key]
        return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the two frozen Dataset2 Setwise seed salts on each "
            "exact-rolling fold and materialize probability-mean candidates."
        )
    )
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--short-none-scores", required=True, type=Path)
    parser.add_argument("--source-rolling-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    config = load_frozen_seed_bag_config(args.frozen_config)
    source_manifest = _read_json(args.source_rolling_manifest)
    source_folds = validate_source_rolling_manifest(source_manifest, config)

    args.output_dir.mkdir(parents=True)
    frozen_copy = args.output_dir / "precommitted-config.json"
    shutil.copyfile(args.frozen_config, frozen_copy)
    if _sha256(frozen_copy) != _sha256(args.frozen_config):
        raise RuntimeError("frozen config copy differs from source")
    started = time.time()

    feature_path = Path(f"{args.train_cache_prefix}.train.npy")
    candidate_path = Path(f"{args.train_cache_prefix}.train-candidates.npy")
    destination_path = Path(f"{args.train_cache_prefix}.train-dst.npy")
    time_path = Path(f"{args.train_cache_prefix}.train-time.npy")
    cache_report = _read_json(args.train_cache_report)
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    candidates = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    destinations = np.load(
        destination_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    event_time = np.load(time_path, mmap_mode="r", allow_pickle=False)
    short_none = np.load(
        args.short_none_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    _validate_training_assets(
        features=features,
        candidates=candidates,
        destinations=destinations,
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
    setwise_indices = tuple(int(index) for index in setwise_result.feature_indices)
    if not setwise_indices:
        raise ValueError("champion Setwise feature indices must be non-empty")
    setwise_hidden_dim = int(state["setwise_hidden_dim"])
    del state
    gc.collect()

    run_contract = {
        "status": "training_without_model_or_weight_selection",
        "integration_id": config.integration_id,
        "precommitted_config": str(frozen_copy.resolve()),
        "precommitted_config_sha256": _sha256(frozen_copy),
        "source_rolling_manifest": str(args.source_rolling_manifest.resolve()),
        "source_rolling_manifest_sha256": _sha256(args.source_rolling_manifest),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_cache": str(feature_path.resolve()),
        "train_cache_sha256": _sha256(feature_path),
        "candidate_sidecar_sha256": _sha256(candidate_path),
        "destination_sidecar_sha256": _sha256(destination_path),
        "time_sidecar_sha256": _sha256(time_path),
        "short_none_scores_sha256": _sha256(args.short_none_scores),
        "folds": [
            {
                "fold_id": fold_id,
                "train_rows": list(train_rows),
                "score_rows": list(score_rows),
                "training_seeds": list(training_seeds(config, fold_index)),
            }
            for fold_index, (fold_id, train_rows, score_rows) in enumerate(config.folds)
        ],
        "weights": list(config.weights),
        "seed_salts": list(config.seed_salts),
        "setwise_epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "early_stop_patience": config.early_stop_patience,
        "gnn_short_column": gnn_column,
        "setwise_feature_indices": list(setwise_indices),
        "setwise_hidden_dim": setwise_hidden_dim,
        "auxiliary_formula": "arithmetic mean of the two new-seed probabilities",
        "external_scores_read": False,
    }
    _write_json(args.output_dir / "run-contract.json", run_contract)
    print(json.dumps(run_contract, ensure_ascii=False, sort_keys=True), flush=True)

    jt.flags.use_cuda = 1
    fold_entries: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold_index, (
        fold_id,
        train_rows,
        score_rows,
    ) in enumerate(config.folds):
        fold_started = time.time()
        train_start, train_stop = train_rows
        score_start, score_stop = score_rows
        train_overlay = _ColumnOverlay(
            features[train_start:train_stop],
            short_none[train_start:train_stop],
            gnn_column,
        )
        score_overlay = _ColumnOverlay(
            features[score_start:score_stop],
            short_none[score_start:score_stop],
            gnn_column,
        )
        model_dir = args.output_dir / "models" / fold_id
        probability_dir = args.output_dir / "seed-probabilities" / fold_id
        model_dir.mkdir(parents=True)
        probability_dir.mkdir(parents=True)
        seed_probabilities: list[np.ndarray] = []
        seed_reports: list[dict[str, Any]] = []

        for salt, seed in zip(
            config.seed_salts,
            training_seeds(config, fold_index),
            strict=True,
        ):
            seed_started = time.time()
            jt.set_global_seed(seed)
            train_view = SetwiseFeatureView(train_overlay)
            score_view = SetwiseFeatureView(score_overlay)
            setwise_config = FusionConfig(
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                hidden_dim=setwise_hidden_dim,
                selection_metric="mrr",
                early_stop_patience=config.early_stop_patience,
            )
            model, result, losses = fit_fusion_mlp_listwise_fixed(
                train_view,
                score_view,
                setwise_config,
                np.random.default_rng(seed),
                verbose=False,
                feature_indices=setwise_indices,
                candidate_name=f"{config.integration_id}_{fold_id}_salt{salt}",
            )
            probabilities = _predict_probabilities(
                model,
                score_view,
                result,
                batch_size=config.batch_size,
            )
            model_path = model_dir / f"setwise-salt{salt}-seed{seed}.npz"
            _save_fusion_result(
                model_path,
                result=result,
                hidden_dim=setwise_hidden_dim,
                source_feature_count=features.shape[-1],
                transform_version=1,
                seed=seed,
                salt=salt,
            )
            probability_path = probability_dir / f"setwise-salt{salt}-seed{seed}.npy"
            np.save(
                probability_path,
                np.asarray(probabilities, dtype=np.float32),
                allow_pickle=False,
            )
            seed_probabilities.append(probabilities)
            seed_reports.append(
                {
                    "seed_salt": salt,
                    "training_seed": seed,
                    "epochs": config.epochs,
                    "losses": list(losses),
                    "model": str(model_path.resolve()),
                    "model_sha256": _sha256(model_path),
                    "probabilities": str(probability_path.resolve()),
                    "probabilities_sha256": _sha256(probability_path),
                    "elapsed_seconds": time.time() - seed_started,
                }
            )
            del model, result, probabilities, train_view, score_view
            _release_jittor()

        auxiliary_probabilities = mean_seed_probabilities(seed_probabilities)
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
        fold_report = {
            "fold_id": fold_id,
            "train_rows": list(train_rows),
            "score_rows": list(score_rows),
            "seed_heads": seed_reports,
            "source_baseline_sha256": source_fold["baseline"]["sha256"],
            "score_artifacts": entry,
            "ranking_metrics_computed_by_producer": False,
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
            train_overlay,
            score_overlay,
            seed_probabilities,
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
    report = {
        "status": "complete",
        "integration_id": config.integration_id,
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "rolling_manifest": str(manifest_path.resolve()),
        "rolling_manifest_sha256": _sha256(manifest_path),
        "ranking_metrics_computed_by_producer": False,
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
                "external_scores_read": False,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _validate_training_assets(
    *,
    features: np.ndarray,
    candidates: np.ndarray,
    destinations: np.ndarray,
    event_time: np.ndarray,
    short_none: np.ndarray,
) -> None:
    if features.shape != EXPECTED_FEATURE_SHAPE:
        raise ValueError(f"unexpected train feature shape: {features.shape}")
    if candidates.shape != EXPECTED_MATRIX_SHAPE:
        raise ValueError(f"unexpected candidate shape: {candidates.shape}")
    if short_none.shape != EXPECTED_MATRIX_SHAPE:
        raise ValueError(f"unexpected short_none score shape: {short_none.shape}")
    if destinations.shape != (EXPECTED_MATRIX_SHAPE[0],):
        raise ValueError(f"unexpected destination shape: {destinations.shape}")
    if event_time.shape != (EXPECTED_MATRIX_SHAPE[0],):
        raise ValueError(f"unexpected time shape: {event_time.shape}")
    if not np.array_equal(candidates[:, 0], destinations):
        raise ValueError("candidate zero differs from destination sidecar")
    if not np.all(event_time[1:] >= event_time[:-1]):
        raise ValueError("training cache must be chronological")


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
