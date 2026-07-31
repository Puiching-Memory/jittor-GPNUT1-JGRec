from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jittor as jt
import lightgbm as lgb
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.listwise_mlp_exact_blend import (
    A2_INTEGRATION_ID,
    A2_ROLLING_FOLDS,
    build_rolling_selection_manifest,
    materialize_fold_candidates,
    validate_frozen_a2_weights,
    validate_rolling_folds,
)
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    FusionResult,
    fit_fusion_mlp_listwise_fixed,
    predict_logits,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

SETWISE_EPOCHS = 4
AUXILIARY_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.001
SEED = 60
SETWISE_WEIGHT = 0.80


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
            "Train fold-exact Setwise baselines and base listwise-MLP "
            "auxiliaries for the frozen A2 rolling-origin protocol."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--short-none-scores", required=True, type=Path)
    parser.add_argument("--source-weight-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    feature_path = Path(f"{args.train_cache_prefix}.train.npy")
    candidate_path = Path(f"{args.train_cache_prefix}.train-candidates.npy")
    destination_path = Path(f"{args.train_cache_prefix}.train-dst.npy")
    time_path = Path(f"{args.train_cache_prefix}.train-time.npy")
    cache_report = _read_json(args.train_cache_report)
    source_weight_config = _read_json(args.source_weight_config)
    weights = validate_frozen_a2_weights(source_weight_config)
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    candidates = np.load(
        candidate_path,
        mmap_mode="r",
        allow_pickle=False,
    )
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
    expected_shape = (200_000, 100)
    if features.shape != (*expected_shape, 63):
        raise ValueError(f"unexpected train feature shape: {features.shape}")
    if candidates.shape != expected_shape:
        raise ValueError(f"unexpected candidate shape: {candidates.shape}")
    if short_none.shape != expected_shape:
        raise ValueError(f"unexpected short_none score shape: {short_none.shape}")
    if event_time.shape != (expected_shape[0],):
        raise ValueError(f"unexpected time shape: {event_time.shape}")
    if not np.array_equal(candidates[:, 0], destinations):
        raise ValueError("candidate zero differs from destination sidecar")
    if not np.all(event_time[1:] >= event_time[:-1]):
        raise ValueError("training cache must be chronological")
    folds = validate_rolling_folds(
        A2_ROLLING_FOLDS,
        row_count=features.shape[0],
    )

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if feature_names != tuple(cache_report["feature_names"]):
        raise ValueError("checkpoint and cache feature schemas differ")
    gnn_column = feature_names.index("gnn_short")
    setwise_result = state.get("setwise_fusion_result")
    setwise_state = state.get("setwise_fusion_state")
    if setwise_result is None or setwise_state is None:
        raise ValueError("champion checkpoint has no Setwise head")
    lgbm_result = state.get("lgbm_result")
    if lgbm_result is None:
        raise ValueError("champion checkpoint has no LightGBM expert")
    if abs(float(lgbm_result.mlp_weight) - SETWISE_WEIGHT) > 1e-12:
        raise ValueError("champion Setwise weight differs from 0.80")
    base_result = state["fusion_result"]
    setwise_indices = tuple(int(index) for index in setwise_result.feature_indices)
    auxiliary_indices = tuple(int(index) for index in base_result.feature_indices)
    lgbm_indices = tuple(int(index) for index in lgbm_result.feature_indices)
    if not setwise_indices or not auxiliary_indices or not lgbm_indices:
        raise ValueError("champion expert feature indices must be non-empty")
    setwise_hidden_dim = int(state["setwise_hidden_dim"])
    auxiliary_hidden_dim = int(state["fusion_hidden_dim"])
    booster = lgb.Booster(model_str=str(lgbm_result.model_text))
    del state
    gc.collect()

    frozen = {
        "status": "frozen_before_rolling_training",
        "integration_id": A2_INTEGRATION_ID,
        "folds": [
            {
                "fold_id": fold.fold_id,
                "train_rows": list(fold.train_rows),
                "score_rows": list(fold.score_rows),
            }
            for fold in folds
        ],
        "weights": list(weights),
        "weights_source": str(args.source_weight_config.resolve()),
        "weights_source_sha256": _sha256(args.source_weight_config),
        "setwise_epochs": SETWISE_EPOCHS,
        "auxiliary_epochs": AUXILIARY_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "setwise_weight": SETWISE_WEIGHT,
        "candidate_formula": ("candidate_w = (1 - w) * fold_champion + w * fold_listwise_mlp"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "train_cache": str(feature_path.resolve()),
        "train_cache_sha256": _sha256(feature_path),
        "candidate_sidecar_sha256": _sha256(candidate_path),
        "time_sidecar_sha256": _sha256(time_path),
        "short_none_scores_sha256": _sha256(args.short_none_scores),
        "gnn_column": gnn_column,
        "setwise_feature_indices": list(setwise_indices),
        "auxiliary_feature_indices": list(auxiliary_indices),
        "lgbm_feature_indices": list(lgbm_indices),
        "external_scores_read": False,
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)
    print(
        json.dumps(frozen, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    jt.flags.use_cuda = 1
    fold_entries: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        fold_started = time.time()
        train_start, train_stop = fold.train_rows
        score_start, score_stop = fold.score_rows
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
        setwise_train = SetwiseFeatureView(train_overlay)
        setwise_score = SetwiseFeatureView(score_overlay)
        setwise_config = FusionConfig(
            epochs=SETWISE_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            weight_decay=0.0,
            hidden_dim=setwise_hidden_dim,
            selection_metric="mrr",
            early_stop_patience=0,
        )
        setwise_model, fold_setwise_result, setwise_losses = fit_fusion_mlp_listwise_fixed(
            setwise_train,
            setwise_score,
            setwise_config,
            np.random.default_rng(SEED + fold_index * 1009),
            verbose=True,
            feature_indices=setwise_indices,
            candidate_name=f"a2_{fold.fold_id}_setwise",
        )
        setwise_probabilities = _predict_probabilities(
            setwise_model,
            setwise_score,
            fold_setwise_result,
            batch_size=BATCH_SIZE,
        )
        model_dir = args.output_dir / "models" / fold.fold_id
        model_dir.mkdir(parents=True)
        setwise_model_path = model_dir / "setwise.npz"
        _save_fusion_result(
            setwise_model_path,
            result=fold_setwise_result,
            hidden_dim=setwise_hidden_dim,
            source_feature_count=features.shape[-1],
            transform_version=1,
        )
        del setwise_model, setwise_train, setwise_score
        _release_jittor()

        auxiliary_config = FusionConfig(
            epochs=AUXILIARY_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            weight_decay=0.0,
            hidden_dim=auxiliary_hidden_dim,
            selection_metric="mrr",
            early_stop_patience=0,
        )
        auxiliary_model, auxiliary_result, auxiliary_losses = fit_fusion_mlp_listwise_fixed(
            train_overlay,
            score_overlay,
            auxiliary_config,
            np.random.default_rng(SEED + fold_index * 1009),
            verbose=True,
            feature_indices=auxiliary_indices,
            candidate_name=f"a2_{fold.fold_id}_auxiliary",
        )
        auxiliary_probabilities = _predict_probabilities(
            auxiliary_model,
            score_overlay,
            auxiliary_result,
            batch_size=BATCH_SIZE,
        )
        auxiliary_model_path = model_dir / "auxiliary-listwise-mlp.npz"
        _save_fusion_result(
            auxiliary_model_path,
            result=auxiliary_result,
            hidden_dim=auxiliary_hidden_dim,
            source_feature_count=features.shape[-1],
            transform_version=0,
        )
        del auxiliary_model
        _release_jittor()

        lgbm_probabilities = _predict_lgbm_probabilities(
            booster,
            features[score_start:score_stop],
            feature_indices=lgbm_indices,
            batch_size=BATCH_SIZE,
        )
        baseline_probabilities = SETWISE_WEIGHT * setwise_probabilities + (1.0 - SETWISE_WEIGHT) * lgbm_probabilities
        entry = materialize_fold_candidates(
            output_dir=args.output_dir / "scores" / fold.fold_id,
            fold_id=fold.fold_id,
            integration_id=A2_INTEGRATION_ID,
            train_time_max=int(event_time[train_stop - 1]),
            score_time_min=int(event_time[score_start]),
            score_time_max=int(event_time[score_stop - 1]),
            baseline_scores=baseline_probabilities,
            auxiliary_scores=auxiliary_probabilities,
            candidate_ids=candidates[score_start:score_stop],
            weights=weights,
        )
        entry["train_rows"] = [train_start, train_stop]
        entry["score_rows"] = [score_start, score_stop]
        fold_entries.append(entry)
        fold_report = {
            "fold_id": fold.fold_id,
            "train_rows": [train_start, train_stop],
            "score_rows": [score_start, score_stop],
            "setwise_epochs": SETWISE_EPOCHS,
            "setwise_losses": list(setwise_losses),
            "setwise_score_mrr": _mrr(setwise_probabilities),
            "setwise_model": str(setwise_model_path.resolve()),
            "setwise_model_sha256": _sha256(setwise_model_path),
            "auxiliary_epochs": AUXILIARY_EPOCHS,
            "auxiliary_losses": list(auxiliary_losses),
            "auxiliary_score_mrr": _mrr(auxiliary_probabilities),
            "auxiliary_model": str(auxiliary_model_path.resolve()),
            "auxiliary_model_sha256": _sha256(auxiliary_model_path),
            "lgbm_score_mrr": _mrr(lgbm_probabilities),
            "baseline_score_mrr": _mrr(baseline_probabilities),
            "score_artifacts": entry,
            "elapsed_seconds": time.time() - fold_started,
        }
        fold_reports.append(fold_report)
        _write_json(
            args.output_dir / "rolling-progress.json",
            {
                "status": "training",
                "completed_folds": len(fold_reports),
                "total_folds": len(folds),
                "folds": fold_reports,
                "external_scores_read": False,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(
            json.dumps(fold_report, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        del (
            train_overlay,
            score_overlay,
            setwise_probabilities,
            auxiliary_probabilities,
            lgbm_probabilities,
            baseline_probabilities,
        )
        _release_jittor()

    manifest_path = args.output_dir / "rolling-manifest.json"
    build_rolling_selection_manifest(
        integration_id=A2_INTEGRATION_ID,
        fold_entries=fold_entries,
        output_path=manifest_path,
    )
    report = {
        "status": "complete",
        "integration_id": A2_INTEGRATION_ID,
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "rolling_manifest": str(manifest_path.resolve()),
        "rolling_manifest_sha256": _sha256(manifest_path),
        "external_scores_read": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "rolling-training-report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


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


def _predict_lgbm_probabilities(
    booster: lgb.Booster,
    features: np.ndarray,
    *,
    feature_indices: tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    probabilities = np.empty(features.shape[:2], dtype=np.float64)
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        source_batch = np.asarray(features[start:stop], dtype=np.float32)
        selected = source_batch[..., feature_indices]
        flat = np.ascontiguousarray(
            selected.reshape(-1, selected.shape[-1]),
            dtype=np.float32,
        )
        logits = booster.predict(flat).reshape(selected.shape[:2])
        probabilities[start:stop] = _softmax(logits)
    return probabilities


def _save_fusion_result(
    path: Path,
    *,
    result: FusionResult,
    hidden_dim: int,
    source_feature_count: int,
    transform_version: int,
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
    }
    payload.update({f"state__{key}": np.asarray(value, dtype=np.float32) for key, value in result.state.items()})
    np.savez_compressed(path, **payload)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _mrr(scores: np.ndarray) -> float:
    ranks = 1 + np.sum(scores[:, 1:] > scores[:, :1], axis=1)
    return float(np.mean(1.0 / ranks))


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
    return json.loads(path.read_text(encoding="utf-8"))


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
