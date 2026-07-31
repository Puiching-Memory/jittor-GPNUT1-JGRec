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
from jgrec.cooccur_lift_external import (
    build_external_manifest,
    validate_locked_external_setup,
)
from jgrec.core.io import read_interactions
from jgrec.rankers.hybrid.cooccur_lift import (
    COOCCUR_LIFT_FEATURE_NAMES,
    CooccurLiftAugmentedView,
    load_frozen_cooccur_lift_config,
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

TRAIN_SHAPE = (200_000, 100)
EXTERNAL_SHAPE = (20_000, 100)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "After a valid rolling lock exists, train the frozen full-origin "
            "cooccur-lift head and materialize exactly one external manifest "
            "without computing any external ranking metric."
        )
    )
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--train-lift-features", required=True, type=Path)
    parser.add_argument("--train-run-contract", required=True, type=Path)
    parser.add_argument("--train-short-none", required=True, type=Path)
    parser.add_argument("--validation-cache-prefix", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--validation-short-none", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--baseline-probabilities", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    setup = validate_locked_external_setup(
        frozen_config_path=args.frozen_config,
        selection_lock_path=args.selection_lock,
    )
    config = load_frozen_cooccur_lift_config(args.frozen_config)
    train_paths = _cache_paths(args.train_cache_prefix, split="train")
    validation_paths = _cache_paths(
        args.validation_cache_prefix,
        split="val",
    )
    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    rolling_contract = _read_json(args.train_run_contract)
    if rolling_contract.get("integration_id") != config.integration_id:
        raise ValueError("rolling run contract integration_id differs")
    if rolling_contract.get("lift_features_sha256") != _sha256(
        args.train_lift_features
    ):
        raise ValueError("training lift features differ from rolling contract")

    train_features = np.load(
        train_paths["features"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_candidates = np.load(
        train_paths["candidates"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_sources = np.load(
        train_paths["sources"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_destinations = np.load(
        train_paths["destinations"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_times = np.load(
        train_paths["times"],
        mmap_mode="r",
        allow_pickle=False,
    )
    train_short_none = np.load(
        args.train_short_none,
        mmap_mode="r",
        allow_pickle=False,
    )
    train_lift = np.load(
        args.train_lift_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_features = np.load(
        validation_paths["features"],
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_candidates = np.load(
        validation_paths["candidates"],
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_sources = np.load(
        validation_paths["sources"],
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_destinations = np.load(
        validation_paths["destinations"],
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_times = np.load(
        validation_paths["times"],
        mmap_mode="r",
        allow_pickle=False,
    )
    validation_short_none = np.load(
        args.validation_short_none,
        mmap_mode="r",
        allow_pickle=False,
    )
    baseline = np.load(
        args.baseline_probabilities,
        mmap_mode="r",
        allow_pickle=False,
    )
    _validate_assets(
        train_features=train_features,
        train_candidates=train_candidates,
        train_sources=train_sources,
        train_destinations=train_destinations,
        train_times=train_times,
        train_short_none=train_short_none,
        train_lift=train_lift,
        validation_features=validation_features,
        validation_candidates=validation_candidates,
        validation_sources=validation_sources,
        validation_destinations=validation_destinations,
        validation_times=validation_times,
        validation_short_none=validation_short_none,
        baseline=baseline,
    )
    training_time_max = int(train_times[-1])
    strict_external_rows = np.asarray(
        validation_times > training_time_max,
        dtype=bool,
    )
    if not np.any(strict_external_rows):
        raise ValueError("external cache has no row strictly after training")

    state = load_checkpoint_dataset(args.checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if feature_names != tuple(train_report["feature_names"]):
        raise ValueError("checkpoint and training cache schemas differ")
    if feature_names != tuple(validation_report["feature_names"]):
        raise ValueError("checkpoint and validation cache schemas differ")
    gnn_column = feature_names.index("gnn_short")
    setwise_hidden_dim = int(state["setwise_hidden_dim"])
    del state
    gc.collect()

    interactions = read_interactions(args.train_csv).sort_by_time()
    time_span = int(interactions.time[-1]) - int(interactions.time[0])
    short_window = float(time_span) * config.short_window_ratio
    args.output_dir.mkdir(parents=True)
    shutil.copyfile(
        args.frozen_config,
        args.output_dir / "precommitted-config.json",
    )
    shutil.copyfile(
        args.selection_lock,
        args.output_dir / "selection-lock.json",
    )
    started = time.time()
    validation_lift_path = args.output_dir / "external-lift-features.npy"
    validation_popularity_path = (
        args.output_dir / "external-positive-dst-causal-popularity.npy"
    )
    native_contract = materialize_compact_cooccur_lift(
        interactions=interactions,
        sources=validation_sources,
        candidates=validation_candidates,
        destinations=validation_destinations,
        event_time=validation_times,
        short_window=short_window,
        lift_path=validation_lift_path,
        positive_popularity_path=validation_popularity_path,
        progress_path=args.output_dir / "external-materialization-progress.json",
        work_dir=args.output_dir,
    )
    validation_lift = np.load(
        validation_lift_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    train_augmented = CooccurLiftAugmentedView(
        train_features,
        short_none_scores=train_short_none,
        gnn_short_column=gnn_column,
        lift_features=train_lift,
    )
    validation_augmented = CooccurLiftAugmentedView(
        validation_features,
        short_none_scores=validation_short_none,
        gnn_short_column=gnn_column,
        lift_features=validation_lift,
    )
    train_view = SetwiseFeatureView(
        train_augmented,
        transform_version=1,
    )
    validation_view = SetwiseFeatureView(
        validation_augmented,
        transform_version=1,
    )
    if train_view.shape[-1] != config.context_feature_count:
        raise ValueError("full-origin Setwise context does not have 195 columns")

    jt.flags.use_cuda = 1
    jt.set_global_seed(setup.full_origin_seed)
    model, result, losses = fit_fusion_mlp_listwise_fixed(
        train_view,
        train_view[:1],
        FusionConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            hidden_dim=setwise_hidden_dim,
            selection_metric=config.selection_metric,
            early_stop_patience=config.early_stop_patience,
        ),
        np.random.default_rng(setup.full_origin_seed),
        verbose=False,
        feature_indices=config.context_feature_indices,
        candidate_name=f"{config.integration_id}_full_origin",
    )
    auxiliary = _predict_probabilities(
        model,
        validation_view,
        result,
        batch_size=config.batch_size,
    )
    candidate = (
        (1.0 - setup.selected_weight) * np.asarray(baseline)
        + setup.selected_weight * auxiliary
    )
    model_path = args.output_dir / (
        f"cooccur-lift-full-origin-seed{setup.full_origin_seed}.npz"
    )
    auxiliary_path = args.output_dir / "external-auxiliary-probabilities.npy"
    baseline_path = args.output_dir / "external-baseline-probabilities.npy"
    candidate_path = args.output_dir / "external-candidate-probabilities.npy"
    _save_fusion_result(
        model_path,
        result=result,
        hidden_dim=setwise_hidden_dim,
        seed=setup.full_origin_seed,
        salt=config.seed_salt,
    )
    np.save(auxiliary_path, auxiliary[strict_external_rows])
    np.save(
        baseline_path,
        np.asarray(baseline)[strict_external_rows],
    )
    np.save(candidate_path, candidate[strict_external_rows])

    fingerprint = hashlib.sha256(
        np.ascontiguousarray(
            validation_candidates[strict_external_rows]
        ).tobytes(order="C")
    ).hexdigest()
    manifest = build_external_manifest(
        contract=setup,
        candidate_fingerprint=fingerprint,
        training_time_max=training_time_max,
        score_time_min=int(validation_times[strict_external_rows][0]),
        score_time_max=int(validation_times[strict_external_rows][-1]),
        baseline_path=baseline_path,
        baseline_sha256=_sha256(baseline_path),
        candidate_path=candidate_path,
        candidate_sha256=_sha256(candidate_path),
    )
    manifest_path = args.output_dir / "external-manifest.json"
    _write_json(manifest_path, manifest)
    report = {
        "schema_version": 1,
        "status": "external_candidate_materialized_metrics_unread",
        "integration_id": setup.integration_id,
        "selected_weight": setup.selected_weight,
        "full_origin_seed": setup.full_origin_seed,
        "selection_lock_sha256": setup.selection_lock_sha256,
        "external_manifest": str(manifest_path.resolve()),
        "external_manifest_sha256": _sha256(manifest_path),
        "model": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "auxiliary_probabilities": str(auxiliary_path.resolve()),
        "auxiliary_probabilities_sha256": _sha256(auxiliary_path),
        "baseline_probabilities": str(baseline_path.resolve()),
        "baseline_probabilities_sha256": _sha256(baseline_path),
        "candidate_probabilities": str(candidate_path.resolve()),
        "candidate_probabilities_sha256": _sha256(candidate_path),
        "external_lift_features": str(validation_lift_path.resolve()),
        "external_lift_features_sha256": _sha256(validation_lift_path),
        "native_materializer": native_contract,
        "losses": list(losses),
        "external_ranking_metrics_computed": False,
        "external_evaluator_invoked": False,
        "external_cache_rows": len(validation_times),
        "boundary_tie_rows_excluded": int(
            np.sum(~strict_external_rows)
        ),
        "strict_external_rows": int(np.sum(strict_external_rows)),
        "external_row_rule": (
            "validation_time > full_origin_training_time_max"
        ),
        "cooccur_time_decay_score_reused": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "external-materialization-report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _cache_paths(prefix: Path, *, split: str) -> dict[str, Path]:
    base = str(prefix)
    return {
        "features": Path(f"{base}.{split}.npy"),
        "candidates": Path(f"{base}.{split}-candidates.npy"),
        "sources": Path(f"{base}.{split}-src.npy"),
        "destinations": Path(f"{base}.{split}-dst.npy"),
        "times": Path(f"{base}.{split}-time.npy"),
    }


def _validate_assets(**assets: np.ndarray) -> None:
    if assets["train_features"].shape != (*TRAIN_SHAPE, 63):
        raise ValueError("unexpected training feature shape")
    if assets["validation_features"].shape != (*EXTERNAL_SHAPE, 63):
        raise ValueError("unexpected external feature shape")
    for name in ("train_candidates", "train_short_none"):
        if assets[name].shape != TRAIN_SHAPE:
            raise ValueError(f"unexpected {name} shape")
    if assets["train_lift"].shape != (*TRAIN_SHAPE, 2):
        raise ValueError("unexpected train_lift shape")
    for name in (
        "validation_candidates",
        "validation_short_none",
        "baseline",
    ):
        if assets[name].shape != EXTERNAL_SHAPE:
            raise ValueError(f"unexpected {name} shape")
    for name, shape in (
        ("train_sources", (TRAIN_SHAPE[0],)),
        ("train_destinations", (TRAIN_SHAPE[0],)),
        ("train_times", (TRAIN_SHAPE[0],)),
        ("validation_sources", (EXTERNAL_SHAPE[0],)),
        ("validation_destinations", (EXTERNAL_SHAPE[0],)),
        ("validation_times", (EXTERNAL_SHAPE[0],)),
    ):
        if assets[name].shape != shape:
            raise ValueError(f"unexpected {name} shape")
    if not np.array_equal(
        assets["train_candidates"][:, 0],
        assets["train_destinations"],
    ):
        raise ValueError("training candidate zero differs from destination")
    if not np.array_equal(
        assets["validation_candidates"][:, 0],
        assets["validation_destinations"],
    ):
        raise ValueError("external candidate zero differs from destination")
    if not np.all(np.isfinite(assets["baseline"])):
        raise ValueError("external baseline probabilities are non-finite")


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
    seed: int,
    salt: int,
) -> None:
    payload = {
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
        "feature_indices": np.asarray(result.feature_indices, dtype=np.int32),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
        "source_feature_count": np.asarray([65], dtype=np.int32),
        "context_transform_version": np.asarray([1], dtype=np.int32),
        "training_seed": np.asarray([seed], dtype=np.int64),
        "seed_salt": np.asarray([salt], dtype=np.int64),
        "feature_names": np.asarray(COOCCUR_LIFT_FEATURE_NAMES),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
