from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.cuda import require_jittor_cuda
from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.base_context_gate import (
    BASE_CONTEXT_INTEGRATION_ID,
    BaseContextBlendProtocol,
    compose_dataset1_final_scores,
    validate_context_only_difference,
)
from jgrec.rankers.hybrid.expert_fusion import ExpertBlendCalibration
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    fit_fusion_mlp_listwise_streaming,
    fit_fusion_mlp_streaming,
)
from jgrec.rankers.hybrid.fusion_lgbm import (
    fit_fusion_lgbm,
    predict_logits_lgbm,
)
from jgrec.rankers.hybrid.setwise import (
    SetwiseFeatureView,
    setwise_context_features,
)
from jgrec.robust_weight_selection import (
    ranking_metrics,
    select_rolling_origin_weight,
)

TUNE_ROWS = 20_000
SELECTION_FOLDS = 3
SETWISE_EPOCHS = 10
SETWISE_PATIENCE = 2
SETWISE_BATCH_SIZE = 256
SETWISE_HIDDEN_DIM = 32
SETWISE_LEARNING_RATE = 0.001
CANDIDATE_KEY = "1.0"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Dataset1 v0-vs-v1 base-context comparison on "
            "three exact integrated rolling-origin folds."
        )
    )
    parser.add_argument("--rolling-manifest", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    require_jittor_cuda(jt)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    rolling = _read_json(args.rolling_manifest)
    _validate_source_manifest(rolling)
    source_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset1",
    )
    source_hash = _sha256(args.source_checkpoint)
    feature_path = Path(rolling["source"]["features"])
    time_path = Path(rolling["source"]["times"])
    _require_hash(
        feature_path,
        rolling["source"]["features_sha256"],
        "rolling feature cache",
    )
    _require_hash(
        time_path,
        rolling["source"]["times_sha256"],
        "rolling time cache",
    )
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    times = np.load(time_path, mmap_mode="r", allow_pickle=False)
    protocol, control_config, candidate_config = _frozen_protocol(
        source_state
    )
    validate_context_only_difference(control_config, candidate_config)
    frozen = {
        "status": "frozen_before_metrics",
        "integration_id": BASE_CONTEXT_INTEGRATION_ID,
        "candidate_key": CANDIDATE_KEY,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": source_hash,
        "rolling_manifest": str(args.rolling_manifest.resolve()),
        "rolling_manifest_sha256": _sha256(args.rolling_manifest),
        "external_paths_present": False,
        "external_metrics_read": False,
        "selection_fold_count": SELECTION_FOLDS,
        "inner_tune_rows": TUNE_ROWS,
        "control_training": control_config,
        "candidate_training": candidate_config,
        "shared_setwise_training": {
            "epochs": SETWISE_EPOCHS,
            "early_stop_patience": SETWISE_PATIENCE,
            "batch_size": SETWISE_BATCH_SIZE,
            "hidden_dim": SETWISE_HIDDEN_DIM,
            "learning_rate": SETWISE_LEARNING_RATE,
            "objective": "listwise_positive_candidate_zero",
        },
        "final_integration": {
            "mlp_weight": protocol.mlp_weight,
            "expert_calibration": asdict(protocol.expert_calibration),
            "time_ramp_power": protocol.time_ramp_power,
        },
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)

    fold_descriptors: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    for fold_payload in rolling["folds"][:SELECTION_FOLDS]:
        fold = _train_and_score_fold(
            features=features,
            times=times,
            fold_payload=fold_payload,
            source_state=source_state,
            protocol=protocol,
            control_config=control_config,
            candidate_config=candidate_config,
            output_dir=args.output_dir,
        )
        fold_descriptors.append(fold["manifest"])
        progress.append(fold["report"])
        _write_json(
            args.output_dir / "rolling-progress.json",
            {
                "status": "running",
                "completed_folds": len(progress),
                "external_metrics_read": False,
                "folds": progress,
                "elapsed_seconds": time.time() - started,
            },
        )
        release_memory()

    selection_manifest = {
        "schema_version": 1,
        "protocol": "exact_integrated_rolling_weight_selection_v1",
        "integration_id": BASE_CONTEXT_INTEGRATION_ID,
        "positive_candidate_column": 0,
        "candidate_semantics": {
            CANDIDATE_KEY: "replace_base_mlp_context_v0_with_v1"
        },
        "frozen_config_sha256": _sha256(
            args.output_dir / "frozen-config.json"
        ),
        "folds": fold_descriptors,
    }
    manifest_path = args.output_dir / "selection-manifest.json"
    _write_json(manifest_path, selection_manifest)
    selection = select_rolling_origin_weight(
        manifest_path=manifest_path,
        output_dir=args.output_dir / "selection",
    )
    final = {
        "status": selection["status"],
        "rolling_pass": selection["status"] == "selected",
        "external_authorized": selection["status"] == "selected",
        "package_authorized": False,
        "source_checkpoint_sha256": source_hash,
        "selection_manifest_sha256": _sha256(manifest_path),
        "selection_report_sha256": _sha256(
            args.output_dir / "selection" / "selection-report.json"
        ),
        "folds": progress,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "rolling-result.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0 if final["rolling_pass"] else 2


def _train_and_score_fold(
    *,
    features: Any,
    times: np.ndarray,
    fold_payload: dict[str, Any],
    source_state: dict[str, Any],
    protocol: BaseContextBlendProtocol,
    control_config: dict[str, Any],
    candidate_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    fold_index = int(fold_payload["index"])
    train_start, train_stop = (
        int(value) for value in fold_payload["train_rows"]
    )
    score_start, score_stop = (
        int(value) for value in fold_payload["score_rows"]
    )
    tune_start = train_stop - TUNE_ROWS
    if tune_start <= train_start:
        raise ValueError("rolling train window is too short for inner tuning")

    source_config = source_state["config"]
    maximum_fit_rows = int(getattr(source_config, "max_train_events", 0))
    fit_start = train_start
    if maximum_fit_rows > 0:
        fit_start = max(fit_start, tune_start - maximum_fit_rows)
    base_train_features = features[fit_start:tune_start]
    setwise_train_features = features[train_start:tune_start]
    tune_features = features[tune_start:train_stop]
    score_features = features[score_start:score_stop]
    feature_indices = tuple(
        int(index) for index in source_state["fusion_result"].feature_indices
    )
    seed = int(control_config["seed"])

    control_model, control_result = fit_fusion_mlp_streaming(
        base_train_features,
        tune_features,
        _fusion_config(control_config),
        np.random.default_rng(seed),
        verbose=True,
        feature_indices=feature_indices,
        candidate_name=f"fold{fold_index}_base_context_v0",
    )
    control_logits = _predict_mlp_streaming(
        control_model,
        score_features,
        control_result,
        context_transform_version=0,
        batch_size=int(control_config["batch_size"]),
    )
    _save_head(
        output_dir / f"fold-{fold_index:02d}-base-context-v0.npz",
        control_result,
        hidden_dim=int(control_config["hidden_dim"]),
        context_transform_version=0,
        fit_rows=(fit_start, tune_start),
        tune_rows=(tune_start, train_stop),
    )
    del control_model
    release_memory()

    candidate_model, candidate_result = fit_fusion_mlp_streaming(
        base_train_features,
        tune_features,
        _fusion_config(candidate_config),
        np.random.default_rng(seed),
        verbose=True,
        feature_indices=feature_indices,
        candidate_name=f"fold{fold_index}_base_context_v1",
    )
    candidate_logits = _predict_mlp_streaming(
        candidate_model,
        score_features,
        candidate_result,
        context_transform_version=1,
        batch_size=int(candidate_config["batch_size"]),
    )
    _save_head(
        output_dir / f"fold-{fold_index:02d}-base-context-v1.npz",
        candidate_result,
        hidden_dim=int(candidate_config["hidden_dim"]),
        context_transform_version=1,
        fit_rows=(fit_start, tune_start),
        tune_rows=(tune_start, train_stop),
    )
    del candidate_model
    release_memory()

    shared_lgbm = fit_fusion_lgbm(
        base_train_features,
        tune_features,
        selection_metric=str(control_config["selection_metric"]),
        verbose=True,
        feature_indices=feature_indices,
        candidate_name=f"fold{fold_index}_shared",
    )
    lgbm_logits = predict_logits_lgbm(
        shared_lgbm.model_text,
        np.asarray(score_features[..., feature_indices]),
    )
    del shared_lgbm
    release_memory()

    setwise_config = FusionConfig(
        epochs=SETWISE_EPOCHS,
        batch_size=SETWISE_BATCH_SIZE,
        lr=SETWISE_LEARNING_RATE,
        weight_decay=0.0,
        hidden_dim=SETWISE_HIDDEN_DIM,
        selection_metric="mrr",
        early_stop_patience=SETWISE_PATIENCE,
        context_transform_version=0,
    )
    setwise_train = SetwiseFeatureView(setwise_train_features)
    setwise_tune = SetwiseFeatureView(tune_features)
    setwise_model, setwise_result, _ = fit_fusion_mlp_listwise_streaming(
        setwise_train,
        setwise_tune,
        setwise_config,
        np.random.default_rng(seed),
        verbose=True,
        feature_indices=tuple(range(setwise_train.shape[-1])),
        candidate_name=f"fold{fold_index}_shared_setwise",
    )
    setwise_logits = _predict_setwise_streaming(
        setwise_model,
        score_features,
        setwise_result,
        batch_size=SETWISE_BATCH_SIZE,
    )
    _save_head(
        output_dir / f"fold-{fold_index:02d}-shared-setwise.npz",
        setwise_result,
        hidden_dim=SETWISE_HIDDEN_DIM,
        context_transform_version=1,
        fit_rows=(train_start, tune_start),
        tune_rows=(tune_start, train_stop),
    )
    del setwise_model
    release_memory()

    score_times = np.asarray(
        times[score_start:score_stop],
        dtype=np.int64,
    )
    comparison = compose_dataset1_final_scores(
        control_mlp_logits=control_logits,
        candidate_mlp_logits=candidate_logits,
        shared_lgbm_logits=lgbm_logits,
        shared_setwise_logits=setwise_logits,
        query_times=score_times,
        protocol=protocol,
        minimum_time=float(score_times.min()),
        maximum_time=float(score_times.max()),
    )
    baseline_path = output_dir / f"fold-{fold_index:02d}-control.npy"
    candidate_path = output_dir / f"fold-{fold_index:02d}-candidate.npy"
    np.save(baseline_path, comparison.control.astype(np.float32))
    np.save(candidate_path, comparison.candidate.astype(np.float32))
    fingerprint = _json_sha256(
        {
            "features_sha256": _read_json(
                Path(output_dir / "frozen-config.json")
            )["rolling_manifest_sha256"],
            "fold": fold_index,
            "score_rows": [score_start, score_stop],
            "score_time_min": int(score_times.min()),
            "score_time_max": int(score_times.max()),
        }
    )
    baseline_metrics = ranking_metrics(comparison.control)
    candidate_metrics = ranking_metrics(
        comparison.candidate,
        baseline_scores=comparison.control,
    )
    time_boundary = fold_payload["time_boundary"]
    report = {
        "fold_id": f"fold-{fold_index:02d}",
        "fit_rows": [fit_start, tune_start],
        "setwise_fit_rows": [train_start, tune_start],
        "tune_rows": [tune_start, train_stop],
        "score_rows": [score_start, score_stop],
        "train_time_max": int(time_boundary["train_time_max"]),
        "score_time_min": int(time_boundary["score_time_min"]),
        "score_time_max": int(time_boundary["score_time_max"]),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "elapsed_seconds": time.time() - started,
    }
    manifest = {
        "fold_id": report["fold_id"],
        "train_time_max": report["train_time_max"],
        "score_time_min": report["score_time_min"],
        "score_time_max": report["score_time_max"],
        "candidate_fingerprint": fingerprint,
        "baseline": _artifact_descriptor(
            baseline_path,
            output_dir=output_dir,
        ),
        "candidates": {
            CANDIDATE_KEY: {
                **_artifact_descriptor(
                    candidate_path,
                    output_dir=output_dir,
                ),
                "integration_id": BASE_CONTEXT_INTEGRATION_ID,
                "candidate_fingerprint": fingerprint,
                "weight": 1.0,
            }
        },
    }
    del (
        control_logits,
        candidate_logits,
        lgbm_logits,
        setwise_logits,
        comparison,
    )
    gc.collect()
    return {"report": report, "manifest": manifest}


def _frozen_protocol(
    state: dict[str, Any],
) -> tuple[BaseContextBlendProtocol, dict[str, Any], dict[str, Any]]:
    lgbm = state.get("lgbm_result")
    if lgbm is None:
        raise ValueError("Dataset1 source checkpoint has no LGBM expert")
    time_ramp = state.get("time_ramp_config")
    if (
        state.get("time_ramp_setwise_result") is None
        or state.get("time_ramp_setwise_fusion_state") is None
        or time_ramp is None
        or float(time_ramp["power"]) != 0.5
    ):
        raise ValueError("Dataset1 source is not the gamma=0.5 champion")
    calibration = ExpertBlendCalibration(
        mode=str(getattr(lgbm, "blend_mode", "probability")),
        mlp_temperature=float(getattr(lgbm, "mlp_temperature", 1.0)),
        lgbm_temperature=float(
            getattr(lgbm, "lgbm_temperature", 1.0)
        ),
        rrf_k=float(getattr(lgbm, "rrf_k", 60.0)),
    )
    training = state["config"]
    control = {
        "context_transform_version": 0,
        "seed": int(training.seed),
        "epochs": int(training.epochs),
        "batch_size": int(training.train_batch_size),
        "learning_rate": float(training.lr),
        "weight_decay": float(training.weight_decay),
        "hidden_dim": int(state["fusion_hidden_dim"]),
        "selection_metric": str(training.selection_metric),
        "early_stop_patience": int(training.early_stop_patience),
        "feature_indices": [
            int(index)
            for index in state["fusion_result"].feature_indices
        ],
    }
    candidate = {**control, "context_transform_version": 1}
    return (
        BaseContextBlendProtocol(
            mlp_weight=float(lgbm.mlp_weight),
            expert_calibration=calibration,
            time_ramp_power=0.5,
        ),
        control,
        candidate,
    )


def _fusion_config(values: dict[str, Any]) -> FusionConfig:
    return FusionConfig(
        epochs=int(values["epochs"]),
        batch_size=int(values["batch_size"]),
        lr=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        hidden_dim=int(values["hidden_dim"]),
        selection_metric=str(values["selection_metric"]),
        early_stop_patience=int(values["early_stop_patience"]),
        context_transform_version=int(
            values["context_transform_version"]
        ),
    )


def _predict_mlp_streaming(
    model: Any,
    features: Any,
    result: Any,
    *,
    context_transform_version: int,
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    indices = tuple(int(index) for index in result.feature_indices)
    with jt.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            batch = np.asarray(
                features[start:stop][..., indices],
                dtype=np.float32,
            )
            if context_transform_version:
                batch = setwise_context_features(
                    batch,
                    transform_version=context_transform_version,
                )
            normalized = ((batch - result.mean) / result.std).astype(
                np.float32,
                copy=False,
            )
            logits = model(jt.array(normalized, dtype=jt.float32))
            scores[start:stop] = np.asarray(
                logits.numpy(),
                dtype=np.float32,
            )
            del batch, normalized, logits
    return scores


def _predict_setwise_streaming(
    model: Any,
    features: Any,
    result: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(features.shape[:2], dtype=np.float32)
    indices = tuple(int(index) for index in result.feature_indices)
    with jt.no_grad():
        for start in range(0, features.shape[0], batch_size):
            stop = min(start + batch_size, features.shape[0])
            batch = setwise_context_features(
                np.asarray(features[start:stop], dtype=np.float32)
            )
            batch = batch[..., indices]
            normalized = ((batch - result.mean) / result.std).astype(
                np.float32,
                copy=False,
            )
            logits = model(jt.array(normalized, dtype=jt.float32))
            scores[start:stop] = np.asarray(
                logits.numpy(),
                dtype=np.float32,
            )
            del batch, normalized, logits
    return scores


def _save_head(
    path: Path,
    result: Any,
    *,
    hidden_dim: int,
    context_transform_version: int,
    fit_rows: tuple[int, int],
    tune_rows: tuple[int, int],
) -> None:
    payload = {
        "mean": np.asarray(result.mean, dtype=np.float32),
        "std": np.asarray(result.std, dtype=np.float32),
        "feature_indices": np.asarray(
            result.feature_indices,
            dtype=np.int32,
        ),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
        "context_transform_version": np.asarray(
            [context_transform_version],
            dtype=np.int32,
        ),
        "fit_rows": np.asarray(fit_rows, dtype=np.int64),
        "tune_rows": np.asarray(tune_rows, dtype=np.int64),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _validate_source_manifest(manifest: dict[str, Any]) -> None:
    protocol = manifest.get("protocol", {})
    if (
        manifest.get("dataset_name") != "dataset1"
        or int(protocol.get("selection_fold_count", -1))
        < SELECTION_FOLDS
        or int(protocol.get("train_window_rows", -1)) != 100_000
        or int(protocol.get("score_rows", -1)) != 25_000
    ):
        raise ValueError("unexpected Dataset1 rolling-origin manifest")


def _artifact_descriptor(
    path: Path,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": str(path.relative_to(output_dir)),
        "sha256": _sha256(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
