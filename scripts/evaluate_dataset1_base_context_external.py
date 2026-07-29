from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
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
from jgrec.rankers.hybrid.base_context_head import (
    BaseContextHeadArtifact,
    save_base_context_head,
)
from jgrec.rankers.hybrid.expert_fusion import ExpertBlendCalibration
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    build_fusion_from_state,
    fit_fusion_mlp_streaming,
)
from jgrec.rankers.hybrid.fusion_lgbm import predict_logits_lgbm
from jgrec.rankers.hybrid.setwise import setwise_context_features
from jgrec.robust_weight_selection import evaluate_locked_external

EXPECTED_EXTERNAL_FEATURE_SHA256 = (
    "43f32c3430eec82180314f889cdfe94b9a8fdb9dc3fd338f7b22f3fa44ad6906"
)
EXPECTED_EXTERNAL_TIME_SHA256 = (
    "2f5a68329f5e75f26acd77bf22e578bb47f51261aaf3c30ee4dca7aefbd89f96"
)
CANDIDATE_KEY = 1.0
RECENT_TRAIN_ROWS = 100_000
TUNE_ROWS = 20_000


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Open the frozen Dataset1 external holdout once for the "
            "rolling-locked base-context v1 candidate."
        )
    )
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--rolling-frozen-config", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--train-features", required=True, type=Path)
    parser.add_argument("--train-times", required=True, type=Path)
    parser.add_argument("--external-features", required=True, type=Path)
    parser.add_argument("--external-times", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    require_jittor_cuda(jt)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    lock = _read_json(args.selection_lock)
    frozen = _read_json(args.rolling_frozen_config)
    _validate_lock_and_frozen(lock, frozen)
    _require_hash(
        args.source_checkpoint,
        frozen["source_checkpoint_sha256"],
        "source checkpoint",
    )
    rolling_manifest_path = Path(frozen["rolling_manifest"])
    _require_hash(
        rolling_manifest_path,
        frozen["rolling_manifest_sha256"],
        "rolling manifest",
    )
    rolling_manifest = _read_json(rolling_manifest_path)
    _require_hash(
        args.train_features,
        rolling_manifest["source"]["features_sha256"],
        "training features",
    )
    _require_hash(
        args.train_times,
        rolling_manifest["source"]["times_sha256"],
        "training times",
    )
    source_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset1",
    )
    protocol = _protocol_from_state(source_state)
    control_training = frozen["control_training"]
    candidate_training = frozen["candidate_training"]
    validate_context_only_difference(
        control_training,
        candidate_training,
    )

    train_features = np.load(
        args.train_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    train_times = np.load(
        args.train_times,
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        train_features.shape[0] < RECENT_TRAIN_ROWS
        or train_times.shape != (train_features.shape[0],)
    ):
        raise ValueError("external training cache has an invalid shape")
    train_stop = int(train_features.shape[0])
    recent_start = train_stop - RECENT_TRAIN_ROWS
    tune_start = train_stop - TUNE_ROWS
    maximum_fit_rows = int(
        getattr(source_state["config"], "max_train_events", 0)
    )
    fit_start = recent_start
    if maximum_fit_rows > 0:
        fit_start = max(fit_start, tune_start - maximum_fit_rows)
    feature_indices = tuple(
        int(index)
        for index in source_state["fusion_result"].feature_indices
    )
    candidate_model, candidate_result = fit_fusion_mlp_streaming(
        train_features[fit_start:tune_start],
        train_features[tune_start:train_stop],
        _fusion_config(candidate_training),
        np.random.default_rng(int(candidate_training["seed"])),
        verbose=True,
        feature_indices=feature_indices,
        candidate_name="external_base_context_v1",
    )
    candidate_head_path = args.output_dir / "base-context-v1-head.npz"
    save_base_context_head(
        candidate_head_path,
        BaseContextHeadArtifact(
            context_transform_version=1,
            hidden_dim=int(candidate_training["hidden_dim"]),
            feature_indices=tuple(candidate_result.feature_indices),
            mean=np.asarray(candidate_result.mean, dtype=np.float32),
            std=np.asarray(candidate_result.std, dtype=np.float32),
            state={
                key: np.asarray(value, dtype=np.float32)
                for key, value in candidate_result.state.items()
            },
            best_val_ap=float(candidate_result.best_val_ap),
            best_val_mrr=float(candidate_result.best_val_mrr),
            candidate_name=str(candidate_result.candidate_name),
            fit_rows=(fit_start, tune_start),
            tune_rows=(tune_start, train_stop),
        ),
    )

    # This receipt is deliberately written before loading any external array.
    open_receipt_path = args.output_dir / "external-score-open-receipt.json"
    _write_json_exclusive(
        open_receipt_path,
        {
            "schema_version": 1,
            "protocol": "dataset1_base_context_external_score_open_v1",
            "opened_at_utc": datetime.now(UTC).isoformat(),
            "selection_lock_sha256": _sha256(args.selection_lock),
            "integration_id": BASE_CONTEXT_INTEGRATION_ID,
            "selected_weight": CANDIDATE_KEY,
            "external_feature_sha256": EXPECTED_EXTERNAL_FEATURE_SHA256,
            "external_time_sha256": EXPECTED_EXTERNAL_TIME_SHA256,
        },
    )
    _require_hash(
        args.external_features,
        EXPECTED_EXTERNAL_FEATURE_SHA256,
        "external features",
    )
    _require_hash(
        args.external_times,
        EXPECTED_EXTERNAL_TIME_SHA256,
        "external times",
    )
    external_features = np.load(
        args.external_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    external_times = np.load(
        args.external_times,
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        external_features.shape != (20_000, 100, 63)
        or external_times.shape != (20_000,)
    ):
        raise ValueError("unexpected Dataset1 external holdout shape")

    source_result = source_state["fusion_result"]
    source_model = build_fusion_from_state(
        input_dim=len(source_result.feature_indices),
        hidden_dim=int(source_state["fusion_hidden_dim"]),
        state=source_state["fusion_state"],
    )
    control_logits = _predict_mlp_streaming(
        source_model,
        external_features,
        source_result,
        context_transform_version=0,
        batch_size=int(control_training["batch_size"]),
    )
    candidate_logits = _predict_mlp_streaming(
        candidate_model,
        external_features,
        candidate_result,
        context_transform_version=1,
        batch_size=int(candidate_training["batch_size"]),
    )
    del source_model, candidate_model
    release_memory()

    lgbm_result = source_state["lgbm_result"]
    lgbm_indices = tuple(
        int(index) for index in lgbm_result.feature_indices
    )
    lgbm_logits = predict_logits_lgbm(
        lgbm_result.model_text,
        np.asarray(external_features[..., lgbm_indices]),
    )
    setwise_result = source_state["time_ramp_setwise_result"]
    setwise_model = build_fusion_from_state(
        input_dim=int(np.asarray(setwise_result.mean).shape[0]),
        hidden_dim=int(source_state["time_ramp_setwise_hidden_dim"]),
        state=source_state["time_ramp_setwise_fusion_state"],
    )
    setwise_logits = _predict_setwise_streaming(
        setwise_model,
        external_features,
        setwise_result,
        batch_size=256,
    )
    del setwise_model
    release_memory()

    comparison = compose_dataset1_final_scores(
        control_mlp_logits=control_logits,
        candidate_mlp_logits=candidate_logits,
        shared_lgbm_logits=lgbm_logits,
        shared_setwise_logits=setwise_logits,
        query_times=external_times,
        protocol=protocol,
        minimum_time=float(external_times.min()),
        maximum_time=float(external_times.max()),
    )
    baseline_path = args.output_dir / "external-control.npy"
    candidate_path = args.output_dir / "external-candidate.npy"
    np.save(baseline_path, comparison.control.astype(np.float32))
    np.save(candidate_path, comparison.candidate.astype(np.float32))
    fingerprint = _json_sha256(
        {
            "external_features_sha256": EXPECTED_EXTERNAL_FEATURE_SHA256,
            "external_times_sha256": EXPECTED_EXTERNAL_TIME_SHA256,
            "shape": list(external_features.shape),
        }
    )
    external_manifest = {
        "schema_version": 1,
        "protocol": "exact_integrated_external_holdout_v1",
        "integration_id": BASE_CONTEXT_INTEGRATION_ID,
        "selected_weight": CANDIDATE_KEY,
        "selection_lock_sha256": _sha256(args.selection_lock),
        "positive_candidate_column": 0,
        "candidate_fingerprint": fingerprint,
        "training_time_max": int(train_times[train_stop - 1]),
        "score_time_min": int(external_times.min()),
        "score_time_max": int(external_times.max()),
        "minimum_train_to_score_gap": 1,
        "baseline": _artifact_descriptor(
            baseline_path,
            output_dir=args.output_dir,
        ),
        "candidate": {
            **_artifact_descriptor(
                candidate_path,
                output_dir=args.output_dir,
            ),
            "integration_id": BASE_CONTEXT_INTEGRATION_ID,
            "candidate_fingerprint": fingerprint,
            "weight": CANDIDATE_KEY,
        },
    }
    manifest_path = args.output_dir / "external-manifest.json"
    _write_json(manifest_path, external_manifest)
    evaluation = evaluate_locked_external(
        manifest_path=manifest_path,
        selection_lock_path=args.selection_lock,
        state_dir=args.output_dir / "external-state",
    )
    final = {
        "status": evaluation["status"],
        "external_pass": evaluation["status"] == "accepted",
        "package_authorized": evaluation["status"] == "accepted",
        "weight_rescan_authorized": False,
        "candidate_head": str(candidate_head_path.resolve()),
        "candidate_head_sha256": _sha256(candidate_head_path),
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "selection_lock_sha256": _sha256(args.selection_lock),
        "external_manifest_sha256": _sha256(manifest_path),
        "external_evaluation_sha256": _sha256(
            args.output_dir
            / "external-state"
            / "external-evaluation-report.json"
        ),
        "fit_rows": [fit_start, tune_start],
        "tune_rows": [tune_start, train_stop],
        "protocol": {
            "integration_id": BASE_CONTEXT_INTEGRATION_ID,
            "candidate_training": candidate_training,
            "final_integration": {
                "mlp_weight": protocol.mlp_weight,
                "expert_calibration": asdict(
                    protocol.expert_calibration
                ),
                "time_ramp_power": protocol.time_ramp_power,
            },
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "external-result.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0 if final["external_pass"] else 2


def _validate_lock_and_frozen(
    lock: dict[str, Any],
    frozen: dict[str, Any],
) -> None:
    if (
        lock.get("protocol")
        != "exact_integrated_weight_selection_lock_v1"
        or lock.get("integration_id") != BASE_CONTEXT_INTEGRATION_ID
        or float(lock.get("selected_weight", -1.0)) != CANDIDATE_KEY
    ):
        raise ValueError("selection lock does not authorize context v1")
    if (
        frozen.get("integration_id") != BASE_CONTEXT_INTEGRATION_ID
        or frozen.get("external_paths_present") is not False
        or frozen.get("external_metrics_read") is not False
    ):
        raise ValueError("rolling frozen config violates external isolation")


def _protocol_from_state(
    state: dict[str, Any],
) -> BaseContextBlendProtocol:
    lgbm = state.get("lgbm_result")
    time_ramp = state.get("time_ramp_config")
    if (
        lgbm is None
        or state.get("time_ramp_setwise_result") is None
        or state.get("time_ramp_setwise_fusion_state") is None
        or time_ramp is None
        or float(time_ramp["power"]) != 0.5
    ):
        raise ValueError("source checkpoint is not the Dataset1 champion")
    return BaseContextBlendProtocol(
        mlp_weight=float(lgbm.mlp_weight),
        expert_calibration=ExpertBlendCalibration(
            mode=str(getattr(lgbm, "blend_mode", "probability")),
            mlp_temperature=float(
                getattr(lgbm, "mlp_temperature", 1.0)
            ),
            lgbm_temperature=float(
                getattr(lgbm, "lgbm_temperature", 1.0)
            ),
            rrf_k=float(getattr(lgbm, "rrf_k", 60.0)),
        ),
        time_ramp_power=0.5,
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
        context_transform_version=1,
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


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


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
