from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jittor as jt
import numpy as np

from jgrec.listwise_mlp_exact_blend import (
    build_rolling_selection_manifest,
    materialize_fold_candidates,
)
from jgrec.rankers.hybrid.fusion import (
    FusionResult,
    build_fusion_from_state,
    predict_logits,
)
from jgrec.rankers.hybrid.setwise import setwise_context_features
from jgrec.service_normalizer import (
    StreamingFeatureNormalizer,
    normalizer_drift_report,
    replace_result_normalizer,
)

INTEGRATION_ID = "post_refit_service_normalizer_calibration_v1"
SETWISE_WEIGHT = 0.80


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate service-distribution normalizer calibration on the "
            "three frozen A2 rolling-origin folds without reading external data."
        )
    )
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--short-none-scores", required=True, type=Path)
    parser.add_argument("--a2-artifact-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    feature_path = Path(f"{args.train_cache_prefix}.train.npy")
    candidate_path = Path(f"{args.train_cache_prefix}.train-candidates.npy")
    event_time_path = Path(f"{args.train_cache_prefix}.train-time.npy")
    frozen = _read_json(args.a2_artifact_dir / "frozen-config.json")
    source_manifest = _read_json(args.a2_artifact_dir / "rolling-manifest.json")
    _validate_frozen_inputs(
        frozen=frozen,
        source_manifest=source_manifest,
        feature_path=feature_path,
        candidate_path=candidate_path,
        short_none_path=args.short_none_scores,
    )

    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    candidates = np.load(
        candidate_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    event_time = np.load(
        event_time_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    short_none = np.load(
        args.short_none_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        features.shape != (200_000, 100, 63)
        or candidates.shape != features.shape[:2]
        or short_none.shape != features.shape[:2]
        or event_time.shape != (features.shape[0],)
    ):
        raise ValueError("frozen A3 cache shapes do not match the A2 protocol")
    gnn_column = int(frozen["gnn_column"])

    protocol = {
        "status": "frozen_before_rolling_metrics",
        "integration_id": INTEGRATION_ID,
        "source_integration_id": source_manifest["integration_id"],
        "candidate_formula": ("0.80 * same_fold_setwise_state_with_service_normalizer + 0.20 * unchanged_fold_lgbm"),
        "normalizer_population": ("all unlabeled candidate features in the fold score interval"),
        "normalizer_passes": 1,
        "prediction_passes": 1,
        "weights_scanned": [1.0],
        "setwise_weight": SETWISE_WEIGHT,
        "batch_size": args.batch_size,
        "external_scores_read": False,
        "feature_cache": str(feature_path.resolve()),
        "feature_cache_sha256": _sha256(feature_path),
        "short_none_scores": str(args.short_none_scores.resolve()),
        "short_none_scores_sha256": _sha256(args.short_none_scores),
        "source_manifest": str((args.a2_artifact_dir / "rolling-manifest.json").resolve()),
        "source_manifest_sha256": _sha256(args.a2_artifact_dir / "rolling-manifest.json"),
    }
    _write_json(args.output_dir / "frozen-config.json", protocol)

    jt.flags.use_cuda = 1
    fold_entries: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in source_manifest["folds"]:
        fold_started = time.time()
        fold_id = str(fold["fold_id"])
        train_start, train_stop = (int(value) for value in fold["train_rows"])
        score_start, score_stop = (int(value) for value in fold["score_rows"])
        model_path = args.a2_artifact_dir / "models" / fold_id / "setwise.npz"
        original = _load_fusion_result(model_path)
        state_sha256 = _state_sha256(original.state)
        service = _service_normalizer(
            features=features,
            short_none=short_none,
            start=score_start,
            stop=score_stop,
            gnn_column=gnn_column,
            result=original,
            batch_size=args.batch_size,
        )
        calibrated = replace_result_normalizer(original, service)
        if calibrated.state is not original.state:
            raise RuntimeError("normalizer calibration replaced model state")
        if _state_sha256(calibrated.state) != state_sha256:
            raise RuntimeError("normalizer calibration mutated model state")
        drift = normalizer_drift_report(
            training_mean=original.mean,
            training_std=original.std,
            service=service,
        )
        model = build_fusion_from_state(
            input_dim=len(original.feature_indices),
            hidden_dim=_load_hidden_dim(model_path),
            state=original.state,
        )
        original_setwise, calibrated_setwise = _score_setwise_pair(
            model=model,
            features=features,
            short_none=short_none,
            start=score_start,
            stop=score_stop,
            gnn_column=gnn_column,
            original=original,
            calibrated=calibrated,
            batch_size=args.batch_size,
        )
        baseline_path = Path(fold["baseline"]["path"])
        baseline = np.load(
            baseline_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        replay_error = float(
            np.max(
                np.abs(original_setwise.sum(axis=1) - 1.0),
                initial=0.0,
            )
        )
        if replay_error > 1e-6:
            raise RuntimeError("original Setwise replay is not normalized")
        lgbm = (np.asarray(baseline, dtype=np.float64) - SETWISE_WEIGHT * original_setwise) / (1.0 - SETWISE_WEIGHT)
        if (
            not np.isfinite(lgbm).all()
            or float(np.min(lgbm)) < -1e-8
            or not np.allclose(
                lgbm.sum(axis=1),
                1.0,
                atol=1e-6,
                rtol=0.0,
            )
        ):
            raise RuntimeError(
                "unchanged LGBM expert could not be recovered exactly from the frozen integrated baseline"
            )
        candidate = SETWISE_WEIGHT * calibrated_setwise + (1.0 - SETWISE_WEIGHT) * lgbm
        fold_output = args.output_dir / "scores" / fold_id
        entry = materialize_fold_candidates(
            output_dir=fold_output,
            fold_id=fold_id,
            integration_id=INTEGRATION_ID,
            train_time_max=int(event_time[train_stop - 1]),
            score_time_min=int(event_time[score_start]),
            score_time_max=int(event_time[score_stop - 1]),
            baseline_scores=baseline,
            auxiliary_scores=candidate,
            candidate_ids=candidates[score_start:score_stop],
            weights=(1.0,),
        )
        entry["train_rows"] = [train_start, train_stop]
        entry["score_rows"] = [score_start, score_stop]
        fold_entries.append(entry)
        fold_report = {
            "fold_id": fold_id,
            "train_rows": [train_start, train_stop],
            "score_rows": [score_start, score_stop],
            "setwise_model": str(model_path.resolve()),
            "setwise_model_sha256": _sha256(model_path),
            "model_state_sha256_before": state_sha256,
            "model_state_sha256_after": _state_sha256(calibrated.state),
            "feature_indices_unchanged": (calibrated.feature_indices == original.feature_indices),
            "normalizer_drift": drift,
            "max_abs_setwise_probability_change": float(
                np.max(
                    np.abs(calibrated_setwise - original_setwise),
                    initial=0.0,
                )
            ),
            "mean_abs_setwise_probability_change": float(np.mean(np.abs(calibrated_setwise - original_setwise))),
            "score_artifacts": entry,
            "external_scores_read": False,
            "elapsed_seconds": time.time() - fold_started,
        }
        fold_reports.append(fold_report)
        _write_json(
            args.output_dir / "rolling-progress.json",
            {
                "status": "scoring",
                "completed_folds": len(fold_reports),
                "total_folds": len(source_manifest["folds"]),
                "folds": fold_reports,
                "external_scores_read": False,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(
            json.dumps(
                fold_report,
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        del (
            model,
            original_setwise,
            calibrated_setwise,
            baseline,
            lgbm,
            candidate,
        )
        _release_jittor()

    manifest_path = args.output_dir / "rolling-manifest.json"
    build_rolling_selection_manifest(
        integration_id=INTEGRATION_ID,
        fold_entries=fold_entries,
        output_path=manifest_path,
    )
    report = {
        "status": "complete",
        "integration_id": INTEGRATION_ID,
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "rolling_manifest": str(manifest_path.resolve()),
        "rolling_manifest_sha256": _sha256(manifest_path),
        "external_scores_read": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "rolling-report.json", report)
    print(
        json.dumps(report, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return 0


def _service_normalizer(
    *,
    features: np.ndarray,
    short_none: np.ndarray,
    start: int,
    stop: int,
    gnn_column: int,
    result: FusionResult,
    batch_size: int,
):
    accumulator = StreamingFeatureNormalizer()
    for batch_start in range(start, stop, batch_size):
        batch_stop = min(batch_start + batch_size, stop)
        selected = _setwise_batch(
            features=features,
            short_none=short_none,
            start=batch_start,
            stop=batch_stop,
            gnn_column=gnn_column,
            indices=result.feature_indices,
        )
        accumulator.update(selected)
    return accumulator.finalize()


def _score_setwise_pair(
    *,
    model: Any,
    features: np.ndarray,
    short_none: np.ndarray,
    start: int,
    stop: int,
    gnn_column: int,
    original: FusionResult,
    calibrated: FusionResult,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (stop - start, features.shape[1])
    original_probabilities = np.empty(shape, dtype=np.float64)
    calibrated_probabilities = np.empty(shape, dtype=np.float64)
    for batch_start in range(start, stop, batch_size):
        batch_stop = min(batch_start + batch_size, stop)
        selected = _setwise_batch(
            features=features,
            short_none=short_none,
            start=batch_start,
            stop=batch_stop,
            gnn_column=gnn_column,
            indices=original.feature_indices,
        )
        destination = slice(batch_start - start, batch_stop - start)
        original_probabilities[destination] = _softmax(
            predict_logits(
                model,
                selected,
                original.mean,
                original.std,
            )
        )
        calibrated_probabilities[destination] = _softmax(
            predict_logits(
                model,
                selected,
                calibrated.mean,
                calibrated.std,
            )
        )
    return original_probabilities, calibrated_probabilities


def _setwise_batch(
    *,
    features: np.ndarray,
    short_none: np.ndarray,
    start: int,
    stop: int,
    gnn_column: int,
    indices: tuple[int, ...],
) -> np.ndarray:
    batch = np.array(
        features[start:stop],
        dtype=np.float32,
        copy=True,
    )
    batch[..., gnn_column] = short_none[start:stop]
    context = setwise_context_features(batch)
    if indices != tuple(range(context.shape[-1])):
        context = context[..., indices]
    return context


def _load_fusion_result(path: Path) -> FusionResult:
    payload = np.load(path, allow_pickle=False)
    try:
        state = {
            key.removeprefix("state__"): np.asarray(
                payload[key],
                dtype=np.float32,
            )
            for key in payload.files
            if key.startswith("state__")
        }
        return FusionResult(
            best_val_ap=0.0,
            best_val_mrr=0.0,
            state=state,
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            feature_indices=tuple(int(value) for value in payload["feature_indices"]),
            candidate_name=path.stem,
        )
    finally:
        payload.close()


def _load_hidden_dim(path: Path) -> int:
    payload = np.load(path, allow_pickle=False)
    try:
        return int(np.asarray(payload["hidden_dim"]).reshape(-1)[0])
    finally:
        payload.close()


def _validate_frozen_inputs(
    *,
    frozen: dict[str, Any],
    source_manifest: dict[str, Any],
    feature_path: Path,
    candidate_path: Path,
    short_none_path: Path,
) -> None:
    if frozen.get("status") != "frozen_before_rolling_training":
        raise ValueError("A2 rolling artifacts were not frozen before training")
    if source_manifest.get("integration_id") != frozen.get("integration_id"):
        raise ValueError("A2 frozen config and rolling manifest differ")
    if len(source_manifest.get("folds", [])) != 3:
        raise ValueError("A3 requires the frozen three A2 folds")
    hashes = {
        "train_cache_sha256": _sha256(feature_path),
        "candidate_sidecar_sha256": _sha256(candidate_path),
        "short_none_scores_sha256": _sha256(short_none_path),
    }
    for key, actual in hashes.items():
        if actual != frozen[key]:
            raise ValueError(f"{key} differs from the frozen A2 input")


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _state_sha256(state: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        values = np.ascontiguousarray(state[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


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
