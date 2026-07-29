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

from jgrec.core.cuda import require_jittor_cuda
from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    fit_fusion_mlp_listwise_fixed,
)
from jgrec.rankers.hybrid.rolling_origin import (
    RollingOriginFold,
    select_candidate_on_rolling_origins,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView
from jgrec.rankers.hybrid.time_ramp import apply_time_ramp
from train_evaluate_dataset1_full100_setwise import (
    _predict_streaming,
    _softmax,
)

POWERS = (0.5, 1.0, 2.0)
MINIMUM_MEAN_DELTA = 0.0002


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    manifest_hash = _require_hash_sidecar(
        args.manifest,
        args.manifest_sha256,
    )
    manifest = _read_json(args.manifest)
    _validate_manifest(manifest)
    selection_folds = [
        _fold_from_payload(payload)
        for payload in manifest["folds"]
        if payload["role"] == "selection"
    ]
    if len(selection_folds) != 3:
        raise ValueError("frozen protocol requires three selection folds")

    feature_path = Path(manifest["source"]["features"])
    time_path = Path(manifest["source"]["times"])
    feature_hash_before = _sha256(feature_path)
    time_hash_before = _sha256(time_path)
    if feature_hash_before != manifest["source"]["features_sha256"]:
        raise ValueError("rolling-origin feature cache hash differs")
    if time_hash_before != manifest["source"]["times_sha256"]:
        raise ValueError("rolling-origin time sidecar hash differs")

    require_jittor_cuda(jt)
    args.output_dir.mkdir(parents=True)
    started = time.time()
    frozen = {
        "status": "frozen_before_training",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "selection_fold_indices": [
            fold.index for fold in selection_folds
        ],
        "gate_fold_metrics_read": False,
        "powers": list(POWERS),
        "minimum_mean_delta": MINIMUM_MEAN_DELTA,
        "training": manifest["protocol"]["head_training"],
        "selection_order": manifest["protocol"]["selection"]["order"],
    }
    _write_json_atomic(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)

    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    times = np.load(time_path, mmap_mode="r", allow_pickle=False)
    fold_reports: list[dict[str, Any]] = []
    baseline_mrrs: list[float] = []
    candidate_mrrs = {
        _power_name(power): [] for power in POWERS
    }
    for fold in selection_folds:
        fold_report, raw_scores, setwise_scores = train_and_score_fold(
            features=features,
            times=times,
            fold=fold,
            output_dir=args.output_dir,
            training=manifest["protocol"]["head_training"],
        )
        score_start, score_stop = fold.score_rows
        fold_times = times[score_start:score_stop]
        baseline_mrrs.append(float(fold_report["raw_mrr"]))
        ramp_metrics: dict[str, Any] = {}
        for power in POWERS:
            name = _power_name(power)
            ramped = apply_time_ramp(
                raw_scores,
                setwise_scores,
                fold_times,
                power=power,
            )
            mrr = _mrr(ramped)
            candidate_mrrs[name].append(mrr)
            ramp_metrics[name] = {
                "power": power,
                "mrr": mrr,
                "delta_vs_raw": mrr - float(
                    fold_report["raw_mrr"]
                ),
            }
        fold_report["time_ramps"] = ramp_metrics
        fold_reports.append(fold_report)
        _write_json_atomic(
            args.output_dir / "selection-progress.json",
            {
                "status": "training_selection_folds",
                "completed_folds": len(fold_reports),
                "gate_fold_metrics_read": False,
                "folds": fold_reports,
                "elapsed_seconds": time.time() - started,
            },
        )
        del raw_scores, setwise_scores
        release_memory()

    selection = select_candidate_on_rolling_origins(
        baseline_mrrs=tuple(baseline_mrrs),
        candidate_mrrs={
            name: tuple(values)
            for name, values in candidate_mrrs.items()
        },
        minimum_mean_delta=MINIMUM_MEAN_DELTA,
        tie_break_order=tuple(
            _power_name(power) for power in POWERS
        ),
    )
    report = {
        "status": (
            "selected"
            if selection.selected_name is not None
            else "no_eligible_candidate"
        ),
        "gate_passed": selection.selected_name is not None,
        "gate_fold_unlocked": selection.selected_name is not None,
        "gate_fold_metrics_read": False,
        "manifest_sha256": manifest_hash,
        "selection": asdict(selection),
        "selected_power": (
            None
            if selection.selected_name is None
            else _name_power(selection.selected_name)
        ),
        "baseline_mrrs": baseline_mrrs,
        "candidate_mrrs": candidate_mrrs,
        "folds": fold_reports,
        "source_hashes_unchanged": bool(
            _sha256(feature_path) == feature_hash_before
            and _sha256(time_path) == time_hash_before
        ),
        "elapsed_seconds": time.time() - started,
    }
    if not report["source_hashes_unchanged"]:
        raise RuntimeError("rolling-origin source artifacts changed")
    report_path = args.output_dir / "selection-report.json"
    _write_json_atomic(report_path, report)
    report_hash = _sha256(report_path)
    (args.output_dir / "selection-report.sha256").write_text(
        f"{report_hash}  {report_path.name}\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if selection.selected_name is not None else 2


def train_and_score_fold(
    *,
    features: Any,
    times: np.ndarray,
    fold: RollingOriginFold,
    output_dir: Path,
    training: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    fold_started = time.time()
    train_start, train_stop = fold.train_rows
    score_start, score_stop = fold.score_rows
    train_features = features[train_start:train_stop]
    score_features = features[score_start:score_stop]
    config = FusionConfig(
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        lr=float(training["learning_rate"]),
        weight_decay=0.0,
        hidden_dim=int(training["hidden_dim"]),
        selection_metric="mrr",
        early_stop_patience=0,
    )
    seed = int(training["seed"])

    raw_model, raw_result, raw_losses = (
        fit_fusion_mlp_listwise_fixed(
            train_features,
            score_features,
            config,
            np.random.default_rng(seed),
            verbose=True,
            feature_indices=tuple(range(features.shape[-1])),
            candidate_name=f"rolling_fold{fold.index}_raw",
        )
    )
    raw_model_path = output_dir / (
        f"fold-{fold.index:02d}-raw-head.npz"
    )
    _save_head_model(
        raw_model_path,
        result=raw_result,
        hidden_dim=config.hidden_dim,
        source_feature_count=int(features.shape[-1]),
        transform_version=0,
        fold=fold,
        seed=seed,
    )
    raw_logits = _predict_streaming(
        raw_model,
        score_features,
        raw_result.mean,
        raw_result.std,
        feature_indices=raw_result.feature_indices,
        batch_size=config.batch_size,
    )
    raw_scores = _softmax(raw_logits)
    del raw_model, raw_logits
    release_memory()

    setwise_train = SetwiseFeatureView(train_features)
    setwise_score = SetwiseFeatureView(score_features)
    setwise_model, setwise_result, setwise_losses = (
        fit_fusion_mlp_listwise_fixed(
            setwise_train,
            setwise_score,
            config,
            np.random.default_rng(seed),
            verbose=True,
            feature_indices=tuple(
                range(setwise_train.shape[-1])
            ),
            candidate_name=f"rolling_fold{fold.index}_setwise",
        )
    )
    setwise_model_path = output_dir / (
        f"fold-{fold.index:02d}-setwise-head.npz"
    )
    _save_head_model(
        setwise_model_path,
        result=setwise_result,
        hidden_dim=config.hidden_dim,
        source_feature_count=int(features.shape[-1]),
        transform_version=1,
        fold=fold,
        seed=seed,
    )
    setwise_logits = _predict_streaming(
        setwise_model,
        setwise_score,
        setwise_result.mean,
        setwise_result.std,
        feature_indices=setwise_result.feature_indices,
        batch_size=config.batch_size,
    )
    setwise_scores = _softmax(setwise_logits)
    del setwise_model, setwise_logits
    release_memory()

    scores_path = output_dir / (
        f"fold-{fold.index:02d}-expert-scores.npz"
    )
    np.savez_compressed(
        scores_path,
        raw=np.asarray(raw_scores, dtype=np.float32),
        setwise=np.asarray(setwise_scores, dtype=np.float32),
        query_times=np.asarray(
            times[score_start:score_stop],
            dtype=np.int64,
        ),
    )
    report = {
        "fold": fold.index,
        "role": fold.role,
        "train_rows": list(fold.train_rows),
        "score_rows": list(fold.score_rows),
        "fixed_epochs": config.epochs,
        "early_stopping": False,
        "raw_epoch_losses": list(raw_losses),
        "setwise_epoch_losses": list(setwise_losses),
        "raw_mrr": _mrr(raw_scores),
        "setwise_mrr": _mrr(setwise_scores),
        "raw_model_sha256": _sha256(raw_model_path),
        "setwise_model_sha256": _sha256(setwise_model_path),
        "expert_scores_sha256": _sha256(scores_path),
        "elapsed_seconds": time.time() - fold_started,
    }
    gc.collect()
    return report, raw_scores, setwise_scores


def _save_head_model(
    path: Path,
    *,
    result: Any,
    hidden_dim: int,
    source_feature_count: int,
    transform_version: int,
    fold: RollingOriginFold,
    seed: int,
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
        "training_seed": np.asarray([seed], dtype=np.int32),
        "train_rows": np.asarray(fold.train_rows, dtype=np.int64),
        "score_rows": np.asarray(fold.score_rows, dtype=np.int64),
    }
    payload.update(
        {
            f"state__{key}": np.asarray(value, dtype=np.float32)
            for key, value in result.state.items()
        }
    )
    np.savez_compressed(path, **payload)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    protocol = manifest.get("protocol", {})
    if (
        manifest.get("status") != "frozen_before_training"
        or manifest.get("dataset_name") != "dataset1"
        or manifest.get("level") != "cached_head_rolling_origin"
        or protocol.get("fold_count") != 4
        or protocol.get("selection_fold_count") != 3
        or protocol.get("train_window_rows") != 100_000
        or protocol.get("score_rows") != 25_000
    ):
        raise ValueError("manifest differs from frozen rolling-origin protocol")


def _fold_from_payload(payload: dict[str, Any]) -> RollingOriginFold:
    return RollingOriginFold(
        index=int(payload["index"]),
        train_rows=tuple(int(value) for value in payload["train_rows"]),
        score_rows=tuple(int(value) for value in payload["score_rows"]),
        role=str(payload["role"]),
    )


def _power_name(power: float) -> str:
    return f"gamma_{float(power):.1f}"


def _name_power(name: str) -> float:
    return float(name.removeprefix("gamma_"))


def _mrr(scores: np.ndarray) -> float:
    values = np.asarray(scores)
    ranks = 1 + np.sum(values[:, 1:] > values[:, :1], axis=1)
    return float(np.mean(1.0 / ranks))


def _require_hash_sidecar(path: Path, sidecar: Path) -> str:
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError("manifest hash differs from sidecar")
    return actual


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
