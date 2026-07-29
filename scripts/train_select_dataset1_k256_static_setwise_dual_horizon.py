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

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.cuda import require_jittor_cuda
from jgrec.core.memory import release_memory
from jgrec.rankers.hybrid.expert_fusion import blend_expert_logits
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    fit_fusion_mlp_listwise_streaming,
    fit_fusion_mlp_streaming,
)
from jgrec.rankers.hybrid.fusion_lgbm import (
    fit_fusion_lgbm,
    predict_logits_lgbm,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView
from jgrec.rankers.hybrid.static_setwise import (
    blend_static_setwise,
    select_dual_horizon_static_weight,
    static_setwise_weight_grid,
)
from jgrec.rankers.hybrid.time_ramp import apply_time_ramp
from jgrec.robust_weight_selection import ranking_metrics
from train_evaluate_dataset1_full100_setwise import _softmax
from train_select_dataset1_base_context_rolling import (
    SETWISE_BATCH_SIZE,
    SETWISE_EPOCHS,
    SETWISE_HIDDEN_DIM,
    SETWISE_LEARNING_RATE,
    SETWISE_PATIENCE,
    TUNE_ROWS,
    _frozen_protocol,
    _fusion_config,
    _predict_mlp_streaming,
    _predict_setwise_streaming,
    _save_head,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--cache-prefix", required=True, type=Path)
    parser.add_argument("--cache-report", required=True, type=Path)
    parser.add_argument("--reference-cache-prefix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    require_jittor_cuda(jt)
    plan_hash = _require_sidecar(args.plan, args.plan_sha256)
    plan = _read_json(args.plan)
    report = _read_json(args.cache_report)
    _validate_inputs(
        plan=plan,
        plan_hash=plan_hash,
        report=report,
        checkpoint=args.source_checkpoint,
        cache_prefix=args.cache_prefix,
        reference_cache_prefix=args.reference_cache_prefix,
    )
    args.output_dir.mkdir(parents=True)
    started = time.time()
    source_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset1",
    )
    protocol, control_config, _ = _frozen_protocol(source_state)
    features_path = Path(f"{args.cache_prefix}.train.npy")
    times_path = Path(f"{args.cache_prefix}.train-time.npy")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    times = np.load(times_path, mmap_mode="r", allow_pickle=False)
    source_hashes = {
        "features": _sha256(features_path),
        "times": _sha256(times_path),
    }
    frozen = {
        "status": "frozen_before_training",
        "external_labels_read": False,
        "plan": str(args.plan.resolve()),
        "plan_sha256": plan_hash,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "cache_report": str(args.cache_report.resolve()),
        "cache_report_sha256": _sha256(args.cache_report),
        "cache_source_hashes": source_hashes,
        "weights": list(static_setwise_weight_grid()),
        "baseline": plan["baseline"],
        "head_training": plan["head_training"],
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)

    fold_results: dict[str, list[dict[str, Any]]] = {
        "near": [],
        "gapped": [],
    }
    for horizon, key in (
        ("near", "near_folds"),
        ("gapped", "gapped_folds"),
    ):
        for fold in plan[key]:
            result = _train_and_score_fold(
                features=features,
                times=times,
                fold=fold,
                source_state=source_state,
                protocol=protocol,
                control_config=control_config,
                global_time_bounds=tuple(plan["baseline"]["global_time_bounds"]),
                output_dir=args.output_dir,
            )
            fold_results[horizon].append(result)
            _write_json(
                args.output_dir / "progress.json",
                {
                    "status": "training_folds",
                    "completed_near": len(fold_results["near"]),
                    "completed_gapped": len(fold_results["gapped"]),
                    "external_labels_read": False,
                    "last_fold": fold["fold_id"],
                    "elapsed_seconds": time.time() - started,
                },
            )

    trials: dict[float, dict[str, tuple[float, ...]]] = {}
    for weight in static_setwise_weight_grid():
        candidate_id = _candidate_id(weight)
        trials[weight] = {
            "near_mrr": tuple(
                fold["candidates"][candidate_id]["delta"]["mrr"]
                for fold in fold_results["near"]
            ),
            "near_ndcg_at_10": tuple(
                fold["candidates"][candidate_id]["delta"]["ndcg_at_10"]
                for fold in fold_results["near"]
            ),
            "gapped_mrr": tuple(
                fold["candidates"][candidate_id]["delta"]["mrr"]
                for fold in fold_results["gapped"]
            ),
            "gapped_ndcg_at_10": tuple(
                fold["candidates"][candidate_id]["delta"]["ndcg_at_10"]
                for fold in fold_results["gapped"]
            ),
        }
    selection = select_dual_horizon_static_weight(trials)
    hashes_unchanged = (
        _sha256(features_path) == source_hashes["features"]
        and _sha256(times_path) == source_hashes["times"]
    )
    if not hashes_unchanged:
        raise RuntimeError("K256 source cache changed during head training")
    selection_report = {
        "status": selection["status"],
        "internal_gate_passed": selection["selected_weight"] is not None,
        "external_authorized": selection["selected_weight"] is not None,
        "external_labels_read": False,
        "selected_weight": selection["selected_weight"],
        "selection": selection,
        "folds": fold_results,
        "plan_sha256": plan_hash,
        "cache_report_sha256": _sha256(args.cache_report),
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "source_hashes_unchanged": hashes_unchanged,
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.output_dir / "selection-report.json"
    _write_json(report_path, selection_report)
    report_hash = _sha256(report_path)
    (args.output_dir / "selection-report.sha256").write_text(
        f"{report_hash}  {report_path.name}\n",
        encoding="ascii",
    )
    if selection["selected_weight"] is not None:
        lock = {
            "status": "selected",
            "selected_weight": selection["selected_weight"],
            "candidate_id": _candidate_id(selection["selected_weight"]),
            "selection_report_sha256": report_hash,
            "plan_sha256": plan_hash,
            "external_labels_read": False,
        }
        lock_path = args.output_dir / "selection-lock.json"
        _write_json(lock_path, lock)
        (args.output_dir / "selection-lock.sha256").write_text(
            f"{_sha256(lock_path)}  {lock_path.name}\n",
            encoding="ascii",
        )
    _write_json(
        args.output_dir / "progress.json",
        {
            "status": selection["status"],
            "selected_weight": selection["selected_weight"],
            "external_authorized": selection["selected_weight"] is not None,
            "elapsed_seconds": selection_report["elapsed_seconds"],
        },
    )
    print(json.dumps(selection_report, ensure_ascii=False, indent=2))
    return 0 if selection["selected_weight"] is not None else 2


def _train_and_score_fold(
    *,
    features: Any,
    times: np.ndarray,
    fold: dict[str, Any],
    source_state: dict[str, Any],
    protocol: Any,
    control_config: dict[str, Any],
    global_time_bounds: tuple[int, int],
    output_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    fold_id = str(fold["fold_id"])
    train_start, train_stop = (int(value) for value in fold["train_rows"])
    score_start, score_stop = (int(value) for value in fold["score_rows"])
    tune_start = train_stop - TUNE_ROWS
    source_config = source_state["config"]
    maximum_fit_rows = int(getattr(source_config, "max_train_events", 0))
    fit_start = train_start
    if maximum_fit_rows > 0:
        fit_start = max(fit_start, tune_start - maximum_fit_rows)
    base_train = features[fit_start:tune_start]
    setwise_train_features = features[train_start:tune_start]
    tune = features[tune_start:train_stop]
    score = features[score_start:score_stop]
    feature_indices = tuple(
        int(index) for index in source_state["fusion_result"].feature_indices
    )
    seed = int(control_config["seed"])
    fold_dir = output_dir / fold_id
    fold_dir.mkdir()

    base_model, base_result = fit_fusion_mlp_streaming(
        base_train,
        tune,
        _fusion_config(control_config),
        np.random.default_rng(seed),
        verbose=True,
        feature_indices=feature_indices,
        candidate_name=f"{fold_id}_base_v0",
    )
    base_logits = _predict_mlp_streaming(
        base_model,
        score,
        base_result,
        context_transform_version=0,
        batch_size=int(control_config["batch_size"]),
    )
    base_head = fold_dir / "base-head.npz"
    _save_head(
        base_head,
        base_result,
        hidden_dim=int(control_config["hidden_dim"]),
        context_transform_version=0,
        fit_rows=(fit_start, tune_start),
        tune_rows=(tune_start, train_stop),
    )
    del base_model
    release_memory()

    lgbm = fit_fusion_lgbm(
        base_train,
        tune,
        selection_metric=str(control_config["selection_metric"]),
        verbose=True,
        feature_indices=feature_indices,
        candidate_name=f"{fold_id}_shared_lgbm",
    )
    lgbm_logits = predict_logits_lgbm(
        lgbm.model_text,
        np.asarray(score[..., feature_indices]),
    )
    lgbm_path = fold_dir / "lgbm-model.txt"
    lgbm_path.write_text(lgbm.model_text, encoding="utf-8")
    del lgbm
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
    setwise_tune = SetwiseFeatureView(tune)
    setwise_model, setwise_result, history = (
        fit_fusion_mlp_listwise_streaming(
            setwise_train,
            setwise_tune,
            setwise_config,
            np.random.default_rng(seed),
            verbose=True,
            feature_indices=tuple(range(setwise_train.shape[-1])),
            candidate_name=f"{fold_id}_setwise_context",
        )
    )
    setwise_logits = _predict_setwise_streaming(
        setwise_model,
        score,
        setwise_result,
        batch_size=SETWISE_BATCH_SIZE,
    )
    setwise_head = fold_dir / "setwise-head.npz"
    _save_head(
        setwise_head,
        setwise_result,
        hidden_dim=SETWISE_HIDDEN_DIM,
        context_transform_version=1,
        fit_rows=(train_start, tune_start),
        tune_rows=(tune_start, train_stop),
    )
    del setwise_model
    release_memory()

    backbone = blend_expert_logits(
        base_logits,
        lgbm_logits,
        protocol.mlp_weight,
        calibration=protocol.expert_calibration,
    )
    setwise_scores = _softmax(setwise_logits)
    score_times = np.asarray(times[score_start:score_stop], dtype=np.int64)
    baseline = apply_time_ramp(
        backbone,
        setwise_scores,
        score_times,
        power=protocol.time_ramp_power,
        minimum_time=float(global_time_bounds[0]),
        maximum_time=float(global_time_bounds[1]),
    )
    baseline_metrics = ranking_metrics(baseline)
    candidates: dict[str, Any] = {}
    for weight in static_setwise_weight_grid():
        candidate = blend_static_setwise(
            backbone,
            setwise_scores,
            weight=weight,
        )
        metrics = ranking_metrics(candidate, baseline_scores=baseline)
        candidates[_candidate_id(weight)] = {
            "weight": weight,
            "metrics": metrics,
            "delta": _metric_delta(metrics, baseline_metrics),
        }
    score_path = fold_dir / "shared-scores.npz"
    np.savez(
        score_path,
        backbone=np.asarray(backbone, dtype=np.float32),
        setwise=np.asarray(setwise_scores, dtype=np.float32),
        time_ramp_baseline=np.asarray(baseline, dtype=np.float32),
        query_times=score_times,
    )
    result = {
        "fold_id": fold_id,
        "train_rows": [train_start, train_stop],
        "base_fit_rows": [fit_start, tune_start],
        "setwise_fit_rows": [train_start, tune_start],
        "tune_rows": [tune_start, train_stop],
        "score_rows": [score_start, score_stop],
        "train_time_max": int(fold["train_time_max"]),
        "score_time_min": int(fold["score_time_min"]),
        "score_time_max": int(fold["score_time_max"]),
        "minimum_gap_seconds": fold.get("minimum_gap_seconds"),
        "actual_gap_seconds": fold.get("actual_gap_seconds"),
        "baseline": baseline_metrics,
        "candidates": candidates,
        "artifacts": {
            "base_head_sha256": _sha256(base_head),
            "lgbm_sha256": _sha256(lgbm_path),
            "setwise_head_sha256": _sha256(setwise_head),
            "shared_scores_sha256": _sha256(score_path),
        },
        "setwise_history": list(history),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(fold_dir / "fold-report.json", result)
    del (
        base_logits,
        lgbm_logits,
        setwise_logits,
        backbone,
        setwise_scores,
        baseline,
    )
    gc.collect()
    return result


def _validate_inputs(
    *,
    plan: dict[str, Any],
    plan_hash: str,
    report: dict[str, Any],
    checkpoint: Path,
    cache_prefix: Path,
    reference_cache_prefix: Path,
) -> None:
    if len(plan_hash) != 64:
        raise ValueError("plan SHA-256 is malformed")
    if (
        plan.get("status") != "frozen_before_training_or_metric_read"
        or plan.get("external_labels_read") is not False
        or len(plan.get("candidate_space", [])) != 16
        or len(plan.get("near_folds", [])) != 3
        or len(plan.get("gapped_folds", [])) != 3
    ):
        raise ValueError("static Setwise plan differs from preregistration")
    if _sha256(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("source checkpoint differs from plan")
    if (
        report.get("status") != "complete"
        or report.get("dataset_name") != "dataset1"
        or report.get("requested_train_rows") != 200_000
        or report.get("prediction_limits")
        != plan["prediction_limits"]
    ):
        raise ValueError("K256 cache report differs from plan")
    features = Path(f"{cache_prefix}.train.npy")
    times = Path(f"{cache_prefix}.train-time.npy")
    candidates = Path(f"{cache_prefix}.train-candidates.npy")
    reference_times = Path(f"{reference_cache_prefix}.train-time.npy")
    reference_candidates = Path(
        f"{reference_cache_prefix}.train-candidates.npy"
    )
    for path in (
        features,
        times,
        candidates,
        reference_times,
        reference_candidates,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if _sha256(times) != plan["cache_times_sha256"]:
        raise ValueError("K256 time sidecar differs from frozen plan")
    if _sha256(times) != _sha256(reference_times):
        raise ValueError("K256 training rows differ from reference cache")
    if _sha256(candidates) != _sha256(reference_candidates):
        raise ValueError("K256 candidate matrix differs from frozen reference")
    values = np.load(features, mmap_mode="r", allow_pickle=False)
    if values.shape != (200_000, 100, 63):
        raise ValueError("K256 feature tensor shape differs")


def _metric_delta(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    return {
        name: float(candidate[name]) - float(baseline[name])
        for name in (
            "mrr",
            "hit_at_1",
            "hit_at_3",
            "hit_at_10",
            "ndcg_at_10",
            "mean_rank",
        )
    }


def _candidate_id(weight: float) -> str:
    return f"static_setwise_w{round(weight * 100):03d}"


def _require_sidecar(path: Path, sidecar: Path) -> str:
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError("plan hash differs from sidecar")
    return actual


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
