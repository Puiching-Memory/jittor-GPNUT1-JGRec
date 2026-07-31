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
from jgrec.feature_mask_validation import (
    FeatureMaskCandidate,
    build_feature_mask_candidates,
)
from jgrec.rankers.hybrid.base_context_gate import (
    BASE_CONTEXT_INTEGRATION_ID,
    BaseContextBlendProtocol,
    compose_dataset1_final_scores,
)
from jgrec.rankers.hybrid.candidate_prior import (
    CANDIDATE_PRIOR_FEATURE_NAMES,
)
from jgrec.rankers.hybrid.config import (
    GRAPH_WINDOW_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SOURCE_PROFILE_FEATURE_NAMES,
    TARGET_WINDOW_FEATURE_NAMES,
    TWO_TOWER_FEATURE_NAMES,
)
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
from jgrec.rankers.hybrid.stats import STAT_FEATURE_NAMES
from jgrec.rankers.hybrid.structure import STRUCTURE_FEATURE_NAMES
from jgrec.robust_weight_selection import ranking_metrics
from jgrec.standard_validation_protocol import (
    freeze_standard_validation_plan,
    select_standard_rolling_candidate,
)
from train_select_dataset1_base_context_rolling import (
    SETWISE_BATCH_SIZE,
    SETWISE_EPOCHS,
    SETWISE_HIDDEN_DIM,
    SETWISE_LEARNING_RATE,
    SETWISE_PATIENCE,
    TUNE_ROWS,
    _artifact_descriptor,
    _frozen_protocol,
    _fusion_config,
    _predict_mlp_streaming,
    _predict_setwise_streaming,
    _read_json,
    _save_head,
    _sha256,
    _validate_source_manifest,
    _write_json,
)

EXPERIMENT_ID = "dataset1_feature_mask_loo_context_v1_g050_20260728"
INTEGRATION_ID = (
    f"{BASE_CONTEXT_INTEGRATION_ID}_feature_mask_leave_one_out_v1"
)
SELECTION_FOLDS = 3
DATASET1_EXTERNAL_TIME_SHA256 = (
    "2f5a68329f5e75f26acd77bf22e578bb47f51261aaf3c30ee4dca7aefbd89f96"
)
DATASET1_EXTERNAL_HORIZON_SECONDS = 129_594_998 - 121_216_367
FEATURE_GROUPS = {
    "stats": STAT_FEATURE_NAMES,
    "prior": CANDIDATE_PRIOR_FEATURE_NAMES,
    "target": TARGET_WINDOW_FEATURE_NAMES,
    "structure": STRUCTURE_FEATURE_NAMES,
    "profile": SOURCE_PROFILE_FEATURE_NAMES,
    "tower": TWO_TOWER_FEATURE_NAMES,
    "gnn": GRAPH_WINDOW_NAMES,
    "seq": SEQUENCE_FEATURE_NAMES,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen Dataset1 full/leave-one-feature-group-out "
            "candidate family on three exact integrated rolling folds."
        )
    )
    parser.add_argument("--rolling-manifest", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    rolling = _read_json(args.rolling_manifest)
    _validate_source_manifest(rolling)
    source_state = load_checkpoint_dataset(
        args.source_checkpoint,
        "dataset1",
    )
    source_checkpoint_sha256 = _sha256(args.source_checkpoint)
    rolling_manifest_sha256 = _sha256(args.rolling_manifest)
    protocol, control_training, candidate_training = _frozen_protocol(
        source_state
    )
    feature_names = tuple(
        str(name) for name in source_state["feature_names"]
    )
    cache_report = _read_json(
        Path(rolling["source"]["train_cache_report"])
    )
    if tuple(cache_report["feature_names"]) != feature_names:
        raise ValueError("checkpoint and rolling cache feature schemas differ")

    shared_config = _shared_candidate_config(
        candidate_training=candidate_training,
        protocol=protocol,
        source_checkpoint_sha256=source_checkpoint_sha256,
        rolling_manifest_sha256=rolling_manifest_sha256,
    )
    candidates = build_feature_mask_candidates(
        feature_names=feature_names,
        feature_groups=FEATURE_GROUPS,
        shared_config=shared_config,
    )

    args.output_dir.mkdir(parents=True)
    candidate_config_path = args.output_dir / "candidate-configs.json"
    _write_json(
        candidate_config_path,
        {
            "status": "frozen_before_metrics",
            "experiment_id": EXPERIMENT_ID,
            "integration_id": INTEGRATION_ID,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "removed_group": candidate.removed_group,
                    "feature_indices": list(candidate.feature_indices),
                    "config": candidate.config,
                    "config_sha256": candidate.config_sha256,
                    "tie_break_priority": (
                        candidate.tie_break_priority
                    ),
                }
                for candidate in candidates
            ],
            "external_holdout_read": False,
        },
    )
    plan_path = args.output_dir / "validation-plan.json"
    _write_json(
        plan_path,
        _validation_plan(candidates),
    )
    plan_preflight = freeze_standard_validation_plan(
        plan_path=plan_path,
        output_dir=args.output_dir / "plan",
    )
    plan_lock_path = (
        args.output_dir / "plan" / "validation-plan-lock.json"
    )
    frozen = {
        "status": "frozen_before_metrics",
        "experiment_id": EXPERIMENT_ID,
        "integration_id": INTEGRATION_ID,
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "rolling_manifest": str(args.rolling_manifest.resolve()),
        "rolling_manifest_sha256": rolling_manifest_sha256,
        "candidate_configs": str(candidate_config_path.resolve()),
        "candidate_configs_sha256": _sha256(candidate_config_path),
        "validation_plan": str(plan_path.resolve()),
        "validation_plan_sha256": _sha256(plan_path),
        "plan_lock": str(plan_lock_path.resolve()),
        "plan_lock_sha256": _sha256(plan_lock_path),
        "plan_preflight": plan_preflight,
        "selection_fold_count": SELECTION_FOLDS,
        "reserved_fold_metrics_read": False,
        "external_holdout_read": False,
        "package_authorized": False,
    }
    _write_json(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2), flush=True)

    feature_path = Path(rolling["source"]["features"])
    time_path = Path(rolling["source"]["times"])
    candidate_path = Path(cache_report["artifacts"]["candidates"]["path"])
    _require_hash(
        feature_path,
        rolling["source"]["features_sha256"],
        "rolling features",
    )
    _require_hash(
        time_path,
        rolling["source"]["times_sha256"],
        "rolling times",
    )
    _require_hash(
        candidate_path,
        cache_report["artifacts"]["candidates"]["sha256"],
        "rolling candidate ids",
    )
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    times = np.load(time_path, mmap_mode="r", allow_pickle=False)
    candidate_ids = np.load(
        candidate_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        features.shape != (200_000, 100, len(feature_names))
        or times.shape != (200_000,)
        or candidate_ids.shape != features.shape[:2]
    ):
        raise ValueError("rolling cache shape differs from frozen protocol")

    require_jittor_cuda(jt)
    started = time.time()
    fold_manifests: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    selection_payloads = [
        payload
        for payload in rolling["folds"]
        if payload["role"] == "selection"
    ]
    if len(selection_payloads) != SELECTION_FOLDS:
        raise ValueError("rolling manifest must have three selection folds")
    for fold_payload in selection_payloads:
        fold = _train_and_score_fold(
            features=features,
            times=times,
            candidate_ids=candidate_ids,
            fold_payload=fold_payload,
            source_state=source_state,
            protocol=protocol,
            control_training=control_training,
            candidate_training=candidate_training,
            candidates=candidates,
            output_dir=args.output_dir,
        )
        fold_manifests.append(fold["manifest"])
        fold_reports.append(fold["report"])
        _write_json(
            args.output_dir / "rolling-progress.json",
            {
                "status": "training_selection_folds",
                "completed_folds": len(fold_reports),
                "candidate_count": len(candidates),
                "reserved_fold_metrics_read": False,
                "external_holdout_read": False,
                "folds": fold_reports,
                "elapsed_seconds": time.time() - started,
            },
        )
        release_memory()

    rolling_score_manifest = {
        "schema_version": 1,
        "protocol": "standard_rolling_origin_scores_v1",
        "experiment_id": EXPERIMENT_ID,
        "integration_id": INTEGRATION_ID,
        "plan_lock_sha256": _sha256(plan_lock_path),
        "positive_candidate_column": 0,
        "folds": fold_manifests,
        "reserved_folds": [
            _reserved_fold(payload)
            for payload in rolling["folds"]
            if payload["role"] == "gate"
        ],
        "external_holdout_read": False,
    }
    score_manifest_path = args.output_dir / "rolling-manifest.json"
    _write_json(score_manifest_path, rolling_score_manifest)
    selection = select_standard_rolling_candidate(
        manifest_path=score_manifest_path,
        plan_lock_path=plan_lock_path,
        output_dir=args.output_dir / "selection",
    )
    result = {
        "status": selection["status"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "rolling_pass": selection["status"] == "selected",
        "reserved_fold_metrics_read": False,
        "external_authorized": selection["status"] == "selected",
        "external_holdout_read": False,
        "package_authorized": False,
        "selection_report_sha256": _sha256(
            args.output_dir
            / "selection"
            / "selection-report.json"
        ),
        "selection_lock_created": (
            args.output_dir
            / "selection"
            / "selection-lock.json"
        ).exists(),
        "folds": fold_reports,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "rolling-result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["rolling_pass"] else 2


def _train_and_score_fold(
    *,
    features: Any,
    times: np.ndarray,
    candidate_ids: np.ndarray,
    fold_payload: dict[str, Any],
    source_state: dict[str, Any],
    protocol: BaseContextBlendProtocol,
    control_training: dict[str, Any],
    candidate_training: dict[str, Any],
    candidates: tuple[FeatureMaskCandidate, ...],
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
    maximum_fit_rows = int(
        getattr(source_state["config"], "max_train_events", 0)
    )
    fit_start = train_start
    if maximum_fit_rows > 0:
        fit_start = max(fit_start, tune_start - maximum_fit_rows)
    base_train_features = features[fit_start:tune_start]
    setwise_train_features = features[train_start:tune_start]
    tune_features = features[tune_start:train_stop]
    score_features = features[score_start:score_stop]
    seed = int(control_training["seed"])

    baseline_indices = tuple(
        int(index)
        for index in source_state["fusion_result"].feature_indices
    )
    baseline_model, baseline_result = fit_fusion_mlp_streaming(
        base_train_features,
        tune_features,
        _fusion_config(control_training),
        np.random.default_rng(seed),
        verbose=True,
        feature_indices=baseline_indices,
        candidate_name=f"fold{fold_index}_current_champion",
    )
    baseline_mlp_logits = _predict_mlp_streaming(
        baseline_model,
        score_features,
        baseline_result,
        context_transform_version=0,
        batch_size=int(control_training["batch_size"]),
    )
    _save_head(
        output_dir / f"fold-{fold_index:02d}-baseline-mlp.npz",
        baseline_result,
        hidden_dim=int(control_training["hidden_dim"]),
        context_transform_version=0,
        fit_rows=(fit_start, tune_start),
        tune_rows=(tune_start, train_stop),
    )
    del baseline_model
    release_memory()

    lgbm_indices = tuple(
        int(index)
        for index in source_state["lgbm_result"].feature_indices
    )
    shared_lgbm = fit_fusion_lgbm(
        base_train_features,
        tune_features,
        selection_metric=str(control_training["selection_metric"]),
        verbose=True,
        feature_indices=lgbm_indices,
        candidate_name=f"fold{fold_index}_shared_lgbm",
    )
    lgbm_logits = predict_logits_lgbm(
        shared_lgbm.model_text,
        np.asarray(score_features[..., lgbm_indices]),
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
    (
        setwise_model,
        setwise_result,
        _setwise_losses,
    ) = fit_fusion_mlp_listwise_streaming(
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
    del setwise_model, setwise_train, setwise_tune
    release_memory()

    score_times = np.asarray(
        times[score_start:score_stop],
        dtype=np.int64,
    )
    fingerprint = _candidate_fingerprint(
        candidate_ids[score_start:score_stop],
    )
    fold_dir = output_dir / "scores" / f"fold-{fold_index:02d}"
    fold_dir.mkdir(parents=True)
    baseline_path = fold_dir / "baseline-current-champion.npy"
    candidate_descriptors: dict[str, dict[str, Any]] = {}
    candidate_metrics: dict[str, dict[str, Any]] = {}
    baseline_metrics: dict[str, Any] | None = None

    for candidate_index, candidate in enumerate(candidates):
        candidate_model, candidate_result = fit_fusion_mlp_streaming(
            base_train_features,
            tune_features,
            _fusion_config(candidate_training),
            np.random.default_rng(seed),
            verbose=True,
            feature_indices=candidate.feature_indices,
            candidate_name=(
                f"fold{fold_index}_{candidate.candidate_id}"
            ),
        )
        candidate_mlp_logits = _predict_mlp_streaming(
            candidate_model,
            score_features,
            candidate_result,
            context_transform_version=1,
            batch_size=int(candidate_training["batch_size"]),
        )
        _save_head(
            output_dir
            / (
                f"fold-{fold_index:02d}-"
                f"{candidate.candidate_id}-mlp.npz"
            ),
            candidate_result,
            hidden_dim=int(candidate_training["hidden_dim"]),
            context_transform_version=1,
            fit_rows=(fit_start, tune_start),
            tune_rows=(tune_start, train_stop),
        )
        comparison = compose_dataset1_final_scores(
            control_mlp_logits=baseline_mlp_logits,
            candidate_mlp_logits=candidate_mlp_logits,
            shared_lgbm_logits=lgbm_logits,
            shared_setwise_logits=setwise_logits,
            query_times=score_times,
            protocol=protocol,
            minimum_time=float(score_times.min()),
            maximum_time=float(score_times.max()),
        )
        if baseline_metrics is None:
            np.save(
                baseline_path,
                np.asarray(comparison.control, dtype=np.float32),
                allow_pickle=False,
            )
            baseline_metrics = ranking_metrics(comparison.control)
        candidate_score_path = (
            fold_dir / f"{candidate.candidate_id}.npy"
        )
        np.save(
            candidate_score_path,
            np.asarray(comparison.candidate, dtype=np.float32),
            allow_pickle=False,
        )
        candidate_descriptors[candidate.candidate_id] = {
            **_artifact_descriptor(
                candidate_score_path,
                output_dir=output_dir,
            ),
            "candidate_id": candidate.candidate_id,
            "config_sha256": candidate.config_sha256,
            "candidate_fingerprint": fingerprint,
        }
        candidate_metrics[candidate.candidate_id] = ranking_metrics(
            comparison.candidate,
            baseline_scores=comparison.control,
        )
        _write_json(
            output_dir
            / f"fold-{fold_index:02d}-candidate-progress.json",
            {
                "fold": fold_index,
                "completed_candidates": candidate_index + 1,
                "candidate_count": len(candidates),
                "latest_candidate": candidate.candidate_id,
                "external_holdout_read": False,
            },
        )
        del (
            candidate_model,
            candidate_mlp_logits,
            candidate_result,
            comparison,
        )
        release_memory()

    if baseline_metrics is None:
        raise RuntimeError("feature-mask candidate set is empty")
    time_boundary = fold_payload["time_boundary"]
    report = {
        "fold_id": f"fold-{fold_index:02d}",
        "role": "selection",
        "fit_rows": [fit_start, tune_start],
        "setwise_fit_rows": [train_start, tune_start],
        "tune_rows": [tune_start, train_stop],
        "score_rows": [score_start, score_stop],
        "train_time_max": int(time_boundary["train_time_max"]),
        "score_time_min": int(time_boundary["score_time_min"]),
        "score_time_max": int(time_boundary["score_time_max"]),
        "baseline": baseline_metrics,
        "candidates": candidate_metrics,
        "elapsed_seconds": time.time() - started,
    }
    manifest = {
        "fold_id": report["fold_id"],
        "role": "selection",
        "train_time_max": report["train_time_max"],
        "score_time_min": report["score_time_min"],
        "score_time_max": report["score_time_max"],
        "candidate_fingerprint": fingerprint,
        "baseline": _artifact_descriptor(
            baseline_path,
            output_dir=output_dir,
        ),
        "candidates": candidate_descriptors,
    }
    del baseline_mlp_logits, lgbm_logits, setwise_logits
    gc.collect()
    return {"report": report, "manifest": manifest}


def _shared_candidate_config(
    *,
    candidate_training: dict[str, Any],
    protocol: BaseContextBlendProtocol,
    source_checkpoint_sha256: str,
    rolling_manifest_sha256: str,
) -> dict[str, Any]:
    training = {
        key: value
        for key, value in candidate_training.items()
        if key != "feature_indices"
    }
    return {
        "integration_id": INTEGRATION_ID,
        "context_transform_version": 1,
        "training": training,
        "final_integration": {
            "mlp_weight": protocol.mlp_weight,
            "expert_calibration": asdict(protocol.expert_calibration),
            "time_ramp_power": protocol.time_ramp_power,
        },
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "rolling_manifest_sha256": rolling_manifest_sha256,
    }


def _validation_plan(
    candidates: tuple[FeatureMaskCandidate, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "standard_validation_plan_v1",
        "experiment_id": EXPERIMENT_ID,
        "candidate_family": (
            "dataset1_exact_integrated_feature_mask_leave_one_out"
        ),
        "candidate_space": [
            {
                "candidate_id": candidate.candidate_id,
                "config_sha256": candidate.config_sha256,
                "tie_break_priority": candidate.tie_break_priority,
            }
            for candidate in candidates
        ],
        "rolling_selection": {
            "minimum_folds": SELECTION_FOLDS,
            "reserved_gate_folds": 1,
            "aggregation": "equal_weight_fold_mean",
            "per_fold_minimum_deltas": {
                "mrr": 0.0,
                "ndcg_at_10": 0.0,
            },
            "mean_minimum_deltas": {
                "mrr": 0.0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_10": 0.0,
                "ndcg_at_10": 0.0,
            },
            "mean_maximum_deltas": {"mean_rank": 0.0},
            "minimum_improved_minus_worsened": 1,
            "selection_order": [
                "maximum_mean_fold_mrr_delta",
                "maximum_worst_fold_mrr_delta",
                "maximum_mean_fold_ndcg_at_10_delta",
                "minimum_tie_break_priority",
            ],
        },
        "external_gate": {
            "holdout_id": "dataset1_external_20k_v1",
            "lineage_sha256": DATASET1_EXTERNAL_TIME_SHA256,
            "deployment_horizon_seconds": (
                DATASET1_EXTERNAL_HORIZON_SECONDS
            ),
            "minimum_horizon_seconds": (
                DATASET1_EXTERNAL_HORIZON_SECONDS
            ),
            "minimum_start_gap_seconds": 63,
            "strictly_increasing_metrics": ["mrr"],
            "minimum_deltas": {
                "mrr": 0.0,
                "hit_at_1": 0.0,
                "hit_at_3": 0.0,
                "hit_at_10": 0.0,
                "ndcg_at_10": 0.0,
            },
            "maximum_deltas": {"mean_rank": 0.0},
            "minimum_improved_minus_worsened": 1,
        },
    }


def _reserved_fold(payload: dict[str, Any]) -> dict[str, Any]:
    boundary = payload["time_boundary"]
    return {
        "fold_id": f"fold-{int(payload['index']):02d}",
        "role": "gate",
        "train_time_max": int(boundary["train_time_max"]),
        "score_time_min": int(boundary["score_time_min"]),
        "score_time_max": int(boundary["score_time_max"]),
    }


def _candidate_fingerprint(candidate_ids: np.ndarray) -> str:
    values = np.ascontiguousarray(candidate_ids)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
