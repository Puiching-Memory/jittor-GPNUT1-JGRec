from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import jittor as jt
import lightgbm as lgb
import numpy as np

from jgrec import cooccur_lift_successor_execution as execution_module
from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.cooccur_lift_automatic_pipeline import (
    build_duel_manifest_bindings,
)
from jgrec.cooccur_lift_successor_execution import (
    build_deterministic_replay_report,
    resolve_bugfixed_v1_fold_baseline,
    validate_successor_execution_contract,
)
from jgrec.rankers.hybrid import fusion as fusion_module
from jgrec.rankers.hybrid.cooccur_lift import (
    FROZEN_FOLDS,
    CooccurLiftAugmentedView,
)
from jgrec.rankers.hybrid.cooccur_lift_successor import (
    ConcatenatedFeatureView,
    CooccurLiftFullOnlyView,
    CooccurLiftGapAwareView,
)
from jgrec.rankers.hybrid.fusion import (
    FusionConfig,
    FusionResult,
    fit_fusion_mlp_listwise_fixed,
    predict_logits,
)
from jgrec.rankers.hybrid.setwise import SetwiseFeatureView

BASE_SEED = 60
SEED_SALT = 30_013
SEED_STRIDE = 1_009
SETWISE_WEIGHT = 0.80
V1_WEIGHT = 0.50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 4
FULL_ONLY_ID = "cooccur_lift_full_only_v2"
GAP_AWARE_ID = "cooccur_lift_gap_aware_v2"


class _ColumnOverlay:
    def __init__(self, source: Any, replacement: np.ndarray, column: int) -> None:
        if tuple(replacement.shape) != tuple(source.shape[:2]):
            raise ValueError("column overlay must match source rows/candidates")
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


class _ZeroShortLiftView:
    def __init__(self, source: Any) -> None:
        if len(source.shape) != 3 or int(source.shape[-1]) != 2:
            raise ValueError("lift source must have two channels")
        self._source = source
        self.shape = tuple(int(value) for value in source.shape)
        self.ndim = 3
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, key: Any) -> np.ndarray:
        values = np.array(self._source[key], dtype=np.float32, copy=True)
        values[..., 1] = 0.0
        return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen near+gapped cooccur-lift successor duel and "
            "write one standard validation manifest without opening external."
        )
    )
    parser.add_argument("--v1-checkpoint", required=True, type=Path)
    parser.add_argument("--validation-plan", required=True, type=Path)
    parser.add_argument("--plan-lock", required=True, type=Path)
    parser.add_argument("--execution-contract", required=True, type=Path)
    parser.add_argument("--bugfixed-v1-contract", required=True, type=Path)
    parser.add_argument("--full-only-config", required=True, type=Path)
    parser.add_argument("--gap-aware-config", required=True, type=Path)
    parser.add_argument("--near-cache-prefix", required=True, type=Path)
    parser.add_argument("--near-cache-report", required=True, type=Path)
    parser.add_argument("--near-short-none", required=True, type=Path)
    parser.add_argument("--near-lift", required=True, type=Path)
    parser.add_argument("--near-v1-manifest", required=True, type=Path)
    parser.add_argument("--gapped-cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    started = time.time()

    plan = _read_json(args.validation_plan)
    plan_lock = _read_json(args.plan_lock)
    execution_contract = _read_json(args.execution_contract)
    bugfixed_v1_contract = _read_json(args.bugfixed_v1_contract)
    execution = validate_successor_execution_contract(execution_contract)
    full_config = _read_json(args.full_only_config)
    gap_config = _read_json(args.gap_aware_config)
    candidate_configs = {
        FULL_ONLY_ID: {
            "path": args.full_only_config,
            "sha256": _sha256(args.full_only_config),
            "payload": full_config,
        },
        GAP_AWARE_ID: {
            "path": args.gap_aware_config,
            "sha256": _sha256(args.gap_aware_config),
            "payload": gap_config,
        },
    }
    _validate_frozen_inputs(
        checkpoint=args.v1_checkpoint,
        validation_plan=args.validation_plan,
        plan=plan,
        plan_lock_path=args.plan_lock,
        plan_lock=plan_lock,
        execution_contract_path=args.execution_contract,
        execution_contract=execution_contract,
        bugfixed_v1_contract_path=args.bugfixed_v1_contract,
        bugfixed_v1_contract=bugfixed_v1_contract,
        near_v1_manifest_path=args.near_v1_manifest,
        near_cache_report_path=args.near_cache_report,
        candidate_configs=candidate_configs,
    )
    args.output_dir.mkdir(parents=True)

    state = load_checkpoint_dataset(args.v1_checkpoint, "dataset2")
    feature_names = tuple(str(name) for name in state["feature_names"])
    if len(feature_names) != 63:
        raise ValueError("v1 checkpoint base schema must have 63 features")
    gnn_column = feature_names.index("gnn_short")
    setwise_result = state.get("setwise_fusion_result")
    lgbm_result = state.get("lgbm_result")
    if setwise_result is None or lgbm_result is None:
        raise ValueError("v1 checkpoint lacks the frozen prior champion experts")
    if abs(float(lgbm_result.mlp_weight) - SETWISE_WEIGHT) > 1e-12:
        raise ValueError("prior champion Setwise weight differs from 0.80")
    setwise_hidden_dim = int(state["setwise_hidden_dim"])
    setwise_indices = tuple(int(value) for value in setwise_result.feature_indices)
    lgbm_indices = tuple(int(value) for value in lgbm_result.feature_indices)
    booster = lgb.Booster(model_str=str(lgbm_result.model_text))
    del state
    gc.collect()

    jt.flags.use_cuda = 0
    near_entries, near_reports = _train_near_folds(
        cache_prefix=args.near_cache_prefix,
        short_none_path=args.near_short_none,
        lift_path=args.near_lift,
        source_manifest_path=args.near_v1_manifest,
        output_dir=args.output_dir / "near",
        gnn_column=gnn_column,
        setwise_hidden_dim=setwise_hidden_dim,
        candidate_configs=candidate_configs,
        replay_rtol=float(execution["rtol"]),
        replay_atol=float(execution["atol"]),
    )
    gapped_entries, gapped_reports = _train_gapped_folds(
        cache_dir=args.gapped_cache_dir,
        fold_specs=plan["far_horizon_validation"]["gapped_fold_specs"],
        output_dir=args.output_dir / "gapped",
        gnn_column=gnn_column,
        setwise_hidden_dim=setwise_hidden_dim,
        setwise_indices=setwise_indices,
        lgbm_indices=lgbm_indices,
        booster=booster,
        candidate_configs=candidate_configs,
        collapsed_fraction=float(
            plan["far_horizon_validation"][
                "deployment_collapsed_fraction"
            ]
        ),
        replay_rtol=float(execution["rtol"]),
        replay_atol=float(execution["atol"]),
        expected_cache_report_sha256=str(
            execution_contract["gapped_cache_report_sha256"]
        ),
    )

    last_near_time = int(near_entries[-1]["score_time_max"])
    manifest = {
        "schema_version": 1,
        "protocol": "standard_rolling_origin_scores_v1",
        "experiment_id": plan["experiment_id"],
        "plan_lock_sha256": _sha256(args.plan_lock),
        **build_duel_manifest_bindings(plan_lock),
        "baseline_id": plan["baseline"]["baseline_id"],
        "baseline_checkpoint_sha256": plan["baseline"]["checkpoint_sha256"],
        "baseline_execution_id": execution_contract[
            "baseline_execution"
        ]["fold_baseline_id"],
        "baseline_execution_contract_sha256": _sha256(
            args.execution_contract
        ),
        "positive_candidate_column": 0,
        "folds": near_entries,
        "reserved_folds": [
            {
                "fold_id": "reserved-gate-metadata",
                "role": "gate",
                "train_time_max": last_near_time,
                "score_time_min": last_near_time + 1,
                "score_time_max": last_near_time + 2,
                "metrics_read": False,
            }
        ],
        "gapped_folds": gapped_entries,
        "external_scores_read": False,
    }
    manifest_path = args.output_dir / "rolling-manifest.json"
    _write_json(manifest_path, manifest)
    report = {
        "status": "complete",
        "protocol": "cooccur_lift_successor_v2_duel_training_v2",
        "validation_plan": str(args.validation_plan.resolve()),
        "validation_plan_sha256": _sha256(args.validation_plan),
        "plan_lock": str(args.plan_lock.resolve()),
        "plan_lock_sha256": _sha256(args.plan_lock),
        "execution_contract": str(args.execution_contract.resolve()),
        "execution_contract_sha256": _sha256(args.execution_contract),
        "bugfixed_v1_contract": str(args.bugfixed_v1_contract.resolve()),
        "bugfixed_v1_contract_sha256": _sha256(args.bugfixed_v1_contract),
        "training_device": execution["training_device"],
        "internal_scoring_device": execution["internal_scoring_device"],
        "deterministic_replay_runs": execution["runs"],
        "deterministic_replay_rtol": execution["rtol"],
        "deterministic_replay_atol": execution["atol"],
        "historical_near_v1_manifest_role": execution["legacy_role"],
        "baseline_execution": execution_contract["baseline_execution"],
        "v1_checkpoint": str(args.v1_checkpoint.resolve()),
        "v1_checkpoint_sha256": _sha256(args.v1_checkpoint),
        "candidate_configs": {
            candidate_id: {
                "path": str(value["path"].resolve()),
                "sha256": value["sha256"],
            }
            for candidate_id, value in candidate_configs.items()
        },
        "near_folds": near_reports,
        "gapped_folds": gapped_reports,
        "rolling_manifest": str(manifest_path.resolve()),
        "rolling_manifest_sha256": _sha256(manifest_path),
        "external_scores_read": False,
        "external_authorized": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "training-report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _train_near_folds(
    *,
    cache_prefix: Path,
    short_none_path: Path,
    lift_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    gnn_column: int,
    setwise_hidden_dim: int,
    candidate_configs: dict[str, dict[str, Any]],
    replay_rtol: float,
    replay_atol: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = np.load(
        Path(f"{cache_prefix}.train.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    candidates = np.load(
        Path(f"{cache_prefix}.train-candidates.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    event_time = np.load(
        Path(f"{cache_prefix}.train-time.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    short_none = np.load(short_none_path, mmap_mode="r", allow_pickle=False)
    lift = np.load(lift_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (200_000, 100, 63):
        raise ValueError("near base feature cache shape drifted")
    if (
        candidates.shape != (200_000, 100)
        or short_none.shape != (200_000, 100)
        or lift.shape != (200_000, 100, 2)
    ):
        raise ValueError("near cache sidecars do not share the frozen shape")
    source_manifest = _read_json(source_manifest_path)
    source_folds = source_manifest["folds"]
    if len(source_folds) != len(FROZEN_FOLDS):
        raise ValueError("near v1 manifest must contain three folds")

    entries: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for fold_index, (
        (fold_id, train_rows, score_rows),
        source_fold,
    ) in enumerate(zip(FROZEN_FOLDS, source_folds, strict=True)):
        fold_started = time.time()
        train_start, train_stop = train_rows
        score_start, score_stop = score_rows
        score_candidates = candidates[score_start:score_stop]
        prior_baseline = _load_source_fold_scores(
            source_fold,
            descriptor=source_fold["baseline"],
            candidate_ids=score_candidates,
        )
        legacy_cuda_v1 = _load_source_fold_scores(
            source_fold,
            descriptor=source_fold["candidates"]["0.5"],
            candidate_ids=score_candidates,
        )
        train_v1 = SetwiseFeatureView(
            CooccurLiftAugmentedView(
                features[train_start:train_stop],
                short_none_scores=short_none[train_start:train_stop],
                gnn_short_column=gnn_column,
                lift_features=lift[train_start:train_stop],
            ),
            transform_version=1,
        )
        score_v1_raw = CooccurLiftAugmentedView(
            features[score_start:score_stop],
            short_none_scores=short_none[score_start:score_stop],
            gnn_short_column=gnn_column,
            lift_features=lift[score_start:score_stop],
        )
        score_v1 = SetwiseFeatureView(score_v1_raw, transform_version=1)
        zero_score_v1 = SetwiseFeatureView(
            CooccurLiftAugmentedView(
                features[score_start:score_stop],
                short_none_scores=short_none[score_start:score_stop],
                gnn_short_column=gnn_column,
                lift_features=_ZeroShortLiftView(
                    lift[score_start:score_stop]
                ),
            ),
            transform_version=1,
        )
        v1_predictions, v1_losses, v1_replay = _fit_head_replayed(
            train_v1,
            {
                "score": score_v1,
                "zero_short": zero_score_v1,
            },
            hidden_dim=setwise_hidden_dim,
            seed=_candidate_seed(fold_index),
            feature_count=195,
            candidate_name=f"v1-replay-{fold_id}",
            replay_rtol=replay_rtol,
            replay_atol=replay_atol,
        )
        v1_evidence = resolve_bugfixed_v1_fold_baseline(
            prior_baseline=prior_baseline,
            cpu_auxiliary=v1_predictions["score"],
            legacy_cuda_v1=legacy_cuda_v1,
            replay=v1_replay,
            weight=V1_WEIGHT,
        )
        v1_baseline = np.asarray(
            v1_evidence["baseline"],
            dtype=np.float64,
        )
        v1_zero_aux = v1_predictions["zero_short"]
        zero_v1_baseline = (
            V1_WEIGHT * prior_baseline + V1_WEIGHT * v1_zero_aux
        )
        del train_v1, score_v1, score_v1_raw, zero_score_v1

        full_train = SetwiseFeatureView(
            CooccurLiftFullOnlyView(
                features[train_start:train_stop],
                short_none_scores=short_none[train_start:train_stop],
                gnn_short_column=gnn_column,
                lift_features=lift[train_start:train_stop],
            ),
            transform_version=1,
        )
        full_score = SetwiseFeatureView(
            CooccurLiftFullOnlyView(
                features[score_start:score_stop],
                short_none_scores=short_none[score_start:score_stop],
                gnn_short_column=gnn_column,
                lift_features=lift[score_start:score_stop],
            ),
            transform_version=1,
        )
        full_predictions, full_losses, full_replay = _fit_head_replayed(
            full_train,
            {"score": full_score},
            hidden_dim=setwise_hidden_dim,
            seed=_candidate_seed(fold_index),
            feature_count=192,
            candidate_name=f"{FULL_ONLY_ID}-{fold_id}",
            replay_rtol=replay_rtol,
            replay_atol=replay_atol,
        )
        full_aux = full_predictions["score"]
        full_candidate = V1_WEIGHT * v1_baseline + V1_WEIGHT * full_aux
        full_zero_candidate = (
            V1_WEIGHT * zero_v1_baseline + V1_WEIGHT * full_aux
        )
        del full_train, full_score

        near_support = np.ones(train_stop - train_start, dtype=np.float32)
        score_support = np.ones(score_stop - score_start, dtype=np.float32)
        gap_train = SetwiseFeatureView(
            CooccurLiftGapAwareView(
                features[train_start:train_stop],
                short_none_scores=short_none[train_start:train_stop],
                gnn_short_column=gnn_column,
                lift_features=lift[train_start:train_stop],
                short_window_supported=near_support,
            ),
            transform_version=1,
        )
        gap_score = SetwiseFeatureView(
            CooccurLiftGapAwareView(
                features[score_start:score_stop],
                short_none_scores=short_none[score_start:score_stop],
                gnn_short_column=gnn_column,
                lift_features=lift[score_start:score_stop],
                short_window_supported=score_support,
            ),
            transform_version=1,
        )
        gap_zero_score = SetwiseFeatureView(
            CooccurLiftGapAwareView(
                features[score_start:score_stop],
                short_none_scores=short_none[score_start:score_stop],
                gnn_short_column=gnn_column,
                lift_features=_ZeroShortLiftView(
                    lift[score_start:score_stop]
                ),
                short_window_supported=np.zeros(
                    score_stop - score_start,
                    dtype=np.float32,
                ),
            ),
            transform_version=1,
        )
        gap_predictions, gap_losses, gap_replay = _fit_head_replayed(
            gap_train,
            {
                "score": gap_score,
                "zero_short": gap_zero_score,
            },
            hidden_dim=setwise_hidden_dim,
            seed=_candidate_seed(fold_index),
            feature_count=198,
            candidate_name=f"{GAP_AWARE_ID}-{fold_id}",
            replay_rtol=replay_rtol,
            replay_atol=replay_atol,
        )
        gap_aux = gap_predictions["score"]
        gap_candidate = V1_WEIGHT * v1_baseline + V1_WEIGHT * gap_aux
        gap_zero_aux = gap_predictions["zero_short"]
        gap_zero_candidate = (
            V1_WEIGHT * zero_v1_baseline + V1_WEIGHT * gap_zero_aux
        )
        del gap_train, gap_score, gap_zero_score

        entry = _write_scored_fold(
            output_dir=output_dir / fold_id,
            fold_id=fold_id,
            role="selection",
            train_time_max=int(event_time[train_stop - 1]),
            score_time_min=int(event_time[score_start]),
            score_time_max=int(event_time[score_stop - 1]),
            candidate_ids=score_candidates,
            baseline=v1_baseline,
            candidate_scores={
                FULL_ONLY_ID: full_candidate,
                GAP_AWARE_ID: gap_candidate,
            },
            candidate_configs=candidate_configs,
            counterfactual={
                "zero_short": {
                    "baseline": zero_v1_baseline,
                    "candidates": {
                        FULL_ONLY_ID: full_zero_candidate,
                        GAP_AWARE_ID: gap_zero_candidate,
                    },
                }
            },
        )
        entry["train_rows"] = list(train_rows)
        entry["score_rows"] = list(score_rows)
        entries.append(entry)
        reports.append(
            {
                "fold_id": fold_id,
                "train_rows": list(train_rows),
                "score_rows": list(score_rows),
                "training_device": "cpu",
                "internal_scoring_device": "cpu",
                "v1_deterministic_replay": v1_replay,
                "v1_legacy_cuda_max_abs_error": v1_evidence[
                    "legacy_cuda_max_abs_error"
                ],
                "v1_legacy_cuda_role": v1_evidence["legacy_cuda_role"],
                "v1_losses": list(v1_losses),
                "full_only_deterministic_replay": full_replay,
                "full_only_losses": list(full_losses),
                "gap_aware_deterministic_replay": gap_replay,
                "gap_aware_losses": list(gap_losses),
                "gap_aware_support_values": [1],
                "elapsed_seconds": time.time() - fold_started,
            }
        )
        del (
            prior_baseline,
            legacy_cuda_v1,
            v1_baseline,
            v1_predictions,
            v1_evidence,
            v1_zero_aux,
            zero_v1_baseline,
            full_predictions,
            full_aux,
            full_candidate,
            full_zero_candidate,
            gap_aux,
            gap_predictions,
            gap_candidate,
            gap_zero_aux,
            gap_zero_candidate,
        )
        gc.collect()
    return entries, reports


def _train_gapped_folds(
    *,
    cache_dir: Path,
    fold_specs: list[dict[str, Any]],
    output_dir: Path,
    gnn_column: int,
    setwise_hidden_dim: int,
    setwise_indices: tuple[int, ...],
    lgbm_indices: tuple[int, ...],
    booster: lgb.Booster,
    candidate_configs: dict[str, dict[str, Any]],
    collapsed_fraction: float,
    replay_rtol: float,
    replay_atol: float,
    expected_cache_report_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache_report_path = cache_dir / "cache-report.json"
    if _sha256(cache_report_path) != expected_cache_report_sha256:
        raise ValueError("gapped cache report differs from execution contract")
    report = _read_json(cache_report_path)
    if report.get("status") != "complete":
        raise ValueError("gapped cache report is incomplete")
    features = np.load(
        cache_dir / "base-features.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    candidates = np.load(
        cache_dir / "candidates.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    short_none = np.load(
        cache_dir / "short-none.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    lift_train_near = np.load(
        cache_dir / "lift-train-near.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    lift_score_gapped = np.load(
        cache_dir / "lift-score-gapped.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if features.shape != (277_035, 100, 63):
        raise ValueError("gapped base cache shape differs from preregistration")
    if candidates.shape != (277_035, 100):
        raise ValueError("gapped candidate cache shape differs")
    if short_none.shape != (277_035, 100):
        raise ValueError("gapped short_none cache shape differs")
    if lift_train_near.shape != (159_804, 100, 2):
        raise ValueError("gapped near-train lift shape differs")
    if lift_score_gapped.shape != (117_231, 100, 2):
        raise ValueError("gapped score lift shape differs")

    entries: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(fold_specs):
        fold_started = time.time()
        fold_id = str(fold["fold_id"])
        train_start, train_stop = map(int, fold["cache_train_rows"])
        score_start, score_stop = map(int, fold["cache_score_rows"])
        score_lift_start = score_start - 159_804
        score_lift_stop = score_stop - 159_804
        score_candidates = candidates[score_start:score_stop]
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
        (
            setwise_predictions,
            setwise_losses,
            setwise_replay,
        ) = _fit_head_replayed(
            setwise_train,
            {"score": setwise_score},
            hidden_dim=setwise_hidden_dim,
            seed=BASE_SEED + fold_index * SEED_STRIDE,
            feature_indices=setwise_indices,
            candidate_name=f"prior-champion-{fold_id}",
            replay_rtol=replay_rtol,
            replay_atol=replay_atol,
        )
        setwise_probabilities = setwise_predictions["score"]
        del setwise_train, setwise_score
        lgbm_probabilities = _predict_lgbm_probabilities(
            booster,
            features[score_start:score_stop],
            feature_indices=lgbm_indices,
        )
        prior_baseline = (
            SETWISE_WEIGHT * setwise_probabilities
            + (1.0 - SETWISE_WEIGHT) * lgbm_probabilities
        )

        v1_train = SetwiseFeatureView(
            CooccurLiftAugmentedView(
                features[train_start:train_stop],
                short_none_scores=short_none[train_start:train_stop],
                gnn_short_column=gnn_column,
                lift_features=lift_train_near[train_start:train_stop],
            ),
            transform_version=1,
        )
        v1_score = SetwiseFeatureView(
            CooccurLiftAugmentedView(
                features[score_start:score_stop],
                short_none_scores=short_none[score_start:score_stop],
                gnn_short_column=gnn_column,
                lift_features=lift_score_gapped[
                    score_lift_start:score_lift_stop
                ],
            ),
            transform_version=1,
        )
        v1_predictions, v1_losses, v1_replay = _fit_head_replayed(
            v1_train,
            {"score": v1_score},
            hidden_dim=setwise_hidden_dim,
            seed=_candidate_seed(fold_index),
            feature_count=195,
            candidate_name=f"v1-baseline-{fold_id}",
            replay_rtol=replay_rtol,
            replay_atol=replay_atol,
        )
        v1_aux = v1_predictions["score"]
        v1_baseline = V1_WEIGHT * prior_baseline + V1_WEIGHT * v1_aux
        del v1_train, v1_score

        full_train = SetwiseFeatureView(
            CooccurLiftFullOnlyView(
                features[train_start:train_stop],
                short_none_scores=short_none[train_start:train_stop],
                gnn_short_column=gnn_column,
                lift_features=lift_train_near[train_start:train_stop],
            ),
            transform_version=1,
        )
        full_score = SetwiseFeatureView(
            CooccurLiftFullOnlyView(
                features[score_start:score_stop],
                short_none_scores=short_none[score_start:score_stop],
                gnn_short_column=gnn_column,
                lift_features=lift_score_gapped[
                    score_lift_start:score_lift_stop
                ],
            ),
            transform_version=1,
        )
        full_predictions, full_losses, full_replay = _fit_head_replayed(
            full_train,
            {"score": full_score},
            hidden_dim=setwise_hidden_dim,
            seed=_candidate_seed(fold_index),
            feature_count=192,
            candidate_name=f"{FULL_ONLY_ID}-{fold_id}",
            replay_rtol=replay_rtol,
            replay_atol=replay_atol,
        )
        full_aux = full_predictions["score"]
        full_candidate = V1_WEIGHT * v1_baseline + V1_WEIGHT * full_aux
        del full_train, full_score

        train_count = train_stop - train_start
        stale_lift = np.load(
            cache_dir / f"lift-train-stale-{fold_id}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if stale_lift.shape != (train_count, 100, 2):
            raise ValueError(f"{fold_id} stale lift shape differs")
        near_gap_raw = CooccurLiftGapAwareView(
            features[train_start:train_stop],
            short_none_scores=short_none[train_start:train_stop],
            gnn_short_column=gnn_column,
            lift_features=lift_train_near[train_start:train_stop],
            short_window_supported=np.ones(train_count, dtype=np.float32),
        )
        stale_gap_raw = CooccurLiftGapAwareView(
            features[train_start:train_stop],
            short_none_scores=short_none[train_start:train_stop],
            gnn_short_column=gnn_column,
            lift_features=stale_lift,
            short_window_supported=np.zeros(train_count, dtype=np.float32),
        )
        gap_train = SetwiseFeatureView(
            ConcatenatedFeatureView((near_gap_raw, stale_gap_raw)),
            transform_version=1,
        )
        gap_score = SetwiseFeatureView(
            CooccurLiftGapAwareView(
                features[score_start:score_stop],
                short_none_scores=short_none[score_start:score_stop],
                gnn_short_column=gnn_column,
                lift_features=lift_score_gapped[
                    score_lift_start:score_lift_stop
                ],
                short_window_supported=np.zeros(
                    score_stop - score_start,
                    dtype=np.float32,
                ),
            ),
            transform_version=1,
        )
        row_weights = np.concatenate(
            (
                np.full(
                    train_count,
                    1.0 - collapsed_fraction,
                    dtype=np.float32,
                ),
                np.full(
                    train_count,
                    collapsed_fraction,
                    dtype=np.float32,
                ),
            )
        )
        gap_predictions, gap_losses, gap_replay = _fit_head_replayed(
            gap_train,
            {"score": gap_score},
            hidden_dim=setwise_hidden_dim,
            seed=_candidate_seed(fold_index),
            feature_count=198,
            candidate_name=f"{GAP_AWARE_ID}-{fold_id}",
            train_row_weights=row_weights,
            replay_rtol=replay_rtol,
            replay_atol=replay_atol,
        )
        gap_aux = gap_predictions["score"]
        gap_candidate = V1_WEIGHT * v1_baseline + V1_WEIGHT * gap_aux
        del gap_train, gap_score

        entry = _write_scored_fold(
            output_dir=output_dir / fold_id,
            fold_id=fold_id,
            role="gapped",
            train_time_max=int(fold["train_time_max"]),
            score_time_min=int(fold["score_time_min"]),
            score_time_max=int(fold["score_time_max"]),
            candidate_ids=score_candidates,
            baseline=v1_baseline,
            candidate_scores={
                FULL_ONLY_ID: full_candidate,
                GAP_AWARE_ID: gap_candidate,
            },
            candidate_configs=candidate_configs,
        )
        entry["deployment_horizon_quantile"] = float(
            fold["deployment_horizon_quantile"]
        )
        entry["train_rows"] = [train_start, train_stop]
        entry["score_rows"] = [score_start, score_stop]
        entries.append(entry)
        reports.append(
            {
                "fold_id": fold_id,
                "train_rows": [train_start, train_stop],
                "score_rows": [score_start, score_stop],
                "gap_seconds": int(fold["gap_seconds"]),
                "training_device": "cpu",
                "internal_scoring_device": "cpu",
                "prior_setwise_deterministic_replay": setwise_replay,
                "prior_setwise_losses": list(setwise_losses),
                "v1_deterministic_replay": v1_replay,
                "v1_losses": list(v1_losses),
                "full_only_deterministic_replay": full_replay,
                "full_only_losses": list(full_losses),
                "gap_aware_deterministic_replay": gap_replay,
                "gap_aware_losses": list(gap_losses),
                "gap_aware_near_copy_weight": 1.0 - collapsed_fraction,
                "gap_aware_collapsed_copy_weight": collapsed_fraction,
                "gap_aware_original_train_rows": train_count,
                "gap_aware_effective_training_rows": 2 * train_count,
                "elapsed_seconds": time.time() - fold_started,
            }
        )
        del (
            train_overlay,
            score_overlay,
            setwise_predictions,
            setwise_probabilities,
            lgbm_probabilities,
            prior_baseline,
            v1_predictions,
            v1_aux,
            v1_baseline,
            full_predictions,
            full_aux,
            full_candidate,
            stale_lift,
            near_gap_raw,
            stale_gap_raw,
            row_weights,
            gap_predictions,
            gap_aux,
            gap_candidate,
        )
        gc.collect()
    return entries, reports


def _fit_head_replayed(
    train_view: Any,
    scoring_views: dict[str, Any],
    *,
    hidden_dim: int,
    seed: int,
    candidate_name: str,
    replay_rtol: float,
    replay_atol: float,
    feature_count: int | None = None,
    feature_indices: tuple[int, ...] | None = None,
    train_row_weights: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], tuple[float, ...], dict[str, Any]]:
    first_model, first_result, first_losses = _fit_head(
        train_view,
        next(iter(scoring_views.values())),
        hidden_dim=hidden_dim,
        seed=seed,
        candidate_name=candidate_name,
        feature_count=feature_count,
        feature_indices=feature_indices,
        train_row_weights=train_row_weights,
    )
    first_predictions = {
        arm_id: _predict_probabilities(first_model, view, first_result)
        for arm_id, view in scoring_views.items()
    }
    first_state = {
        key: np.array(value, copy=True)
        for key, value in first_result.state.items()
    }
    del first_model, first_result
    _release_jittor()

    second_model, second_result, second_losses = _fit_head(
        train_view,
        next(iter(scoring_views.values())),
        hidden_dim=hidden_dim,
        seed=seed,
        candidate_name=candidate_name,
        feature_count=feature_count,
        feature_indices=feature_indices,
        train_row_weights=train_row_weights,
    )
    second_predictions = {
        arm_id: _predict_probabilities(second_model, view, second_result)
        for arm_id, view in scoring_views.items()
    }
    replay = build_deterministic_replay_report(
        first_state=first_state,
        second_state=second_result.state,
        first_losses=first_losses,
        second_losses=second_losses,
        first_predictions=first_predictions,
        second_predictions=second_predictions,
        rtol=replay_rtol,
        atol=replay_atol,
    )
    del second_model, second_result, second_predictions
    _release_jittor()
    if not replay["matched"]:
        raise RuntimeError(
            f"{candidate_name} deterministic CPU replay failed: "
            f"{json.dumps(replay, sort_keys=True)}"
        )
    return first_predictions, first_losses, replay


def _fit_head(
    train_view: Any,
    score_view: Any,
    *,
    hidden_dim: int,
    seed: int,
    candidate_name: str,
    feature_count: int | None = None,
    feature_indices: tuple[int, ...] | None = None,
    train_row_weights: np.ndarray | None = None,
) -> tuple[Any, FusionResult, tuple[float, ...]]:
    jt.flags.use_cuda = 0
    if feature_indices is None:
        if feature_count is None:
            raise ValueError("feature_count or feature_indices is required")
        feature_indices = tuple(range(feature_count))
    jt.set_global_seed(seed)
    config = FusionConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        weight_decay=0.0,
        hidden_dim=hidden_dim,
        selection_metric="mrr",
        early_stop_patience=0,
    )
    return fit_fusion_mlp_listwise_fixed(
        train_view,
        score_view,
        config,
        np.random.default_rng(seed),
        verbose=False,
        feature_indices=feature_indices,
        candidate_name=candidate_name,
        train_row_weights=train_row_weights,
    )


def _predict_probabilities(
    model: Any,
    features: Any,
    result: FusionResult,
) -> np.ndarray:
    jt.flags.use_cuda = 0
    rows = int(features.shape[0])
    probabilities = np.empty(
        (rows, int(features.shape[1])),
        dtype=np.float32,
    )
    for start in range(0, rows, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, rows)
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
    features: Any,
    *,
    feature_indices: tuple[int, ...],
) -> np.ndarray:
    rows = int(features.shape[0])
    probabilities = np.empty((rows, int(features.shape[1])), dtype=np.float32)
    for start in range(0, rows, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, rows)
        batch = np.asarray(
            features[start:stop][..., feature_indices],
            dtype=np.float32,
        )
        logits = np.asarray(
            booster.predict(batch.reshape(-1, batch.shape[-1])),
            dtype=np.float64,
        ).reshape(batch.shape[:2])
        probabilities[start:stop] = _softmax(logits)
    return probabilities


def _write_scored_fold(
    *,
    output_dir: Path,
    fold_id: str,
    role: str,
    train_time_max: int,
    score_time_min: int,
    score_time_max: int,
    candidate_ids: np.ndarray,
    baseline: np.ndarray,
    candidate_scores: dict[str, np.ndarray],
    candidate_configs: dict[str, dict[str, Any]],
    counterfactual: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    fingerprint = hashlib.sha256(
        np.ascontiguousarray(candidate_ids).tobytes(order="C")
    ).hexdigest()
    baseline_path = output_dir / "baseline-v1.npy"
    _save_array(baseline_path, baseline)
    descriptors: dict[str, dict[str, Any]] = {}
    for candidate_id, scores in candidate_scores.items():
        path = output_dir / f"{candidate_id}.npy"
        _save_array(path, scores)
        descriptors[candidate_id] = _candidate_descriptor(
            path,
            candidate_id=candidate_id,
            config_sha256=candidate_configs[candidate_id]["sha256"],
            candidate_fingerprint=fingerprint,
        )
    entry: dict[str, Any] = {
        "fold_id": fold_id,
        "role": role,
        "train_time_max": int(train_time_max),
        "score_time_min": int(score_time_min),
        "score_time_max": int(score_time_max),
        "candidate_fingerprint": fingerprint,
        "baseline": _artifact_descriptor(baseline_path),
        "candidates": descriptors,
    }
    if counterfactual is not None:
        arms: dict[str, Any] = {}
        for arm_id, arm in counterfactual.items():
            arm_dir = output_dir / arm_id
            arm_dir.mkdir()
            arm_baseline_path = arm_dir / "baseline-v1.npy"
            _save_array(arm_baseline_path, arm["baseline"])
            arm_candidates: dict[str, Any] = {}
            for candidate_id, scores in arm["candidates"].items():
                path = arm_dir / f"{candidate_id}.npy"
                _save_array(path, scores)
                arm_candidates[candidate_id] = _candidate_descriptor(
                    path,
                    candidate_id=candidate_id,
                    config_sha256=(
                        candidate_configs[candidate_id]["sha256"]
                    ),
                    candidate_fingerprint=fingerprint,
                )
            arms[arm_id] = {
                "baseline": _artifact_descriptor(arm_baseline_path),
                "candidates": arm_candidates,
            }
        entry["counterfactual_arms"] = arms
    _write_json(output_dir / "fold-score-report.json", entry)
    return entry


def _load_source_fold_scores(
    source_fold: dict[str, Any],
    *,
    descriptor: dict[str, Any],
    candidate_ids: np.ndarray,
) -> np.ndarray:
    path = Path(descriptor["path"])
    if _sha256(path) != descriptor["sha256"]:
        raise ValueError(f"{source_fold['fold_id']} source score hash differs")
    fingerprint = hashlib.sha256(
        np.ascontiguousarray(candidate_ids).tobytes(order="C")
    ).hexdigest()
    if fingerprint != source_fold["candidate_fingerprint"]:
        raise ValueError(
            f"{source_fold['fold_id']} source candidate fingerprint differs"
        )
    scores = np.load(path, mmap_mode="r", allow_pickle=False)
    if scores.shape != candidate_ids.shape:
        raise ValueError(f"{source_fold['fold_id']} source score shape differs")
    return np.asarray(scores, dtype=np.float64)


def _validate_frozen_inputs(
    *,
    checkpoint: Path,
    validation_plan: Path,
    plan: dict[str, Any],
    plan_lock_path: Path,
    plan_lock: dict[str, Any],
    execution_contract_path: Path,
    execution_contract: dict[str, Any],
    bugfixed_v1_contract_path: Path,
    bugfixed_v1_contract: dict[str, Any],
    near_v1_manifest_path: Path,
    near_cache_report_path: Path,
    candidate_configs: dict[str, dict[str, Any]],
) -> None:
    if _sha256(checkpoint) != plan["baseline"]["checkpoint_sha256"]:
        raise ValueError("v1 checkpoint differs from validation plan")
    if _sha256(validation_plan) != plan_lock["source_plan_sha256"]:
        raise ValueError("validation plan differs from the frozen lock")
    if _sha256(plan_lock_path) == plan.get("plan_lock_sha256"):
        raise ValueError("validation plan must not embed a mutable lock hash")
    frozen_hashes = {
        "validation_plan_sha256": _sha256(validation_plan),
        "plan_lock_sha256": _sha256(plan_lock_path),
        "bugfixed_v1_contract_sha256": _sha256(bugfixed_v1_contract_path),
        "historical_near_v1_manifest_sha256": _sha256(
            near_v1_manifest_path
        ),
        "near_cache_report_sha256": _sha256(near_cache_report_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "execution_module_sha256": _sha256(
            Path(execution_module.__file__).resolve()
        ),
        "fusion_source_sha256": _sha256(
            Path(fusion_module.__file__).resolve()
        ),
    }
    for key, actual in frozen_hashes.items():
        if execution_contract.get(key) != actual:
            raise ValueError(f"successor execution {key} differs")
    near_report = _read_json(near_cache_report_path)
    limits = near_report.get("prediction_limits")
    if (
        near_report.get("status") != "complete"
        or near_report.get("checkpoint_sha256")
        != execution_contract.get("near_cache_checkpoint_sha256")
        or not isinstance(limits, dict)
        or limits.get("structure_predict_neighbor_limit") != 512
        or limits.get("source_profile_predict_history_limit") != 512
    ):
        raise ValueError("successor near cache K512 lineage differs")
    if (
        execution_contract.get("weighted_normalizer_required") is not True
        or execution_contract.get(
            "weighted_normalizer_bug_bypass_authorized"
        )
        is not False
    ):
        raise ValueError("successor weighted normalizer contract differs")
    if execution_contract.get("path_sha256") == _sha256(
        execution_contract_path
    ):
        raise ValueError("execution contract must not embed its own hash")
    implementation = bugfixed_v1_contract.get("implementation_contract")
    if (
        not isinstance(implementation, dict)
        or implementation.get("training_device") != "cpu"
        or implementation.get("gpu_training_authorized") is not False
    ):
        raise ValueError("bugfixed V1 CPU training contract differs")
    expected = {
        item["candidate_id"]: item
        for item in plan["candidate_space"]
    }
    if set(expected) != set(candidate_configs):
        raise ValueError("candidate config set differs from validation plan")
    for candidate_id, config in candidate_configs.items():
        if config["sha256"] != expected[candidate_id]["config_sha256"]:
            raise ValueError(f"{candidate_id} config hash differs")
        if config["payload"]["candidate_id"] != candidate_id:
            raise ValueError(f"{candidate_id} config identity differs")
        if config["payload"]["baseline"]["checkpoint_sha256"] != _sha256(
            checkpoint
        ):
            raise ValueError(f"{candidate_id} baseline binding differs")
    if plan["rescan_policy"]["v1_family_weight_rescan_authorized"]:
        raise ValueError("v1 weight rescan must remain prohibited")
    if float(plan["rescan_policy"]["selected_weight"]) != V1_WEIGHT:
        raise ValueError("successor integration weight must remain 0.50")


def _candidate_seed(fold_index: int) -> int:
    return BASE_SEED + fold_index * SEED_STRIDE + SEED_SALT


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def _artifact_descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _candidate_descriptor(
    path: Path,
    *,
    candidate_id: str,
    config_sha256: str,
    candidate_fingerprint: str,
) -> dict[str, str]:
    return {
        **_artifact_descriptor(path),
        "candidate_id": candidate_id,
        "config_sha256": config_sha256,
        "candidate_fingerprint": candidate_fingerprint,
    }


def _save_array(path: Path, values: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_jittor() -> None:
    try:
        jt.sync_all(True)
        jt.gc()
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
