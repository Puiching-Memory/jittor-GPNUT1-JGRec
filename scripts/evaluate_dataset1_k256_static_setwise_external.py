from __future__ import annotations

import argparse
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
    build_fusion_from_state,
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
    evaluate_external_safety_deltas,
    static_setwise_weight_grid,
)
from jgrec.rankers.hybrid.time_ramp import apply_time_ramp
from jgrec.robust_weight_selection import ranking_metrics
from train_evaluate_dataset1_full100_setwise import (
    _predict_streaming,
    _softmax,
)
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

TRAIN_ROWS = 200_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--selection-lock-sha256", required=True, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--external-cache-prefix", required=True, type=Path)
    parser.add_argument("--external-cache-report", required=True, type=Path)
    parser.add_argument("--reference-external-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    plan_hash = _require_sidecar(args.plan, args.plan_sha256)
    lock_hash = _require_sidecar(
        args.selection_lock,
        args.selection_lock_sha256,
    )
    plan = _read_json(args.plan)
    lock = _read_json(args.selection_lock)
    selection = _read_json(args.selection_report)
    train_report = _read_json(args.train_cache_report)
    external_report = _read_json(args.external_cache_report)
    reference_external = _read_json(args.reference_external_report)
    selected_weight = _preflight(
        plan=plan,
        plan_hash=plan_hash,
        lock=lock,
        lock_hash=lock_hash,
        selection=selection,
        selection_report=args.selection_report,
        checkpoint=args.source_checkpoint,
        train_report=train_report,
        external_report=external_report,
        reference_external=reference_external,
    )
    require_jittor_cuda(jt)
    args.output_dir.mkdir(parents=True)
    started = time.time()
    state = load_checkpoint_dataset(args.source_checkpoint, "dataset1")
    protocol, control_config, _ = _frozen_protocol(state)
    train_path = Path(f"{args.train_cache_prefix}.train.npy")
    train_features = np.load(
        train_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if train_features.shape != (TRAIN_ROWS, 100, 63):
        raise ValueError("full-origin K256 train tensor shape differs")

    tune_start = TRAIN_ROWS - TUNE_ROWS
    max_fit_rows = int(getattr(state["config"], "max_train_events", 0))
    fit_start = (
        max(0, tune_start - max_fit_rows)
        if max_fit_rows > 0
        else 0
    )
    base_train = train_features[fit_start:tune_start]
    setwise_train_features = train_features[:tune_start]
    tune = train_features[tune_start:TRAIN_ROWS]
    feature_indices = tuple(
        int(index) for index in state["fusion_result"].feature_indices
    )
    seed = int(control_config["seed"])

    candidate_base_model, candidate_base_result = (
        fit_fusion_mlp_streaming(
            base_train,
            tune,
            _fusion_config(control_config),
            np.random.default_rng(seed),
            verbose=True,
            feature_indices=feature_indices,
            candidate_name="dataset1_k256_static_full_origin_base",
        )
    )
    candidate_base_head = args.output_dir / "candidate-base-head.npz"
    _save_head(
        candidate_base_head,
        candidate_base_result,
        hidden_dim=int(control_config["hidden_dim"]),
        context_transform_version=0,
        fit_rows=(fit_start, tune_start),
        tune_rows=(tune_start, TRAIN_ROWS),
    )

    candidate_lgbm = fit_fusion_lgbm(
        base_train,
        tune,
        selection_metric=str(control_config["selection_metric"]),
        verbose=True,
        feature_indices=feature_indices,
        candidate_name="dataset1_k256_static_full_origin_lgbm",
    )
    candidate_lgbm_path = args.output_dir / "candidate-lgbm-model.txt"
    candidate_lgbm_path.write_text(
        candidate_lgbm.model_text,
        encoding="utf-8",
    )

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
    candidate_setwise_model, candidate_setwise_result, setwise_history = (
        fit_fusion_mlp_listwise_streaming(
            setwise_train,
            setwise_tune,
            setwise_config,
            np.random.default_rng(seed),
            verbose=True,
            feature_indices=tuple(range(setwise_train.shape[-1])),
            candidate_name="dataset1_k256_static_full_origin_setwise",
        )
    )
    candidate_setwise_head = args.output_dir / "candidate-setwise-head.npz"
    _save_head(
        candidate_setwise_head,
        candidate_setwise_result,
        hidden_dim=SETWISE_HIDDEN_DIM,
        context_transform_version=1,
        fit_rows=(0, tune_start),
        tune_rows=(tune_start, TRAIN_ROWS),
    )

    receipt = {
        "status": "external_opened_once",
        "decision_role": "safety_gate_only",
        "effect_size_estimation_authorized": False,
        "plan_sha256": plan_hash,
        "selection_lock_sha256": lock_hash,
        "selection_report_sha256": _sha256(args.selection_report),
        "selected_weight": selected_weight,
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "external_cache_report_sha256": _sha256(
            args.external_cache_report
        ),
        "opened_unix_seconds": time.time(),
    }
    receipt_path = args.output_dir / "external-open-receipt.json"
    _write_json_exclusive(receipt_path, receipt)

    external_path = Path(f"{args.external_cache_prefix}.val.npy")
    external_time_path = Path(
        f"{args.external_cache_prefix}.val-time.npy"
    )
    external_candidate_path = Path(
        f"{args.external_cache_prefix}.val-candidates.npy"
    )
    _require_report_hash(
        external_path,
        external_report,
        "features",
    )
    _require_report_hash(
        external_time_path,
        external_report,
        "time",
    )
    _require_report_hash(
        external_candidate_path,
        external_report,
        "candidates",
    )
    external_features = np.load(
        external_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    external_times = np.load(
        external_time_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if external_features.shape != (20_000, 100, 63):
        raise ValueError("external K256 tensor shape differs")

    candidate_base_logits = _predict_mlp_streaming(
        candidate_base_model,
        external_features,
        candidate_base_result,
        context_transform_version=0,
        batch_size=int(control_config["batch_size"]),
    )
    candidate_lgbm_logits = predict_logits_lgbm(
        candidate_lgbm.model_text,
        np.asarray(external_features[..., feature_indices]),
    )
    candidate_setwise_logits = _predict_setwise_streaming(
        candidate_setwise_model,
        external_features,
        candidate_setwise_result,
        batch_size=SETWISE_BATCH_SIZE,
    )
    candidate_backbone = blend_expert_logits(
        candidate_base_logits,
        candidate_lgbm_logits,
        protocol.mlp_weight,
        calibration=protocol.expert_calibration,
    )
    candidate_scores = blend_static_setwise(
        candidate_backbone,
        _softmax(candidate_setwise_logits),
        weight=selected_weight,
    )

    source_base_result = state["fusion_result"]
    source_base_model = build_fusion_from_state(
        input_dim=len(source_base_result.feature_indices),
        hidden_dim=int(state["fusion_hidden_dim"]),
        state=state["fusion_state"],
    )
    source_base_logits = _predict_streaming(
        source_base_model,
        external_features,
        source_base_result.mean,
        source_base_result.std,
        feature_indices=tuple(source_base_result.feature_indices),
        batch_size=256,
    )
    source_lgbm = state["lgbm_result"]
    source_lgbm_logits = predict_logits_lgbm(
        source_lgbm.model_text,
        np.asarray(
            external_features[..., source_lgbm.feature_indices]
        ),
    )
    source_backbone = blend_expert_logits(
        source_base_logits,
        source_lgbm_logits,
        protocol.mlp_weight,
        calibration=protocol.expert_calibration,
    )
    source_setwise_result = state["time_ramp_setwise_result"]
    source_setwise_model = build_fusion_from_state(
        input_dim=len(source_setwise_result.feature_indices),
        hidden_dim=int(state["time_ramp_setwise_hidden_dim"]),
        state=state["time_ramp_setwise_fusion_state"],
    )
    source_setwise_logits = _predict_streaming(
        source_setwise_model,
        SetwiseFeatureView(external_features),
        source_setwise_result.mean,
        source_setwise_result.std,
        feature_indices=tuple(source_setwise_result.feature_indices),
        batch_size=256,
    )
    baseline_scores = apply_time_ramp(
        source_backbone,
        _softmax(source_setwise_logits),
        np.asarray(external_times, dtype=np.int64),
        power=0.5,
        minimum_time=float(np.min(external_times)),
        maximum_time=float(np.max(external_times)),
    )

    baseline_metrics = ranking_metrics(baseline_scores)
    candidate_metrics = ranking_metrics(
        candidate_scores,
        baseline_scores=baseline_scores,
    )
    deltas = _metric_delta(candidate_metrics, baseline_metrics)
    movement = candidate_metrics["query_movements"]
    gate = evaluate_external_safety_deltas(
        deltas,
        improved=int(movement["improved"]),
        worsened=int(movement["worsened"]),
    )
    np.save(
        args.output_dir / "external-baseline.npy",
        np.asarray(baseline_scores, dtype=np.float32),
    )
    np.save(
        args.output_dir / "external-candidate.npy",
        np.asarray(candidate_scores, dtype=np.float32),
    )
    result = {
        "status": "accepted" if gate["accepted"] else "rejected",
        "accepted": gate["accepted"],
        "package_authorized": gate["accepted"],
        "decision_role": "safety_gate_only",
        "effect_size_estimation_authorized": False,
        "selected_weight": selected_weight,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_candidate_minus_baseline": deltas,
        "gate": gate,
        "receipt_sha256": _sha256(receipt_path),
        "plan_sha256": plan_hash,
        "selection_lock_sha256": lock_hash,
        "source_checkpoint_sha256": _sha256(args.source_checkpoint),
        "full_origin": {
            "base_fit_rows": [fit_start, tune_start],
            "setwise_fit_rows": [0, tune_start],
            "tune_rows": [tune_start, TRAIN_ROWS],
            "base_head_sha256": _sha256(candidate_base_head),
            "lgbm_sha256": _sha256(candidate_lgbm_path),
            "setwise_head_sha256": _sha256(candidate_setwise_head),
            "setwise_history": list(setwise_history),
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "external-evaluation-report.json", result)
    release_memory()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if gate["accepted"] else 2


def _preflight(
    *,
    plan: dict[str, Any],
    plan_hash: str,
    lock: dict[str, Any],
    lock_hash: str,
    selection: dict[str, Any],
    selection_report: Path,
    checkpoint: Path,
    train_report: dict[str, Any],
    external_report: dict[str, Any],
    reference_external: dict[str, Any],
) -> float:
    if (
        plan.get("external_labels_read") is not False
        or plan.get("external_gate", {}).get("decision_role")
        != "safety_gate_only"
        or plan.get("external_gate", {}).get(
            "effect_size_estimation_authorized"
        )
        is not False
    ):
        raise ValueError("plan does not authorize a safety-only external")
    if (
        lock.get("status") != "selected"
        or lock.get("external_labels_read") is not False
        or lock.get("plan_sha256") != plan_hash
        or lock.get("selection_report_sha256")
        != _sha256(selection_report)
        or not selection.get("internal_gate_passed")
        or not selection.get("external_authorized")
    ):
        raise ValueError("internal selection did not authorize external")
    weight = float(lock["selected_weight"])
    if (
        weight not in static_setwise_weight_grid()
        or float(selection["selected_weight"]) != weight
    ):
        raise ValueError("selected weight differs across lock/report")
    if (
        _sha256(checkpoint) != plan["checkpoint_sha256"]
        or train_report.get("prediction_limits")
        != plan["prediction_limits"]
        or external_report.get("prediction_limits")
        != plan["prediction_limits"]
        or external_report.get("train_cache_report_sha256")
        != _sha256(Path(external_report["train_cache_report"]))
    ):
        raise ValueError("K256 cache/checkpoint lineage differs")
    current_candidates = external_report["artifacts"]["candidates"]
    reference_candidates = reference_external["artifacts"]["candidates"]
    if current_candidates["sha256"] != reference_candidates["sha256"]:
        raise ValueError("external candidate matrix differs from reference")
    if len(lock_hash) != 64:
        raise ValueError("selection lock SHA-256 is malformed")
    return weight


def _require_report_hash(
    path: Path,
    report: dict[str, Any],
    name: str,
) -> None:
    expected = report["artifacts"][name]["sha256"]
    if _sha256(path) != expected:
        raise ValueError(f"external {name} hash differs from report")


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


def _require_sidecar(path: Path, sidecar: Path) -> str:
    expected = sidecar.read_text(encoding="ascii").split()[0]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError("artifact hash differs from sidecar")
    return actual


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
