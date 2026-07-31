from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_METRIC_NAMES = (
    "mrr",
    "hit_at_1",
    "hit_at_3",
    "hit_at_10",
    "ndcg_at_10",
    "mean_rank",
)
_TOLERANCE = 1e-15


def ranking_metrics(
    scores: np.ndarray,
    *,
    baseline_scores: np.ndarray | None = None,
    positive_candidate_column: int = 0,
) -> dict[str, Any]:
    values = _validated_scores(scores, label="scores")
    ranks = _positive_ranks(values, positive_candidate_column)
    metrics: dict[str, Any] = {
        "mrr": float(np.mean(1.0 / ranks)),
        "hit_at_1": float(np.mean(ranks <= 1)),
        "hit_at_3": float(np.mean(ranks <= 3)),
        "hit_at_10": float(np.mean(ranks <= 10)),
        "ndcg_at_10": float(
            np.mean(
                np.where(
                    ranks <= 10,
                    1.0 / np.log2(ranks + 1.0),
                    0.0,
                )
            )
        ),
        "mean_rank": float(np.mean(ranks)),
    }
    if baseline_scores is not None:
        baseline = _validated_scores(
            baseline_scores,
            label="baseline_scores",
        )
        if baseline.shape != values.shape:
            raise ValueError("baseline_scores and scores must have identical shapes")
        baseline_ranks = _positive_ranks(
            baseline,
            positive_candidate_column,
        )
        metrics["query_movements"] = {
            "improved": int(np.sum(ranks < baseline_ranks)),
            "unchanged": int(np.sum(ranks == baseline_ranks)),
            "worsened": int(np.sum(ranks > baseline_ranks)),
        }
    return metrics


def select_rolling_origin_weight(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    manifest = _read_json(manifest_path)
    integration_id, positive_column, folds, weight_keys = _validate_selection_manifest(manifest)
    baseline_scores = [
        _load_score_artifact(
            fold["baseline"],
            manifest_dir=manifest_path.parent,
            label=f"{fold['fold_id']} baseline",
        )
        for fold in folds
    ]
    baseline_shapes = [baseline.shape for baseline in baseline_scores]
    for baseline in baseline_scores:
        _validate_positive_column(
            positive_column,
            candidate_count=baseline.shape[1],
        )
    pooled_baseline = np.concatenate(baseline_scores, axis=0)
    pooled_baseline_metrics = ranking_metrics(
        pooled_baseline,
        positive_candidate_column=positive_column,
    )

    weight_reports: dict[str, dict[str, Any]] = {}
    candidate_hashes: dict[str, list[dict[str, str]]] = {}
    for weight_key in weight_keys:
        candidate_scores: list[np.ndarray] = []
        fold_reports: list[dict[str, Any]] = []
        hashes: list[dict[str, str]] = []
        for fold, baseline, baseline_shape in zip(
            folds,
            baseline_scores,
            baseline_shapes,
            strict=True,
        ):
            candidate_artifact = fold["candidates"][weight_key]
            candidate = _load_score_artifact(
                candidate_artifact,
                manifest_dir=manifest_path.parent,
                label=f"{fold['fold_id']} weight {weight_key}",
            )
            if candidate.shape != baseline_shape:
                raise ValueError(
                    f"{fold['fold_id']} weight {weight_key} score shape "
                    f"{candidate.shape} differs from baseline "
                    f"{baseline_shape}"
                )
            baseline_metrics = ranking_metrics(
                baseline,
                positive_candidate_column=positive_column,
            )
            candidate_metrics = ranking_metrics(
                candidate,
                baseline_scores=baseline,
                positive_candidate_column=positive_column,
            )
            fold_reports.append(
                {
                    "fold_id": fold["fold_id"],
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "delta_candidate_minus_baseline": _metric_deltas(
                        candidate_metrics,
                        baseline_metrics,
                    ),
                }
            )
            candidate_scores.append(candidate)
            hashes.append(
                {
                    "fold_id": fold["fold_id"],
                    "score_sha256": candidate_artifact["sha256"],
                }
            )
        pooled_candidate = np.concatenate(candidate_scores, axis=0)
        pooled_candidate_metrics = ranking_metrics(
            pooled_candidate,
            baseline_scores=pooled_baseline,
            positive_candidate_column=positive_column,
        )
        pooled_deltas = _metric_deltas(
            pooled_candidate_metrics,
            pooled_baseline_metrics,
        )
        mrr_deltas = [fold_report["delta_candidate_minus_baseline"]["mrr"] for fold_report in fold_reports]
        ndcg_deltas = [fold_report["delta_candidate_minus_baseline"]["ndcg_at_10"] for fold_report in fold_reports]
        gates = {
            "all_folds_mrr_non_decreasing": all(delta >= -_TOLERANCE for delta in mrr_deltas),
            "all_folds_ndcg_at_10_non_decreasing": all(delta >= -_TOLERANCE for delta in ndcg_deltas),
            "pooled_hit_at_1_non_decreasing": (pooled_deltas["hit_at_1"] >= -_TOLERANCE),
            "pooled_hit_at_3_non_decreasing": (pooled_deltas["hit_at_3"] >= -_TOLERANCE),
            "pooled_hit_at_10_non_decreasing": (pooled_deltas["hit_at_10"] >= -_TOLERANCE),
            "pooled_mean_rank_non_increasing": (pooled_deltas["mean_rank"] <= _TOLERANCE),
            "pooled_improved_queries_exceed_worsened": (
                pooled_candidate_metrics["query_movements"]["improved"]
                > pooled_candidate_metrics["query_movements"]["worsened"]
            ),
        }
        failed_gates = [name for name, passed in gates.items() if not passed]
        weight_reports[weight_key] = {
            "weight": float(weight_key),
            "eligible": not failed_gates,
            "failed_gates": failed_gates,
            "gates": gates,
            "folds": fold_reports,
            "pooled": {
                "baseline": pooled_baseline_metrics,
                "candidate": pooled_candidate_metrics,
                "delta_candidate_minus_baseline": pooled_deltas,
            },
            "stability": {
                "fold_mrr_deltas": mrr_deltas,
                "fold_ndcg_at_10_deltas": ndcg_deltas,
                "worst_fold_mrr_delta": float(min(mrr_deltas)),
                "median_fold_mrr_delta": float(np.median(mrr_deltas)),
            },
        }
        candidate_hashes[weight_key] = hashes

    eligible_keys = [key for key, weight_report in weight_reports.items() if weight_report["eligible"]]
    selected_key = (
        max(
            eligible_keys,
            key=lambda key: _selection_key(weight_reports[key]),
        )
        if eligible_keys
        else None
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "exact_integrated_rolling_weight_selection_v1",
        "status": "selected" if selected_key is not None else "rejected",
        "integration_id": integration_id,
        "selection_manifest": str(manifest_path.resolve()),
        "selection_manifest_sha256": _sha256(manifest_path),
        "fold_count": len(folds),
        "positive_candidate_column": positive_column,
        "stability_gate": {
            "fold_metrics": [
                "mrr_non_decreasing",
                "ndcg_at_10_non_decreasing",
            ],
            "pooled_metrics": [
                "hit_at_1_non_decreasing",
                "hit_at_3_non_decreasing",
                "hit_at_10_non_decreasing",
                "mean_rank_non_increasing",
                "improved_queries_exceed_worsened",
            ],
            "selection_order": [
                "maximum_worst_fold_mrr_delta",
                "maximum_median_fold_mrr_delta",
                "maximum_pooled_mrr_delta",
                "minimum_weight",
            ],
        },
        "weights": weight_reports,
        "selected_weight": (float(selected_key) if selected_key is not None else None),
        "external_holdout_read": False,
    }
    output_dir.mkdir(parents=True)
    _write_json_exclusive(output_dir / "selection-report.json", report)
    if selected_key is not None:
        lock = {
            "schema_version": 1,
            "protocol": "exact_integrated_weight_selection_lock_v1",
            "integration_id": integration_id,
            "selection_manifest_sha256": report["selection_manifest_sha256"],
            "selected_weight": float(selected_key),
            "selected_candidate_scores": candidate_hashes[selected_key],
            "selection_rule": report["stability_gate"]["selection_order"],
            "external_holdout_read": False,
        }
        _write_json_exclusive(output_dir / "selection-lock.json", lock)
    return report


def evaluate_locked_external(
    *,
    manifest_path: Path,
    selection_lock_path: Path,
    state_dir: Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    selection_lock_path = Path(selection_lock_path)
    state_dir = Path(state_dir)
    manifest = _read_json(manifest_path)
    lock = _read_json(selection_lock_path)
    _validate_external_contract(
        manifest=manifest,
        lock=lock,
        selection_lock_sha256=_sha256(selection_lock_path),
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = state_dir / "external-open-receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"external holdout already opened: {receipt_path}")
    receipt = {
        "schema_version": 1,
        "protocol": "exact_integrated_external_open_receipt_v1",
        "opened_at_utc": datetime.now(UTC).isoformat(),
        "external_manifest_sha256": _sha256(manifest_path),
        "selection_lock_sha256": _sha256(selection_lock_path),
        "integration_id": lock["integration_id"],
        "selected_weight": float(lock["selected_weight"]),
    }
    _write_json_exclusive(receipt_path, receipt)

    baseline = _load_score_artifact(
        manifest["baseline"],
        manifest_dir=manifest_path.parent,
        label="external baseline",
    )
    candidate = _load_score_artifact(
        manifest["candidate"],
        manifest_dir=manifest_path.parent,
        label="external candidate",
    )
    if candidate.shape != baseline.shape:
        raise ValueError("external candidate score shape differs from baseline")
    positive_column = int(manifest.get("positive_candidate_column", 0))
    baseline_metrics = ranking_metrics(
        baseline,
        positive_candidate_column=positive_column,
    )
    candidate_metrics = ranking_metrics(
        candidate,
        baseline_scores=baseline,
        positive_candidate_column=positive_column,
    )
    deltas = _metric_deltas(candidate_metrics, baseline_metrics)
    gates = {
        "mrr_strictly_increasing": deltas["mrr"] > _TOLERANCE,
        "ndcg_at_10_non_decreasing": (deltas["ndcg_at_10"] >= -_TOLERANCE),
        "hit_at_1_non_decreasing": (deltas["hit_at_1"] >= -_TOLERANCE),
        "hit_at_3_non_decreasing": (deltas["hit_at_3"] >= -_TOLERANCE),
        "hit_at_10_non_decreasing": (deltas["hit_at_10"] >= -_TOLERANCE),
        "mean_rank_non_increasing": (deltas["mean_rank"] <= _TOLERANCE),
        "improved_queries_exceed_worsened": (
            candidate_metrics["query_movements"]["improved"] > candidate_metrics["query_movements"]["worsened"]
        ),
    }
    accepted = all(gates.values())
    report = {
        "schema_version": 1,
        "protocol": "exact_integrated_external_evaluation_v1",
        "status": "accepted" if accepted else "rejected",
        "integration_id": lock["integration_id"],
        "selected_weight": float(lock["selected_weight"]),
        "selection_lock_sha256": receipt["selection_lock_sha256"],
        "external_manifest_sha256": receipt["external_manifest_sha256"],
        "external_open_receipt": str(receipt_path.resolve()),
        "external_gap": {
            "training_time_max": manifest["training_time_max"],
            "score_time_min": manifest["score_time_min"],
            "score_time_max": manifest["score_time_max"],
            "minimum_train_to_score_gap": manifest["minimum_train_to_score_gap"],
            "actual_train_to_score_gap": (manifest["score_time_min"] - manifest["training_time_max"]),
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_candidate_minus_baseline": deltas,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "weight_rescan_authorized": False,
        "leaderboard_tuning_authorized": False,
    }
    _write_json_exclusive(
        state_dir / "external-evaluation-report.json",
        report,
    )
    return report


def _validate_selection_manifest(
    manifest: dict[str, Any],
) -> tuple[str, int, list[dict[str, Any]], list[str]]:
    if manifest.get("protocol") != "exact_integrated_rolling_weight_selection_v1":
        raise ValueError("unexpected rolling selection protocol")
    integration_id = manifest.get("integration_id")
    if not isinstance(integration_id, str) or not integration_id:
        raise ValueError("integration_id must be a non-empty string")
    positive_column = int(manifest.get("positive_candidate_column", 0))
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) < 3:
        raise ValueError("rolling selection requires at least three folds")
    weight_keys: list[str] | None = None
    previous_score_max: float | int | None = None
    previous_train_max: float | int | None = None
    fold_ids: set[str] = set()
    for fold in folds:
        fold_id = fold.get("fold_id")
        if not isinstance(fold_id, str) or not fold_id:
            raise ValueError("every fold requires a fold_id")
        if fold_id in fold_ids:
            raise ValueError(f"duplicate fold_id: {fold_id}")
        fold_ids.add(fold_id)
        train_max = fold["train_time_max"]
        score_min = fold["score_time_min"]
        score_max = fold["score_time_max"]
        if not train_max < score_min <= score_max:
            raise ValueError(f"{fold_id} score interval must be strictly after training")
        if previous_score_max is not None and score_min <= previous_score_max:
            raise ValueError(f"{fold_id} score interval must be strictly after the previous fold")
        if previous_train_max is not None and train_max <= previous_train_max:
            raise ValueError(f"{fold_id} training origin must increase monotonically")
        previous_score_max = score_max
        previous_train_max = train_max
        fingerprint = fold.get("candidate_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"{fold_id} requires a candidate_fingerprint")
        _validate_artifact_descriptor(fold.get("baseline"), fold_id)
        candidates = fold.get("candidates")
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError(f"{fold_id} requires candidate scores")
        current_keys = sorted(
            candidates,
            key=lambda value: float(value),
        )
        if weight_keys is None:
            weight_keys = current_keys
            _validate_weight_keys(weight_keys)
        elif current_keys != weight_keys:
            raise ValueError("all folds must provide the same candidate weights")
        for weight_key, candidate in candidates.items():
            _validate_artifact_descriptor(
                candidate,
                f"{fold_id} weight {weight_key}",
            )
            if candidate.get("integration_id") != integration_id:
                raise ValueError(
                    f"{fold_id} weight {weight_key} integration_id differs from the exact integrated candidate family"
                )
            if candidate.get("candidate_fingerprint") != fingerprint:
                raise ValueError(
                    f"{fold_id} weight {weight_key} candidate_fingerprint differs from the baseline candidate order"
                )
    assert weight_keys is not None
    return integration_id, positive_column, folds, weight_keys


def _validate_external_contract(
    *,
    manifest: dict[str, Any],
    lock: dict[str, Any],
    selection_lock_sha256: str,
) -> None:
    if lock.get("protocol") != "exact_integrated_weight_selection_lock_v1":
        raise ValueError("selection lock has an unexpected protocol")
    if manifest.get("protocol") != "exact_integrated_external_holdout_v1":
        raise ValueError("external manifest has an unexpected protocol")
    if manifest.get("selection_lock_sha256") != selection_lock_sha256:
        raise ValueError("external selection lock hash does not match")
    if manifest.get("integration_id") != lock.get("integration_id"):
        raise ValueError("external integration_id differs from selection lock")
    if float(manifest.get("selected_weight")) != float(lock.get("selected_weight")):
        raise ValueError("external selected weight differs from selection lock")
    candidate = manifest.get("candidate")
    _validate_artifact_descriptor(candidate, "external candidate")
    _validate_artifact_descriptor(manifest.get("baseline"), "external baseline")
    if candidate.get("integration_id") != lock.get("integration_id"):
        raise ValueError("external candidate integration_id differs from selection lock")
    if float(candidate.get("weight")) != float(lock.get("selected_weight")):
        raise ValueError("external candidate weight differs from selection lock")
    if candidate.get("candidate_fingerprint") != manifest.get("candidate_fingerprint"):
        raise ValueError("external candidate_fingerprint differs from the baseline order")
    train_max = manifest["training_time_max"]
    score_min = manifest["score_time_min"]
    score_max = manifest["score_time_max"]
    minimum_gap = manifest["minimum_train_to_score_gap"]
    if minimum_gap <= 0:
        raise ValueError("external minimum gap must be positive")
    if not train_max < score_min <= score_max:
        raise ValueError("external score interval must follow training")
    if score_min - train_max < minimum_gap:
        raise ValueError("external train-to-score gap is shorter than the frozen minimum")


def _selection_key(weight_report: dict[str, Any]) -> tuple[float, ...]:
    stability = weight_report["stability"]
    pooled_delta = weight_report["pooled"]["delta_candidate_minus_baseline"]
    return (
        stability["worst_fold_mrr_delta"],
        stability["median_fold_mrr_delta"],
        pooled_delta["mrr"],
        -float(weight_report["weight"]),
    )


def _metric_deltas(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    return {name: float(candidate[name] - baseline[name]) for name in _METRIC_NAMES}


def _positive_ranks(
    scores: np.ndarray,
    positive_candidate_column: int,
) -> np.ndarray:
    _validate_positive_column(
        positive_candidate_column,
        candidate_count=scores.shape[1],
    )
    positive = scores[
        :,
        positive_candidate_column : positive_candidate_column + 1,
    ]
    return 1 + np.sum(scores > positive, axis=1)


def _validate_positive_column(
    positive_candidate_column: int,
    *,
    candidate_count: int,
) -> None:
    if not 0 <= positive_candidate_column < candidate_count:
        raise ValueError("positive_candidate_column is outside score matrix")


def _validated_scores(
    scores: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(f"{label} must be a non-empty 2D score matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain only finite values")
    return values


def _load_score_artifact(
    descriptor: dict[str, Any],
    *,
    manifest_dir: Path,
    label: str,
) -> np.ndarray:
    _validate_artifact_descriptor(descriptor, label)
    path = Path(descriptor["path"])
    if not path.is_absolute():
        path = manifest_dir / path
    actual_sha256 = _sha256(path)
    if actual_sha256 != descriptor["sha256"]:
        raise ValueError(f"{label} hash mismatch: actual={actual_sha256} expected={descriptor['sha256']}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return _validated_scores(values, label=label)


def _validate_artifact_descriptor(
    descriptor: Any,
    label: str,
) -> None:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{label} artifact descriptor must be an object")
    if not isinstance(descriptor.get("path"), str):
        raise ValueError(f"{label} artifact requires a path")
    sha256 = descriptor.get("sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError(f"{label} artifact requires a lowercase SHA-256")


def _validate_weight_keys(weight_keys: list[str]) -> None:
    values = [float(key) for key in weight_keys]
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values) or len(values) != len(set(values)):
        raise ValueError("candidate weights must be unique finite values in [0, 1]")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
