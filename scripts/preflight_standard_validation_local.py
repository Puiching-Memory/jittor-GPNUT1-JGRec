from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SECONDS_PER_DAY = 24 * 60 * 60
EXPECTED_DEPLOYMENT_DAYS = 468
TIME_LOCAL_SHORT_WINDOW_SECONDS = 17_038_080
TIME_LOCAL_COLLAPSED_ROW_FRACTION = 0.39971972363446745
TIME_LOCAL_GAPPED_SPECS = (
    ("gapped-p75", 0.75, 251 * SECONDS_PER_DAY),
    ("gapped-p90", 0.90, 308 * SECONDS_PER_DAY),
    ("gapped-p100", 1.00, 349 * SECONDS_PER_DAY),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the local Dataset2 rolling/external metadata needed by "
            "the standard validation protocol without opening score arrays "
            "or external labels."
        )
    )
    parser.add_argument(
        "--rolling-report",
        type=Path,
        default=Path(
            "result/dataset2_listwise_mlp_exact_rolling_20260728/"
            "artifacts/rolling-training-report.json"
        ),
    )
    parser.add_argument(
        "--external-preflight",
        type=Path,
        default=Path(
            "result/dataset2_partial_listwise_expert_blend_20260728/"
            "preflight-report.json"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    rolling = _read_json(args.rolling_report)
    external = _read_json(args.external_preflight)
    folds = rolling.get("folds")
    if not isinstance(folds, list) or len(folds) < 3:
        raise ValueError("local rolling report requires at least three folds")
    fold_audit = []
    previous_score_max: int | None = None
    for fold in folds:
        artifact = fold["score_artifacts"]
        train_max = int(artifact["train_time_max"])
        score_min = int(artifact["score_time_min"])
        score_max = int(artifact["score_time_max"])
        if not train_max < score_min <= score_max:
            raise ValueError(f"{fold['fold_id']} is not causal")
        if previous_score_max is not None and score_min <= previous_score_max:
            raise ValueError("rolling score windows overlap")
        previous_score_max = score_max
        fold_audit.append(
            {
                "fold_id": fold["fold_id"],
                "train_time_max": train_max,
                "score_time_min": score_min,
                "score_time_max": score_max,
                "score_span_days": (
                    (score_max - score_min) / SECONDS_PER_DAY
                ),
                "candidate_count": len(artifact["candidates"]),
            }
        )

    validation = external["validation"]
    external_min, external_max = (
        int(value) for value in validation["time_range"]
    )
    final_origin = int(folds[-1]["score_artifacts"]["score_time_max"])
    external_horizon_seconds = external_max - final_origin
    expected_horizon_seconds = (
        EXPECTED_DEPLOYMENT_DAYS * SECONDS_PER_DAY
    )
    if external_min != final_origin:
        raise ValueError(
            "external validation does not start at the final rolling origin"
        )
    if external_horizon_seconds < expected_horizon_seconds:
        raise ValueError(
            "external metadata does not cover the 468-day deployment horizon"
        )
    if external.get("metrics_read") is not False:
        raise ValueError("external preflight already consumed metrics")

    report = {
        "schema_version": 1,
        "protocol": "standard_validation_local_preflight_v1",
        "status": (
            "ready_for_nonlocal_candidate_preregistration_"
            "far_horizon_pending_for_time_local"
        ),
        "rolling_report": str(args.rolling_report.resolve()),
        "rolling_report_sha256": _sha256(args.rolling_report),
        "rolling_selection": {
            "fold_count": len(folds),
            "minimum_fold_count_passed": len(folds) >= 3,
            "aggregation": "equal_weight_fold_mean",
            "folds": fold_audit,
        },
        "external_gate": {
            "holdout_id": "dataset2_external_20k_v1",
            "lineage_sha256": external["hashes"]["validation_time"],
            "score_time_min": external_min,
            "score_time_max": external_max,
            "reference_origin_time": final_origin,
            "actual_horizon_seconds": external_horizon_seconds,
            "actual_horizon_days": (
                external_horizon_seconds / SECONDS_PER_DAY
            ),
            "minimum_horizon_seconds": expected_horizon_seconds,
            "minimum_horizon_days": EXPECTED_DEPLOYMENT_DAYS,
            "historically_reused_holdout": True,
            "statistical_independence": "limited",
        },
        "time_local_validation": {
            "status": "far_horizon_folds_required_before_preregistration",
            "ready_for_time_local_candidate_preregistration": False,
            "short_window_seconds": TIME_LOCAL_SHORT_WINDOW_SECONDS,
            "short_window_days": (
                TIME_LOCAL_SHORT_WINDOW_SECONDS / SECONDS_PER_DAY
            ),
            "deployment_collapsed_row_fraction": (
                TIME_LOCAL_COLLAPSED_ROW_FRACTION
            ),
            "current_gapped_fold_count": 0,
            "required_gapped_fold_specs": [
                {
                    "fold_id": fold_id,
                    "deployment_horizon_quantile": quantile,
                    "minimum_gap_seconds": gap_seconds,
                    "minimum_gap_days": gap_seconds / SECONDS_PER_DAY,
                }
                for fold_id, quantile, gap_seconds in TIME_LOCAL_GAPPED_SPECS
            ],
            "zero_short_counterfactual": {
                "optional": True,
                "participates_in_selection": False,
            },
            "external_interpretation": {
                "decision_role": "safety_gate_only",
                "effect_size_estimation_authorized": False,
                "calibration_discount_factor": 19.5,
            },
        },
        "candidate_preregistered": False,
        "selection_metrics_read": False,
        "reserved_fold_metrics_read": False,
        "external_holdout_read_by_this_preflight": False,
        "selection_lock_created": False,
        "external_open_receipt_created": False,
        "package_authorized": False,
        "next_action": (
            "for a time-local candidate, materialize the preregistered "
            "gapped folds before freezing any successor-v2 candidate plan"
        ),
    }
    args.output_dir.mkdir(parents=True)
    _write_json_exclusive(
        args.output_dir / "preflight-report.json",
        report,
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_exclusive(
    path: Path,
    payload: dict[str, Any],
) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
