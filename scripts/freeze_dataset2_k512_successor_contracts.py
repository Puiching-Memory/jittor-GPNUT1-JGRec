from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from jgrec.rankers.hybrid.full100_training import (
    validate_joint_cache_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze fresh K512 cooccur-lift V1, duel, and external "
            "execution contracts before each metric-bearing stage."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    v1 = subparsers.add_parser("bugfixed-v1")
    v1.add_argument("--base-contract", required=True, type=Path)
    v1.add_argument("--frozen-config", required=True, type=Path)
    v1.add_argument("--selection-lock", required=True, type=Path)
    v1.add_argument("--source-checkpoint", required=True, type=Path)
    v1.add_argument("--train-cache-report", required=True, type=Path)
    v1.add_argument(
        "--validation-cache-report",
        required=True,
        type=Path,
    )
    v1.add_argument("--train-lift-features", required=True, type=Path)
    v1.add_argument("--train-short-none", required=True, type=Path)
    v1.add_argument("--fusion-source", required=True, type=Path)
    v1.add_argument("--output", required=True, type=Path)

    duel = subparsers.add_parser("duel")
    duel.add_argument("--base-contract", required=True, type=Path)
    duel.add_argument("--validation-plan", required=True, type=Path)
    duel.add_argument("--plan-lock", required=True, type=Path)
    duel.add_argument("--bugfixed-v1-contract", required=True, type=Path)
    duel.add_argument("--near-v1-manifest", required=True, type=Path)
    duel.add_argument("--near-cache-report", required=True, type=Path)
    duel.add_argument("--gapped-cache-report", required=True, type=Path)
    duel.add_argument("--runner", required=True, type=Path)
    duel.add_argument("--execution-module", required=True, type=Path)
    duel.add_argument("--fusion-source", required=True, type=Path)
    duel.add_argument("--pipeline-script", required=True, type=Path)
    duel.add_argument("--output", required=True, type=Path)

    external = subparsers.add_parser("external")
    external.add_argument("--base-contract", required=True, type=Path)
    external.add_argument("--candidate-config", required=True, type=Path)
    external.add_argument("--selection-lock", required=True, type=Path)
    external.add_argument("--bugfixed-v1-contract", required=True, type=Path)
    external.add_argument(
        "--bugfixed-v1-training-report",
        required=True,
        type=Path,
    )
    external.add_argument("--bugfixed-v1-model", required=True, type=Path)
    external.add_argument("--source-checkpoint", required=True, type=Path)
    external.add_argument("--train-cache-report", required=True, type=Path)
    external.add_argument(
        "--validation-cache-report",
        required=True,
        type=Path,
    )
    external.add_argument("--train-lift-features", required=True, type=Path)
    external.add_argument("--train-short-none", required=True, type=Path)
    external.add_argument(
        "--validation-short-none",
        required=True,
        type=Path,
    )
    external.add_argument("--train-csv", required=True, type=Path)
    external.add_argument(
        "--prior-external-probabilities",
        required=True,
        type=Path,
    )
    external.add_argument("--materializer-script", required=True, type=Path)
    external.add_argument("--external-module", required=True, type=Path)
    external.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "bugfixed-v1":
        payload = _build_bugfixed_v1_contract(args)
    elif args.command == "duel":
        payload = _build_duel_contract(args)
    else:
        payload = _build_external_contract(args)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": "frozen",
                "command": args.command,
                "output": str(args.output.resolve()),
                "sha256": _sha256(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _build_bugfixed_v1_contract(
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = _read_json(args.base_contract)
    train_report = _read_json(args.train_cache_report)
    validation_report = _read_json(args.validation_cache_report)
    _validate_k512_cache_reports(train_report, validation_report)
    joint_validation = validate_joint_cache_reports(
        train_report,
        validation_report,
    )
    train_joint = _require_mapping(train_report, "joint_build")
    artifacts = _require_mapping(train_report, "artifacts")
    feature = _require_mapping(artifacts, "features")

    payload = copy.deepcopy(base)
    payload["candidate_id"] = (
        "cooccur_lift_aux_expert_v1_k512_weighted_normalizer_refit_20260729"
    )
    payload["purpose"] = (
        "Rebuild unchanged V1 on a fresh K512 200k/20k cache after the "
        "weighted listwise normalizer correction."
    )
    payload["parent_contract_sha256"] = _sha256(args.base_contract)
    payload["lineage"] = {
        "cache": "fresh_k512_joint_200k_train_20k_validation",
        "normalizer": "row_weighted_mean_and_std",
        "external_scores_read": False,
        "joint_cache_validation": joint_validation,
    }
    training_assets = _require_mapping(payload, "training_assets")
    training_assets.update(
        {
            "frozen_config": str(args.frozen_config.resolve()),
            "frozen_config_sha256": _sha256(args.frozen_config),
            "selection_lock": str(args.selection_lock.resolve()),
            "selection_lock_sha256": _sha256(args.selection_lock),
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "source_checkpoint_sha256": _sha256(
                args.source_checkpoint
            ),
            "train_cache_report": str(
                args.train_cache_report.resolve()
            ),
            "train_cache_report_sha256": _sha256(
                args.train_cache_report
            ),
            "validation_cache_report": str(
                args.validation_cache_report.resolve()
            ),
            "validation_cache_report_sha256": _sha256(
                args.validation_cache_report
            ),
            "train_feature_sha256": str(feature["sha256"]),
            "train_lift_features": str(
                args.train_lift_features.resolve()
            ),
            "train_lift_features_sha256": _sha256(
                args.train_lift_features
            ),
            "train_short_none": str(args.train_short_none.resolve()),
            "train_short_none_sha256": _sha256(
                args.train_short_none
            ),
            "joint_cache_provenance_required": True,
            "joint_build_id": str(train_joint["id"]),
            "joint_build_pid": int(train_joint["pid"]),
        }
    )
    implementation = _require_mapping(
        payload,
        "implementation_contract",
    )
    implementation["fusion_source"] = str(args.fusion_source.resolve())
    implementation["fusion_source_sha256"] = _sha256(
        args.fusion_source
    )
    implementation["weighted_normalizer_required"] = True
    implementation["weighted_normalizer_bug_bypass_authorized"] = False
    return payload


def _build_duel_contract(args: argparse.Namespace) -> dict[str, Any]:
    payload = copy.deepcopy(_read_json(args.base_contract))
    near_report = _read_json(args.near_cache_report)
    limits = _require_mapping(near_report, "prediction_limits")
    if (
        limits.get("structure_predict_neighbor_limit") != 512
        or limits.get("source_profile_predict_history_limit") != 512
    ):
        raise ValueError("near cache report is not K512")
    payload["amendment_reason"] = (
        "Exact execution replay after the weighted listwise normalizer fix "
        "and a fresh K512 200k cache. Candidate shapes, folds, weights, "
        "seeds, capacities, tolerances, gates, and tie-breaks are unchanged."
    )
    payload["baseline_execution"]["full_origin_online_score"] = None
    payload["baseline_execution"][
        "full_origin_external_role"
    ] = "not_read_during_internal_selection"
    payload.update(
        {
            "validation_plan_sha256": _sha256(args.validation_plan),
            "plan_lock_sha256": _sha256(args.plan_lock),
            "bugfixed_v1_contract_sha256": _sha256(
                args.bugfixed_v1_contract
            ),
            "historical_near_v1_manifest_sha256": _sha256(
                args.near_v1_manifest
            ),
            "near_cache_report_sha256": _sha256(
                args.near_cache_report
            ),
            "near_cache_checkpoint_sha256": str(
                near_report["checkpoint_sha256"]
            ),
            "gapped_cache_report_sha256": _sha256(
                args.gapped_cache_report
            ),
            "runner_sha256": _sha256(args.runner),
            "execution_module_sha256": _sha256(
                args.execution_module
            ),
            "fusion_source_sha256": _sha256(args.fusion_source),
            "pipeline_script_sha256": _sha256(
                args.pipeline_script
            ),
            "weighted_normalizer_required": True,
            "weighted_normalizer_bug_bypass_authorized": False,
            "external_authorized": False,
        }
    )
    return payload


def _build_external_contract(
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = copy.deepcopy(_read_json(args.base_contract))
    payload.pop("post_gate_sha256", None)
    payload.pop("package_contract", None)
    paths = {
        "candidate_config": args.candidate_config,
        "selection_lock": args.selection_lock,
        "bugfixed_v1_contract": args.bugfixed_v1_contract,
        "bugfixed_v1_training_report": (
            args.bugfixed_v1_training_report
        ),
        "bugfixed_v1_model": args.bugfixed_v1_model,
        "source_checkpoint": args.source_checkpoint,
        "train_cache_report": args.train_cache_report,
        "validation_cache_report": args.validation_cache_report,
        "train_lift_features": args.train_lift_features,
        "train_short_none": args.train_short_none,
        "validation_short_none": args.validation_short_none,
        "train_csv": args.train_csv,
        "prior_external_probabilities": (
            args.prior_external_probabilities
        ),
        "materializer_script": args.materializer_script,
        "external_module": args.external_module,
    }
    payload["external_authorization_source"] = (
        "explicit_user_instruction_automatic_k512_pipeline_20260729"
    )
    payload["input_sha256"] = {
        name: _sha256(path) for name, path in paths.items()
    }
    payload["lineage"] = {
        "cache": "fresh_k512_joint_200k_train_20k_validation",
        "normalizer": "row_weighted_mean_and_std",
        "selection_lock_sha256": _sha256(args.selection_lock),
        "external_role": "safety_gate_only",
    }
    payload["tolerance_relaxation_authorized"] = False
    payload["external_authorized"] = True
    payload["maximum_external_opens"] = 1
    prohibited = payload.get("prohibited")
    if isinstance(prohibited, list):
        payload["prohibited"] = [
            value
            for value in prohibited
            if value != "package generation after any failed external gate"
        ]
        payload["prohibited"].append(
            "submission package generation by this automatic pipeline"
        )
    return payload


def _validate_k512_cache_reports(
    train_report: dict[str, Any],
    validation_report: dict[str, Any],
) -> None:
    for role, report in (
        ("train", train_report),
        ("validation", validation_report),
    ):
        if report.get("status") != "complete":
            raise ValueError(f"{role} cache report is incomplete")
        limits = _require_mapping(report, "prediction_limits")
        if (
            limits.get("structure_predict_neighbor_limit") != 512
            or limits.get("source_profile_predict_history_limit") != 512
        ):
            raise ValueError(f"{role} cache report is not K512")
    if (
        train_report.get("checkpoint_sha256")
        != validation_report.get("checkpoint_sha256")
    ):
        raise ValueError("joint cache checkpoint differs")


def _require_mapping(
    value: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    result = value.get(name)
    if not isinstance(result, dict):
        raise ValueError(f"missing object: {name}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
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
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
