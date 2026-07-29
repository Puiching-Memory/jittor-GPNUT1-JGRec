from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.cooccur_lift_successor_external import (
    validate_successor_external_setup,
)
from jgrec.standard_validation_protocol import _validate_external_contract


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the standard external contract and score artifacts "
            "without creating an open receipt or computing ranking metrics."
        )
    )
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--materialization-report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if args.state_dir.exists():
        raise FileExistsError(
            f"external state exists before one-shot open: {args.state_dir}"
        )
    setup = validate_successor_external_setup(
        candidate_config_path=args.candidate_config,
        selection_lock_path=args.selection_lock,
    )
    selection_lock = _read_json(args.selection_lock)
    manifest = _read_json(args.manifest)
    materialization = _read_json(args.materialization_report)
    _validate_external_contract(
        manifest=manifest,
        selection_lock=selection_lock,
        selection_lock_sha256=_sha256(args.selection_lock),
    )
    if (
        materialization.get("status")
        != "external_candidate_materialized_metrics_unread"
        or materialization.get("external_ranking_metrics_computed") is not False
        or materialization.get("external_evaluator_invoked") is not False
        or materialization.get("external_manifest_sha256")
        != _sha256(args.manifest)
        or materialization.get("decision_role") != "safety_gate_only"
        or materialization.get("effect_size_estimation_authorized") is not False
    ):
        raise ValueError("external materialization evidence differs")
    support = manifest.get("short_window_support")
    if support != {
        "collapsed_fraction": 0.0,
        "supported_rows": 19_981,
        "total_rows": 19_981,
        "unique_values": [1],
    }:
        raise ValueError("external support is not the frozen zero-collapse state")

    shapes: dict[str, list[int]] = {}
    for name in ("baseline", "candidate"):
        descriptor = manifest[name]
        path = Path(descriptor["path"])
        if _sha256(path) != descriptor["sha256"]:
            raise ValueError(f"{name} score hash differs")
        scores = np.load(path, mmap_mode="r", allow_pickle=False)
        if scores.shape != (19_981, 100):
            raise ValueError(f"{name} external score shape differs")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{name} external scores are non-finite")
        maximum_row_sum_error = float(
            np.max(np.abs(np.asarray(scores).sum(axis=1) - 1.0))
        )
        if maximum_row_sum_error > 5e-6:
            raise ValueError(f"{name} external scores are not normalized")
        shapes[name] = list(scores.shape)

    report = {
        "schema_version": 1,
        "protocol": "cooccur_lift_successor_v2_external_preflight_v1",
        "status": "ready_for_exactly_one_external_open",
        "decision_role": "safety_gate_only",
        "effect_size_estimation_authorized": False,
        "selection_lock_sha256": setup.selection_lock_sha256,
        "candidate_id": setup.candidate_id,
        "candidate_config_sha256": setup.config_sha256,
        "external_manifest_sha256": _sha256(args.manifest),
        "external_state_absent": True,
        "external_support_collapsed_fraction": 0.0,
        "score_shapes": shapes,
        "ranking_metrics_computed": False,
        "maximum_external_opens": 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
