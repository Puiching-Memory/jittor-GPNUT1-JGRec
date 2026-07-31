from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.rankers.hybrid.time_ramp import (
    apply_time_ramp,
    passes_time_ramp_gate,
)

SLICE_STOPS = (6_667, 13_334)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--selection-report-sha256", required=True, type=Path)
    parser.add_argument("--validation-expert-scores", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--minimum-full-delta", type=float, default=0.001)
    args = parser.parse_args()

    if args.output_report.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_report}")
    selection_report = _read_json(args.selection_report)
    if not selection_report.get("gate_passed"):
        raise RuntimeError("selection did not unlock the slice2 gate")
    expected_report_hash = (
        args.selection_report_sha256.read_text(encoding="ascii").split()[0]
    )
    if _sha256(args.selection_report) != expected_report_hash:
        raise ValueError("selection report hash differs from sidecar")
    expected_scores_hash = selection_report["artifacts"][
        "validation_expert_scores_sha256"
    ]
    if _sha256(args.validation_expert_scores) != expected_scores_hash:
        raise ValueError("validation expert scores differ from selection report")

    payload = np.load(args.validation_expert_scores, allow_pickle=False)
    champion = np.asarray(payload["champion"], dtype=np.float64)
    setwise = np.asarray(payload["setwise"], dtype=np.float64)
    query_times = np.asarray(payload["query_times"], dtype=np.int64)
    power_value = selection_report["selection"]["selected_power"]
    if power_value is None:
        raise RuntimeError("selection report has no selected power")
    power = float(power_value)
    candidate = apply_time_ramp(
        champion,
        setwise,
        query_times,
        power=power,
    )
    result = passes_time_ramp_gate(
        champion,
        candidate,
        slice_stops=SLICE_STOPS,
        minimum_full_delta=args.minimum_full_delta,
    )
    selected_trial = next(
        trial
        for trial in selection_report["selection"]["trials"]
        if float(trial["power"]) == power
    )
    for index in range(2):
        if (
            abs(
                float(result.slice_deltas[index])
                - float(selected_trial["slice_deltas"][index])
            )
            > 1e-12
        ):
            raise RuntimeError("gate cannot reproduce selected prefix metrics")

    report = {
        "status": "passed" if result.passed else "rejected",
        "gate_passed": result.passed,
        "package_authorized": result.passed,
        "package_generated": False,
        "selection_report_sha256": expected_report_hash,
        "selected_power": power,
        "gate": asdict(result),
        "decision": (
            "continue_to_production_package"
            if result.passed
            else "stop_dataset1_time_ramp"
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


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
