from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.rankers.hybrid.config import TwoTowerConfig
from jgrec.rankers.hybrid.tower_optimization_experiment import (
    paired_rank_movements,
    positive_ranks,
    ranking_metrics,
    two_tower_screen_config,
    two_tower_screen_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebase Dataset2 Two-Tower screen arms onto a same-code control."
        )
    )
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument(
        "--candidate-dir",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    control_report = _read_json(
        args.control_dir / "evaluation-report.json"
    )
    if control_report["arm"] != "control":
        raise ValueError("--control-dir must contain the matched control arm")
    control_scores = np.load(
        args.control_dir / "candidate-scores.npy",
        allow_pickle=False,
    )
    slices = tuple(
        slice(int(start), int(stop))
        for start, stop in control_report["frozen_config"][
            "validation_slices"
        ]
    )
    if len(slices) != 3:
        raise ValueError("the frozen Stage-1 screen must have three slices")
    control_evaluation = _evaluate(control_scores, slices)
    control_config = TwoTowerConfig(
        **control_report["frozen_config"]["candidate_config"]
    )

    arms: dict[str, Any] = {}
    passing: list[str] = []
    for candidate_dir in args.candidate_dir:
        candidate_report = _read_json(
            candidate_dir / "evaluation-report.json"
        )
        arm = str(candidate_report["arm"])
        if arm in arms or arm == "control":
            raise ValueError(f"duplicate or invalid candidate arm: {arm}")
        expected_config = asdict(
            two_tower_screen_config(control_config, arm)
        )
        actual_config = candidate_report["frozen_config"][
            "candidate_config"
        ]
        if actual_config != expected_config:
            raise ValueError(
                f"candidate arm {arm} differs from its frozen factors"
            )
        candidate_scores = np.load(
            candidate_dir / "candidate-scores.npy",
            allow_pickle=False,
        )
        if candidate_scores.shape != control_scores.shape:
            raise ValueError(f"candidate arm {arm} score shape differs")
        candidate_evaluation = _evaluate(candidate_scores, slices)
        movements = paired_rank_movements(
            positive_ranks(control_scores),
            positive_ranks(candidate_scores),
        )
        slice_movements = [
            paired_rank_movements(
                positive_ranks(control_scores[part]),
                positive_ranks(candidate_scores[part]),
            )
            for part in slices
        ]
        gate = two_tower_screen_gate(
            control_evaluation["full"],
            candidate_evaluation["full"],
            control_evaluation["slices"],
            candidate_evaluation["slices"],
            movements,
        )
        if gate["passed"]:
            passing.append(arm)
        arms[arm] = {
            "evaluation": candidate_evaluation,
            "delta": {
                "full": _metric_delta(
                    candidate_evaluation["full"],
                    control_evaluation["full"],
                ),
                "slices": [
                    _metric_delta(candidate_part, control_part)
                    for control_part, candidate_part in zip(
                        control_evaluation["slices"],
                        candidate_evaluation["slices"],
                        strict=True,
                    )
                ],
            },
            "query_movements": movements,
            "slice_query_movements": slice_movements,
            "gate": gate,
            "score_sha256": _sha256_array(candidate_scores),
            "source_report_sha256": _sha256_file(
                candidate_dir / "evaluation-report.json"
            ),
        }

    selected = passing[0] if len(passing) == 1 else None
    report = {
        "status": "complete",
        "comparison": "same-code same-seed matched control",
        "control": {
            "evaluation": control_evaluation,
            "score_sha256": _sha256_array(control_scores),
            "config": asdict(control_config),
            "source_report_sha256": _sha256_file(
                args.control_dir / "evaluation-report.json"
            ),
        },
        "arms": arms,
        "passing_arms": passing,
        "selected_arm": selected,
        "selection_status": (
            "selected"
            if selected is not None
            else "no_pass"
            if not passing
            else "ambiguous_multiple_pass"
        ),
        "exact_integrated_rolling_authorized": selected is not None,
        "external_opened": False,
        "package_generated": False,
        "policy": {
            "exactly_one_passing_arm_required": True,
            "no_selection_by_mrr_only": True,
            "historical_control_reports_are_diagnostic_only": True,
        },
    }
    _write_json(args.output_dir / "matched-screen-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _evaluate(
    scores: np.ndarray,
    slices: tuple[slice, ...],
) -> dict[str, Any]:
    return {
        "full": ranking_metrics(scores),
        "slices": [ranking_metrics(scores[part]) for part in slices],
        "score_shape": list(scores.shape),
    }


def _metric_delta(
    candidate: dict[str, float],
    control: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key] - control[key])
        for key in control
    }


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
