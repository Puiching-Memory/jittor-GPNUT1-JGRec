from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.rankers.hybrid.partial_listwise_blend import (
    choose_forward_winner,
    evaluate_final_gate,
    evaluate_forward_gate,
    scan_auxiliary_weights,
    select_auxiliary_weight,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen slice0 -> slice1 -> single-winner slice2 "
            "partial-listwise blend protocol."
        )
    )
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--score-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_paths = {
        "mlp_scan": args.output_dir / "listwise-mlp-slice0-scan.json",
        "tower_scan": (
            args.output_dir / "listwise-two-tower-slice0-scan.json"
        ),
        "mlp_selection": args.output_dir / "listwise-mlp-selection.json",
        "tower_selection": (
            args.output_dir / "listwise-two-tower-selection.json"
        ),
        "mlp_forward": args.output_dir / "listwise-mlp-forward-gate.json",
        "tower_forward": (
            args.output_dir / "listwise-two-tower-forward-gate.json"
        ),
        "winner": args.output_dir / "winner-lock.json",
        "evaluation": args.output_dir / "evaluation-report.json",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite selection artifacts: {existing}"
        )
    frozen = _read_json(args.frozen_config)
    score_report = _read_json(args.score_report)
    if frozen.get("status") != "frozen_before_any_partial_blend_metric":
        raise ValueError("selection config was not frozen before metrics")
    if (
        score_report.get("status") != "passed"
        or not score_report.get("champion_reproduced")
    ):
        raise ValueError("aligned score report did not reproduce the champion")

    score_artifacts = score_report["score_artifacts"]
    score_paths = {
        name: Path(details["path"])
        for name, details in score_artifacts.items()
    }
    for name, path in score_paths.items():
        actual = _sha256_file(path)
        expected = score_artifacts[name]["sha256"]
        if actual != expected:
            raise ValueError(
                f"{name} score hash mismatch: {actual} != {expected}"
            )
    champion = np.load(
        score_paths["champion"],
        mmap_mode="r",
        allow_pickle=False,
    )
    experts = {
        "listwise_mlp": np.load(
            score_paths["listwise_mlp"],
            mmap_mode="r",
            allow_pickle=False,
        ),
        "listwise_two_tower": np.load(
            score_paths["listwise_two_tower"],
            mmap_mode="r",
            allow_pickle=False,
        ),
    }
    if any(expert.shape != champion.shape for expert in experts.values()):
        raise ValueError("aligned expert score shapes differ")
    if list(champion.shape) != score_report["candidate_shape"]:
        raise ValueError("score shape differs from the frozen manifest")

    weights = tuple(float(value) for value in frozen["weights"])
    slices = tuple(
        tuple(int(value) for value in frozen["slices"][f"slice_{index}"])
        for index in range(3)
    )
    candidate_manifest_sha256 = score_report[
        "candidate_manifest_sha256"
    ]
    selections: dict[str, dict[str, Any]] = {}
    selection_errors: dict[str, str] = {}
    for expert_name, expert in experts.items():
        scan = scan_auxiliary_weights(
            champion_slice0=champion[slice(*slices[0])],
            expert_slice0=expert[slice(*slices[0])],
            weights=weights,
        )
        scan["expert_name"] = expert_name
        scan["candidate_manifest_sha256"] = candidate_manifest_sha256
        _write_json(
            (
                output_paths["mlp_scan"]
                if expert_name == "listwise_mlp"
                else output_paths["tower_scan"]
            ),
            scan,
        )
        try:
            selection = select_auxiliary_weight(
                expert_name=expert_name,
                champion_slice0=champion[slice(*slices[0])],
                expert_slice0=expert[slice(*slices[0])],
                candidate_manifest_sha256=candidate_manifest_sha256,
                weights=weights,
            )
        except ValueError as error:
            selection_errors[expert_name] = str(error)
            continue
        selections[expert_name] = selection
        _write_json(
            (
                output_paths["mlp_selection"]
                if expert_name == "listwise_mlp"
                else output_paths["tower_selection"]
            ),
            selection,
        )

    forward_reports: list[dict[str, Any]] = []
    for expert_name, selection in selections.items():
        expert = experts[expert_name]
        report = evaluate_forward_gate(
            selection=selection,
            champion_slice0=champion[slice(*slices[0])],
            expert_slice0=expert[slice(*slices[0])],
            champion_slice1=champion[slice(*slices[1])],
            expert_slice1=expert[slice(*slices[1])],
            candidate_manifest_sha256=candidate_manifest_sha256,
            minimum_prefix_delta=float(
                frozen["forward_gate"][
                    "combined_slice_0_slice_1_delta_min"
                ]
            ),
        )
        forward_reports.append(report)
        _write_json(
            (
                output_paths["mlp_forward"]
                if expert_name == "listwise_mlp"
                else output_paths["tower_forward"]
            ),
            report,
        )

    passed_forward = [
        report for report in forward_reports if report["passed"]
    ]
    if not passed_forward:
        report = {
            "status": "rejected_before_slice_2",
            "gate_passed": False,
            "slice_2_metrics_read": False,
            "selection_errors": selection_errors,
            "forward_reports": forward_reports,
            "package_authorized": False,
        }
        _write_json(output_paths["evaluation"], report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    winner = choose_forward_winner(passed_forward)
    winner["schema_version"] = 1
    winner["candidate_manifest_sha256"] = candidate_manifest_sha256
    winner["winner_lock_sha256"] = _canonical_sha256(winner)
    _write_json(output_paths["winner"], winner)

    winner_name = winner["expert_name"]
    final_report = evaluate_final_gate(
        selection=selections[winner_name],
        forward_report=next(
            report
            for report in passed_forward
            if report["expert_name"] == winner_name
        ),
        champion_scores=champion,
        expert_scores=experts[winner_name],
        slices=slices,
        expected_selection_lock_sha256=winner[
            "selection_lock_sha256"
        ],
        minimum_full_delta=float(
            frozen["final_gate"]["minimum_full_delta"]
        ),
    )
    report = {
        "status": "passed" if final_report["passed"] else "rejected",
        "gate_passed": bool(final_report["passed"]),
        "slice_2_metrics_read": True,
        "selection_errors": selection_errors,
        "forward_reports": forward_reports,
        "winner": winner,
        "final_gate": final_report,
        "package_authorized": bool(final_report["passed"]),
    }
    _write_json(output_paths["evaluation"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if final_report["passed"] else 2


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
