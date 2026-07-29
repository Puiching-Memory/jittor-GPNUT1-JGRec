from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.rankers.hybrid.conservative_window_blend import (
    conservative_window_scores,
    evaluate_conservative_window_gate,
    select_conservative_window_blend_on_prefix,
)
from jgrec.rankers.hybrid.window_diversity import blend_expert_subset

ALPHAS = (0.05, 0.10, 0.20, 0.30)
FIRST_SLICE_STOP = 6_667
SELECTION_STOP = 13_334
VALIDATION_ROWS = 20_000
SETWISE_WEIGHT = 0.80
MINIMUM_PREFIX_DELTA = 0.0001
MINIMUM_FULL_DELTA = 0.0002
SELECTED_EXPERTS = (
    "recent100k",
    "recent200k",
    "recent200k_decay100k",
)
EXPECTED_DATASET1_SHA256 = (
    "81285eddfba612b9c05e96075be29a15718d8a94839d143e3083bf5353644369"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select or gate a conservative Dataset2 window residual."
    )
    parser.add_argument("phase", choices=("select", "gate"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            "result/dataset2_setwise_window_diversity_20260726"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "result/dataset2_conservative_window_blend_20260726"
        ),
    )
    parser.add_argument(
        "--frozen-dataset1-csv",
        type=Path,
        default=Path(
            "result/d1_time_ramp_g050_d2_setwise_w080_seed60_20260726/"
            "csv/dataset1.csv"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.phase == "select":
        return _select(args)
    return _gate(args)


def _select(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = args.output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output_dir / "frozen-config.json"
    selection_path = args.output_dir / "selection-report.json"
    if frozen_path.exists() or selection_path.exists():
        raise FileExistsError(
            f"refusing to overwrite conservative selection in {args.output_dir}"
        )

    source = _load_and_verify_source(args.source_dir)
    _require_hash(
        args.frozen_dataset1_csv,
        EXPECTED_DATASET1_SHA256,
        "frozen Dataset1 CSV",
    )
    frozen = {
        "status": "frozen_before_selection",
        "protocol_version": 1,
        "scope": "Dataset2 cached validation probability residual only",
        "formula": (
            "champion + alpha * (fixed_window_candidate - champion)"
        ),
        "alphas": list(ALPHAS),
        "selection_rows": [0, SELECTION_STOP],
        "visible_slices": [
            [0, FIRST_SLICE_STOP],
            [FIRST_SLICE_STOP, SELECTION_STOP],
        ],
        "forward_rows": [SELECTION_STOP, VALIDATION_ROWS],
        "minimum_prefix_delta": MINIMUM_PREFIX_DELTA,
        "minimum_full_delta": MINIMUM_FULL_DELTA,
        "selection_policy": (
            "both visible slice deltas non-negative and prefix delta at least "
            "minimum; then higher prefix MRR; exact tie prefers smaller alpha"
        ),
        "gate_policy": (
            "all three slice deltas non-negative and full delta at least "
            "minimum"
        ),
        "champion": (
            "0.8 * recent200k + 0.2 * LightGBM"
        ),
        "fixed_window_candidate": (
            "0.8 * mean(recent100k,recent200k,"
            "recent200k_decay100k) + 0.2 * LightGBM"
        ),
        "selected_experts": list(SELECTED_EXPERTS),
        "source": source["frozen_source"],
        "frozen_dataset1_csv": str(
            args.frozen_dataset1_csv.resolve()
        ),
        "frozen_dataset1_csv_sha256": EXPECTED_DATASET1_SHA256,
        "package_only_after_gate": True,
    }
    _write_json_atomic(frozen_path, frozen)

    probabilities = _load_probabilities(source)
    champion_scores, window_scores = _build_source_scores(probabilities)
    source_prefix = select_conservative_window_blend_on_prefix(
        champion_scores,
        window_scores,
        alphas=(1.0,),
        first_slice_stop=FIRST_SLICE_STOP,
        selection_stop=SELECTION_STOP,
        minimum_prefix_delta=0.0,
    )
    source_candidate = source_prefix.candidates[0]
    _require_close(
        source_prefix.baseline_prefix_mrr,
        float(source["selection"]["baseline_recent200k_selection_mrr"]),
        "source champion prefix MRR",
    )
    _require_close(
        source_candidate.prefix_mrr,
        float(source["selection"]["selection_mrr"]),
        "source window prefix MRR",
    )

    selection = select_conservative_window_blend_on_prefix(
        champion_scores,
        window_scores,
        alphas=ALPHAS,
        first_slice_stop=FIRST_SLICE_STOP,
        selection_stop=SELECTION_STOP,
        minimum_prefix_delta=MINIMUM_PREFIX_DELTA,
    )
    source_hashes_unchanged = _source_hashes_unchanged(source)
    if not source_hashes_unchanged:
        raise RuntimeError("source artifacts changed during selection")
    selected_alpha = selection.selected_alpha
    report = {
        "status": (
            "locked_before_forward_gate"
            if selected_alpha is not None
            else "no_eligible_candidate"
        ),
        "gate_unlocked": selected_alpha is not None,
        "forward_metrics_read": selection.forward_metrics_read,
        "selected_alpha": selected_alpha,
        "selection_rows": [0, SELECTION_STOP],
        "forward_rows": [SELECTION_STOP, VALIDATION_ROWS],
        "baseline_prefix_mrr": selection.baseline_prefix_mrr,
        "baseline_slice_mrrs": list(selection.baseline_slice_mrrs),
        "minimum_prefix_delta": selection.minimum_prefix_delta,
        "candidates": [asdict(candidate) for candidate in selection.candidates],
        "source_window_prefix_mrr": source_candidate.prefix_mrr,
        "source_window_prefix_delta": source_candidate.prefix_delta,
        "source_hashes_unchanged": source_hashes_unchanged,
        "frozen_config": str(frozen_path.resolve()),
        "frozen_config_sha256": _sha256(frozen_path),
    }
    _write_json_atomic(selection_path, report)
    selection_sha = _sha256(selection_path)
    _write_text_atomic(
        args.output_dir / "selection-report.sha256",
        f"{selection_sha}  selection-report.json\n",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _gate(args: argparse.Namespace) -> int:
    frozen_path = args.output_dir / "frozen-config.json"
    selection_path = args.output_dir / "selection-report.json"
    selection_sha_path = args.output_dir / "selection-report.sha256"
    report_path = args.output_dir / "evaluation-report.json"
    if report_path.exists():
        raise FileExistsError(
            f"refusing to overwrite conservative gate report: {report_path}"
        )
    frozen = _read_json(frozen_path)
    selection = _read_json(selection_path)
    locked_selection_sha = (
        selection_sha_path.read_text(encoding="utf-8").split()[0]
    )
    _require_hash(
        selection_path,
        locked_selection_sha,
        "locked selection report",
    )
    if selection.get("status") != "locked_before_forward_gate":
        raise ValueError("selection report did not unlock the forward gate")
    if selection.get("forward_metrics_read") is not False:
        raise ValueError("selection report read forward metrics")
    if _sha256(frozen_path) != selection["frozen_config_sha256"]:
        raise ValueError("frozen config changed after selection")
    selected_alpha = float(selection["selected_alpha"])
    if selected_alpha not in ALPHAS:
        raise ValueError("locked alpha is outside the frozen grid")

    source = _load_and_verify_source(Path(args.source_dir))
    if source["frozen_source"] != frozen["source"]:
        raise ValueError("source artifacts differ from frozen config")
    _require_hash(
        args.frozen_dataset1_csv,
        frozen["frozen_dataset1_csv_sha256"],
        "frozen Dataset1 CSV",
    )
    probabilities = _load_probabilities(source)
    champion_scores, window_scores = _build_source_scores(probabilities)

    source_gate = evaluate_conservative_window_gate(
        champion_scores,
        window_scores,
        selected_alpha=1.0,
        first_slice_stop=FIRST_SLICE_STOP,
        selection_stop=SELECTION_STOP,
        minimum_full_delta=0.0,
    )
    _require_source_gate_metrics(source_gate, source["evaluation"])
    gate = evaluate_conservative_window_gate(
        champion_scores,
        window_scores,
        selected_alpha=selected_alpha,
        first_slice_stop=FIRST_SLICE_STOP,
        selection_stop=SELECTION_STOP,
        minimum_full_delta=MINIMUM_FULL_DELTA,
    )
    candidate_scores = conservative_window_scores(
        champion_scores,
        window_scores,
        alpha=selected_alpha,
    )
    prediction_path = (
        args.output_dir / "artifacts" / "validation-conservative-blend.npy"
    )
    np.save(prediction_path, candidate_scores.astype(np.float32))
    source_hashes_unchanged = _source_hashes_unchanged(source)
    if not source_hashes_unchanged:
        raise RuntimeError("source artifacts changed during gate")
    report = {
        "status": "passed" if gate.passed else "rejected",
        "gate_passed": gate.passed,
        "production_followup_authorized": gate.passed,
        "package_generated": False,
        "selection_report": str(selection_path.resolve()),
        "selection_report_sha256": locked_selection_sha,
        "selected_alpha": selected_alpha,
        "champion": {
            "full": gate.baseline_full_mrr,
            "slice_0": gate.baseline_slice_mrrs[0],
            "slice_1": gate.baseline_slice_mrrs[1],
            "slice_2": gate.baseline_slice_mrrs[2],
        },
        "candidate": {
            "full": gate.candidate_full_mrr,
            "slice_0": gate.candidate_slice_mrrs[0],
            "slice_1": gate.candidate_slice_mrrs[1],
            "slice_2": gate.candidate_slice_mrrs[2],
        },
        "delta_vs_champion": {
            "full": gate.full_delta,
            "slice_0": gate.slice_deltas[0],
            "slice_1": gate.slice_deltas[1],
            "slice_2": gate.slice_deltas[2],
        },
        "gate": {
            "minimum_full_delta": gate.minimum_full_delta,
            "full_delta_passed": bool(
                gate.full_delta + 1e-12 >= gate.minimum_full_delta
            ),
            "all_three_slices_non_decreasing": all(
                delta + 1e-12 >= 0.0 for delta in gate.slice_deltas
            ),
        },
        "source_hashes_unchanged": source_hashes_unchanged,
        "selected_prediction": str(prediction_path.resolve()),
        "selected_prediction_sha256": _sha256(prediction_path),
        "frozen_dataset1_csv": str(
            args.frozen_dataset1_csv.resolve()
        ),
        "frozen_dataset1_csv_sha256": frozen[
            "frozen_dataset1_csv_sha256"
        ],
    }
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if gate.passed else 2


def _load_and_verify_source(source_dir: Path) -> dict[str, Any]:
    selection_path = source_dir / "selection-report.json"
    selection_sha_path = source_dir / "selection-report.sha256"
    evaluation_path = source_dir / "evaluation-report.json"
    locked_sha = selection_sha_path.read_text(encoding="utf-8").split()[0]
    _require_hash(selection_path, locked_sha, "source selection report")
    selection = _read_json(selection_path)
    evaluation = _read_json(evaluation_path)
    if tuple(selection["selected_experts"]) != SELECTED_EXPERTS:
        raise ValueError("source selected expert subset differs")
    if tuple(evaluation["selected_experts"]) != SELECTED_EXPERTS:
        raise ValueError("source evaluation expert subset differs")
    if evaluation.get("selection_report_sha256") != locked_sha:
        raise ValueError("source evaluation does not reference locked selection")

    artifact_paths = {
        "lightgbm": Path(selection["secondary_probabilities"]),
        **{
            name: Path(
                selection["experts"][name]["validation_probabilities"]
            )
            for name in SELECTED_EXPERTS
        },
    }
    artifact_hashes = {
        "lightgbm": str(selection["secondary_probabilities_sha256"]),
        **{
            name: str(
                selection["experts"][name][
                    "validation_probabilities_sha256"
                ]
            )
            for name in SELECTED_EXPERTS
        },
    }
    for name, path in artifact_paths.items():
        _require_hash(path, artifact_hashes[name], f"{name} probabilities")
    frozen_source = {
        "directory": str(source_dir.resolve()),
        "selection_report": str(selection_path.resolve()),
        "selection_report_sha256": locked_sha,
        "evaluation_report": str(evaluation_path.resolve()),
        "evaluation_report_sha256": _sha256(evaluation_path),
        "probabilities": {
            name: {
                "path": str(path.resolve()),
                "sha256": artifact_hashes[name],
            }
            for name, path in artifact_paths.items()
        },
    }
    return {
        "selection": selection,
        "evaluation": evaluation,
        "artifact_paths": artifact_paths,
        "artifact_hashes": artifact_hashes,
        "frozen_source": frozen_source,
    }


def _load_probabilities(source: dict[str, Any]) -> dict[str, np.ndarray]:
    probabilities = {
        name: np.load(path, mmap_mode="r", allow_pickle=False)
        for name, path in source["artifact_paths"].items()
    }
    expected_shape = (VALIDATION_ROWS, 100)
    if any(values.shape != expected_shape for values in probabilities.values()):
        raise ValueError("source probability shape differs from 20000x100")
    return probabilities


def _build_source_scores(
    probabilities: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    experts = {
        name: probabilities[name]
        for name in SELECTED_EXPERTS
    }
    champion = blend_expert_subset(
        experts,
        probabilities["lightgbm"],
        selected_experts=("recent200k",),
        expert_weight=SETWISE_WEIGHT,
    )
    window = blend_expert_subset(
        experts,
        probabilities["lightgbm"],
        selected_experts=SELECTED_EXPERTS,
        expert_weight=SETWISE_WEIGHT,
    )
    return champion, window


def _require_source_gate_metrics(
    gate: Any,
    evaluation: dict[str, Any],
) -> None:
    champion = evaluation["champion"]
    candidate = evaluation["candidate"]
    _require_close(gate.baseline_full_mrr, champion["full"], "champion full")
    _require_close(gate.candidate_full_mrr, candidate["full"], "window full")
    for index in range(3):
        _require_close(
            gate.baseline_slice_mrrs[index],
            champion[f"slice_{index}"],
            f"champion slice{index}",
        )
        _require_close(
            gate.candidate_slice_mrrs[index],
            candidate[f"slice_{index}"],
            f"window slice{index}",
        )


def _source_hashes_unchanged(source: dict[str, Any]) -> bool:
    frozen = source["frozen_source"]
    if _sha256(Path(frozen["selection_report"])) != frozen[
        "selection_report_sha256"
    ]:
        return False
    if _sha256(Path(frozen["evaluation_report"])) != frozen[
        "evaluation_report_sha256"
    ]:
        return False
    return all(
        _sha256(Path(item["path"])) == item["sha256"]
        for item in frozen["probabilities"].values()
    )


def _require_close(actual: float, expected: float, label: str) -> None:
    if abs(float(actual) - float(expected)) > 1e-10:
        raise ValueError(
            f"{label} mismatch: actual={actual} expected={expected}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: actual={actual} expected={expected}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    _write_text_atomic(path, f"{text}\n")


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
