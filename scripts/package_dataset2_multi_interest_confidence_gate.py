from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.core.io import discover_datasets
from jgrec.core.types import DatasetResult, TrainingReport
from jgrec.rankers.hybrid.multi_interest_gate import (
    EXPERT_SCORE_DESCRIPTOR_NAMES,
    MULTI_INTEREST_GATE_DESCRIPTOR_NAMES,
    ConfidenceGateConfig,
    expert_score_descriptors,
    fit_confidence_gate,
    predict_confidence_gate,
    route_query_experts,
)
from jgrec.rankers.hybrid.segment_fusion import QUERY_SEGMENT_FEATURE_NAMES
from jgrec.submission import (
    expected_test_rows,
    validate_submission_file,
    write_zip,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-report", required=True, type=Path)
    parser.add_argument("--paired-diagnostics", required=True, type=Path)
    parser.add_argument("--champion-dataset1", required=True, type=Path)
    parser.add_argument("--champion-dataset2", required=True, type=Path)
    parser.add_argument("--candidate-dataset2", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()

    report = _read_json(args.oof_report)
    if (
        report.get("status") != "passed"
        or not report.get("gate_passed")
        or not report.get("package_authorized")
    ):
        raise RuntimeError("OOF report does not authorize packaging")
    selected = report["selected"]
    selected_config = selected["config"]
    config = ConfidenceGateConfig(
        max_depth=int(selected_config["max_depth"]),
        min_samples_leaf=int(selected_config["min_samples_leaf"]),
        minimum_predicted_lift=float(
            selected_config["minimum_predicted_lift"]
        ),
    )
    frozen = report["frozen_config"]
    if (
        config.minimum_predicted_lift
        < float(frozen["minimum_confidence_threshold"])
        or float(selected["coverage"])
        > float(frozen["maximum_gate_coverage"])
    ):
        raise RuntimeError("selected OOF gate is not high-confidence")

    diagnostics = np.load(args.paired_diagnostics, allow_pickle=False)
    descriptors = np.asarray(diagnostics["descriptors"], dtype=np.float32)
    descriptor_names = tuple(
        str(value) for value in diagnostics["descriptor_names"]
    )
    if descriptor_names != MULTI_INTEREST_GATE_DESCRIPTOR_NAMES:
        raise ValueError("paired descriptor schema differs")
    champion_rr = np.asarray(diagnostics["champion_rr"], dtype=np.float64)
    candidate_rr = np.asarray(diagnostics["candidate_rr"], dtype=np.float64)
    rewards = candidate_rr - champion_rr
    gate_model = fit_confidence_gate(
        descriptors,
        rewards,
        config,
        descriptor_names=descriptor_names,
        seed=60,
    )
    validation_use_candidate, validation_predicted_lift = (
        predict_confidence_gate(
            gate_model,
            descriptors,
            descriptor_names=descriptor_names,
        )
    )
    validation_delta = _gated_delta_slices(
        champion_rr,
        candidate_rr,
        validation_use_candidate,
    )
    minimum_full_delta = float(frozen["minimum_full_delta"])
    if (
        validation_delta["full"] + 1e-12 < minimum_full_delta
        or any(
            validation_delta[f"slice_{index}"] < 0.0
            for index in range(3)
        )
    ):
        raise RuntimeError("full-validation production gate failed")

    estimator = pickle.loads(gate_model.model_bytes)
    used_indices = sorted(
        {
            int(index)
            for index in estimator.tree_.feature
            if int(index) >= 0
        }
    )
    score_start = len(QUERY_SEGMENT_FEATURE_NAMES)
    score_stop = score_start + len(EXPERT_SCORE_DESCRIPTOR_NAMES)
    if any(
        index < score_start or index >= score_stop
        for index in used_indices
    ):
        used_names = [descriptor_names[index] for index in used_indices]
        raise RuntimeError(
            "final gate requires unavailable test descriptors: "
            f"{used_names}"
        )

    champion_scores = np.loadtxt(
        args.champion_dataset2,
        delimiter=",",
        dtype=np.float32,
        ndmin=2,
    )
    candidate_scores = np.loadtxt(
        args.candidate_dataset2,
        delimiter=",",
        dtype=np.float32,
        ndmin=2,
    )
    if (
        champion_scores.shape != candidate_scores.shape
        or champion_scores.ndim != 2
        or champion_scores.shape[1] != 100
    ):
        raise ValueError("Dataset2 expert CSV shapes differ")
    test_descriptors = np.zeros(
        (champion_scores.shape[0], len(descriptor_names)),
        dtype=np.float32,
    )
    test_descriptors[:, score_start:score_stop] = (
        expert_score_descriptors(champion_scores, candidate_scores)
    )
    test_use_candidate, test_predicted_lift = predict_confidence_gate(
        gate_model,
        test_descriptors,
        descriptor_names=descriptor_names,
    )
    gated_scores = route_query_experts(
        champion_scores,
        candidate_scores,
        test_use_candidate,
    )
    if not np.array_equal(
        gated_scores[~test_use_candidate],
        champion_scores[~test_use_candidate],
    ):
        raise RuntimeError("champion fallback is not exact")
    if not np.array_equal(
        gated_scores[test_use_candidate],
        candidate_scores[test_use_candidate],
    ):
        raise RuntimeError("candidate routing is not query-exact")

    datasets = {
        dataset.name: dataset
        for dataset in discover_datasets(args.data_dir)
    }
    csv_dir = args.output_dir / "csv"
    csv_dir.mkdir()
    dataset1_output = csv_dir / "dataset1.csv"
    dataset2_output = csv_dir / "dataset2.csv"
    shutil.copyfile(args.champion_dataset1, dataset1_output)
    np.savetxt(
        dataset2_output,
        gated_scores,
        delimiter=",",
        fmt="%.8f",
    )
    dataset1_rows = expected_test_rows(datasets["dataset1"])
    dataset2_rows = expected_test_rows(datasets["dataset2"])
    validate_submission_file(dataset1_output, expected_rows=dataset1_rows)
    validate_submission_file(dataset2_output, expected_rows=dataset2_rows)

    gate_path = args.output_dir / "confidence-gate.pkl"
    with gate_path.open("wb") as handle:
        pickle.dump(gate_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dataset1_result = DatasetResult(
        name="dataset1",
        rows=dataset1_rows,
        output_path=dataset1_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    dataset2_result = DatasetResult(
        name="dataset2",
        rows=dataset2_rows,
        output_path=dataset2_output,
        training_report=TrainingReport(model_name="hybrid"),
    )
    zip_path = args.output_dir / "result.zip"
    write_zip([dataset1_result, dataset2_result], zip_path)
    candidate_report = {
        "status": "complete",
        "mode": "query_level_multi_interest_confidence_gate",
        "oof_report": str(args.oof_report.resolve()),
        "oof_report_sha256": _sha256(args.oof_report),
        "selected_oof_gate": selected,
        "production_config": selected_config,
        "production_validation_delta": validation_delta,
        "production_validation_coverage": float(
            validation_use_candidate.mean()
        ),
        "production_validation_predicted_lift": {
            "min": float(validation_predicted_lift.min()),
            "mean": float(validation_predicted_lift.mean()),
            "max": float(validation_predicted_lift.max()),
        },
        "used_descriptor_indices": used_indices,
        "used_descriptor_names": [
            descriptor_names[index] for index in used_indices
        ],
        "test_gate_rows": int(test_use_candidate.sum()),
        "test_gate_coverage": float(test_use_candidate.mean()),
        "test_predicted_lift": {
            "min": float(test_predicted_lift.min()),
            "mean": float(test_predicted_lift.mean()),
            "max": float(test_predicted_lift.max()),
        },
        "champion_fallback_exact": True,
        "dataset1_rows": dataset1_rows,
        "dataset1_sha256": _sha256(dataset1_output),
        "dataset2_rows": dataset2_rows,
        "dataset2_sha256": _sha256(dataset2_output),
        "confidence_gate_sha256": _sha256(gate_path),
        "result_zip": str(zip_path.resolve()),
        "result_zip_bytes": zip_path.stat().st_size,
        "result_zip_sha256": _sha256(zip_path),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output_dir / "candidate-report.json", candidate_report)
    print(json.dumps(candidate_report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _gated_delta_slices(
    champion_rr: np.ndarray,
    candidate_rr: np.ndarray,
    use_candidate: np.ndarray,
) -> dict[str, float]:
    rewards = candidate_rr - champion_rr
    routed_rewards = np.where(use_candidate, rewards, 0.0)
    slices = np.array_split(
        np.arange(rewards.size, dtype=np.int64),
        3,
    )
    return {
        "full": float(routed_rewards.mean()),
        **{
            f"slice_{index}": float(routed_rewards[indices].mean())
            for index, indices in enumerate(slices)
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
