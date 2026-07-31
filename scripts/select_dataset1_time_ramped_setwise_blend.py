from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.cuda import require_jittor_cuda
from jgrec.rankers.hybrid.time_ramp import select_time_ramp_on_prefix
from train_evaluate_dataset1_full100_setwise import (
    _champion_components,
    _read_json,
    _sha256,
    _write_json_atomic,
)

EXPECTED_ROWS = 20_000
EXPECTED_CANDIDATES = 100
FIRST_SLICE_STOP = 6_667
SELECTION_STOP = 13_334
POWERS = (0.5, 1.0, 2.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-features", required=True, type=Path)
    parser.add_argument("--validation-times", required=True, type=Path)
    parser.add_argument("--validation-cache-report", required=True, type=Path)
    parser.add_argument("--setwise-prediction", required=True, type=Path)
    parser.add_argument("--source-evaluation-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--minimum-prefix-delta", type=float, default=0.0002)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    require_jittor_cuda(jt)

    source_report = _read_json(args.source_evaluation_report)
    cache_report = _read_json(args.validation_cache_report)
    expected_checkpoint_hash = source_report["frozen_config"][
        "checkpoint_sha256"
    ]
    expected_feature_hash = source_report["frozen_config"][
        "validation_features_sha256"
    ]
    expected_prediction_hash = source_report["models"]["recent_100k"][
        "validation_prediction_sha256"
    ]
    expected_time_hash = cache_report["artifacts"]["time"]["sha256"]
    _require_hash(args.checkpoint, expected_checkpoint_hash, "checkpoint")
    _require_hash(
        args.validation_features,
        expected_feature_hash,
        "validation features",
    )
    _require_hash(
        args.setwise_prediction,
        expected_prediction_hash,
        "Setwise prediction",
    )
    _require_hash(args.validation_times, expected_time_hash, "validation time")

    frozen = {
        "status": "frozen_before_prediction",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": expected_checkpoint_hash,
        "validation_features": str(args.validation_features.resolve()),
        "validation_features_sha256": expected_feature_hash,
        "validation_times": str(args.validation_times.resolve()),
        "validation_times_sha256": expected_time_hash,
        "setwise_prediction": str(args.setwise_prediction.resolve()),
        "setwise_prediction_sha256": expected_prediction_hash,
        "source_evaluation_report_sha256": _sha256(
            args.source_evaluation_report
        ),
        "formula": "w=((time-min_time)/(max_time-min_time))**power",
        "powers": list(POWERS),
        "first_slice_rows": [0, FIRST_SLICE_STOP],
        "second_slice_rows": [FIRST_SLICE_STOP, SELECTION_STOP],
        "forward_rows": [SELECTION_STOP, EXPECTED_ROWS],
        "minimum_prefix_delta": args.minimum_prefix_delta,
        "slice0_and_slice1_non_decreasing": True,
        "forward_metrics_read": False,
        "final_gate": {
            "minimum_full_delta": 0.001,
            "all_three_slices_non_decreasing": True,
        },
    }
    _write_json_atomic(args.output_dir / "frozen-config.json", frozen)
    print(json.dumps(frozen, indent=2), flush=True)

    features = np.load(
        args.validation_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    times = np.load(
        args.validation_times,
        mmap_mode="r",
        allow_pickle=False,
    )
    expert = np.load(
        args.setwise_prediction,
        mmap_mode="r",
        allow_pickle=False,
    )
    if features.shape != (EXPECTED_ROWS, EXPECTED_CANDIDATES, 63):
        raise ValueError(f"unexpected validation features: {features.shape}")
    if expert.shape != (EXPECTED_ROWS, EXPECTED_CANDIDATES):
        raise ValueError(f"unexpected Setwise prediction: {expert.shape}")
    if times.shape != (EXPECTED_ROWS,):
        raise ValueError(f"unexpected validation times: {times.shape}")
    if np.any(np.diff(times) < 0):
        raise ValueError("validation times must be non-decreasing")

    state = load_checkpoint_dataset(args.checkpoint, "dataset1")
    champion, _ = _champion_components(
        state,
        features,
        batch_size=args.batch_size,
    )
    champion = np.asarray(champion, dtype=np.float32)
    expert = np.asarray(expert, dtype=np.float32)
    _require_prefix_metrics(
        champion,
        source_report["baseline"],
        "champion",
    )
    _require_prefix_metrics(
        expert,
        source_report["candidate"],
        "recent-100k Setwise",
    )

    selection = select_time_ramp_on_prefix(
        champion,
        expert,
        times,
        powers=POWERS,
        first_slice_stop=FIRST_SLICE_STOP,
        selection_stop=SELECTION_STOP,
        minimum_prefix_delta=args.minimum_prefix_delta,
    )
    scores_path = args.output_dir / "validation-expert-scores.npz"
    np.savez_compressed(
        scores_path,
        champion=champion,
        setwise=expert,
        query_times=np.asarray(times, dtype=np.int64),
    )
    selected = selection.selected_power is not None
    report = {
        "status": "selected" if selected else "no_eligible_candidate",
        "gate_passed": selected,
        "slice2_unlocked": selected,
        "slice2_metrics_read": False,
        "frozen_config": frozen,
        "selection": asdict(selection),
        "artifacts": {
            "validation_expert_scores_sha256": _sha256(scores_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.output_dir / "selection-report.json"
    _write_json_atomic(report_path, report)
    report_hash = _sha256(report_path)
    (args.output_dir / "selection-report.sha256").write_text(
        f"{report_hash}  {report_path.name}\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


def _require_prefix_metrics(
    probabilities: np.ndarray,
    expected: dict[str, float],
    label: str,
) -> None:
    actual = {
        "slice_0": _mrr(probabilities[:FIRST_SLICE_STOP]),
        "slice_1": _mrr(
            probabilities[FIRST_SLICE_STOP:SELECTION_STOP]
        ),
    }
    for name, value in actual.items():
        if abs(value - float(expected[name])) > 1e-10:
            raise RuntimeError(
                f"{label} reproduction failed for {name}: "
                f"{value} != {expected[name]}"
            )


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )


def _mrr(probabilities: np.ndarray) -> float:
    ranks = 1 + np.sum(
        probabilities[:, 1:] > probabilities[:, :1],
        axis=1,
    )
    return float(np.mean(1.0 / ranks))


if __name__ == "__main__":
    raise SystemExit(main())
