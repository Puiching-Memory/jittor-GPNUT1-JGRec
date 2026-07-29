from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.rankers.hybrid.rolling_origin import (
    sliding_rolling_origin_folds,
    validate_rolling_origin_times,
)

EXPECTED_FEATURE_SHA256 = (
    "a8f4b5d71dedd1b5aa89a9f0c40e1501afc882929d036ddaaa73076e5be6a6ef"
)
ROW_COUNT = 200_000
TRAIN_WINDOW_ROWS = 100_000
SCORE_ROWS = 25_000
FOLD_COUNT = 4
SELECTION_FOLD_COUNT = 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache-prefix", required=True, type=Path)
    parser.add_argument("--train-cache-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists() or args.output.with_suffix(
        f"{args.output.suffix}.sha256"
    ).exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    report = _read_json(args.train_cache_report)
    if (
        report.get("dataset_name") != "dataset1"
        or report.get("train_selection") != "recent"
        or int(report.get("requested_train_rows", -1)) != ROW_COUNT
    ):
        raise ValueError(
            "rolling-origin manifest requires Dataset1 recent-200k cache"
        )

    features_path = Path(f"{args.train_cache_prefix}.train.npy")
    times_path = Path(f"{args.train_cache_prefix}.train-time.npy")
    rows_path = Path(
        f"{args.train_cache_prefix}.train-row-indices.npy"
    )
    feature_hash = _require_report_hash(
        features_path,
        report,
        "features",
        "training features",
    )
    time_hash = _require_report_hash(
        times_path,
        report,
        "time",
        "training times",
    )
    row_indices_hash = _require_report_hash(
        rows_path,
        report,
        "row_indices",
        "training row indices",
    )
    if feature_hash != EXPECTED_FEATURE_SHA256:
        raise ValueError("training features are not the frozen Dataset1 cache")

    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    times = np.load(times_path, mmap_mode="r", allow_pickle=False)
    row_indices = np.load(rows_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (ROW_COUNT, 100, 63):
        raise ValueError(
            f"unexpected rolling-origin feature shape: {features.shape}"
        )
    if times.shape != (ROW_COUNT,) or row_indices.shape != (ROW_COUNT,):
        raise ValueError("rolling-origin sidecars do not align")
    if np.any(np.diff(row_indices) != 1):
        raise ValueError(
            "rolling-origin row indices must be one contiguous window"
        )

    folds = sliding_rolling_origin_folds(
        row_count=ROW_COUNT,
        train_window_rows=TRAIN_WINDOW_ROWS,
        score_rows=SCORE_ROWS,
        fold_count=FOLD_COUNT,
        step_rows=SCORE_ROWS,
        selection_fold_count=SELECTION_FOLD_COUNT,
    )
    time_boundaries = validate_rolling_origin_times(times, folds)
    fold_payloads = []
    for fold, boundary in zip(
        folds,
        time_boundaries,
        strict=True,
    ):
        train_start, train_stop = fold.train_rows
        score_start, score_stop = fold.score_rows
        fold_payloads.append(
            {
                **asdict(fold),
                "time_boundary": asdict(boundary),
                "interaction_train_rows": [
                    int(row_indices[train_start]),
                    int(row_indices[train_stop - 1]) + 1,
                ],
                "interaction_score_rows": [
                    int(row_indices[score_start]),
                    int(row_indices[score_stop - 1]) + 1,
                ],
            }
        )

    manifest = {
        "status": "frozen_before_training",
        "dataset_name": "dataset1",
        "level": "cached_head_rolling_origin",
        "limitation": (
            "feature encoder is frozen before all cached queries; raw and "
            "Setwise heads are retrained causally at every origin"
        ),
        "source": {
            "train_cache_report": str(
                args.train_cache_report.resolve()
            ),
            "train_cache_report_sha256": _sha256(
                args.train_cache_report
            ),
            "features": str(features_path.resolve()),
            "features_sha256": feature_hash,
            "times": str(times_path.resolve()),
            "times_sha256": time_hash,
            "row_indices": str(rows_path.resolve()),
            "row_indices_sha256": row_indices_hash,
            "checkpoint": report["checkpoint"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "encoder_context_interaction_row": int(
                report["split"]["context_end"]
            ),
            "cached_interaction_rows": [
                int(row_indices[0]),
                int(row_indices[-1]) + 1,
            ],
        },
        "protocol": {
            "row_count": ROW_COUNT,
            "candidate_count": 100,
            "feature_count": 63,
            "train_window_rows": TRAIN_WINDOW_ROWS,
            "score_rows": SCORE_ROWS,
            "step_rows": SCORE_ROWS,
            "fold_count": FOLD_COUNT,
            "selection_fold_count": SELECTION_FOLD_COUNT,
            "gate_fold_count": FOLD_COUNT - SELECTION_FOLD_COUNT,
            "head_training": {
                "seed": 60,
                "epochs": 4,
                "early_stopping": False,
                "batch_size": 256,
                "hidden_dim": 32,
                "learning_rate": 0.001,
            },
            "candidates": {
                "control": "raw_63_feature_listwise_mlp",
                "expert": "setwise_v1_189_feature_listwise_mlp",
                "time_ramp_powers": [0.5, 1.0, 2.0],
            },
            "selection": {
                "minimum_mean_delta": 0.0002,
                "every_selection_fold_non_decreasing": True,
                "order": [
                    "higher_worst_fold_delta",
                    "higher_mean_delta",
                    "larger_power",
                ],
                "forward_metrics_read": False,
            },
            "gate": {
                "forward_fold_non_decreasing": True,
                "minimum_all_fold_mean_delta": 0.0002,
            },
        },
        "folds": fold_payloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.output, manifest)
    manifest_hash = _sha256(args.output)
    args.output.with_suffix(
        f"{args.output.suffix}.sha256"
    ).write_text(
        f"{manifest_hash}  {args.output.name}\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _require_report_hash(
    path: Path,
    report: dict[str, Any],
    artifact_name: str,
    label: str,
) -> str:
    expected = str(report["artifacts"][artifact_name]["sha256"])
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: actual={actual} expected={expected}"
        )
    return actual


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
