from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset


def _load(base: Path, suffix: str) -> np.ndarray:
    return np.load(f"{base}{suffix}", mmap_mode="r", allow_pickle=False)


def main() -> int:
    if len(sys.argv) not in (3, 7, 9):
        raise SystemExit(
            "usage: diagnose CURRENT_BASE REFERENCE_BASE "
            "[CURRENT_TRAIN_REPORT REFERENCE_TRAIN_REPORT "
            "CURRENT_VAL_REPORT REFERENCE_VAL_REPORT] "
            "[CURRENT_CHECKPOINT REFERENCE_CHECKPOINT]"
        )
    current_base = Path(sys.argv[1])
    reference_base = Path(sys.argv[2])
    current = _load(current_base, ".val-candidates.npy")
    reference = _load(reference_base, ".val-candidates.npy")
    current_dst = _load(current_base, ".val-dst.npy")
    reference_dst = _load(reference_base, ".val-dst.npy")
    difference = np.asarray(current) != np.asarray(reference)
    differing_rows = np.flatnonzero(np.any(difference, axis=1))
    intersections = np.fromiter(
        (
            len(
                set(map(int, current[row])).intersection(
                    map(int, reference[row])
                )
            )
            for row in range(len(current))
        ),
        dtype=np.int64,
        count=len(current),
    )
    aligned_fields = {}
    for suffix in (
        ".val-src.npy",
        ".val-dst.npy",
        ".val-time.npy",
        ".val-row-indices.npy",
    ):
        left = _load(current_base, suffix)
        right = _load(reference_base, suffix)
        aligned_fields[suffix] = {
            "shape_equal": left.shape == right.shape,
            "dtype_equal": left.dtype == right.dtype,
            "exact_equal": bool(np.array_equal(left, right)),
        }
    first = int(differing_rows[0]) if len(differing_rows) else None
    report = {
        "shape_current": list(current.shape),
        "shape_reference": list(reference.shape),
        "dtype_current": str(current.dtype),
        "dtype_reference": str(reference.dtype),
        "differing_rows": int(len(differing_rows)),
        "differing_cells": int(difference.sum()),
        "differing_cells_by_column": difference.sum(axis=0).tolist(),
        "first_diff_row": first,
        "first_diff_columns": (
            np.flatnonzero(difference[first]).tolist()
            if first is not None
            else []
        ),
        "current_first": (
            current[first, :12].tolist() if first is not None else []
        ),
        "reference_first": (
            reference[first, :12].tolist() if first is not None else []
        ),
        "positive_col_current_ok": bool(
            np.array_equal(current[:, 0], current_dst)
        ),
        "positive_col_reference_ok": bool(
            np.array_equal(reference[:, 0], reference_dst)
        ),
        "destinations_equal": bool(
            np.array_equal(current_dst, reference_dst)
        ),
        "intersection_mean": float(intersections.mean()),
        "intersection_min": int(intersections.min()),
        "intersection_max": int(intersections.max()),
        "rows_exact_as_sets": int(
            np.sum(intersections == current.shape[1])
        ),
        "aligned_fields": aligned_fields,
    }
    if len(sys.argv) >= 7:
        report_paths = [Path(value) for value in sys.argv[3:]]
        loaded_reports = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in report_paths
        ]
        current_train, reference_train, current_val, reference_val = (
            loaded_reports
        )
        report["report_comparison"] = {
            "train_candidate_checksum_equal": (
                current_train.get("candidate_checksum")
                == reference_train.get("candidate_checksum")
            ),
            "train_rng_state_equal": (
                current_train.get("fusion_rng_state_after_build")
                == reference_train.get("fusion_rng_state_after_build")
            ),
            "train_current_rng_state": current_train.get(
                "fusion_rng_state_after_build"
            ),
            "train_reference_rng_state": reference_train.get(
                "fusion_rng_state_after_build"
            ),
            "validation_candidate_checksum_equal": (
                current_val.get("candidate_checksum")
                == reference_val.get("candidate_checksum")
            ),
            "validation_rng_state_equal": (
                current_val.get("post_validation_rng_state")
                == reference_val.get("post_validation_rng_state")
            ),
            "validation_current_rng_state": current_val.get(
                "post_validation_rng_state"
            ),
            "validation_reference_rng_state": reference_val.get(
                "post_validation_rng_state"
            ),
            "current_checkpoint": current_train.get("checkpoint"),
            "reference_checkpoint": reference_train.get("checkpoint"),
            "current_checkpoint_sha256": current_train.get(
                "checkpoint_sha256"
            ),
            "reference_checkpoint_sha256": reference_train.get(
                "checkpoint_sha256"
            ),
            "current_prediction_limits": current_train.get(
                "prediction_limits"
            ),
            "reference_prediction_limits": reference_train.get(
                "prediction_limits"
            ),
            "current_validation_protocol": current_val.get("protocol"),
            "reference_validation_protocol": reference_val.get("protocol"),
        }
    if len(sys.argv) == 9:
        current_config = asdict(
            load_checkpoint_dataset(Path(sys.argv[7]), "dataset2")["config"]
        )
        reference_config = asdict(
            load_checkpoint_dataset(Path(sys.argv[8]), "dataset2")["config"]
        )
        report["checkpoint_config_differences"] = {
            key: {
                "current": current_config.get(key),
                "reference": reference_config.get(key),
            }
            for key in sorted(set(current_config) | set(reference_config))
            if current_config.get(key) != reference_config.get(key)
        }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
