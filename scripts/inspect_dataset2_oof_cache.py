from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-prefix", required=True, type=Path)
    parser.add_argument("--warmup-rows", type=int, default=40_000)
    parser.add_argument("--fold-rows", type=int, default=40_000)
    parser.add_argument("--fold-count", type=int, default=4)
    args = parser.parse_args()

    times = np.load(
        Path(f"{args.cache_prefix}.train-time.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    row_indices = np.load(
        Path(f"{args.cache_prefix}.train-row-indices.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_rows = args.warmup_rows + args.fold_rows * args.fold_count
    if times.ndim != 1 or row_indices.shape != times.shape:
        raise ValueError("OOF cache sidecars must be aligned vectors")
    if times.shape[0] != expected_rows:
        raise ValueError(
            f"OOF cache has {times.shape[0]} rows; expected {expected_rows}"
        )
    target_boundaries = [
        args.warmup_rows + index * args.fold_rows
        for index in range(args.fold_count)
    ]
    aligned_boundaries = [
        int(np.searchsorted(times, times[target], side="left"))
        for target in target_boundaries
    ]
    score_stops = [*aligned_boundaries[1:], int(times.shape[0])]
    folds = []
    for index, (score_start, score_stop) in enumerate(
        zip(aligned_boundaries, score_stops, strict=True)
    ):
        folds.append(
            {
                "index": index,
                "target_score_start": target_boundaries[index],
                "train_rows": [0, score_start],
                "score_rows": [score_start, score_stop],
                "train_time_max": int(times[score_start - 1]),
                "score_time_min": int(times[score_start]),
                "score_time_max": int(times[score_stop - 1]),
                "equal_origin_timestamp": bool(
                    times[score_start - 1] == times[score_start]
                ),
            }
        )
    report = {
        "rows": int(times.shape[0]),
        "time_dtype": str(times.dtype),
        "time_min": int(times[0]),
        "time_max": int(times[-1]),
        "time_decreases": int(
            np.count_nonzero(
                np.diff(times.astype(np.int64, copy=False)) < 0
            )
        ),
        "row_index_decreases": int(
            np.count_nonzero(
                np.diff(row_indices.astype(np.int64, copy=False)) < 0
            )
        ),
        "row_indices_unique": bool(
            np.unique(row_indices).size == row_indices.size
        ),
        "strict_origin_boundaries": all(
            fold["train_time_max"] < fold["score_time_min"]
            for fold in folds
        ),
        "oof_rows": int(times.shape[0] - aligned_boundaries[0]),
        "folds": folds,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
