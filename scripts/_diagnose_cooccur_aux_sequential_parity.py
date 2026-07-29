from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_test_queries
from jgrec.rankers.registry import create_ranker

ROOT = Path(__file__).resolve().parent.parent
TARGET_ROW = 97_576
BATCH_SIZE = 4_096


def _selected_csv_rows(path: Path, rows: np.ndarray) -> np.ndarray:
    positions = {int(row): index for index, row in enumerate(rows)}
    selected = np.empty((len(rows), 100), dtype=np.float64)
    remaining = len(rows)
    with path.open("rb") as handle:
        for row_index, line in enumerate(handle):
            output_index = positions.get(row_index)
            if output_index is None:
                continue
            selected[output_index] = np.fromstring(
                line.decode("ascii"),
                sep=",",
                dtype=np.float64,
            )
            remaining -= 1
            if remaining == 0:
                break
    if remaining:
        raise ValueError("champion replay is missing selected rows")
    return selected


def _summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    errors = np.abs(actual - expected)
    return {
        "maximum_absolute_error": float(errors.max()),
        "mean_absolute_error": float(errors.mean()),
        "values_above_5e_7": int(np.count_nonzero(errors > 5e-7)),
        "rows_above_5e_7": int(
            np.count_nonzero(np.any(errors > 5e-7, axis=1))
        ),
        "top1_disagreements": int(
            np.count_nonzero(
                np.argmax(actual, axis=1) != np.argmax(expected, axis=1)
            )
        ),
    }


started = time.time()
queries = read_test_queries(ROOT / "data" / "dataset2" / "test.csv")
order = np.argsort(queries.src, kind="stable")
inverse = np.empty(len(order), dtype=np.int64)
inverse[order] = np.arange(len(order), dtype=np.int64)
target_position = int(inverse[TARGET_ROW])
target_start = target_position // BATCH_SIZE * BATCH_SIZE

checkpoint_state = load_checkpoint_dataset(
    ROOT
    / "checkpoints"
    / "d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729.pkl",
    "dataset2",
)
ranker = create_ranker("hybrid", None)
ranker.hydrate(checkpoint_state)
del checkpoint_state
gc.collect()
jt.flags.use_cuda = 1

target_rows: np.ndarray | None = None
target_actual: np.ndarray | None = None
for start in range(0, target_start + 1, BATCH_SIZE):
    stop = min(start + BATCH_SIZE, len(order))
    rows = order[start:stop]
    actual = ranker.predict_batch(queries[rows])
    print(
        json.dumps(
            {
                "status": "sequential_prefix",
                "start": start,
                "stop": stop,
                "target_start": target_start,
                "elapsed_seconds": time.time() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if start == target_start:
        target_rows = rows
        target_actual = actual

if target_rows is None or target_actual is None:
    raise RuntimeError("target batch was not evaluated")
champion = _selected_csv_rows(
    ROOT
    / "result"
    / "d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_tiesafe_v2_20260728"
    / "csv"
    / "dataset2.csv",
    target_rows,
)
auxiliary = np.load(
    ROOT
    / "result"
    / "dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2"
    / "online-materialization-source-grouped-b4096"
    / "test-auxiliary-probabilities.npy",
    mmap_mode="r",
)
expected = (
    0.5 * champion
    + 0.5 * np.asarray(auxiliary[target_rows], dtype=np.float64)
)
target_offset = int(np.flatnonzero(target_rows == TARGET_ROW)[0])
report = {
    "status": "passed",
    "target_row": TARGET_ROW,
    "target_position": target_position,
    "target_batch_start": target_start,
    "sequential_batches": target_start // BATCH_SIZE + 1,
    "batch_summary": _summary(target_actual, expected),
    "target_maximum_absolute_error": float(
        np.abs(
            target_actual[target_offset] - expected[target_offset]
        ).max()
    ),
    "elapsed_seconds": time.time() - started,
}
if (
    report["batch_summary"]["maximum_absolute_error"] > 5e-7
    or report["batch_summary"]["top1_disagreements"] != 0
):
    report["status"] = "failed"
print(json.dumps(report, indent=2, sort_keys=True), flush=True)
if report["status"] != "passed":
    raise SystemExit(1)
