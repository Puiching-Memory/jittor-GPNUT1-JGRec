from __future__ import annotations

import gc
import json
from pathlib import Path

import jittor as jt
import numpy as np

from jgrec.contest_checkpoint import load_checkpoint_dataset
from jgrec.core.io import read_test_queries
from jgrec.rankers.hybrid.cooccur_lift_checkpoint import (
    predict_cooccur_lift_auxiliary_probabilities,
)
from jgrec.rankers.registry import create_ranker

ROOT = Path(__file__).resolve().parent.parent
TARGET_ROW = 97_576
BATCH_SIZE = 4_096


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


queries = read_test_queries(ROOT / "data" / "dataset2" / "test.csv")
order = np.argsort(queries.src, kind="stable")
inverse = np.empty(len(order), dtype=np.int64)
inverse[order] = np.arange(len(order), dtype=np.int64)
position = int(inverse[TARGET_ROW])
start = position // BATCH_SIZE * BATCH_SIZE
rows = order[start : start + BATCH_SIZE]
batch_queries = queries[rows]
target_offset = int(np.flatnonzero(rows == TARGET_ROW)[0])
expected_all = np.load(
    ROOT
    / "result"
    / "dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2"
    / "online-materialization-source-grouped-b4096"
    / "test-auxiliary-probabilities.npy",
    mmap_mode="r",
)
expected = np.asarray(expected_all[rows], dtype=np.float64)

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

base = ranker.impl.encoder.features_for_query_array(batch_queries)
state = ranker.impl.cooccur_lift_auxiliary_state
model = ranker.impl.cooccur_lift_auxiliary_model
before = predict_cooccur_lift_auxiliary_probabilities(
    state,
    model,
    base,
    batch_queries,
)
saved_state = ranker.impl.cooccur_lift_auxiliary_state
ranker.impl.cooccur_lift_auxiliary_state = None
baseline = ranker.predict_batch(batch_queries)
ranker.impl.cooccur_lift_auxiliary_state = saved_state
integrated = ranker.predict_batch(batch_queries)
after = predict_cooccur_lift_auxiliary_probabilities(
    state,
    model,
    base,
    batch_queries,
)
champion = _selected_csv_rows(
    ROOT
    / "result"
    / "d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_tiesafe_v2_20260728"
    / "csv"
    / "dataset2.csv",
    rows,
)
expected_integrated = 0.5 * champion + 0.5 * expected

print(
    json.dumps(
        {
            "target_row": TARGET_ROW,
            "target_offset": target_offset,
            "source_grouped_batch_start": start,
            "rows": len(rows),
            "before_champion": _summary(before, expected),
            "after_champion": _summary(after, expected),
            "coexisting_baseline_vs_champion_replay": _summary(
                baseline,
                champion,
            ),
            "integrated_vs_reference_recomposition": _summary(
                integrated,
                expected_integrated,
            ),
            "target": {
                "maximum_before_error": float(
                    np.abs(
                        before[target_offset] - expected[target_offset]
                    ).max()
                ),
                "maximum_after_error": float(
                    np.abs(
                        after[target_offset] - expected[target_offset]
                    ).max()
                ),
                "maximum_baseline_error": float(
                    np.abs(
                        baseline[target_offset] - champion[target_offset]
                    ).max()
                ),
                "maximum_integrated_error": float(
                    np.abs(
                        integrated[target_offset]
                        - expected_integrated[target_offset]
                    ).max()
                ),
            },
        },
        indent=2,
        sort_keys=True,
    )
)
