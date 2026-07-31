from __future__ import annotations

import hashlib
import heapq
import json
import zipfile
from pathlib import Path

import numpy as np

from replay_dataset2_cooccur_lift_checkpoint import (
    _compare_with_zip_member,
)

ROOT = Path(__file__).resolve().parent.parent
REPLAY_ROOT = (
    ROOT
    / "result"
    / "dataset2_cooccur_lift_online_promotion_20260729"
    / "double-replay-retry2"
)
ONLINE_ZIP = (
    ROOT
    / "result"
    / "d1_time_ramp_g050_d2_cooccur_lift_aux_w050_20260729"
    / "result.zip"
)
CHAMPION_ZIP = (
    ROOT
    / "result"
    / "d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_seed60_20260727"
    / "result.zip"
)
CHAMPION_REPLAY = (
    ROOT
    / "result"
    / "d1_time_ramp_g050_d2_gnn_short_none_e50_edges40000_setwise_w080_tiesafe_v2_20260728"
    / "csv"
    / "dataset2.csv"
)
ONLINE_AUXILIARY = (
    ROOT
    / "result"
    / "dataset2_cooccur_lift_aux_expert_v1_20260728_compact_retry2"
    / "online-materialization-source-grouped-b4096"
    / "test-auxiliary-probabilities.npy"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _largest_differences(
    replay_path: Path,
    online_zip: Path,
) -> dict[str, object]:
    largest: list[tuple[float, int, int, float, float]] = []
    values_above_tolerance = 0
    rows_above_tolerance = 0
    with (
        zipfile.ZipFile(online_zip) as archive,
        archive.open("dataset2.csv") as online,
        replay_path.open("rb") as replay,
    ):
        for row_index, (online_line, replay_line) in enumerate(
            zip(online, replay, strict=True)
        ):
            online_values = np.fromstring(
                online_line.decode("ascii"),
                sep=",",
                dtype=np.float64,
            )
            replay_values = np.fromstring(
                replay_line.decode("ascii"),
                sep=",",
                dtype=np.float64,
            )
            errors = np.abs(online_values - replay_values)
            over = errors > 5e-7
            values_above_tolerance += int(over.sum())
            rows_above_tolerance += int(over.any())
            for candidate_index in np.argpartition(errors, -1)[-1:]:
                item = (
                    float(errors[candidate_index]),
                    row_index,
                    int(candidate_index),
                    float(online_values[candidate_index]),
                    float(replay_values[candidate_index]),
                )
                if len(largest) < 20:
                    heapq.heappush(largest, item)
                elif item > largest[0]:
                    heapq.heapreplace(largest, item)
    return {
        "values_above_tolerance": values_above_tolerance,
        "rows_above_tolerance": rows_above_tolerance,
        "largest": [
            {
                "absolute_error": error,
                "row_index": row_index,
                "candidate_index": candidate_index,
                "online": online,
                "replay": replay,
            }
            for error, row_index, candidate_index, online, replay in sorted(
                largest,
                reverse=True,
            )
        ],
    }


def _recomposition_summary(
    replay_path: Path,
    champion_replay_path: Path,
    auxiliary_path: Path,
) -> dict[str, object]:
    auxiliary = np.load(auxiliary_path, mmap_mode="r")
    rows = 0
    values = 0
    maximum_error = 0.0
    absolute_error_sum = 0.0
    values_above_tolerance = 0
    rows_above_tolerance = 0
    top1_disagreements = 0
    with (
        replay_path.open("rb") as replay,
        champion_replay_path.open("rb") as champion,
    ):
        for row_index, (replay_line, champion_line) in enumerate(
            zip(replay, champion, strict=True)
        ):
            replay_values = np.fromstring(
                replay_line.decode("ascii"),
                sep=",",
                dtype=np.float64,
            )
            champion_values = np.fromstring(
                champion_line.decode("ascii"),
                sep=",",
                dtype=np.float64,
            )
            recomposed = (
                0.5 * champion_values
                + 0.5 * np.asarray(auxiliary[row_index], dtype=np.float64)
            )
            errors = np.abs(replay_values - recomposed)
            maximum_error = max(maximum_error, float(errors.max()))
            absolute_error_sum += float(errors.sum())
            values += int(errors.size)
            over = errors > 5e-7
            values_above_tolerance += int(over.sum())
            rows_above_tolerance += int(over.any())
            top1_disagreements += int(
                int(np.argmax(replay_values))
                != int(np.argmax(recomposed))
            )
            rows += 1
    if rows != len(auxiliary):
        raise ValueError("auxiliary and replay row counts differ")
    return {
        "rows": rows,
        "values": values,
        "maximum_absolute_error": maximum_error,
        "mean_absolute_error": absolute_error_sum / values,
        "values_above_tolerance": values_above_tolerance,
        "rows_above_tolerance": rows_above_tolerance,
        "top1_disagreements": top1_disagreements,
    }


replay_a = REPLAY_ROOT / "replay-a" / "dataset2.csv"
replay_b = REPLAY_ROOT / "replay-b" / "dataset2.csv"
print(
    json.dumps(
        {
            "replay_a_sha256": _sha256(replay_a),
            "replay_b_sha256": _sha256(replay_b),
            "comparison": _compare_with_zip_member(
                replay_a,
                ONLINE_ZIP,
                member="dataset2.csv",
            ),
            "largest_differences": _largest_differences(
                replay_a,
                ONLINE_ZIP,
            ),
            "champion_replay_vs_online_champion": (
                _compare_with_zip_member(
                    CHAMPION_REPLAY,
                    CHAMPION_ZIP,
                    member="dataset2.csv",
                )
            ),
            "runtime_vs_champion_replay_plus_stored_auxiliary": (
                _recomposition_summary(
                    replay_a,
                    CHAMPION_REPLAY,
                    ONLINE_AUXILIARY,
                )
            ),
        },
        indent=2,
        sort_keys=True,
    )
)
