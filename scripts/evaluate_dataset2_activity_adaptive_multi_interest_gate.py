from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.rankers.hybrid.segment_fusion import (
    QUERY_SEGMENT_FEATURE_NAMES,
    query_segment_features,
)

SLICE_2_START = 13_334


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--selection-report-sha256", required=True, type=Path)
    parser.add_argument("--validation-scores", required=True, type=Path)
    parser.add_argument("--validation-features", required=True, type=Path)
    parser.add_argument("--feature-names-json", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()

    if args.output_report.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_report}")
    selection = _read_json(args.selection_report)
    if not selection.get("gate_passed"):
        raise RuntimeError("selection did not authorize the slice2 gate")
    expected_report_hash = (
        args.selection_report_sha256.read_text(encoding="ascii").split()[0]
    )
    if _sha256(args.selection_report) != expected_report_hash:
        raise ValueError("selection report hash differs from sidecar")
    if (
        _sha256(args.validation_scores)
        != selection["artifacts"]["validation_scores_sha256"]
    ):
        raise ValueError("validation score hash differs from selection report")

    payload = np.load(args.validation_scores, allow_pickle=False)
    champion = np.asarray(payload["champion"], dtype=np.float64)
    old_multi_interest = np.asarray(
        payload["old_multi_interest"],
        dtype=np.float64,
    )
    adaptive = np.asarray(payload["adaptive"], dtype=np.float64)
    if (
        champion.shape != (20_000, 100)
        or old_multi_interest.shape != champion.shape
        or adaptive.shape != champion.shape
    ):
        raise ValueError("validation score arrays have unexpected shapes")
    validation_features = np.load(
        args.validation_features,
        mmap_mode="r",
        allow_pickle=False,
    )
    feature_names_payload = _read_json(args.feature_names_json)
    feature_names = tuple(feature_names_payload["feature_names"])

    selected = slice(SLICE_2_START, 20_000)
    champion_mrr = _mrr(champion[selected])
    old_mrr = _mrr(old_multi_interest[selected])
    adaptive_mrr = _mrr(adaptive[selected])
    descriptors = query_segment_features(
        validation_features[selected],
        feature_names,
    )
    activity = descriptors[
        :,
        QUERY_SEGMENT_FEATURE_NAMES.index("source_activity"),
    ]
    lower, middle, upper = np.quantile(activity, (0.25, 0.5, 0.75))
    masks = {
        "q1": activity <= lower,
        "q2": (activity > lower) & (activity <= middle),
        "q3": (activity > middle) & (activity <= upper),
        "q4": activity > upper,
    }
    segments: dict[str, Any] = {}
    for name, mask in masks.items():
        segment_champion = _mrr(champion[selected][mask])
        segment_old = _mrr(old_multi_interest[selected][mask])
        segment_adaptive = _mrr(adaptive[selected][mask])
        segments[name] = {
            "rows": int(mask.sum()),
            "champion_mrr": segment_champion,
            "old_multi_interest_mrr": segment_old,
            "adaptive_mrr": segment_adaptive,
            "adaptive_delta_vs_champion": (
                segment_adaptive - segment_champion
            ),
            "adaptive_delta_vs_old_multi_interest": (
                segment_adaptive - segment_old
            ),
        }
    passed = bool(
        adaptive_mrr - old_mrr >= 0.0
        and segments["q4"]["adaptive_delta_vs_champion"] >= 0.0
    )
    report = {
        "status": "passed" if passed else "rejected",
        "gate_passed": passed,
        "package_authorized": False,
        "selection_report_sha256": expected_report_hash,
        "slice_2": {
            "rows": 20_000 - SLICE_2_START,
            "champion_mrr": champion_mrr,
            "old_multi_interest_mrr": old_mrr,
            "adaptive_mrr": adaptive_mrr,
            "adaptive_delta_vs_champion": adaptive_mrr - champion_mrr,
            "adaptive_delta_vs_old_multi_interest": adaptive_mrr - old_mrr,
        },
        "source_activity_quantiles": [float(lower), float(middle), float(upper)],
        "source_activity_segments": segments,
        "decision": (
            "continue_to_production_routing"
            if passed
            else "stop_activity_adaptive_multi_interest"
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


def _mrr(probabilities: np.ndarray) -> float:
    ranks = 1 + np.sum(
        probabilities[:, 1:] > probabilities[:, :1],
        axis=1,
    )
    return float(np.mean(1.0 / ranks))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
