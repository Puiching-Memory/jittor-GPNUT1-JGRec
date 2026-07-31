from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.partial_listwise_submission import ranking_metric_panel
from jgrec.rankers.hybrid.partial_listwise_blend import (
    blend_partial_listwise,
)

SLICES = {
    "slice_0": (0, 6667),
    "slice_1": (6667, 13334),
    "slice_2": (13334, 20000),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report a multi-metric risk panel for the already frozen "
            "Dataset2 partial-listwise online candidate."
        )
    )
    parser.add_argument("--champion-scores", required=True, type=Path)
    parser.add_argument("--expert-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--auxiliary-weight", type=float, default=0.20)
    parser.add_argument("--selection-lock-sha256", required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    champion = np.load(
        args.champion_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    expert = np.load(
        args.expert_scores,
        mmap_mode="r",
        allow_pickle=False,
    )
    if champion.shape != (20_000, 100) or expert.shape != champion.shape:
        raise ValueError(
            f"unexpected aligned score shapes: {champion.shape}, {expert.shape}"
        )
    candidate = blend_partial_listwise(
        champion,
        expert,
        auxiliary_weight=args.auxiliary_weight,
    )
    report: dict[str, Any] = {
        "status": "risk_profile_only_not_a_packaging_gate",
        "candidate_frozen_before_report": True,
        "expert_name": "listwise_two_tower",
        "auxiliary_weight": float(args.auxiliary_weight),
        "selection_lock_sha256": args.selection_lock_sha256,
        "score_sources": {
            "champion": {
                "path": str(args.champion_scores.resolve()),
                "sha256": _sha256(args.champion_scores),
            },
            "expert": {
                "path": str(args.expert_scores.resolve()),
                "sha256": _sha256(args.expert_scores),
            },
        },
        "segments": {},
    }
    segments = {"full": (0, champion.shape[0]), **SLICES}
    for name, (start, stop) in segments.items():
        baseline_metrics = ranking_metric_panel(champion[start:stop])
        candidate_metrics = ranking_metric_panel(candidate[start:stop])
        baseline_ranks = _ranks(champion[start:stop])
        candidate_ranks = _ranks(candidate[start:stop])
        report["segments"][name] = {
            "rows": [start, stop],
            "champion": baseline_metrics,
            "candidate": candidate_metrics,
            "delta_candidate_minus_champion": {
                metric: candidate_metrics[metric] - baseline_metrics[metric]
                for metric in baseline_metrics
            },
            "rank_movements": {
                "improved": int(np.sum(candidate_ranks < baseline_ranks)),
                "unchanged": int(np.sum(candidate_ranks == baseline_ranks)),
                "worsened": int(np.sum(candidate_ranks > baseline_ranks)),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _ranks(scores: np.ndarray) -> np.ndarray:
    return 1 + np.sum(scores[:, 1:] > scores[:, :1], axis=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
