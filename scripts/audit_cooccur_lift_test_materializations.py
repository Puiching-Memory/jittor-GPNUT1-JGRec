from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare completed rows shared by two interrupted or complete "
            "cooccur-lift test probability materializations."
        )
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    reference = np.load(args.reference, mmap_mode="r", allow_pickle=False)
    candidate = np.load(args.candidate, mmap_mode="r", allow_pickle=False)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: {reference.shape} != {candidate.shape}"
        )
    reference_ready = np.isclose(
        reference.sum(axis=1),
        1.0,
        atol=5e-6,
        rtol=0.0,
    )
    candidate_ready = np.isclose(
        candidate.sum(axis=1),
        1.0,
        atol=5e-6,
        rtol=0.0,
    )
    shared = reference_ready & candidate_ready
    if not np.any(shared):
        raise ValueError("materializations have no completed rows in common")
    difference = np.abs(reference[shared] - candidate[shared])
    report = {
        "status": "passed",
        "shared_completed_rows": int(shared.sum()),
        "reference_completed_rows": int(reference_ready.sum()),
        "candidate_completed_rows": int(candidate_ready.sum()),
        "maximum_absolute_probability_difference": float(
            difference.max()
        ),
        "mean_absolute_probability_difference": float(difference.mean()),
        "top1_disagreement_count": int(
            np.sum(
                np.argmax(reference[shared], axis=1)
                != np.argmax(candidate[shared], axis=1)
            )
        ),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
