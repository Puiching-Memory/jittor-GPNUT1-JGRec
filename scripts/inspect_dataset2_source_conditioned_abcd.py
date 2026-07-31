from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()

    fold_metrics: dict[str, list[float]] = {}
    scales: dict[str, dict[str, Any]] = {}
    for variant in "ABCD":
        values = []
        for fold in range(3):
            artifact_dir = (
                args.result_dir
                / "folds"
                / f"variant-{variant}"
                / f"fold-{fold}"
            )
            with (artifact_dir / "report.json").open(
                "r",
                encoding="utf-8",
            ) as handle:
                report = json.load(handle)
            values.append(float(report["score_metrics"]["full"]))
            if variant != "A":
                scales[f"{variant}{fold}"] = _scale_values(
                    artifact_dir / "model.npz"
                )
        fold_metrics[variant] = values
    scales["full"] = _scale_values(
        args.result_dir / "full" / "model.npz"
    )
    output = {
        "fold_metrics": fold_metrics,
        "three_fold_means": {
            variant: float(np.mean(values))
            for variant, values in fold_metrics.items()
        },
        "learned_scales": scales,
    }
    print(json.dumps(output, indent=2))
    return 0


def _scale_values(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            key.removeprefix("state__"): np.asarray(
                payload[key]
            ).tolist()
            for key in payload.files
            if "scale" in key
        }


if __name__ == "__main__":
    raise SystemExit(main())
