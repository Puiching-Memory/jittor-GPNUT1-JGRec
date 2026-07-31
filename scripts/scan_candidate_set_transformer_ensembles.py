from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="NAME=validation-logits.npy",
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    models = dict(_parse_model(value) for value in args.model)
    if len(models) < 2:
        raise ValueError("ensemble scan requires at least two models")
    logits = {
        name: np.load(path, allow_pickle=False)
        for name, path in models.items()
    }
    expected_shape = next(iter(logits.values())).shape
    if any(value.shape != expected_shape for value in logits.values()):
        raise ValueError("Candidate-Set Transformer logits do not align")
    baseline = np.load(args.baseline, allow_pickle=False)
    if baseline.shape != expected_shape:
        raise ValueError("baseline scores do not align with model logits")
    selection_stop = (expected_shape[0] * 2 + 2) // 3
    weights = np.arange(0.0, 1.0 + args.step / 2.0, args.step)
    probability = {
        name: _softmax(value) for name, value in logits.items()
    }
    trials = []
    for left, right in combinations(models, 2):
        for weight in weights:
            scores = (
                float(weight) * probability[left]
                + (1.0 - float(weight)) * probability[right]
            )
            metrics = _metrics(scores)
            trials.append(
                {
                    "left": left,
                    "right": right,
                    "left_weight": float(weight),
                    "method": "probability",
                    "selection_mrr": _mrr(scores[:selection_stop]),
                    "metrics": metrics,
                }
            )
    selected = max(
        trials,
        key=lambda row: (
            row["selection_mrr"],
            row["metrics"]["full"],
        ),
    )
    report = {
        "protocol": "selection_first_two_slices_forward_third_slice",
        "rank_ensemble": (
            "forbidden: exact percentile-rank ties inflate MRR when the "
            "validation positive is stored at candidate position zero"
        ),
        "selection_rows": [0, selection_stop],
        "forward_rows": [selection_stop, expected_shape[0]],
        "baseline": _metrics(baseline),
        "single_models": {
            name: _metrics(values)
            for name, values in probability.items()
        },
        "selected": selected,
        "delta_vs_baseline": {
            key: selected["metrics"][key] - _metrics(baseline)[key]
            for key in selected["metrics"]
        },
        "top_selection_trials": sorted(
            trials,
            key=lambda row: (
                row["selection_mrr"],
                row["metrics"]["full"],
            ),
            reverse=True,
        )[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _parse_model(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError("--model must use NAME=PATH")
    return name, Path(path)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - values.max(
        axis=1,
        keepdims=True,
    )
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _mrr(scores: np.ndarray) -> float:
    ranks = 1 + (scores[:, 1:] > scores[:, :1]).sum(axis=1)
    return float(np.mean(1.0 / ranks))


def _metrics(scores: np.ndarray) -> dict[str, float]:
    parts = np.array_split(scores, 3, axis=0)
    return {
        "full": _mrr(scores),
        **{
            f"slice_{index}": _mrr(part)
            for index, part in enumerate(parts)
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
