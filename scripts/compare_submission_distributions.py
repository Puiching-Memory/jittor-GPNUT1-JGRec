"""Compare two submission CSVs row-by-row and report distribution shifts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> np.ndarray:
    """Read a submission CSV into a (n_queries, 100) float32 array."""
    with path.open("r", encoding="utf-8") as f:
        rows = [line.rstrip("\n").split(",") for line in f if line.strip()]
    data = np.asarray(rows, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 100:
        raise ValueError(f"unexpected shape {data.shape} for {path}")
    return data


def rank_scores(scores: np.ndarray) -> np.ndarray:
    """Return descending ranks (0 = highest score)."""
    return np.argsort(np.argsort(-scores, axis=1), axis=1)


def top1_overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.argmax(a, axis=1) == np.argmax(b, axis=1)))


def top5_overlap(a: np.ndarray, b: np.ndarray) -> float:
    top_a = np.argpartition(-a, 5, axis=1)[:, :5]
    top_b = np.argpartition(-b, 5, axis=1)[:, :5]
    overlaps = []
    for ra, rb in zip(top_a, top_b):
        overlaps.append(len(set(ra.tolist()) & set(rb.tolist())) / 5.0)
    return float(np.mean(overlaps))


def compare(base: np.ndarray, new: np.ndarray, *, label: str) -> dict[str, float]:
    base_sum = base.sum(axis=1)
    new_sum = new.sum(axis=1)
    base_max = base.max(axis=1)
    new_max = new.max(axis=1)

    base_rank = rank_scores(base)
    new_rank = rank_scores(new)

    return {
        f"{label}_queries": float(base.shape[0]),
        f"{label}_mean_sum_base": float(base_sum.mean()),
        f"{label}_mean_sum_new": float(new_sum.mean()),
        f"{label}_sum_delta_mean": float((new_sum - base_sum).mean()),
        f"{label}_mean_max_base": float(base_max.mean()),
        f"{label}_mean_max_new": float(new_max.mean()),
        f"{label}_max_delta_mean": float((new_max - base_max).mean()),
        f"{label}_mean_kl_to_uniform_base": float(np.mean(np.log(base * 100.0) * base)),
        f"{label}_mean_kl_to_uniform_new": float(np.mean(np.log(new * 100.0) * new)),
        f"{label}_top1_overlap": top1_overlap(base, new),
        f"{label}_top5_overlap": top5_overlap(base, new),
        f"{label}_mean_rank_delta": float(np.abs(new_rank - base_rank).mean()),
        f"{label}_correlation_mean_per_query": float(np.mean(
            [np.corrcoef(base[i], new[i])[0, 1] for i in range(base.shape[0])]
        )),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two submission CSV distributions")
    parser.add_argument("base", type=Path, help="base submission CSV")
    parser.add_argument("new", type=Path, help="new submission CSV")
    parser.add_argument("--dataset", default="dataset", help="label prefix")
    args = parser.parse_args()

    base_arr = read_csv(args.base)
    new_arr = read_csv(args.new)
    if base_arr.shape != new_arr.shape:
        raise ValueError(f"shape mismatch: {base_arr.shape} vs {new_arr.shape}")

    metrics = compare(base_arr, new_arr, label=args.dataset)
    max_key_len = max(len(k) for k in metrics)
    for key in sorted(metrics):
        value = metrics[key]
        if "overlap" in key or "correlation" in key:
            print(f"{key:<{max_key_len}} {value:.6f}")
        else:
            print(f"{key:<{max_key_len}} {value:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
