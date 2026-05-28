from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from jgrec.data import read_interactions, read_test_queries
from jgrec.stats import TemporalStats


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark TemporalStats fit and candidate feature construction.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing dataset*/train.csv and test.csv.")
    parser.add_argument("--dataset", default="dataset2", help="Dataset directory name under --data-dir.")
    parser.add_argument("--limit-queries", type=int, default=8192, help="Number of test queries used for feature benchmark.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of measured repeats.")
    parser.add_argument("--warmups", type=int, default=1, help="Number of unmeasured feature-construction warmups.")
    parser.add_argument("--recent-window", type=int, default=32, help="TemporalStats recent destination window.")
    args = parser.parse_args()

    train_path = args.data_dir / args.dataset / "train.csv"
    test_path = args.data_dir / args.dataset / "test.csv"
    interactions = list(read_interactions(train_path))
    queries = []
    for query in read_test_queries(test_path):
        queries.append(query)
        if len(queries) >= args.limit_queries:
            break

    if not interactions:
        raise ValueError(f"no interactions loaded from {train_path}")
    if not queries:
        raise ValueError(f"no queries loaded from {test_path}")

    fit_times: list[float] = []
    for _ in range(args.repeats):
        stats = TemporalStats(recent_window=args.recent_window)
        start = time.perf_counter()
        stats.fit(interactions)
        fit_times.append(time.perf_counter() - start)

    stats = TemporalStats(recent_window=args.recent_window)
    stats.fit(interactions)
    cold_feature_times: list[float] = []
    cold_features = np.empty((0, 0, 0), dtype=np.float32)
    for _ in range(args.repeats):
        cold_stats = TemporalStats(recent_window=args.recent_window)
        cold_stats.fit(interactions)
        start = time.perf_counter()
        cold_features = cold_stats.features_for_queries(queries)
        cold_feature_times.append(time.perf_counter() - start)

    for _ in range(args.warmups):
        features = stats.features_for_queries(queries)

    warm_feature_times: list[float] = []
    features = np.empty((0, 0, 0), dtype=np.float32)
    for _ in range(args.repeats):
        start = time.perf_counter()
        features = stats.features_for_queries(queries)
        warm_feature_times.append(time.perf_counter() - start)

    checksum = float(np.sum(features, dtype=np.float64))
    cold_checksum = float(np.sum(cold_features, dtype=np.float64))
    print(f"dataset={args.dataset}")
    print(f"train_interactions={len(interactions)}")
    print(f"queries={len(queries)}")
    print(f"candidates={len(queries) * len(queries[0].candidates)}")
    print(f"feature_shape={tuple(features.shape)}")
    print(f"feature_checksum={checksum:.8f}")
    print(f"cold_feature_checksum={cold_checksum:.8f}")
    print(f"fit_median={statistics.median(fit_times):.6f}s")
    print(f"fit_times={_format_times(fit_times)}")
    print(f"features_cold_median={statistics.median(cold_feature_times):.6f}s")
    print(f"features_cold_times={_format_times(cold_feature_times)}")
    print(f"features_warm_median={statistics.median(warm_feature_times):.6f}s")
    print(f"features_warm_times={_format_times(warm_feature_times)}")
    return 0


def _format_times(values: list[float]) -> str:
    return ",".join(f"{value:.6f}" for value in values)


if __name__ == "__main__":
    raise SystemExit(main())
