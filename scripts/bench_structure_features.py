from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from jgrec.core.io import read_interactions, read_test_queries
from jgrec.core.types import Interaction
from jgrec.rankers.hybrid.config import StructureTowerConfig
from jgrec.rankers.hybrid.structure import StructureFeatureTower


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark hybrid structure feature query speed.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="dataset2")
    parser.add_argument("--limit-rows", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-fit-events", type=int, default=0)
    parser.add_argument("--no-future-only", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.data_dir / args.dataset
    train_array = read_interactions(dataset_dir / "train.csv")
    if args.max_fit_events > 0 and len(train_array) > args.max_fit_events:
        train_array = train_array[-args.max_fit_events :]
    interactions = [Interaction(src=int(src), dst=int(dst), time=int(time)) for src, dst, time in train_array]
    query_array = read_test_queries(dataset_dir / "test.csv", max_rows=args.limit_rows)
    queries = list(query_array)

    tower = StructureFeatureTower(
        StructureTowerConfig(future_only_transition_cooccur=not args.no_future_only)
    )
    fit_start = perf_counter()
    tower.fit(interactions, rng=np.random.default_rng(0), verbose=False)
    fit_elapsed = perf_counter() - fit_start

    checksum = 0.0
    rows = 0
    query_start = perf_counter()
    batch_size = max(int(args.batch_size), 1)
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        features = tower.features_for_queries(batch)
        checksum += float(features.sum(dtype=np.float64))
        rows += len(batch)
    elapsed = perf_counter() - query_start
    rows_per_sec = rows / elapsed if elapsed > 0 else float("inf")

    print(f"dataset={args.dataset}")
    print(f"future_only={not args.no_future_only}")
    print(f"train_events={len(interactions)}")
    print(f"rows={rows}")
    print(f"batch_size={batch_size}")
    print(f"fit_elapsed={fit_elapsed:.3f}s")
    print(f"query_elapsed={elapsed:.3f}s")
    print(f"rows_per_sec={rows_per_sec:.3f}")
    print(f"feature_checksum={checksum:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
