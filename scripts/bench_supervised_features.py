from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jgrec.core.io import discover_datasets, read_interactions
from jgrec.idmap import NodeIdMap
from jgrec.rankers.hybrid.config import TrainingConfig
from jgrec.rankers.hybrid.ranker import HybridFeatureEncoder, SupervisedFeatureBuilder


def main() -> int:
    args = _parse_args()
    dataset = _dataset_path(args.data_dir, args.dataset)
    interactions = read_interactions(dataset.train_path).sort_by_time()
    if args.max_fit_events > 0 and len(interactions) > args.max_fit_events:
        interactions = interactions.tail(args.max_fit_events)

    split = max(2, int(len(interactions) * 0.75))
    context_events = interactions[:split]
    positives = interactions[split : split + args.events]
    if not positives:
        raise ValueError("no positive events selected for supervised feature benchmark")

    config = TrainingConfig(
        supervised_feature_batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        negative_sampling_workers=args.workers,
        candidate_prior_enabled=True,
        structure_enabled=True,
        two_tower_enabled=False,
        gnn_enabled=False,
        seq_enabled=False,
        verbose=False,
    )
    encoder = HybridFeatureEncoder(
        id_map=NodeIdMap.from_interactions(interactions),
        recent_window=args.recent_window,
        candidate_prior_config=config.candidate_prior_config(),
        structure_config=config.structure_config(),
        two_tower_config=config.two_tower_config(),
        graph_config=config.graph_config(),
        sequence_config=config.sequence_config(),
    )
    encoder.fit(context_events, rng=np.random.default_rng(args.seed), verbose=False)
    dst_pool = np.unique(interactions.dst).astype(np.int64, copy=False)
    builder = SupervisedFeatureBuilder(encoder=encoder, dst_pool=dst_pool, config=config)
    rng = np.random.default_rng(args.seed)

    feature_checksum = 0.0
    candidate_checksum = 0
    start_time = perf_counter()
    for start in range(0, len(positives), args.batch_size):
        end = min(start + args.batch_size, len(positives))
        features, batch_candidate_checksum = builder.features_for_events(positives[start:end], rng)
        feature_checksum += float(np.asarray(features, dtype=np.float64).sum())
        candidate_checksum += batch_candidate_checksum
    elapsed = perf_counter() - start_time
    rows_per_second = len(positives) / max(elapsed, 1e-9)
    rss_mb = _rss_mb()

    print(f"dataset={dataset.name}")
    print(f"rows={len(positives)}")
    print(f"batch_size={args.batch_size}")
    print(f"num_negatives={args.num_negatives}")
    print(f"workers={args.workers}")
    print(f"elapsed={elapsed:.3f}s")
    print(f"rows_per_second={rows_per_second:.2f}")
    print(f"sample_elapsed={builder.sample_elapsed:.3f}s")
    print(f"encode_elapsed={builder.encode_elapsed:.3f}s")
    print(f"feature_checksum={feature_checksum:.6f}")
    print(f"candidate_checksum={candidate_checksum}")
    print(f"negative_checksum={builder.negative_checksum}")
    if rss_mb is not None:
        print(f"rss_mb={rss_mb:.1f}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark hybrid supervised feature construction.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="dataset2")
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-negatives", type=int, default=63)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-fit-events", type=int, default=50_000)
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _dataset_path(data_dir: Path, dataset_name: str):
    datasets = discover_datasets(data_dir)
    for dataset in datasets:
        if dataset.name == dataset_name:
            return dataset
    available = ", ".join(dataset.name for dataset in datasets)
    raise ValueError(f"unknown dataset {dataset_name!r}; available: {available}")


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
