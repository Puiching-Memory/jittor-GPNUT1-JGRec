from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from jgrec.core.io import read_interactions
from jgrec.core.memory import release_memory
from jgrec.core.types import DatasetPaths, FitContext
from jgrec.rankers.temporal_graph.config import TemporalGraphTrainingConfig
from jgrec.rankers.temporal_graph.ranker import TemporalGraphRanker


def main() -> int:
    args = _parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    for dataset_name in args.datasets:
        dataset = _dataset_paths(args.data_dir, dataset_name)
        interactions = read_interactions(dataset.train_path)
        for group in args.groups:
            config = TemporalGraphTrainingConfig(
                seed=args.seed,
                verbose=not args.quiet,
                max_fit_events=args.max_fit_events,
                max_train_events=args.max_train_events,
                max_val_events=args.max_val_events,
                num_negatives=args.num_negatives,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                history_len=args.history_len,
                candidate_history_len=args.candidate_history_len,
                hidden_size=args.hidden_size,
                layers=args.layers,
                heads=args.heads,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                selection_metric=args.selection_metric,
                training_candidates="test_like",
                validation_candidates="test_like",
                candidate_recent_feature_group=group,
                refit_full=False,
            )
            started = time.perf_counter()
            ranker = TemporalGraphRanker()
            report = ranker.fit(
                interactions,
                training_config=config,
                context=FitContext(dataset=dataset, seed=args.seed, verbose=not args.quiet),
            )
            payload = {
                "dataset": dataset_name,
                "candidate_recent_feature_group": group,
                "elapsed_sec": time.perf_counter() - started,
                "best_val_ap": report.best_val_ap,
                "best_val_mrr": report.best_val_mrr,
                "best_epoch": report.metrics.get("best_epoch", 0.0),
                "config": asdict(config),
            }
            with args.output_path.open("a") as output:
                output.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            print(
                f"{dataset_name} group={group} "
                f"ap={report.best_val_ap:.6f} mrr={report.best_val_mrr:.6f} "
                f"epoch={report.metrics.get('best_epoch', 0.0):.0f}"
            )
            del ranker
            release_memory()
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate temporal-graph recent candidate prior features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=["dataset1", "dataset2"])
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["none", "recency_rank"],
        default=["none", "recency_rank"],
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("result/ablation/temporal_graph_recent_feature_group_seed60.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--max-fit-events", type=int, default=10_000)
    parser.add_argument("--max-train-events", type=int, default=1_000)
    parser.add_argument("--max-val-events", type=int, default=200)
    parser.add_argument("--num-negatives", type=int, default=99)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--candidate-history-len", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=["ap", "mrr"], default="ap")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _dataset_paths(data_dir: Path, name: str) -> DatasetPaths:
    root = data_dir / name
    return DatasetPaths(
        name=name,
        root=root,
        train_path=root / "train.csv",
        test_path=root / "test.csv",
    )


if __name__ == "__main__":
    raise SystemExit(main())
