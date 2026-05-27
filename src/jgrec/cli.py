from __future__ import annotations

import argparse
from pathlib import Path

import jittor as jt
from rich.panel import Panel
from rich.table import Table

from .data import discover_datasets
from .logging import console
from .model import TrainingConfig
from .submission import (
    build_dataset_submission,
    expected_test_rows,
    validate_submission_file,
    write_zip,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build JGRec dynamic recommendation submission files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing dataset*/train.csv and test.csv.")
    parser.add_argument("--recent-window", type=int, default=32, help="Number of recent destinations kept per source.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Number of test queries scored in one Jittor batch.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional smoke-test limit per dataset.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Tail ratio of train.csv reserved for local validation.")
    parser.add_argument("--context-ratio", type=float, default=0.75, help="Prefix ratio used as causal context before supervised train events.")
    parser.add_argument("--max-train-events", type=int, default=20_000, help="Max supervised events sampled for reranker training per dataset.")
    parser.add_argument("--max-val-events", type=int, default=5_000, help="Max validation events sampled for local MRR per dataset.")
    parser.add_argument("--num-negatives", type=int, default=31, help="Negative candidates per positive event during local training.")
    parser.add_argument("--max-fit-events", type=int, default=0, help="Optional tail limit for all model fitting events; 0 means full train.csv.")
    parser.add_argument("--epochs", type=int, default=5, help="Fusion MLP training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=512, help="Fusion MLP training batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Sequence tower and fusion MLP learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Model weight decay.")
    parser.add_argument("--fusion-hidden-dim", type=int, default=64, help="Hidden width of the final fusion MLP.")
    parser.add_argument("--disable-gnn", action="store_true", help="Disable JittorGeometric graph towers.")
    parser.add_argument("--gnn-model", choices=("xsimgcl", "lightgcn"), default="xsimgcl", help="JittorGeometric graph backbone.")
    parser.add_argument("--gnn-embedding-dim", type=int, default=128, help="Graph tower embedding dimension.")
    parser.add_argument("--gnn-layers", type=int, default=2, help="Graph propagation layers.")
    parser.add_argument("--gnn-epochs", type=int, default=3, help="Graph tower training epochs per time window.")
    parser.add_argument("--gnn-batch-size", type=int, default=2048, help="Graph tower BPR batch size.")
    parser.add_argument("--gnn-max-graph-edges", type=int, default=0, help="Optional tail edge limit for each graph tower; 0 means all window edges.")
    parser.add_argument("--gnn-max-train-edges", type=int, default=40_000, help="Max graph edges sampled per graph epoch; 0 means all edges.")
    parser.add_argument("--gnn-lr", type=float, default=0.001, help="Graph tower learning rate.")
    parser.add_argument("--gnn-reg-weight", type=float, default=1e-5, help="Graph tower embedding regularization.")
    parser.add_argument("--gnn-cl-rate", type=float, default=1e-4, help="XSimGCL contrastive loss weight.")
    parser.add_argument("--disable-seq", action="store_true", help="Disable SASRec sequence tower.")
    parser.add_argument("--seq-epochs", type=int, default=3, help="SASRec training epochs.")
    parser.add_argument("--seq-batch-size", type=int, default=512, help="SASRec training batch size.")
    parser.add_argument("--seq-max-samples", type=int, default=50_000, help="Max SASRec training samples per dataset; 0 means all.")
    parser.add_argument("--seq-max-len", type=int, default=64, help="Max source history length for SASRec.")
    parser.add_argument("--seq-hidden-size", type=int, default=128, help="SASRec hidden size.")
    parser.add_argument("--seq-layers", type=int, default=2, help="SASRec transformer layers.")
    parser.add_argument("--seq-heads", type=int, default=4, help="SASRec attention heads.")
    parser.add_argument("--seq-dropout", type=float, default=0.2, help="SASRec dropout.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for validation sampling and negative sampling.")
    parser.add_argument("--quiet-ranker", action="store_true", help="Suppress per-epoch reranker logs.")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA for this run.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip output format validation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jt.flags.use_cuda = 0 if args.cpu else 1

    training_config = TrainingConfig(
        val_ratio=args.val_ratio,
        context_ratio=args.context_ratio,
        max_train_events=args.max_train_events,
        max_val_events=args.max_val_events,
        num_negatives=args.num_negatives,
        max_fit_events=args.max_fit_events,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        verbose=not args.quiet_ranker,
        gnn_enabled=not args.disable_gnn,
        gnn_model=args.gnn_model,
        gnn_embedding_dim=args.gnn_embedding_dim,
        gnn_layers=args.gnn_layers,
        gnn_epochs=args.gnn_epochs,
        gnn_batch_size=args.gnn_batch_size,
        gnn_max_graph_edges=args.gnn_max_graph_edges,
        gnn_max_train_edges=args.gnn_max_train_edges,
        gnn_lr=args.gnn_lr,
        gnn_reg_weight=args.gnn_reg_weight,
        gnn_cl_rate=args.gnn_cl_rate,
        seq_enabled=not args.disable_seq,
        seq_epochs=args.seq_epochs,
        seq_batch_size=args.seq_batch_size,
        seq_max_samples=args.seq_max_samples,
        seq_max_len=args.seq_max_len,
        seq_hidden_size=args.seq_hidden_size,
        seq_layers=args.seq_layers,
        seq_heads=args.seq_heads,
        seq_dropout=args.seq_dropout,
        fusion_hidden_dim=args.fusion_hidden_dim,
    )
    run_name = _build_run_name(args, training_config)
    run_dir = Path("result") / run_name
    csv_dir = run_dir / "csv"
    zip_path = run_dir / "result.zip"
    console.print(_run_panel(run_dir, zip_path, args, training_config))

    datasets = discover_datasets(args.data_dir)
    results = []
    result_table = _result_table()
    for dataset in datasets:
        console.rule(f"[bold]{dataset.name}")
        console.print(f"[cyan]train[/cyan] {dataset.train_path}")
        console.print(f"[cyan]test [/cyan] {dataset.test_path}")
        result = build_dataset_submission(
            dataset=dataset,
            output_dir=csv_dir,
            recent_window=args.recent_window,
            batch_size=args.batch_size,
            training_config=training_config,
            limit_rows=args.limit_rows,
        )
        results.append(result)
        report = result.training_report

        if not args.skip_validate:
            expected_rows = None if args.limit_rows is not None else expected_test_rows(dataset)
            validate_submission_file(result.output_path, expected_rows=expected_rows)
        result_table.add_row(
            dataset.name,
            str(report.train_events),
            str(report.val_events),
            f"{report.best_val_mrr:.5f}",
            report.selected_fusion or "unknown",
            str(len(report.feature_names)),
            str(result.rows),
            str(result.output_path),
        )
        console.print(f"[green]wrote[/green] {result.rows} rows -> {result.output_path}")

    write_zip(results, zip_path)
    console.print(result_table)
    console.print(f"[bold green]archive[/bold green] {zip_path}")
    return 0


def _build_run_name(args: argparse.Namespace, config: TrainingConfig) -> str:
    parts = [
        f"rw{args.recent_window}",
        f"bs{args.batch_size}",
        f"vr{_num(config.val_ratio)}",
        f"cr{_num(config.context_ratio)}",
        f"tr{config.max_train_events}",
        f"va{config.max_val_events}",
        f"neg{config.num_negatives}",
        f"fit{config.max_fit_events}",
        f"ep{config.epochs}",
        f"tbs{config.train_batch_size}",
        f"lr{_num(config.lr)}",
        f"wd{_num(config.weight_decay)}",
        f"fh{config.fusion_hidden_dim}",
        f"gnn{config.gnn_model if config.gnn_enabled else 'off'}",
        f"ge{config.gnn_epochs}",
        f"gd{config.gnn_embedding_dim}",
        f"gl{config.gnn_layers}",
        f"gmge{config.gnn_max_graph_edges}",
        f"gmte{config.gnn_max_train_edges}",
        f"seq{'on' if config.seq_enabled else 'off'}",
        f"se{config.seq_epochs}",
        f"sd{config.seq_hidden_size}",
        f"sl{config.seq_max_len}",
        f"s{config.seed}",
    ]
    if args.limit_rows is not None:
        parts.append(f"limit{args.limit_rows}")
    if args.cpu:
        parts.append("cpu")
    return "-".join(parts)


def _num(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _run_panel(
    run_dir: Path,
    zip_path: Path,
    args: argparse.Namespace,
    config: TrainingConfig,
) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("output", str(run_dir))
    table.add_row("archive", str(zip_path))
    table.add_row("device", "cpu" if args.cpu else "cuda")
    table.add_row("gnn", config.gnn_model if config.gnn_enabled else "off")
    table.add_row("sequence", "on" if config.seq_enabled else "off")
    table.add_row("limit_rows", str(args.limit_rows) if args.limit_rows is not None else "full")
    table.add_row("max_fit_events", str(config.max_fit_events) if config.max_fit_events else "full")
    return Panel(table, title="JGRec build", border_style="blue")


def _result_table() -> Table:
    table = Table(title="Dataset Results")
    table.add_column("dataset", style="cyan")
    table.add_column("train", justify="right")
    table.add_column("val", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("fusion")
    table.add_column("features", justify="right")
    table.add_column("rows", justify="right")
    table.add_column("csv")
    return table


if __name__ == "__main__":
    raise SystemExit(main())
