from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal

import tyro
from rich.panel import Panel
from rich.table import Table

from .core.io import discover_datasets
from .core.runner import build_dataset_submission
from .logging import console
from .rankers.craft.config import CRAFTBaselineConfig
from .rankers.hybrid import TrainingConfig
from .rankers.registry import create_ranker
from .rankers.third_party import ThirdPartyRankerConfig
from .submission import expected_test_rows, validate_submission_file, write_zip

ModelName = Literal["hybrid", "craft", "third_party"]
SelectionMetric = Literal["ap", "mrr"]
GNNModel = Literal["xsimgcl", "lightgcn"]


@dataclass(frozen=True)
class CLIConfig:
    """Build JGRec dynamic recommendation submission files."""

    model: ModelName = "hybrid"
    data_dir: Path = Path("data")
    recent_window: int = 32
    batch_size: int = 2048
    limit_rows: int | None = None
    val_ratio: float = 0.15
    context_ratio: float = 0.75
    max_train_events: int = 20_000
    max_val_events: int = 5_000
    num_negatives: int = 31
    max_fit_events: int = 0
    epochs: int = 5
    train_batch_size: int = 512
    lr: float = 0.001
    weight_decay: float = 0.0
    selection_metric: SelectionMetric = "ap"
    early_stop: int = 10
    fusion_hidden_dim: int = 64
    disable_gnn: bool = False
    gnn_model: GNNModel = "xsimgcl"
    gnn_embedding_dim: int = 128
    gnn_layers: int = 2
    gnn_epochs: int = 3
    gnn_batch_size: int = 2048
    gnn_max_graph_edges: int = 0
    gnn_max_train_edges: int = 40_000
    gnn_lr: float = 0.001
    gnn_reg_weight: float = 1e-5
    gnn_cl_rate: float = 1e-4
    disable_seq: bool = False
    seq_epochs: int = 3
    seq_batch_size: int = 512
    seq_max_samples: int = 50_000
    seq_max_len: int = 64
    seq_hidden_size: int = 128
    seq_layers: int = 2
    seq_heads: int = 4
    seq_dropout: float = 0.2
    craft_neighbors: int = 30
    craft_hidden_size: int = 64
    third_cooccur_k: int = 16
    seed: int = 42
    quiet_ranker: bool = False
    cpu: bool = False
    skip_validate: bool = False


def main(argv: list[str] | None = None) -> int:
    args = tyro.cli(CLIConfig, args=argv)
    import jittor as jt

    jt.flags.use_cuda = 0 if args.cpu else 1

    ranker_config = _ranker_config(args)
    run_name = _build_run_name(args, ranker_config)
    run_dir = Path("result") / run_name
    csv_dir = run_dir / "csv"
    zip_path = run_dir / "result.zip"
    console.print(_run_panel(run_dir, zip_path, args, ranker_config))

    datasets = discover_datasets(args.data_dir)
    results = []
    result_table = _result_table()
    for dataset in datasets:
        console.rule(f"[bold]{dataset.name}")
        console.print(f"[cyan]train[/cyan] {dataset.train_path}")
        console.print(f"[cyan]test [/cyan] {dataset.test_path}")
        ranker = create_ranker(args.model, ranker_config)
        result = build_dataset_submission(
            dataset=dataset,
            ranker=ranker,
            output_dir=csv_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=not args.quiet_ranker,
            limit_rows=args.limit_rows,
        )
        results.append(result)
        report = result.training_report

        if not args.skip_validate:
            expected_rows = None if args.limit_rows is not None else expected_test_rows(dataset)
            validate_submission_file(result.output_path, expected_rows=expected_rows)
        result_table.add_row(
            dataset.name,
            report.model_name or args.model,
            str(report.train_events),
            str(report.val_events),
            f"{report.best_val_ap:.5f}",
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


def _ranker_config(args: CLIConfig):
    if args.model == "craft":
        return CRAFTBaselineConfig(
            val_ratio=args.val_ratio,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            lr=args.lr,
            early_stop_patience=args.early_stop,
            num_neighbors=args.craft_neighbors,
            hidden_size=args.craft_hidden_size,
        )
    if args.model == "third_party":
        return ThirdPartyRankerConfig(
            val_ratio=args.val_ratio,
            context_ratio=args.context_ratio,
            max_train_events=args.max_train_events,
            max_val_events=args.max_val_events,
            num_negatives=args.num_negatives,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.fusion_hidden_dim,
            selection_metric=args.selection_metric,
            early_stop_patience=args.early_stop,
            seed=args.seed,
            verbose=not args.quiet_ranker,
            recent_window=args.recent_window,
            cooccur_recent_k=args.third_cooccur_k,
        )
    return TrainingConfig(
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
        selection_metric=args.selection_metric,
        early_stop_patience=args.early_stop,
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


def _build_run_name(args: CLIConfig, config) -> str:
    rows = f"sample-{args.limit_rows}-rows" if args.limit_rows is not None else "full"
    parts = [_slug(args.model), rows, "cpu" if args.cpu else "cuda", f"seed-{args.seed}"]
    if args.model == "hybrid":
        parts.extend(
            [
                f"gnn-{_slug(config.gnn_model) if config.gnn_enabled else 'off'}",
                f"sequence-{'on' if config.seq_enabled else 'off'}",
            ]
        )
    parts.append(_config_digest(args, config))
    return "_".join(parts)


def _config_digest(args: CLIConfig, config) -> str:
    payload = {
        "cli": _jsonable(args),
        "ranker": _jsonable(config),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=4).hexdigest()


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _slug(value: object) -> str:
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def _run_panel(run_dir: Path, zip_path: Path, args: CLIConfig, config) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("output", str(run_dir))
    table.add_row("archive", str(zip_path))
    table.add_row("model", args.model)
    table.add_row("device", "cpu" if args.cpu else "cuda")
    table.add_row("selection_metric", getattr(config, "selection_metric", "ap"))
    table.add_row("early_stop", str(getattr(config, "early_stop_patience", args.early_stop)))
    table.add_row("limit_rows", str(args.limit_rows) if args.limit_rows is not None else "full")
    if args.model == "hybrid":
        table.add_row("gnn", config.gnn_model if config.gnn_enabled else "off")
        table.add_row("sequence", "on" if config.seq_enabled else "off")
        table.add_row("max_fit_events", str(config.max_fit_events) if config.max_fit_events else "full")
    return Panel(table, title="JGRec build", border_style="blue")


def _result_table() -> Table:
    table = Table(title="Dataset Results")
    table.add_column("dataset", style="cyan")
    table.add_column("model")
    table.add_column("train", justify="right")
    table.add_column("val", justify="right")
    table.add_column("AP", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("fusion")
    table.add_column("features", justify="right")
    table.add_column("rows", justify="right")
    table.add_column("csv")
    return table


if __name__ == "__main__":
    raise SystemExit(main())
