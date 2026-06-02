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
from .rankers.registry import create_ranker
from .rankers.temporal_graph import TemporalGraphTrainingConfig
from .submission import expected_test_rows, validate_submission_file, write_zip

ModelName = Literal["temporal-graph", "craft"]
SelectionMetric = Literal["ap", "mrr"]


@dataclass(frozen=True)
class CLIConfig:
    """Build JGRec dynamic recommendation submission files."""

    model: ModelName = "temporal-graph"
    data_dir: Path = Path("data")
    batch_size: int = 2048
    limit_rows: int | None = None
    val_ratio: float = 0.15
    max_train_events: int = 20_000
    max_val_events: int = 5_000
    num_negatives: int = 99
    max_fit_events: int = 0
    epochs: int = 8
    train_batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 0.0
    selection_metric: SelectionMetric = "ap"
    early_stop: int = 10
    history_len: int = 64
    candidate_history_len: int = 32
    hidden_size: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.15
    no_refit_full: bool = False
    craft_neighbors: int = 30
    craft_hidden_size: int = 64
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
    return TemporalGraphTrainingConfig(
        val_ratio=args.val_ratio,
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
        history_len=args.history_len,
        candidate_history_len=args.candidate_history_len,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        refit_full=not args.no_refit_full,
    )


def _build_run_name(args: CLIConfig, config) -> str:
    rows = f"sample-{args.limit_rows}-rows" if args.limit_rows is not None else "full"
    parts = [_slug(args.model), rows, "cpu" if args.cpu else "cuda", f"seed-{args.seed}"]
    if args.model == "temporal-graph":
        parts.extend(
            [
                f"hist-{config.history_len}",
                f"candhist-{config.candidate_history_len}",
                f"dim-{config.hidden_size}",
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
    if args.model == "temporal-graph":
        table.add_row("history_len", str(config.history_len))
        table.add_row("candidate_history_len", str(config.candidate_history_len))
        table.add_row("hidden_size", str(config.hidden_size))
        table.add_row("layers", str(config.layers))
        table.add_row("heads", str(config.heads))
        table.add_row("refit_full", "on" if config.refit_full else "off")
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
