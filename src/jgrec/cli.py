from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal

import jittor as jt
import tyro
from rich.panel import Panel
from rich.table import Table

from .core.io import discover_datasets
from .core.memory import configure_memory_log, log_memory, release_memory
from .core.runner import build_dataset_submission
from .core.types import DatasetPaths, DatasetResult, TrainingReport
from .logging import console
from .rankers.craft.config import CRAFTBaselineConfig
from .rankers.hybrid.config import TrainingConfig
from .rankers.registry import create_ranker
from .submission import expected_test_rows, validate_submission_file, write_zip

ModelName = Literal["hybrid", "craft"]
SelectionMetric = Literal["ap", "mrr"]
GNNModel = Literal["xsimgcl", "lightgcn"]
GNNEdgeWeighting = Literal["none", "repeat", "time_decay"]


@dataclass(frozen=True)
class CLIConfig:
    """Build JGRec dynamic recommendation submission files."""

    model: ModelName = "hybrid"
    data_dir: Path = Path("data")
    dataset: str = ""
    run_name: str = ""
    resume_existing: bool = False
    recent_window: int = 32
    batch_size: int = 2048
    limit_rows: int | None = None
    val_ratio: float = 0.15
    context_ratio: float = 0.75
    max_train_events: int = 20_000
    max_val_events: int = 5_000
    supervised_feature_batch_size: int = 4096
    supervised_feature_memmap: bool = False
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
    gnn_edge_weighting: GNNEdgeWeighting = "none"
    gnn_time_decay_ratio: float = 0.05
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
    seq_score_batch_size: int = 1024
    seq_max_samples: int = 50_000
    seq_max_len: int = 64
    seq_hidden_size: int = 128
    seq_layers: int = 2
    seq_heads: int = 4
    seq_dropout: float = 0.2
    disable_two_tower: bool = False
    two_tower_embedding_dim: int = 64
    two_tower_hidden_dim: int = 64
    two_tower_epochs: int = 3
    two_tower_batch_size: int = 512
    two_tower_score_batch_size: int = 2048
    two_tower_max_samples: int = 50_000
    hard_negative_ratio: float = 0.5
    popular_negative_ratio: float = 0.25
    negative_sampling_workers: int = 0
    craft_neighbors: int = 30
    craft_hidden_size: int = 64
    seed: int = 42
    quiet_ranker: bool = False
    cpu: bool = False
    skip_validate: bool = False
    encoder_state_cache: bool = True
    auto_strategy: bool = True
    disable_candidate_prior: bool = False
    disable_target_window: bool = False
    target_window_fractions: str = "0.01,0.05,0.20,1.00"
    test_candidate_negative_ratio: float = 0.0
    disable_structure: bool = False
    disable_structure_cooccur: bool = False
    disable_structure_transition: bool = False
    structure_cooccur_history_limit: int = 128
    disable_source_profile: bool = False
    disable_source_profile_deterministic: bool = False
    disable_source_profile_item2vec: bool = False
    source_profile_embedding_dim: int = 64
    source_profile_epochs: int = 3
    source_profile_batch_size: int = 2048
    source_profile_score_batch_size: int = 8192
    source_profile_max_samples: int = 100_000
    source_profile_window_size: int = 16
    source_profile_recent_k: int = 32


def main(argv: list[str] | None = None) -> int:
    args = tyro.cli(CLIConfig, args=argv)
    jt.flags.use_cuda = 0 if args.cpu else 1

    ranker_config = _ranker_config(args)
    run_name = args.run_name or _build_run_name(args, ranker_config)
    run_dir = Path("result") / run_name
    csv_dir = run_dir / "csv"
    zip_path = run_dir / "result.zip"
    configure_memory_log(run_dir / "memory.log")
    log_memory("cli_start", enabled=not args.quiet_ranker)
    console.print(_run_panel(run_dir, zip_path, args, ranker_config))

    datasets = discover_datasets(args.data_dir)
    selected_datasets = _select_datasets(datasets, args.dataset)
    results = []
    result_table = _result_table()
    for dataset in datasets:
        output_path = csv_dir / f"{dataset.name}.csv"
        selected = dataset.name in selected_datasets
        if args.resume_existing and output_path.exists() and (not selected or args.limit_rows is None):
            result = _reuse_existing_result(dataset, output_path, args)
            results.append(result)
            _add_result_row(result_table, result, args.model, reused=True)
            console.print(f"[yellow]reused[/yellow] {result.rows} rows -> {result.output_path}")
            continue
        if not selected:
            continue

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

        if not args.skip_validate:
            expected_rows = None if args.limit_rows is not None else expected_test_rows(dataset)
            validate_submission_file(result.output_path, expected_rows=expected_rows)
        _add_result_row(result_table, result, args.model)
        console.print(f"[green]wrote[/green] {result.rows} rows -> {result.output_path}")
        del ranker
        release_memory()

    if args.resume_existing and len(results) != len(datasets):
        present = {result.name for result in results}
        missing = ", ".join(dataset.name for dataset in datasets if dataset.name not in present)
        raise RuntimeError(f"resume output is incomplete; missing CSV for: {missing}")

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
    return TrainingConfig(
        val_ratio=args.val_ratio,
        context_ratio=args.context_ratio,
        max_train_events=args.max_train_events,
        max_val_events=args.max_val_events,
        supervised_feature_batch_size=args.supervised_feature_batch_size,
        supervised_feature_memmap=args.supervised_feature_memmap,
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
        encoder_state_cache_enabled=args.encoder_state_cache,
        auto_strategy_enabled=args.auto_strategy,
        candidate_prior_enabled=not args.disable_candidate_prior,
        target_window_enabled=not args.disable_target_window,
        target_window_fractions=_parse_target_window_fractions(args.target_window_fractions),
        test_candidate_negative_ratio=args.test_candidate_negative_ratio,
        structure_enabled=not args.disable_structure,
        structure_cooccur_enabled=not args.disable_structure_cooccur,
        structure_transition_enabled=not args.disable_structure_transition,
        structure_cooccur_history_limit=args.structure_cooccur_history_limit,
        source_profile_enabled=not args.disable_source_profile,
        source_profile_deterministic_enabled=not args.disable_source_profile_deterministic,
        source_profile_item2vec_enabled=not args.disable_source_profile_item2vec,
        source_profile_embedding_dim=args.source_profile_embedding_dim,
        source_profile_epochs=args.source_profile_epochs,
        source_profile_batch_size=args.source_profile_batch_size,
        source_profile_score_batch_size=args.source_profile_score_batch_size,
        source_profile_max_samples=args.source_profile_max_samples,
        source_profile_window_size=args.source_profile_window_size,
        source_profile_recent_k=args.source_profile_recent_k,
        gnn_enabled=not args.disable_gnn,
        gnn_model=args.gnn_model,
        gnn_edge_weighting=args.gnn_edge_weighting,
        gnn_time_decay_ratio=args.gnn_time_decay_ratio,
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
        seq_score_batch_size=args.seq_score_batch_size,
        seq_max_samples=args.seq_max_samples,
        seq_max_len=args.seq_max_len,
        seq_hidden_size=args.seq_hidden_size,
        seq_layers=args.seq_layers,
        seq_heads=args.seq_heads,
        seq_dropout=args.seq_dropout,
        two_tower_enabled=not args.disable_two_tower,
        two_tower_embedding_dim=args.two_tower_embedding_dim,
        two_tower_hidden_dim=args.two_tower_hidden_dim,
        two_tower_epochs=args.two_tower_epochs,
        two_tower_batch_size=args.two_tower_batch_size,
        two_tower_score_batch_size=args.two_tower_score_batch_size,
        two_tower_max_samples=args.two_tower_max_samples,
        fusion_hidden_dim=args.fusion_hidden_dim,
        hard_negative_ratio=args.hard_negative_ratio,
        popular_negative_ratio=args.popular_negative_ratio,
        negative_sampling_workers=args.negative_sampling_workers,
    )


def _build_run_name(args: CLIConfig, config) -> str:
    rows = f"sample-{args.limit_rows}-rows" if args.limit_rows is not None else "full"
    parts = [_slug(args.model), rows, "cpu" if args.cpu else "cuda", f"seed-{args.seed}"]
    if args.model == "hybrid":
        parts.extend(
            [
                f"gnn-{_slug(config.gnn_model) if config.gnn_enabled else 'off'}",
                f"edges-{_slug(config.gnn_edge_weighting)}" if config.gnn_enabled else "edges-off",
                f"auto-{'on' if config.auto_strategy_enabled else 'off'}",
                f"prior-{'on' if config.candidate_prior_enabled else 'off'}",
                f"target-{'on' if config.target_window_enabled else 'off'}",
                f"profile-{'on' if config.source_profile_enabled else 'off'}",
                f"tower-{'on' if config.two_tower_enabled else 'off'}",
                f"sequence-{'on' if config.seq_enabled else 'off'}",
            ]
        )
    parts.append(_config_digest(args, config))
    return "_".join(parts)


def _config_digest(args: CLIConfig, config) -> str:
    cli_payload = _jsonable(args)
    for operational_key in ("dataset", "run_name", "resume_existing", "encoder_state_cache"):
        if isinstance(cli_payload, dict):
            cli_payload.pop(operational_key, None)
    ranker_payload = _jsonable(config)
    if isinstance(ranker_payload, dict):
        ranker_payload.pop("encoder_state_cache_enabled", None)
    payload = {
        "cli": cli_payload,
        "ranker": ranker_payload,
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
        table.add_row("gnn_edge_weighting", config.gnn_edge_weighting if config.gnn_enabled else "off")
        table.add_row("auto_strategy", "on" if config.auto_strategy_enabled else "off")
        table.add_row("encoder_cache", "on" if config.encoder_state_cache_enabled else "off")
        table.add_row("candidate_prior", "on" if config.candidate_prior_enabled else "off")
        table.add_row("target_window", "on" if config.target_window_enabled else "off")
        table.add_row("target_window_fractions", ",".join(f"{value:.2f}" for value in config.target_window_fractions))
        table.add_row("auto_mode", config.auto_mode if config.auto_mode != "manual" else "pending")
        table.add_row("candidate_unseen_dst_rate", f"{config.profile_candidate_unseen_dst_rate:.5f}")
        table.add_row("holdout_pair_hit_rate", f"{config.profile_holdout_pair_hit_rate:.5f}")
        table.add_row("test_candidate_negative_ratio", f"{config.test_candidate_negative_ratio:.2f}")
        table.add_row("source_profile", "on" if config.source_profile_enabled else "off")
        table.add_row(
            "source_profile_item2vec",
            "on" if config.source_profile_enabled and config.source_profile_item2vec_enabled else "off",
        )
        table.add_row("source_profile_epochs/max_samples", f"{config.source_profile_epochs}/{config.source_profile_max_samples}")
        table.add_row("two_tower", "on" if config.two_tower_enabled else "off")
        table.add_row("sequence", "on" if config.seq_enabled else "off")
        table.add_row("structure", "on" if config.structure_enabled else "off")
        table.add_row("structure_cooccur", "on" if config.structure_cooccur_enabled else "off")
        table.add_row("structure_transition", "on" if config.structure_transition_enabled else "off")
        table.add_row("structure_cooccur_history_limit", str(config.structure_cooccur_history_limit))
        table.add_row("max_fit_events", str(config.max_fit_events) if config.max_fit_events else "full")
        table.add_row("supervised_feature_batch_size", str(config.supervised_feature_batch_size))
        table.add_row("supervised_feature_memmap", "on" if config.supervised_feature_memmap else "off")
        table.add_row("negative_sampling_workers", str(config.negative_sampling_workers))
    return Panel(table, title="JGRec build", border_style="blue")


def _select_datasets(datasets: list[DatasetPaths], dataset_name: str) -> set[str]:
    if not dataset_name:
        return {dataset.name for dataset in datasets}
    selected = dataset_name.strip()
    names = {dataset.name for dataset in datasets}
    if selected not in names:
        raise ValueError(f"unknown dataset '{selected}', available: {', '.join(sorted(names))}")
    return {selected}


def _parse_target_window_fractions(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    fractions = tuple(float(part) for part in parts)
    if len(fractions) != 4:
        raise ValueError("--target-window-fractions must contain exactly four comma-separated numbers")
    if any(fraction <= 0.0 for fraction in fractions):
        raise ValueError("--target-window-fractions values must be positive")
    return fractions


def _reuse_existing_result(dataset: DatasetPaths, output_path: Path, args: CLIConfig) -> DatasetResult:
    expected_rows = args.limit_rows if args.limit_rows is not None else expected_test_rows(dataset)
    validate_submission_file(output_path, expected_rows=expected_rows)
    return DatasetResult(
        name=dataset.name,
        rows=expected_rows,
        output_path=output_path,
        training_report=TrainingReport(model_name="reused", selected_fusion="existing"),
    )


def _add_result_row(table: Table, result: DatasetResult, fallback_model: str, reused: bool = False) -> None:
    report = result.training_report
    metrics = report.metrics
    table.add_row(
        result.name,
        report.model_name or fallback_model,
        "-" if reused else str(report.train_events),
        "-" if reused else str(report.val_events),
        "-" if reused else f"{report.best_val_ap:.5f}",
        "-" if reused else f"{report.best_val_mrr:.5f}",
        report.selected_fusion or "unknown",
        str(len(report.feature_names)),
        "-" if reused else _auto_mode_label(metrics.get("auto_mode_code", 0.0)),
        "-" if reused else f"{metrics.get('holdout_pair_hit_rate', 0.0):.5f}",
        "-" if reused else f"{metrics.get('candidate_unseen_dst_rate', 0.0):.5f}",
        "-" if reused else f"{metrics.get('test_candidate_negative_ratio', 0.0):.2f}",
        str(result.rows),
        str(result.output_path),
    )


def _auto_mode_label(code: float) -> str:
    if code == 1.0:
        return "repeat_memory"
    if code == 2.0:
        return "balanced"
    if code == 3.0:
        return "new_link_cold"
    return "manual"


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
    table.add_column("auto")
    table.add_column("pair_hit", justify="right")
    table.add_column("unseen", justify="right")
    table.add_column("test_neg", justify="right")
    table.add_column("rows", justify="right")
    table.add_column("csv")
    return table


if __name__ == "__main__":
    raise SystemExit(main())
