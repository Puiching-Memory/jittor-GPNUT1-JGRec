from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal

from rich.panel import Panel
from rich.table import Table

from .contest_checkpoint import (
    ContestCheckpointWriter,
    load_checkpoint_dataset,
    load_checkpoint_metadata,
)
from .core.io import discover_datasets
from .core.memory import configure_memory_log, log_memory, release_memory
from .core.runner import build_dataset_submission
from .core.types import DatasetPaths, DatasetResult, TrainingReport
from .logging import console
from .rankers.craft.config import CRAFTBaselineConfig
from .rankers.hybrid.config import TrainingConfig
from .rankers.registry import create_ranker
from .submission import expected_test_rows, validate_submission_file, write_zip

ModelName = Literal["hybrid", "hybrid-heuristic", "craft", "temporal-graph"]
SelectionMetric = Literal["ap", "mrr"]
GNNModel = Literal["xsimgcl", "lightgcn"]
FusionMode = Literal["mlp", "lgbm", "ensemble"]
GNNEdgeWeighting = Literal["none", "repeat", "time_decay"]
CandidateProtocol = Literal["random", "test_like"]
CandidateRecentFeatureGroup = Literal["none", "recency_rank"]


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
    epochs: int = 15
    train_batch_size: int = 512
    lr: float = 0.001
    weight_decay: float = 0.0
    selection_metric: SelectionMetric = "ap"
    early_stop: int = 3
    fusion_hidden_dim: int = 64
    fusion_mode: FusionMode = "mlp"
    disable_gnn: bool = False
    gnn_model: GNNModel = "xsimgcl"
    gnn_edge_weighting: GNNEdgeWeighting = "none"
    gnn_time_decay_ratio: float = 0.05
    gnn_embedding_dim: int = 128
    gnn_layers: int = 2
    gnn_epochs: int = 50
    gnn_early_stop_patience: int = 3
    gnn_batch_size: int = 2048
    gnn_max_graph_edges: int = 0
    gnn_max_train_edges: int = 40_000
    gnn_lr: float = 0.001
    gnn_reg_weight: float = 1e-5
    gnn_cl_rate: float = 1e-4
    disable_seq: bool = False
    seq_epochs: int = 50
    seq_early_stop_patience: int = 3
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
    two_tower_epochs: int = 50
    two_tower_early_stop_patience: int = 3
    two_tower_batch_size: int = 512
    two_tower_score_batch_size: int = 2048
    two_tower_max_samples: int = 50_000
    hard_negative_ratio: float = 0.5
    popular_negative_ratio: float = 0.25
    negative_sampling_workers: int = 0
    craft_neighbors: int = 30
    craft_hidden_size: int = 64
    history_len: int = 64
    candidate_history_len: int = 32
    hidden_size: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.15
    training_candidates: CandidateProtocol = "test_like"
    validation_candidates: CandidateProtocol = "test_like"
    candidate_recent_feature_group: CandidateRecentFeatureGroup = "recency_rank"
    refit_full: bool = True
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
    structure_predict_neighbor_limit: int = 0
    disable_source_profile: bool = False
    disable_source_profile_deterministic: bool = False
    disable_source_profile_item2vec: bool = False
    source_profile_embedding_dim: int = 64
    source_profile_epochs: int = 50
    source_profile_early_stop_patience: int = 3
    source_profile_batch_size: int = 2048
    source_profile_score_batch_size: int = 8192
    source_profile_max_samples: int = 100_000
    source_profile_window_size: int = 16
    source_profile_recent_k: int = 32
    source_profile_predict_history_limit: int = 0
    prediction_cache_max_mb: int = 512
    save_checkpoint: Path | None = None
    load_checkpoint: Path | None = None


def main(argv: list[str] | None = None) -> int:
    import tyro  # noqa: PLC0415

    args = tyro.cli(CLIConfig, args=argv)
    _validate_checkpoint_args(args)
    _validate_device_args(args)
    _configure_jittor_device(args)

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
    checkpoint_writer = _checkpoint_writer(args, datasets)
    checkpoint_metadata = _loaded_checkpoint_metadata(args)
    results = []
    result_table = _result_table()
    try:
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
            if args.load_checkpoint is not None:
                _hydrate_ranker(
                    ranker,
                    load_checkpoint_dataset(args.load_checkpoint, dataset.name),
                )
            after_fit = None
            if checkpoint_writer is not None and dataset.name not in checkpoint_writer.written_datasets:
                after_fit = _checkpoint_after_fit(checkpoint_writer, dataset.name)
            result = build_dataset_submission(
                dataset=dataset,
                ranker=ranker,
                output_dir=csv_dir,
                batch_size=args.batch_size,
                seed=args.seed,
                verbose=not args.quiet_ranker,
                limit_rows=args.limit_rows,
                fit_ranker=args.load_checkpoint is None,
                after_fit=after_fit,
            )
            results.append(result)

            if not args.skip_validate:
                expected_rows = None if args.limit_rows is not None else expected_test_rows(dataset)
                validate_submission_file(result.output_path, expected_rows=expected_rows)
            _add_result_row(result_table, result, args.model)
            console.print(f"[green]wrote[/green] {result.rows} rows -> {result.output_path}")
            del ranker
            release_memory()
    finally:
        _finish_checkpoint_run(checkpoint_writer)

    if args.resume_existing and len(results) != len(datasets):
        present = {result.name for result in results}
        missing = ", ".join(dataset.name for dataset in datasets if dataset.name not in present)
        raise RuntimeError(f"resume output is incomplete; missing CSV for: {missing}")

    write_zip(results, zip_path)
    console.print(result_table)
    console.print(f"[bold green]archive[/bold green] {zip_path}")
    if checkpoint_metadata is not None:
        console.print(f"[bold green]loaded checkpoint[/bold green] {args.load_checkpoint}")
    return 0


def _checkpoint_writer(args: CLIConfig, datasets: list[DatasetPaths]) -> ContestCheckpointWriter | None:
    if args.save_checkpoint is None:
        return None
    return ContestCheckpointWriter(
        args.save_checkpoint,
        model_name=args.model,
        expected_datasets=tuple(dataset.name for dataset in datasets),
    )


def _loaded_checkpoint_metadata(args: CLIConfig) -> dict | None:
    if args.load_checkpoint is None:
        return None
    metadata = load_checkpoint_metadata(args.load_checkpoint)
    if metadata["model_name"] != args.model:
        raise ValueError(f"checkpoint model is {metadata['model_name']}, but --model is {args.model}")
    return metadata


def _finish_checkpoint_run(writer: ContestCheckpointWriter | None) -> None:
    if writer is None or writer.is_closed:
        return
    if writer.is_complete:
        writer.finalize()
        console.print(f"[bold green]checkpoint[/bold green] {writer.path}")
    else:
        writer.close_partial()
        pending = ", ".join(name for name in writer.expected_datasets if name not in writer.written_datasets)
        console.print(
            f"[yellow]checkpoint partial[/yellow] {writer.path.with_suffix(f'{writer.path.suffix}.tmp')} "
            f"(pending: {pending})"
        )


def _checkpoint_after_fit(writer: ContestCheckpointWriter, dataset_name: str):
    def save_dataset_state(ranker) -> None:
        writer.add_dataset(dataset_name, _snapshot_ranker(ranker))
        if writer.is_complete:
            writer.finalize()
            console.print(f"[bold green]checkpoint[/bold green] {writer.path}")

    return save_dataset_state


def _snapshot_ranker(ranker) -> dict:
    snapshot = getattr(ranker, "snapshot", None)
    if not callable(snapshot):
        raise TypeError(f"ranker does not support contest checkpoints: {ranker.name}")
    return snapshot()


def _hydrate_ranker(ranker, state: dict) -> None:
    hydrate = getattr(ranker, "hydrate", None)
    if not callable(hydrate):
        raise TypeError(f"ranker does not support contest checkpoints: {ranker.name}")
    hydrate(state)


def _hybrid_config_kwargs(args: CLIConfig) -> dict:
    return {
        "val_ratio": args.val_ratio,
        "context_ratio": args.context_ratio,
        "max_train_events": args.max_train_events,
        "max_val_events": args.max_val_events,
        "supervised_feature_batch_size": args.supervised_feature_batch_size,
        "supervised_feature_memmap": args.supervised_feature_memmap,
        "num_negatives": args.num_negatives,
        "max_fit_events": args.max_fit_events,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "selection_metric": args.selection_metric,
        "early_stop_patience": args.early_stop,
        "seed": args.seed,
        "verbose": not args.quiet_ranker,
        "encoder_state_cache_enabled": args.encoder_state_cache,
        "auto_strategy_enabled": args.auto_strategy,
        "candidate_prior_enabled": not args.disable_candidate_prior,
        "target_window_enabled": not args.disable_target_window,
        "target_window_fractions": _parse_target_window_fractions(args.target_window_fractions),
        "test_candidate_negative_ratio": args.test_candidate_negative_ratio,
        "structure_enabled": not args.disable_structure,
        "structure_cooccur_enabled": not args.disable_structure_cooccur,
        "structure_transition_enabled": not args.disable_structure_transition,
        "structure_cooccur_history_limit": args.structure_cooccur_history_limit,
        "structure_predict_neighbor_limit": args.structure_predict_neighbor_limit,
        "source_profile_enabled": not args.disable_source_profile,
        "source_profile_deterministic_enabled": not args.disable_source_profile_deterministic,
        "source_profile_item2vec_enabled": not args.disable_source_profile_item2vec,
        "source_profile_embedding_dim": args.source_profile_embedding_dim,
        "source_profile_epochs": args.source_profile_epochs,
        "source_profile_early_stop_patience": args.source_profile_early_stop_patience,
        "source_profile_batch_size": args.source_profile_batch_size,
        "source_profile_score_batch_size": args.source_profile_score_batch_size,
        "source_profile_max_samples": args.source_profile_max_samples,
        "source_profile_window_size": args.source_profile_window_size,
        "source_profile_recent_k": args.source_profile_recent_k,
        "source_profile_predict_history_limit": args.source_profile_predict_history_limit,
        "prediction_cache_max_bytes": max(int(args.prediction_cache_max_mb), 0) * 1024 * 1024,
        "gnn_enabled": not args.disable_gnn,
        "gnn_model": args.gnn_model,
        "gnn_edge_weighting": args.gnn_edge_weighting,
        "gnn_time_decay_ratio": args.gnn_time_decay_ratio,
        "gnn_embedding_dim": args.gnn_embedding_dim,
        "gnn_layers": args.gnn_layers,
        "gnn_epochs": args.gnn_epochs,
        "gnn_early_stop_patience": args.gnn_early_stop_patience,
        "gnn_batch_size": args.gnn_batch_size,
        "gnn_max_graph_edges": args.gnn_max_graph_edges,
        "gnn_max_train_edges": args.gnn_max_train_edges,
        "gnn_lr": args.gnn_lr,
        "gnn_reg_weight": args.gnn_reg_weight,
        "gnn_cl_rate": args.gnn_cl_rate,
        "seq_enabled": not args.disable_seq,
        "seq_epochs": args.seq_epochs,
        "seq_early_stop_patience": args.seq_early_stop_patience,
        "seq_batch_size": args.seq_batch_size,
        "seq_score_batch_size": args.seq_score_batch_size,
        "seq_max_samples": args.seq_max_samples,
        "seq_max_len": args.seq_max_len,
        "seq_hidden_size": args.seq_hidden_size,
        "seq_layers": args.seq_layers,
        "seq_heads": args.seq_heads,
        "seq_dropout": args.seq_dropout,
        "two_tower_enabled": not args.disable_two_tower,
        "two_tower_embedding_dim": args.two_tower_embedding_dim,
        "two_tower_hidden_dim": args.two_tower_hidden_dim,
        "two_tower_epochs": args.two_tower_epochs,
        "two_tower_early_stop_patience": args.two_tower_early_stop_patience,
        "two_tower_batch_size": args.two_tower_batch_size,
        "two_tower_score_batch_size": args.two_tower_score_batch_size,
        "two_tower_max_samples": args.two_tower_max_samples,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "fusion_mode": args.fusion_mode,
        "hard_negative_ratio": args.hard_negative_ratio,
        "popular_negative_ratio": args.popular_negative_ratio,
        "negative_sampling_workers": args.negative_sampling_workers,
    }


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
    if args.model == "temporal-graph":
        from .rankers.temporal_graph.config import TemporalGraphTrainingConfig  # noqa: PLC0415

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
            training_candidates=args.training_candidates,
            validation_candidates=args.validation_candidates,
            candidate_recent_feature_group=args.candidate_recent_feature_group,
            refit_full=args.refit_full,
        )
    if args.model == "hybrid-heuristic":
        from .rankers.hybrid_heuristic.config import TrainingConfig as HeuristicTrainingConfig  # noqa: PLC0415

        return HeuristicTrainingConfig(
            **_hybrid_config_kwargs(args),
        )
    return TrainingConfig(
        **_hybrid_config_kwargs(args),
    )


def _validate_device_args(args: CLIConfig) -> None:
    if args.cpu:
        raise ValueError("--cpu is disabled; this project requires CUDA")
    if args.model == "temporal-graph" and args.cpu:
        raise ValueError("temporal-graph requires CUDA; do not pass --cpu")


def _validate_checkpoint_args(args: CLIConfig) -> None:
    if args.save_checkpoint is not None and args.load_checkpoint is not None:
        raise ValueError("--save-checkpoint and --load-checkpoint cannot be used together")
    checkpoint_path = args.save_checkpoint or args.load_checkpoint
    if checkpoint_path is None:
        return
    if args.model != "hybrid":
        raise ValueError("contest checkpoints are only supported for --model hybrid")
    if checkpoint_path.suffix.lower() != ".pkl":
        raise ValueError("contest checkpoint must use the .pkl extension")
    if args.save_checkpoint is not None and args.resume_existing:
        raise ValueError("--save-checkpoint cannot be combined with --resume-existing")


def _configure_jittor_device(args: CLIConfig) -> None:
    import jittor as jt  # noqa: PLC0415

    if args.cpu:
        jt.flags.use_cuda = 0
    else:
        if not jt.has_cuda:
            raise RuntimeError("CUDA is not available but --cpu was not passed; this project requires CUDA")
        jt.flags.use_cuda = 1


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
    elif args.model == "temporal-graph":
        parts.extend(
            [
                f"hist-{config.history_len}",
                f"candhist-{config.candidate_history_len}",
                f"dim-{config.hidden_size}",
                f"layers-{config.layers}",
                f"heads-{config.heads}",
            ]
        )
    parts.append(_config_digest(args, config))
    return "_".join(parts)


def _config_digest(args: CLIConfig, config) -> str:
    cli_payload = _jsonable(args)
    for operational_key in (
        "dataset",
        "run_name",
        "resume_existing",
        "encoder_state_cache",
        "save_checkpoint",
        "load_checkpoint",
        "prediction_cache_max_mb",
    ):
        if isinstance(cli_payload, dict):
            cli_payload.pop(operational_key, None)
    ranker_payload = _jsonable(config)
    if isinstance(ranker_payload, dict):
        ranker_payload.pop("encoder_state_cache_enabled", None)
        ranker_payload.pop("prediction_cache_max_bytes", None)
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
        table.add_row(
            "source_profile_epochs/max_samples", f"{config.source_profile_epochs}/{config.source_profile_max_samples}"
        )
        table.add_row("two_tower", "on" if config.two_tower_enabled else "off")
        table.add_row("sequence", "on" if config.seq_enabled else "off")
        table.add_row("structure", "on" if config.structure_enabled else "off")
        table.add_row("structure_cooccur", "on" if config.structure_cooccur_enabled else "off")
        table.add_row("structure_transition", "on" if config.structure_transition_enabled else "off")
        table.add_row("structure_cooccur_history_limit", str(config.structure_cooccur_history_limit))
        table.add_row(
            "structure_predict_neighbor_limit",
            str(config.structure_predict_neighbor_limit) if config.structure_predict_neighbor_limit else "off",
        )
        table.add_row(
            "source_profile_predict_history_limit",
            str(config.source_profile_predict_history_limit) if config.source_profile_predict_history_limit else "off",
        )
        table.add_row("max_fit_events", str(config.max_fit_events) if config.max_fit_events else "full")
        table.add_row("supervised_feature_batch_size", str(config.supervised_feature_batch_size))
        table.add_row("supervised_feature_memmap", "on" if config.supervised_feature_memmap else "off")
        table.add_row("prediction_cache_max_mb", str(args.prediction_cache_max_mb))
        table.add_row("negative_sampling_workers", str(config.negative_sampling_workers))
    elif args.model == "temporal-graph":
        table.add_row("max_fit_events", str(config.max_fit_events) if config.max_fit_events else "full")
        table.add_row("max_train_events", str(config.max_train_events))
        table.add_row("max_val_events", str(config.max_val_events))
        table.add_row("num_negatives", str(config.num_negatives))
        table.add_row("history_len", str(config.history_len))
        table.add_row("candidate_history_len", str(config.candidate_history_len))
        table.add_row("hidden_size", str(config.hidden_size))
        table.add_row("layers/heads", f"{config.layers}/{config.heads}")
        table.add_row("training_candidates", config.training_candidates)
        table.add_row("validation_candidates", config.validation_candidates)
        table.add_row("candidate_recent_feature_group", config.candidate_recent_feature_group)
        table.add_row("refit_full", "on" if config.refit_full else "off")
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
