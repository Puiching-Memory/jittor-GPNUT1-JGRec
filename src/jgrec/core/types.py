from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Interaction:
    src: int
    dst: int
    time: int


@dataclass(frozen=True)
class TestQuery:
    src: int
    time: int
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    root: Path
    train_path: Path
    test_path: Path


@dataclass(frozen=True)
class FitContext:
    dataset: DatasetPaths
    seed: int = 42
    limit_rows: int | None = None
    verbose: bool = True


@dataclass(frozen=True)
class TrainingReport:
    train_events: int = 0
    val_events: int = 0
    best_val_ap: float = 0.0
    best_val_mrr: float = 0.0
    weights: tuple[float, ...] = ()
    feature_names: tuple[str, ...] = ()
    selected_fusion: str = ""
    model_name: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetResult:
    name: str
    rows: int
    output_path: Path
    training_report: TrainingReport

