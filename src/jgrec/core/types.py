from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

InteractionArray = NDArray[np.int32]
INTERACTION_SRC = 0
INTERACTION_DST = 1
INTERACTION_TIME = 2


@dataclass(frozen=True)
class TestQueryArray:
    __test__: ClassVar[bool] = False

    src: NDArray[np.int32]
    time: NDArray[np.int32]
    candidates: NDArray[np.int32]

    def __post_init__(self) -> None:
        src = np.asarray(self.src, dtype=np.int32)
        time = np.asarray(self.time, dtype=np.int32)
        candidates = np.asarray(self.candidates, dtype=np.int32)
        if src.ndim != 1:
            raise ValueError(f"test query src must be 1-D, got shape {src.shape}")
        if time.ndim != 1:
            raise ValueError(f"test query time must be 1-D, got shape {time.shape}")
        if candidates.ndim != 2:
            raise ValueError(f"test query candidates must be 2-D, got shape {candidates.shape}")
        if len(src) != len(time) or len(src) != candidates.shape[0]:
            raise ValueError(
                "test query arrays must have matching rows: "
                f"src={len(src)}, time={len(time)}, candidates={candidates.shape[0]}"
            )
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "candidates", candidates)

    def __len__(self) -> int:
        return int(self.src.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.candidates.shape[1])

    def rows(self, start: int, stop: int) -> TestQueryArray:
        return TestQueryArray(
            src=self.src[start:stop],
            time=self.time[start:stop],
            candidates=self.candidates[start:stop],
        )


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
