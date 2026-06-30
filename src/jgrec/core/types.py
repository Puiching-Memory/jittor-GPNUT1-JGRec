from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

INTERACTION_SRC = 0
INTERACTION_DST = 1
INTERACTION_TIME = 2
InteractionArray = np.ndarray


@dataclass(frozen=True, slots=True)
class Interaction:
    src: int
    dst: int
    time: int


@dataclass(frozen=True, slots=True)
class InteractionTable:
    src: np.ndarray
    dst: np.ndarray
    time: np.ndarray

    def __post_init__(self) -> None:
        src = np.asarray(self.src, dtype=np.int32)
        dst = np.asarray(self.dst, dtype=np.int32)
        time = np.asarray(self.time, dtype=np.int32)
        if src.ndim != 1 or dst.ndim != 1 or time.ndim != 1:
            raise ValueError("src, dst, and time must be 1D arrays")
        if src.shape[0] != dst.shape[0] or src.shape[0] != time.shape[0]:
            raise ValueError("src, dst, and time must have the same row count")
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "dst", dst)
        object.__setattr__(self, "time", time)

    @classmethod
    def empty(cls) -> InteractionTable:
        return cls(
            src=np.empty(0, dtype=np.int32),
            dst=np.empty(0, dtype=np.int32),
            time=np.empty(0, dtype=np.int32),
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> InteractionTable:
        rows = np.asarray(values, dtype=np.int32)
        if rows.size == 0:
            return cls.empty()
        if rows.ndim != 2 or rows.shape[1] != 3:
            raise ValueError("interaction array must have shape (n, 3)")
        return cls(src=rows[:, 0], dst=rows[:, 1], time=rows[:, 2])

    @classmethod
    def from_events(cls, events: list[Interaction]) -> InteractionTable:
        if not events:
            return cls.empty()
        return cls(
            src=np.asarray([event.src for event in events], dtype=np.int32),
            dst=np.asarray([event.dst for event in events], dtype=np.int32),
            time=np.asarray([event.time for event in events], dtype=np.int32),
        )

    def __len__(self) -> int:
        return int(self.src.shape[0])

    def __getitem__(self, item) -> InteractionTable:
        if isinstance(item, (int, np.integer)):
            raise TypeError("InteractionTable does not support integer row indexing; use a slice or take()")
        return InteractionTable(src=self.src[item], dst=self.dst[item], time=self.time[item])

    def sort_by_time(self) -> InteractionTable:
        if len(self) <= 1:
            return self
        if np.all(self.time[:-1] <= self.time[1:]):
            return self
        order = np.argsort(self.time, kind="stable")
        return self.take(order)

    def tail(self, max_rows: int) -> InteractionTable:
        if max_rows <= 0:
            return InteractionTable.empty()
        if len(self) <= max_rows:
            return self
        return self[-max_rows:]

    def take(self, indices: np.ndarray) -> InteractionTable:
        return InteractionTable(src=self.src[indices], dst=self.dst[indices], time=self.time[indices])

    def to_events(self) -> list[Interaction]:
        return [
            Interaction(src=int(src), dst=int(dst), time=int(time))
            for src, dst, time in zip(self.src, self.dst, self.time, strict=True)
        ]

    def to_array(self) -> np.ndarray:
        if len(self) == 0:
            return np.empty((0, 3), dtype=np.int32)
        return np.column_stack((self.src, self.dst, self.time)).astype(np.int32, copy=False)


@dataclass(frozen=True, slots=True)
class TestQuery:
    __test__: ClassVar[bool] = False

    src: int
    time: int
    candidates: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TestQueryArray:
    __test__: ClassVar[bool] = False

    src: np.ndarray
    time: np.ndarray
    candidates: np.ndarray

    def __post_init__(self) -> None:
        src = np.asarray(self.src, dtype=np.int32)
        time = np.asarray(self.time, dtype=np.int32)
        candidates = np.asarray(self.candidates, dtype=np.int32)
        if candidates.ndim == 1:
            candidates = candidates.reshape((1, -1)) if candidates.size else candidates.reshape((0, 0))
        if src.ndim != 1 or time.ndim != 1 or candidates.ndim != 2:
            raise ValueError("TestQueryArray expects src/time vectors and a candidate matrix")
        if len(src) != len(time) or len(src) != candidates.shape[0]:
            raise ValueError("TestQueryArray row counts do not match")
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "candidates", candidates)

    @classmethod
    def from_queries(cls, queries: list[TestQuery] | tuple[TestQuery, ...]) -> TestQueryArray:
        if not queries:
            return cls(
                src=np.empty(0, dtype=np.int32),
                time=np.empty(0, dtype=np.int32),
                candidates=np.empty((0, 0), dtype=np.int32),
            )
        candidate_count = len(queries[0].candidates)
        candidates = np.empty((len(queries), candidate_count), dtype=np.int32)
        src = np.empty(len(queries), dtype=np.int32)
        time = np.empty(len(queries), dtype=np.int32)
        for row_idx, query in enumerate(queries):
            if len(query.candidates) != candidate_count:
                raise ValueError("all queries in a batch must have the same candidate count")
            src[row_idx] = int(query.src)
            time[row_idx] = int(query.time)
            candidates[row_idx] = np.asarray(query.candidates, dtype=np.int32)
        return cls(src=src, time=time, candidates=candidates)

    @property
    def candidate_count(self) -> int:
        return int(self.candidates.shape[1]) if self.candidates.ndim == 2 else 0

    def __len__(self) -> int:
        return int(self.src.shape[0])

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            row = int(index)
            return TestQuery(
                src=int(self.src[row]),
                time=int(self.time[row]),
                candidates=tuple(int(candidate) for candidate in self.candidates[row]),
            )
        return TestQueryArray(
            src=self.src[index],
            time=self.time[index],
            candidates=self.candidates[index],
        )

@dataclass(frozen=True, slots=True)
class DatasetPaths:
    name: str
    root: Path
    train_path: Path
    test_path: Path


@dataclass(frozen=True, slots=True)
class FitContext:
    dataset: DatasetPaths
    seed: int = 42
    limit_rows: int | None = None
    verbose: bool = True
    load_checkpoint_path: Path | None = None
    save_checkpoint_path: Path | None = None


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class DatasetResult:
    name: str
    rows: int
    output_path: Path
    training_report: TrainingReport
