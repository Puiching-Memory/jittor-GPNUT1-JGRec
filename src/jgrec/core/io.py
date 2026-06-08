from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np

from .types import DatasetPaths, InteractionTable, TestQueryArray


def discover_datasets(data_dir: Path) -> list[DatasetPaths]:
    """Find dataset directories containing train.csv and test.csv."""
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    datasets: list[DatasetPaths] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        train_path = child / "train.csv"
        test_path = child / "test.csv"
        if train_path.exists() and test_path.exists():
            datasets.append(
                DatasetPaths(
                    name=child.name,
                    root=child,
                    train_path=train_path,
                    test_path=test_path,
                )
            )

    if not datasets:
        raise FileNotFoundError(f"no dataset*/train.csv and test.csv pairs found under {data_dir}")
    return datasets


def read_interactions(path: Path) -> InteractionTable:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        required = {"src", "dst", "time"}
        if header is None or not required.issubset(header):
            raise ValueError(f"{path} must contain columns: src,dst,time")

    usecols = (header.index("src"), header.index("dst"), header.index("time"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        interactions = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=usecols,
            dtype=np.int32,
            ndmin=2,
        )
    interactions = np.asarray(interactions, dtype=np.int32)
    if interactions.size == 0:
        return InteractionTable.empty()
    return InteractionTable.from_array(interactions)


def read_test_queries(path: Path, max_rows: int | None = None) -> TestQueryArray:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or len(header) < 3:
            raise ValueError(f"{path} must contain columns: src,time,c1,...,c100")
        if header[0] != "src" or header[1] != "time":
            raise ValueError(f"{path} first columns must be src,time")
        expected_candidates = len(header) - 2
        if expected_candidates != 100:
            raise ValueError(f"{path} must contain exactly 100 candidate columns, got {expected_candidates}")

    if max_rows is not None and max_rows <= 0:
        return TestQueryArray(
            src=np.empty(0, dtype=np.int32),
            time=np.empty(0, dtype=np.int32),
            candidates=np.empty((0, expected_candidates), dtype=np.int32),
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        data = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            dtype=np.int32,
            ndmin=2,
            max_rows=max_rows,
        )
    data = np.asarray(data, dtype=np.int32)
    if data.size == 0:
        return TestQueryArray(
            src=np.empty(0, dtype=np.int32),
            time=np.empty(0, dtype=np.int32),
            candidates=np.empty((0, expected_candidates), dtype=np.int32),
        )
    if data.shape[1] != len(header):
        raise ValueError(f"{path} has {data.shape[1]} columns, expected {len(header)}")
    return TestQueryArray(
        src=data[:, 0],
        time=data[:, 1],
        candidates=data[:, 2:],
    )


def count_csv_data_rows(path: Path) -> int:
    with path.open("r", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)
