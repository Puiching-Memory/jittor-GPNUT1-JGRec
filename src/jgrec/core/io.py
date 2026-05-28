from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .types import DatasetPaths, Interaction, TestQuery


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


def read_interactions(path: Path) -> Iterable[Interaction]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"src", "dst", "time"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: src,dst,time")

        for row in reader:
            yield Interaction(src=int(row["src"]), dst=int(row["dst"]), time=int(row["time"]))


def read_test_queries(path: Path) -> Iterable[TestQuery]:
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

        for row_idx, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path}:{row_idx} has {len(row)} columns, expected {len(header)}")
            yield TestQuery(
                src=int(row[0]),
                time=int(row[1]),
                candidates=tuple(int(value) for value in row[2:]),
            )


def count_csv_data_rows(path: Path) -> int:
    with path.open("r", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)

