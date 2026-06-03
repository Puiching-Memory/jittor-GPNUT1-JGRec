from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np

from .core.io import count_csv_data_rows
from .core.types import DatasetPaths, DatasetResult


def write_zip(results: list[DatasetResult], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for result in results:
            zf.write(result.output_path, arcname=result.output_path.name)


def validate_submission_file(csv_path: Path, expected_rows: int | None = None) -> None:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
    rows = 0 if first_row is None else count_csv_data_rows(csv_path) + 1
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"{csv_path} has {rows} rows, expected {expected_rows}")
    if rows == 0:
        return
    if len(first_row) != 100:
        raise ValueError(f"{csv_path}:1 has {len(first_row)} columns, expected 100")

    try:
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, ndmin=2)
    except ValueError as exc:
        message = str(exc)
        if "the number of columns changed" in message:
            raise ValueError(f"{csv_path} has inconsistent column count") from exc
        raise
    if data.shape != (rows, 100):
        if data.shape[1] != 100:
            raise ValueError(f"{csv_path} has {data.shape[1]} columns, expected 100")
        raise ValueError(f"{csv_path} has {data.shape[0]} rows, expected {rows}")
    if np.any((data < 0.0) | (data > 1.0)):
        raise ValueError(f"{csv_path} contains probability outside [0, 1]")


def expected_test_rows(dataset: DatasetPaths) -> int:
    return count_csv_data_rows(dataset.test_path)
