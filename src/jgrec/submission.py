from __future__ import annotations

import csv
import zipfile
from pathlib import Path

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
        rows = 0
        for line_number, row in enumerate(reader, start=1):
            rows += 1
            if len(row) != 100:
                raise ValueError(f"{csv_path}:{line_number} has {len(row)} columns, expected 100")
            values = [float(item) for item in row]
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError(f"{csv_path}:{line_number} contains probability outside [0, 1]")
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(f"{csv_path} has {rows} rows, expected {expected_rows}")


def expected_test_rows(dataset: DatasetPaths) -> int:
    return count_csv_data_rows(dataset.test_path)
