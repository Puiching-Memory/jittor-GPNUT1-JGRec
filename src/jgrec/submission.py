from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .data import DatasetPaths, count_csv_data_rows, read_interactions, read_test_queries
from .data import TestQuery
from .model import HeuristicJittorRanker


@dataclass(frozen=True)
class DatasetResult:
    name: str
    rows: int
    output_path: Path


def build_dataset_submission(
    dataset: DatasetPaths,
    output_dir: Path,
    recent_window: int = 32,
    batch_size: int = 2048,
    limit_rows: int | None = None,
) -> DatasetResult:
    interactions = list(read_interactions(dataset.train_path))
    ranker = HeuristicJittorRanker(recent_window=recent_window)
    ranker.fit(interactions)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset.name}.csv"

    row_count = 0
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        batch: list[TestQuery] = []
        for query in read_test_queries(dataset.test_path):
            batch.append(query)
            should_flush = len(batch) >= batch_size
            if limit_rows is not None and row_count + len(batch) >= limit_rows:
                should_flush = True

            if should_flush:
                if limit_rows is not None:
                    batch = batch[: limit_rows - row_count]
                row_count += _write_batch(writer, ranker, batch)
                batch.clear()
                if limit_rows is not None and row_count >= limit_rows:
                    break
        if batch and (limit_rows is None or row_count < limit_rows):
            if limit_rows is not None:
                batch = batch[: limit_rows - row_count]
            row_count += _write_batch(writer, ranker, batch)

    return DatasetResult(name=dataset.name, rows=row_count, output_path=output_path)


def _write_batch(
    writer: csv.writer,
    ranker: HeuristicJittorRanker,
    batch: list[TestQuery],
) -> int:
    probs_batch = ranker.predict_batch(batch)
    for probs in probs_batch:
        writer.writerow([f"{value:.8f}" for value in probs])
    return len(batch)


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
