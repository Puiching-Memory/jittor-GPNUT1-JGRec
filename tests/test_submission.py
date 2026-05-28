import csv
import zipfile

import pytest

from jgrec.core.types import DatasetPaths, DatasetResult, TrainingReport
from jgrec.submission import expected_test_rows, validate_submission_file, write_zip


def _write_rows(path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_validate_submission_file_accepts_100_probabilities_per_row(tmp_path):
    csv_path = tmp_path / "dataset1.csv"
    _write_rows(csv_path, [[0.01] * 100, [0.0] * 100])

    validate_submission_file(csv_path, expected_rows=2)


def test_validate_submission_file_rejects_wrong_column_count(tmp_path):
    csv_path = tmp_path / "dataset1.csv"
    _write_rows(csv_path, [[0.01] * 99])

    with pytest.raises(ValueError, match="has 99 columns, expected 100"):
        validate_submission_file(csv_path)


def test_validate_submission_file_rejects_probability_outside_unit_interval(tmp_path):
    csv_path = tmp_path / "dataset1.csv"
    _write_rows(csv_path, [[0.01] * 99 + [1.5]])

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        validate_submission_file(csv_path)


def test_validate_submission_file_rejects_unexpected_row_count(tmp_path):
    csv_path = tmp_path / "dataset1.csv"
    _write_rows(csv_path, [[0.01] * 100])

    with pytest.raises(ValueError, match="has 1 rows, expected 2"):
        validate_submission_file(csv_path, expected_rows=2)


def test_expected_test_rows_counts_test_csv_data_rows(tmp_path):
    dataset_root = tmp_path / "dataset1"
    test_path = dataset_root / "test.csv"
    test_path.parent.mkdir()
    with test_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([["src", "time"], ["1", "2"], ["3", "4"]])
    dataset = DatasetPaths("dataset1", dataset_root, dataset_root / "train.csv", test_path)

    assert expected_test_rows(dataset) == 2


def test_write_zip_stores_flat_csv_names(tmp_path):
    csv_path = tmp_path / "csv" / "dataset1.csv"
    _write_rows(csv_path, [[0.01] * 100])
    result = DatasetResult(
        name="dataset1",
        rows=1,
        output_path=csv_path,
        training_report=TrainingReport(),
    )
    zip_path = tmp_path / "result" / "result.zip"

    write_zip([result], zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == ["dataset1.csv"]
        assert zf.read("dataset1.csv").decode().startswith("0.01")
