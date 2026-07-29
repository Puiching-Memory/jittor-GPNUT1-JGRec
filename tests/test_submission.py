import csv
import hashlib
import zipfile
from pathlib import Path

import pytest

from jgrec.core.types import DatasetPaths, DatasetResult, TrainingReport
from jgrec.submission import (
    compose_submission_package,
    expected_test_rows,
    validate_submission_file,
    write_zip,
)


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


def test_compose_submission_package_selects_each_dataset_from_its_named_source(
    tmp_path: Path,
) -> None:
    segment_dataset1 = ",".join(["0.01"] * 100).encode()
    stale_dataset2 = ",".join(["0.02"] * 100).encode()
    stale_dataset1 = ",".join(["0.03"] * 100).encode()
    champion_dataset2 = ",".join(["0.04"] * 100).encode()
    dataset1_source = tmp_path / "dataset1-source.zip"
    dataset2_source = tmp_path / "dataset2-source.zip"
    with zipfile.ZipFile(dataset1_source, "w") as archive:
        archive.writestr("dataset1.csv", segment_dataset1)
        archive.writestr("dataset2.csv", stale_dataset2)
    with zipfile.ZipFile(dataset2_source, "w") as archive:
        archive.writestr("dataset1.csv", stale_dataset1)
        archive.writestr("dataset2.csv", champion_dataset2)

    report = compose_submission_package(
        dataset1_source_zip=dataset1_source,
        dataset2_source_zip=dataset2_source,
        output_dir=tmp_path / "composed",
        expected_rows={"dataset1": 1, "dataset2": 1},
        expected_sha256={
            "dataset1": hashlib.sha256(segment_dataset1).hexdigest(),
            "dataset2": hashlib.sha256(champion_dataset2).hexdigest(),
        },
    )

    assert (tmp_path / "composed/csv/dataset1.csv").read_bytes() == segment_dataset1
    assert (tmp_path / "composed/csv/dataset2.csv").read_bytes() == champion_dataset2
    with zipfile.ZipFile(tmp_path / "composed/result.zip") as archive:
        assert archive.namelist() == ["dataset1.csv", "dataset2.csv"]
        assert archive.read("dataset1.csv") == segment_dataset1
        assert archive.read("dataset2.csv") == champion_dataset2
    assert report["dataset1"]["sha256"] == hashlib.sha256(segment_dataset1).hexdigest()
    assert report["dataset2"]["sha256"] == hashlib.sha256(champion_dataset2).hexdigest()
