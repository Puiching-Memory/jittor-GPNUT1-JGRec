import csv

import numpy as np
import pytest

from jgrec.core.io import count_csv_data_rows, discover_datasets, read_interactions, read_test_queries


def _candidate_header(count: int = 100) -> list[str]:
    return ["src", "time", *(f"c{idx}" for idx in range(1, count + 1))]


def _candidate_row(src: int = 1, time: int = 10, count: int = 100) -> list[str]:
    return [str(src), str(time), *(str(idx) for idx in range(count))]


def _write_csv(path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_discover_datasets_returns_sorted_complete_dataset_pairs(tmp_path):
    data_dir = tmp_path / "data"
    _write_csv(data_dir / "dataset2" / "train.csv", [["src", "dst", "time"]])
    _write_csv(data_dir / "dataset2" / "test.csv", [_candidate_header()])
    _write_csv(data_dir / "dataset1" / "train.csv", [["src", "dst", "time"]])
    _write_csv(data_dir / "dataset1" / "test.csv", [_candidate_header()])
    _write_csv(data_dir / "incomplete" / "train.csv", [["src", "dst", "time"]])

    datasets = discover_datasets(data_dir)

    assert [dataset.name for dataset in datasets] == ["dataset1", "dataset2"]
    assert datasets[0].train_path == data_dir / "dataset1" / "train.csv"
    assert datasets[1].test_path == data_dir / "dataset2" / "test.csv"


def test_discover_datasets_raises_for_missing_data_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="data directory not found"):
        discover_datasets(tmp_path / "missing")


def test_read_interactions_accepts_extra_columns(tmp_path):
    path = tmp_path / "train.csv"
    _write_csv(
        path,
        [
            ["src", "dst", "time", "split"],
            ["2", "3", "4", "train"],
            ["5", "8", "13", "val"],
        ],
    )

    interactions = read_interactions(path)
    np.testing.assert_array_equal(interactions.src, np.asarray([2, 5], dtype=np.int32))
    np.testing.assert_array_equal(interactions.dst, np.asarray([3, 8], dtype=np.int32))
    np.testing.assert_array_equal(interactions.time, np.asarray([4, 13], dtype=np.int32))


def test_read_interactions_requires_core_columns(tmp_path):
    path = tmp_path / "train.csv"
    _write_csv(path, [["src", "time"], ["1", "2"]])

    with pytest.raises(ValueError, match="must contain columns: src,dst,time"):
        read_interactions(path)


def test_read_test_queries_parses_exactly_100_candidates(tmp_path):
    path = tmp_path / "test.csv"
    _write_csv(path, [_candidate_header(), _candidate_row(src=7, time=11)])

    queries = read_test_queries(path)
    np.testing.assert_array_equal(queries.src, np.asarray([7], dtype=np.int32))
    np.testing.assert_array_equal(queries.time, np.asarray([11], dtype=np.int32))
    np.testing.assert_array_equal(queries.candidates, np.asarray([list(range(100))], dtype=np.int32))


def test_read_test_queries_requires_exactly_100_candidates(tmp_path):
    path = tmp_path / "test.csv"
    _write_csv(path, [_candidate_header(count=99), _candidate_row(count=99)])

    with pytest.raises(ValueError, match="exactly 100 candidate columns"):
        read_test_queries(path)


def test_read_test_queries_validates_row_width(tmp_path):
    path = tmp_path / "test.csv"
    _write_csv(path, [_candidate_header(), ["1", "2", "3"]])

    with pytest.raises(ValueError):
        read_test_queries(path)


def test_count_csv_data_rows_ignores_header(tmp_path):
    path = tmp_path / "rows.csv"
    _write_csv(path, [["a"], ["1"], ["2"]])

    assert count_csv_data_rows(path) == 2
