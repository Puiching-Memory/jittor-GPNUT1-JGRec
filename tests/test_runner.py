import csv

import numpy as np

from jgrec.core import runner
from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetPaths, FitContext, InteractionTable, TestQueryArray, TrainingReport
from jgrec.submission import validate_submission_file


class DummyRanker:
    name = "dummy"

    def __init__(self) -> None:
        self.fit_interactions: InteractionTable | None = None
        self.fit_context: FitContext | None = None
        self.batch_sizes: list[int] = []

    def fit(self, interactions: InteractionTable, context: FitContext) -> TrainingReport:
        self.fit_interactions = interactions
        self.fit_context = context
        return TrainingReport(model_name=self.name, train_events=len(interactions))

    def predict_batch(self, queries: TestQueryArray) -> np.ndarray:
        self.batch_sizes.append(len(queries))
        row = np.linspace(-0.5, 1.5, queries.candidate_count, dtype=np.float32)
        return np.tile(row, (len(queries), 1))


def _write_train_csv(path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(
            [
                ["src", "dst", "time"],
                ["1", "10", "100"],
                ["2", "20", "200"],
            ]
        )


def _write_test_csv(path, row_count: int) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "time", *(f"c{idx}" for idx in range(1, 101))])
        for row_idx in range(row_count):
            writer.writerow([str(row_idx), str(row_idx + 1000), *(str(value) for value in range(100))])


def test_build_dataset_submission_limits_rows_and_clips_predictions(tmp_path):
    dataset_root = tmp_path / "dataset1"
    dataset_root.mkdir()
    train_path = dataset_root / "train.csv"
    test_path = dataset_root / "test.csv"
    _write_train_csv(train_path)
    _write_test_csv(test_path, row_count=3)
    dataset = DatasetPaths("dataset1", dataset_root, train_path, test_path)
    ranker = DummyRanker()

    result = build_dataset_submission(
        dataset=dataset,
        ranker=ranker,
        output_dir=tmp_path / "out",
        batch_size=2,
        seed=7,
        verbose=False,
        limit_rows=2,
    )

    assert result.name == "dataset1"
    assert result.rows == 2
    assert result.training_report.model_name == "dummy"
    assert ranker.fit_interactions is not None
    np.testing.assert_array_equal(ranker.fit_interactions.to_array(), np.asarray([[1, 10, 100], [2, 20, 200]], dtype=np.int32))
    assert ranker.fit_context == FitContext(dataset=dataset, seed=7, limit_rows=2, verbose=False)
    assert ranker.batch_sizes == [2]
    validate_submission_file(result.output_path, expected_rows=2)

    with result.output_path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "0.00000000"
    assert rows[0][-1] == "1.00000000"


def test_build_dataset_submission_logs_predict_progress_without_changing_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "PREDICT_PROGRESS_INTERVAL", 2)
    dataset_root = tmp_path / "dataset1"
    dataset_root.mkdir()
    train_path = dataset_root / "train.csv"
    test_path = dataset_root / "test.csv"
    _write_train_csv(train_path)
    _write_test_csv(test_path, row_count=5)
    dataset = DatasetPaths("dataset1", dataset_root, train_path, test_path)
    ranker = DummyRanker()

    result = build_dataset_submission(
        dataset=dataset,
        ranker=ranker,
        output_dir=tmp_path / "out",
        batch_size=2,
        seed=7,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert result.rows == 5
    assert ranker.batch_sizes == [2, 2, 1]
    assert "[predict] dataset=dataset1 rows=2 batch=2" in captured.out
    assert "[predict] dataset=dataset1 rows=4 batch=2" in captured.out
    validate_submission_file(result.output_path, expected_rows=5)


def test_build_dataset_submission_can_predict_from_loaded_ranker_without_fitting(tmp_path):
    dataset_root = tmp_path / "dataset1"
    dataset_root.mkdir()
    train_path = dataset_root / "train.csv"
    test_path = dataset_root / "test.csv"
    _write_train_csv(train_path)
    _write_test_csv(test_path, row_count=2)
    dataset = DatasetPaths("dataset1", dataset_root, train_path, test_path)
    ranker = DummyRanker()
    ranker.training_report = TrainingReport(model_name=ranker.name, train_events=99)

    result = build_dataset_submission(
        dataset=dataset,
        ranker=ranker,
        output_dir=tmp_path / "out",
        batch_size=2,
        verbose=False,
        fit_ranker=False,
    )

    assert ranker.fit_interactions is None
    assert result.training_report == ranker.training_report
    assert result.rows == 2


def test_build_dataset_submission_calls_after_fit_before_prediction(tmp_path):
    dataset_root = tmp_path / "dataset1"
    dataset_root.mkdir()
    train_path = dataset_root / "train.csv"
    test_path = dataset_root / "test.csv"
    _write_train_csv(train_path)
    _write_test_csv(test_path, row_count=1)
    dataset = DatasetPaths("dataset1", dataset_root, train_path, test_path)
    ranker = DummyRanker()
    observations = []

    build_dataset_submission(
        dataset=dataset,
        ranker=ranker,
        output_dir=tmp_path / "out",
        verbose=False,
        after_fit=lambda fitted: observations.append((fitted is ranker, tuple(ranker.batch_sizes))),
    )

    assert observations == [(True, ())]
