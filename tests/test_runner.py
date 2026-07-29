import csv
import tempfile
from pathlib import Path

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


class SourceValueRanker(DummyRanker):
    def __init__(self, *, fail_after_batches: int | None = None) -> None:
        super().__init__()
        self.seen_sources: list[int] = []
        self.fail_after_batches = fail_after_batches

    def predict_batch(self, queries: TestQueryArray) -> np.ndarray:
        self.batch_sizes.append(len(queries))
        self.seen_sources.extend(int(src) for src in queries.src)
        if self.fail_after_batches is not None and len(self.batch_sizes) > self.fail_after_batches:
            raise RuntimeError("scheduled prediction failed")
        return np.repeat((queries.src.astype(np.float64) / 100.0)[:, None], queries.candidate_count, axis=1)


class SourceScheduledRanker(SourceValueRanker):
    def prediction_order(self, queries: TestQueryArray) -> np.ndarray:
        return np.argsort(queries.src, kind="stable")


class TiedPriorRanker(DummyRanker):
    def predict_batch(self, queries: TestQueryArray) -> np.ndarray:
        return np.tile(
            np.asarray([0.5, 0.5, 0.5, 0.25], dtype=np.float64),
            (len(queries), 25),
        )

    def prediction_tie_break_prior(self, queries: TestQueryArray) -> np.ndarray:
        return np.tile(
            np.asarray([1.0, 3.0, 2.0, 0.0], dtype=np.float64),
            (len(queries), 25),
        )


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


def _write_test_sources(path, sources: list[int]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "time", *(f"c{idx}" for idx in range(1, 101))])
        for row_idx, src in enumerate(sources):
            writer.writerow([str(src), str(row_idx + 1000), *(str(value) for value in range(100))])


def _capture_prediction_memmaps(tmp_path, monkeypatch) -> list[Path]:
    temp_dir = tmp_path / "prediction-temp"
    temp_dir.mkdir()
    created_paths: list[Path] = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, dir=temp_dir, **kwargs)
        created_paths.append(Path(path))
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", recording_mkstemp)
    return created_paths


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
    numeric = np.asarray(rows[0], dtype=np.float64)
    assert numeric.min() == 0.0
    assert numeric.max() == 1.0
    assert np.unique(numeric).size == numeric.size


def test_build_dataset_submission_breaks_exact_ties_by_prior_and_round_trips(tmp_path):
    dataset_root = tmp_path / "dataset1"
    dataset_root.mkdir()
    train_path = dataset_root / "train.csv"
    test_path = dataset_root / "test.csv"
    _write_train_csv(train_path)
    _write_test_csv(test_path, row_count=1)
    dataset = DatasetPaths("dataset1", dataset_root, train_path, test_path)

    result = build_dataset_submission(
        dataset=dataset,
        ranker=TiedPriorRanker(),
        output_dir=tmp_path / "out",
        verbose=False,
    )

    persisted = np.loadtxt(result.output_path, delimiter=",", ndmin=2)
    assert np.unique(persisted[0]).size == persisted.shape[1]
    assert persisted[0, 1] > persisted[0, 2] > persisted[0, 0]
    assert persisted[0, :3].min() > persisted[0, 3]


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


def test_build_dataset_submission_schedules_prediction_but_writes_original_order(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PREDICTION_MEMMAP_FLUSH_INTERVAL", 1)
    dataset_root = tmp_path / "dataset1"
    dataset_root.mkdir()
    train_path = dataset_root / "train.csv"
    test_path = dataset_root / "test.csv"
    _write_train_csv(train_path)
    original_sources = [30, 10, 30, 20, 10]
    _write_test_sources(test_path, original_sources)
    dataset = DatasetPaths("dataset1", dataset_root, train_path, test_path)
    ranker = SourceScheduledRanker()
    memmap_paths = _capture_prediction_memmaps(tmp_path, monkeypatch)

    result = build_dataset_submission(
        dataset=dataset,
        ranker=ranker,
        output_dir=tmp_path / "out",
        batch_size=2,
        verbose=False,
    )

    assert ranker.batch_sizes == [2, 2, 1]
    assert ranker.seen_sources == [10, 10, 20, 30, 30]
    predictions = np.loadtxt(result.output_path, delimiter=",")
    np.testing.assert_array_equal(predictions[:, 0], np.asarray(original_sources, dtype=np.float64) / 100.0)
    assert len(memmap_paths) == 1
    assert not memmap_paths[0].exists()

    reference = build_dataset_submission(
        dataset=dataset,
        ranker=SourceValueRanker(),
        output_dir=tmp_path / "reference",
        batch_size=2,
        verbose=False,
    )
    assert result.output_path.read_bytes() == reference.output_path.read_bytes()


def test_build_dataset_submission_removes_prediction_memmap_after_failure(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset1"
    dataset_root.mkdir()
    train_path = dataset_root / "train.csv"
    test_path = dataset_root / "test.csv"
    _write_train_csv(train_path)
    _write_test_sources(test_path, [30, 10, 20])
    dataset = DatasetPaths("dataset1", dataset_root, train_path, test_path)
    ranker = SourceScheduledRanker(fail_after_batches=1)
    memmap_paths = _capture_prediction_memmaps(tmp_path, monkeypatch)

    with np.testing.assert_raises_regex(RuntimeError, "scheduled prediction failed"):
        build_dataset_submission(
            dataset=dataset,
            ranker=ranker,
            output_dir=tmp_path / "out",
            batch_size=2,
            verbose=False,
        )

    assert len(memmap_paths) == 1
    assert not memmap_paths[0].exists()
