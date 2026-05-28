import csv

import numpy as np

from jgrec.core.runner import build_dataset_submission
from jgrec.core.types import DatasetPaths, FitContext, Interaction, TestQuery as Query, TrainingReport
from jgrec.submission import validate_submission_file


class DummyRanker:
    name = "dummy"

    def __init__(self) -> None:
        self.fit_interactions: list[Interaction] = []
        self.fit_context: FitContext | None = None
        self.batch_sizes: list[int] = []

    def fit(self, interactions: list[Interaction], context: FitContext) -> TrainingReport:
        self.fit_interactions = interactions
        self.fit_context = context
        return TrainingReport(model_name=self.name, train_events=len(interactions))

    def predict_batch(self, queries: list[Query]) -> np.ndarray:
        self.batch_sizes.append(len(queries))
        row = np.linspace(-0.5, 1.5, len(queries[0].candidates), dtype=np.float32)
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
    assert ranker.fit_interactions == [
        Interaction(src=1, dst=10, time=100),
        Interaction(src=2, dst=20, time=200),
    ]
    assert ranker.fit_context == FitContext(dataset=dataset, seed=7, limit_rows=2, verbose=False)
    assert ranker.batch_sizes == [2]
    validate_submission_file(result.output_path, expected_rows=2)

    with result.output_path.open("r", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "0.00000000"
    assert rows[0][-1] == "1.00000000"
