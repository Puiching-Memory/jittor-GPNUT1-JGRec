from __future__ import annotations

from pathlib import Path

import numpy as np

from jgrec.rankers.base import Ranker

from .io import read_interactions, read_test_queries
from .types import DatasetPaths, DatasetResult, FitContext, TestQueryArray


def build_dataset_submission(
    dataset: DatasetPaths,
    ranker: Ranker,
    output_dir: Path,
    batch_size: int = 2048,
    seed: int = 42,
    verbose: bool = True,
    limit_rows: int | None = None,
) -> DatasetResult:
    interactions = read_interactions(dataset.train_path)
    report = ranker.fit(
        interactions,
        FitContext(
            dataset=dataset,
            seed=seed,
            limit_rows=limit_rows,
            verbose=verbose,
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset.name}.csv"

    row_count = 0
    queries = read_test_queries(dataset.test_path, max_rows=limit_rows)
    total_rows = len(queries) if limit_rows is None else min(len(queries), limit_rows)
    with output_path.open("w", newline="") as f:
        for start in range(0, total_rows, batch_size):
            stop = min(start + batch_size, total_rows)
            row_count += _write_batch(f, ranker, queries.rows(start, stop))

    return DatasetResult(
        name=dataset.name,
        rows=row_count,
        output_path=output_path,
        training_report=report,
    )


def _write_batch(output_file, ranker: Ranker, batch: TestQueryArray) -> int:
    probs_batch = ranker.predict_batch(batch)
    if probs_batch.shape != (len(batch), batch.candidate_count):
        raise ValueError(
            "ranker returned invalid prediction shape: "
            f"{probs_batch.shape}, expected {(len(batch), batch.candidate_count)}"
        )
    probs_batch = np.clip(probs_batch, 0.0, 1.0)
    np.savetxt(output_file, probs_batch, delimiter=",", fmt="%.8f")
    return len(batch)
