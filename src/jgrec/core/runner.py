from __future__ import annotations

from pathlib import Path

import numpy as np

from jgrec.rankers.base import Ranker

from .io import read_interactions, read_test_queries
from .types import DatasetPaths, DatasetResult, FitContext, TestQuery


def build_dataset_submission(
    dataset: DatasetPaths,
    ranker: Ranker,
    output_dir: Path,
    batch_size: int = 2048,
    seed: int = 42,
    verbose: bool = True,
    limit_rows: int | None = None,
) -> DatasetResult:
    interactions = list(read_interactions(dataset.train_path))
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
    with output_path.open("w", newline="") as f:
        batch: list[TestQuery] = []
        for query in read_test_queries(dataset.test_path):
            batch.append(query)
            should_flush = len(batch) >= batch_size
            if limit_rows is not None and row_count + len(batch) >= limit_rows:
                should_flush = True

            if should_flush:
                if limit_rows is not None:
                    batch = batch[: limit_rows - row_count]
                row_count += _write_batch(f, ranker, batch)
                batch.clear()
                if limit_rows is not None and row_count >= limit_rows:
                    break
        if batch and (limit_rows is None or row_count < limit_rows):
            if limit_rows is not None:
                batch = batch[: limit_rows - row_count]
            row_count += _write_batch(f, ranker, batch)

    return DatasetResult(
        name=dataset.name,
        rows=row_count,
        output_path=output_path,
        training_report=report,
    )


def _write_batch(output_file, ranker: Ranker, batch: list[TestQuery]) -> int:
    probs_batch = ranker.predict_batch(batch)
    if probs_batch.shape != (len(batch), len(batch[0].candidates)):
        raise ValueError(
            "ranker returned invalid prediction shape: "
            f"{probs_batch.shape}, expected {(len(batch), len(batch[0].candidates))}"
        )
    probs_batch = np.clip(probs_batch, 0.0, 1.0)
    np.savetxt(output_file, probs_batch, delimiter=",", fmt="%.8f")
    return len(batch)

