from __future__ import annotations

from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np

from jgrec.rankers.base import Ranker

from .io import read_interactions, read_test_queries
from .memory import log_event, log_memory, release_memory
from .types import DatasetPaths, DatasetResult, FitContext, TestQuery

PREDICT_PROGRESS_INTERVAL = 10_000


def build_dataset_submission(
    dataset: DatasetPaths,
    ranker: Ranker,
    output_dir: Path,
    batch_size: int = 2048,
    seed: int = 42,
    verbose: bool = True,
    limit_rows: int | None = None,
) -> DatasetResult:
    log_memory(f"read_train_start:{dataset.name}", enabled=verbose)
    interactions = _read_fit_interactions(dataset, ranker)
    log_memory(f"read_train_done:{dataset.name}", enabled=verbose)
    report = ranker.fit(
        interactions,
        FitContext(
            dataset=dataset,
            seed=seed,
            limit_rows=limit_rows,
            verbose=verbose,
        ),
    )
    del interactions
    release_memory()
    log_memory(f"post_fit:{dataset.name}", enabled=verbose)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset.name}.csv"

    row_count = 0
    predict_start = perf_counter()
    next_progress_row = PREDICT_PROGRESS_INTERVAL
    log_memory(f"predict_start:{dataset.name}", enabled=verbose)
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
                batch_rows = _write_batch(f, ranker, batch)
                row_count += batch_rows
                next_progress_row = _log_predict_progress(
                    dataset=dataset,
                    output_path=output_path,
                    row_count=row_count,
                    batch_rows=batch_rows,
                    elapsed=perf_counter() - predict_start,
                    next_progress_row=next_progress_row,
                    verbose=verbose,
                )
                batch.clear()
                if limit_rows is not None and row_count >= limit_rows:
                    break
        if batch and (limit_rows is None or row_count < limit_rows):
            if limit_rows is not None:
                batch = batch[: limit_rows - row_count]
            batch_rows = _write_batch(f, ranker, batch)
            row_count += batch_rows
            _log_predict_progress(
                dataset=dataset,
                output_path=output_path,
                row_count=row_count,
                batch_rows=batch_rows,
                elapsed=perf_counter() - predict_start,
                next_progress_row=next_progress_row,
                verbose=verbose,
            )

    log_memory(f"predict_done:{dataset.name}", enabled=verbose)
    return DatasetResult(
        name=dataset.name,
        rows=row_count,
        output_path=output_path,
        training_report=report,
    )


def _read_fit_interactions(dataset: DatasetPaths, ranker: Ranker):
    max_fit_events = int(getattr(getattr(ranker, "config", None), "max_fit_events", 0) or 0)
    if max_fit_events <= 0:
        return list(read_interactions(dataset.train_path))
    return list(deque(read_interactions(dataset.train_path), maxlen=max_fit_events))


def _write_batch(output_file, ranker: Ranker, batch: list[TestQuery]) -> int:
    probs_batch = ranker.predict_batch(batch)
    if probs_batch.shape != (len(batch), len(batch[0].candidates)):
        raise ValueError(
            "ranker returned invalid prediction shape: "
            f"{probs_batch.shape}, expected {(len(batch), len(batch[0].candidates))}"
        )
    np.clip(probs_batch, 0.0, 1.0, out=probs_batch)
    np.savetxt(output_file, probs_batch, delimiter=",", fmt="%.8f")
    batch_rows = len(batch)
    del probs_batch
    release_memory()
    return batch_rows


def _log_predict_progress(
    dataset: DatasetPaths,
    output_path: Path,
    row_count: int,
    batch_rows: int,
    elapsed: float,
    next_progress_row: int,
    verbose: bool,
) -> int:
    if not verbose or row_count < next_progress_row:
        return next_progress_row

    log_event(
        f"[predict] dataset={dataset.name} rows={row_count} "
        f"batch={batch_rows} elapsed={elapsed:.1f}s csv={output_path}",
        enabled=True,
    )
    while row_count >= next_progress_row:
        next_progress_row += PREDICT_PROGRESS_INTERVAL
    return next_progress_row

