from __future__ import annotations

import mmap as mmap_module
import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from time import perf_counter

import numpy as np

from jgrec.rankers.base import Ranker

from .io import read_interactions, read_test_queries
from .memory import log_event, log_memory, release_memory
from .prediction_ties import break_prediction_ties
from .types import DatasetPaths, DatasetResult, FitContext, TestQueryArray, TrainingReport

PREDICT_PROGRESS_INTERVAL = 10_000
PREDICTION_MEMMAP_FLUSH_INTERVAL = 8


def build_dataset_submission(
    dataset: DatasetPaths,
    ranker: Ranker,
    output_dir: Path,
    batch_size: int = 2048,
    seed: int = 42,
    verbose: bool = True,
    limit_rows: int | None = None,
    fit_ranker: bool = True,
    after_fit: Callable[[Ranker], None] | None = None,
) -> DatasetResult:
    if fit_ranker:
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
        if after_fit is not None:
            after_fit(ranker)
        del interactions
        release_memory()
        log_memory(f"post_fit:{dataset.name}", enabled=verbose)
    else:
        report = getattr(ranker, "training_report", None)
        if report is None:
            report = TrainingReport(model_name=ranker.name)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset.name}.csv"

    row_count = 0
    predict_start = perf_counter()
    next_progress_row = PREDICT_PROGRESS_INTERVAL
    log_memory(f"predict_start:{dataset.name}", enabled=verbose)
    queries = read_test_queries(dataset.test_path, max_rows=limit_rows)
    prediction_order = _prediction_order(ranker, queries)
    with output_path.open("w", newline="") as f:
        if prediction_order is None:
            for start in range(0, len(queries), batch_size):
                batch = queries[start : start + batch_size]
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
        else:
            row_count = _write_scheduled_batches(
                output_file=f,
                ranker=ranker,
                queries=queries,
                prediction_order=prediction_order,
                batch_size=batch_size,
                dataset=dataset,
                output_path=output_path,
                predict_start=predict_start,
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
    interactions = read_interactions(dataset.train_path)
    if max_fit_events <= 0:
        return interactions
    return interactions.tail(max_fit_events)


def _write_batch(output_file, ranker: Ranker, batch: TestQueryArray) -> int:
    probs_batch = _predict_batch(ranker, batch)
    _write_probabilities(output_file, probs_batch)
    return len(batch)


def _predict_batch(ranker: Ranker, batch: TestQueryArray) -> np.ndarray:
    probs_batch = ranker.predict_batch(batch)
    if probs_batch.shape != (len(batch), batch.candidate_count):
        raise ValueError(
            "ranker returned invalid prediction shape: "
            f"{probs_batch.shape}, expected {(len(batch), batch.candidate_count)}"
        )
    probs_batch = np.asarray(probs_batch, dtype=np.float64)
    np.clip(probs_batch, 0.0, 1.0, out=probs_batch)
    priority_for_queries = getattr(ranker, "prediction_tie_break_prior", None)
    priorities = priority_for_queries(batch) if callable(priority_for_queries) else None
    stabilized, _ = break_prediction_ties(
        probs_batch,
        priorities=priorities,
        candidate_ids=batch.candidates,
    )
    return stabilized


def _write_probabilities(output_file, probabilities: np.ndarray) -> None:
    # 17 significant digits round-trip float64 nextafter tie-breaks.
    np.savetxt(output_file, probabilities, delimiter=",", fmt="%.17g")


def _prediction_order(ranker: Ranker, queries: TestQueryArray) -> np.ndarray | None:
    if not queries:
        return None
    order_for_queries = getattr(ranker, "prediction_order", None)
    if not callable(order_for_queries):
        return None
    order = order_for_queries(queries)
    if order is None:
        return None

    order = np.asarray(order)
    if order.shape != (len(queries),) or not np.issubdtype(order.dtype, np.integer):
        raise ValueError("ranker prediction order must be a one-dimensional integer permutation")
    order = order.astype(np.intp, copy=False)
    if np.any(order < 0) or np.any(order >= len(queries)):
        raise ValueError("ranker prediction order contains an out-of-range row index")
    seen = np.zeros(len(queries), dtype=bool)
    seen[order] = True
    if not np.all(seen):
        raise ValueError("ranker prediction order must contain every row exactly once")
    return order


def _write_scheduled_batches(
    output_file,
    ranker: Ranker,
    queries: TestQueryArray,
    prediction_order: np.ndarray,
    batch_size: int,
    dataset: DatasetPaths,
    output_path: Path,
    predict_start: float,
    next_progress_row: int,
    verbose: bool,
) -> int:
    fd, temp_name = tempfile.mkstemp(prefix="jgrec-predictions-", suffix=".dat")
    os.close(fd)
    probabilities: np.memmap | None = None
    row_count = 0
    try:
        probabilities = np.memmap(
            temp_name,
            mode="w+",
            dtype=np.float64,
            shape=(len(queries), queries.candidate_count),
        )
        for start in range(0, len(queries), batch_size):
            row_indices = prediction_order[start : start + batch_size]
            probs_batch = _predict_batch(ranker, queries[row_indices])
            probabilities[row_indices] = probs_batch
            batch_id = start // batch_size + 1
            if batch_id % PREDICTION_MEMMAP_FLUSH_INTERVAL == 0:
                _flush_and_release_memmap_pages(probabilities)
            batch_rows = len(row_indices)
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
        _flush_and_release_memmap_pages(probabilities)
        for start in range(0, len(queries), batch_size):
            _write_probabilities(output_file, probabilities[start : start + batch_size])
            batch_id = start // batch_size + 1
            if batch_id % PREDICTION_MEMMAP_FLUSH_INTERVAL == 0:
                _release_memmap_pages(probabilities)
        return row_count
    finally:
        if probabilities is not None:
            mmap = getattr(probabilities, "_mmap", None)
            if mmap is not None:
                mmap.close()
        Path(temp_name).unlink(missing_ok=True)


def _flush_and_release_memmap_pages(matrix: np.memmap) -> None:
    matrix.flush()
    _release_memmap_pages(matrix)


def _release_memmap_pages(matrix: np.memmap) -> None:
    mapping = getattr(matrix, "_mmap", None)
    madvise = getattr(mapping, "madvise", None)
    dont_need = getattr(mmap_module, "MADV_DONTNEED", None)
    if not callable(madvise) or dont_need is None:
        return
    with suppress(OSError, ValueError):
        madvise(dont_need)


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
