from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from jgrec.rankers.base import Ranker

from .io import read_interactions, read_test_queries
from .memory import log_event, log_memory, release_memory
from .types import DatasetPaths, DatasetResult, FitContext, TestQueryArray, TrainingReport

PREDICT_PROGRESS_INTERVAL = 10_000


def build_dataset_submission(
    dataset: DatasetPaths,
    ranker: Ranker,
    output_dir: Path,
    batch_size: int = 2048,
    seed: int = 42,
    verbose: bool = True,
    limit_rows: int | None = None,
    checkpoint_dir: Path | None = None,
    load_checkpoint_dir: Path | None = None,
    model_name: str = "",
    full_model_dir: Path | None = None,
    load_full_model_dir: Path | None = None,
    recompute_test_profile: bool = False,
) -> DatasetResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset.name}.csv"

    if dataset.train_path.exists() and dataset.train_path.stat().st_size > 0:
        log_memory(f"read_train_start:{dataset.name}", enabled=verbose)
        interactions = _read_fit_interactions(dataset, ranker)
        log_memory(f"read_train_done:{dataset.name}", enabled=verbose)

        save_checkpoint_path = None
        if checkpoint_dir is not None:
            name = model_name or ranker.name
            save_checkpoint_path = checkpoint_dir / f"{dataset.name}_{name}.npz"
        load_checkpoint_path = None
        if load_checkpoint_dir is not None:
            name = model_name or ranker.name
            load_checkpoint_path = load_checkpoint_dir / f"{dataset.name}_{name}.npz"

        report = ranker.fit(
            interactions,
            FitContext(
                dataset=dataset,
                seed=seed,
                limit_rows=limit_rows,
                verbose=verbose,
                load_checkpoint_path=load_checkpoint_path,
                save_checkpoint_path=save_checkpoint_path,
            ),
        )
        if full_model_dir is not None:
            model_out = full_model_dir / dataset.name
            _save_full_model(ranker, model_out, verbose=verbose)
        del interactions
    else:
        log_event(f"[test-only] {dataset.name}: no train.csv, loading full model", enabled=verbose)
        if load_full_model_dir is None:
            raise FileNotFoundError(f"train.csv missing for {dataset.name} and --load-full-model-dir not provided")
        model_in = load_full_model_dir / dataset.name
        _load_full_model(ranker, model_in, verbose=verbose)
        report = TrainingReport(
            train_events=0,
            val_events=0,
            best_val_ap=0.0,
            best_val_mrr=0.0,
            feature_names=(),
            selected_fusion="test-only",
            model_name=model_name or ranker.name,
        )

    if recompute_test_profile:
        _recompute_test_profile_for_ranker(ranker, dataset.test_path, verbose=verbose)

    release_memory()
    log_memory(f"post_fit:{dataset.name}", enabled=verbose)

    row_count = 0
    predict_start = perf_counter()
    next_progress_row = PREDICT_PROGRESS_INTERVAL
    log_memory(f"predict_start:{dataset.name}", enabled=verbose)
    queries = read_test_queries(dataset.test_path, max_rows=limit_rows)
    with output_path.open("w", newline="") as f:
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
    probs_batch = ranker.predict_batch(batch)
    if probs_batch.shape != (len(batch), batch.candidate_count):
        raise ValueError(
            "ranker returned invalid prediction shape: "
            f"{probs_batch.shape}, expected {(len(batch), batch.candidate_count)}"
        )
    np.clip(probs_batch, 0.0, 1.0, out=probs_batch)
    np.savetxt(output_file, probs_batch, delimiter=",", fmt="%.8f")
    return len(batch)


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


def _save_full_model(ranker: Ranker, model_out: Path, *, verbose: bool = True) -> None:
    save_fn = getattr(ranker, "save_full_model", None)
    if save_fn is None:
        log_event(f"[full-model] ranker {ranker.name} does not support full model save", enabled=verbose)
        return
    save_fn(model_out)
    log_event(f"[full-model] saved {model_out}", enabled=verbose)


def _load_full_model(ranker: Ranker, model_in: Path, *, verbose: bool = True) -> None:
    load_fn = getattr(ranker, "load_full_model", None)
    if load_fn is None:
        raise RuntimeError(f"ranker {ranker.name} does not support full model loading")
    load_fn(model_in)
    log_event(f"[full-model] loaded {model_in}", enabled=verbose)


def _recompute_test_profile_for_ranker(ranker: Ranker, test_path: Path, *, verbose: bool = True) -> None:
    """Recompute test-candidate-specific signals from a new test set.

    Only supported for hybrid rankers at the moment.
    """
    impl = getattr(ranker, "impl", None)
    if impl is None:
        return
    encoder = getattr(impl, "encoder", None)
    if encoder is None:
        return
    prior = getattr(encoder, "candidate_prior", None)
    if prior is None or not getattr(prior, "config", None) or not prior.config.enabled:
        return

    from collections import Counter

    from jgrec.core.io import read_test_queries

    queries = read_test_queries(test_path)
    if len(queries) == 0:
        return
    candidates = queries.candidates.astype(np.int64, copy=False).reshape(-1)
    values, counts = np.unique(candidates, return_counts=True)
    test_counts = Counter({int(value): int(count) for value, count in zip(values, counts, strict=True)})
    prior.fit_from_counts(set(prior.train_dst), test_counts)
    log_event(f"[test-profile] recomputed candidate_prior for {test_path}", enabled=verbose)
