from __future__ import annotations

from collections.abc import Container
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
from typing import Protocol

import numpy as np

from jgrec.logging import log


class _DestinationView(Protocol):
    @property
    def cutoff(self) -> int: ...


class _SourceView(Protocol):
    @property
    def visible_dsts(self) -> np.ndarray: ...


class NegativeSamplingIndex(Protocol):
    def source_view(self, src: int, query_time: int) -> _SourceView:
        ...

    def destination_view(self, dst: int, query_time: int) -> _DestinationView:
        ...

    def transition_count(self, left_dst: int, right_dst: int, query_time: int) -> int:
        ...

    def cooccur_count(self, src: int, dst: int, query_time: int) -> int:
        ...


@dataclass(frozen=True)
class NegativeSamplingContext:
    index: NegativeSamplingIndex
    dst_values: tuple[int, ...]
    test_candidate_values: np.ndarray | None = None
    test_candidate_weights: np.ndarray | None = None


@dataclass(frozen=True)
class NegativeSamplingJob:
    src: int
    positive_dst: int
    query_time: int


STRUCTURAL_CANDIDATE_LIMIT = 512
POPULAR_CANDIDATE_LIMIT = 2048
PARALLEL_PROGRESS_INTERVAL = 10_000
DENSE_CANDIDATE_POOL_LIMIT = 10_000_000


@dataclass(frozen=True)
class DenseCandidatePool:
    mask: np.ndarray

    def __contains__(self, value: object) -> bool:
        try:
            item = int(value)
        except (TypeError, ValueError):
            return False
        return 0 <= item < self.mask.shape[0] and bool(self.mask[item])


def build_candidate_pool(*arrays: np.ndarray | tuple[int, ...]) -> Container[int]:
    values: list[np.ndarray] = []
    max_value = -1
    min_value = 0
    for array in arrays:
        current = np.asarray(array, dtype=np.int64)
        if current.size == 0:
            continue
        values.append(current)
        max_value = max(max_value, int(current.max()))
        min_value = min(min_value, int(current.min()))
    if not values:
        return set()
    if min_value >= 0 and max_value <= DENSE_CANDIDATE_POOL_LIMIT:
        mask = np.zeros(max_value + 1, dtype=bool)
        for current in values:
            mask[current] = True
        return DenseCandidatePool(mask=mask)
    return {int(value) for current in values for value in current}


def sample_mixed_negatives_batch(
    jobs: list[NegativeSamplingJob],
    context: NegativeSamplingContext,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
    hard_negative_ratio: float = 0.5,
    popular_negative_ratio: float = 0.25,
    test_candidate_negative_ratio: float = 0.0,
    candidate_pool: Container[int] | None = None,
    workers: int = 0,
    verbose: bool = False,
    label: str = "negative-sampling",
) -> list[tuple[int, ...]]:
    if not jobs:
        return []

    worker_count = _resolve_worker_count(workers, len(jobs))
    if worker_count <= 1:
        return [
            sample_mixed_negatives(
                src=job.src,
                positive_dst=job.positive_dst,
                query_time=job.query_time,
                context=context,
                dst_pool=dst_pool,
                num_negatives=num_negatives,
                rng=rng,
                hard_negative_ratio=hard_negative_ratio,
                popular_negative_ratio=popular_negative_ratio,
                test_candidate_negative_ratio=test_candidate_negative_ratio,
                candidate_pool=candidate_pool,
            )
            for job in jobs
        ]

    seeds = rng.integers(0, 2**32 - 1, size=len(jobs), dtype=np.uint32)
    chunks = _negative_sampling_chunks(jobs, seeds, worker_count)
    results_by_chunk: list[tuple[tuple[int, ...], ...] | None] = [None] * len(chunks)
    completed = 0
    next_progress = PARALLEL_PROGRESS_INTERVAL
    show_progress = verbose and len(jobs) >= PARALLEL_PROGRESS_INTERVAL

    if show_progress:
        log(
            f"[negative-sampling] {label} workers={worker_count} jobs={len(jobs)} chunks={len(chunks)}",
            enabled=True,
        )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _sample_negative_chunk,
                chunk,
                context,
                dst_pool,
                num_negatives,
                hard_negative_ratio,
                popular_negative_ratio,
                test_candidate_negative_ratio,
                candidate_pool,
            )
            for chunk in chunks
        ]
        for future in as_completed(futures):
            chunk_index, chunk_results = future.result()
            results_by_chunk[chunk_index] = chunk_results
            completed += len(chunk_results)
            if show_progress and completed >= next_progress:
                log(
                    f"[negative-sampling] {label} completed={completed}/{len(jobs)}",
                    enabled=True,
                )
                while completed >= next_progress:
                    next_progress += PARALLEL_PROGRESS_INTERVAL

    results: list[tuple[int, ...]] = []
    for chunk_results in results_by_chunk:
        if chunk_results is None:
            raise RuntimeError("negative sampling worker did not return a chunk")
        results.extend(chunk_results)
    if show_progress:
        log(f"[negative-sampling] {label} done jobs={len(results)}", enabled=True)
    return results


def sample_mixed_negatives_batch_seeded(
    jobs: list[NegativeSamplingJob],
    seeds: np.ndarray,
    context: NegativeSamplingContext,
    dst_pool: np.ndarray,
    num_negatives: int,
    hard_negative_ratio: float = 0.5,
    popular_negative_ratio: float = 0.25,
    test_candidate_negative_ratio: float = 0.0,
    candidate_pool: Container[int] | None = None,
    workers: int = 0,
) -> list[tuple[int, ...]]:
    if not jobs:
        return []
    if len(jobs) != len(seeds):
        raise ValueError("negative sampling jobs and seeds must have the same length")

    seed_values = tuple(int(seed) for seed in seeds)
    worker_count = _resolve_worker_count(workers, len(jobs))
    if worker_count <= 1:
        return [
            sample_mixed_negatives(
                src=job.src,
                positive_dst=job.positive_dst,
                query_time=job.query_time,
                context=context,
                dst_pool=dst_pool,
                num_negatives=num_negatives,
                rng=np.random.default_rng(seed),
                hard_negative_ratio=hard_negative_ratio,
                popular_negative_ratio=popular_negative_ratio,
                test_candidate_negative_ratio=test_candidate_negative_ratio,
                candidate_pool=candidate_pool,
            )
            for job, seed in zip(jobs, seed_values, strict=True)
        ]

    chunks = _negative_sampling_chunks(jobs, np.asarray(seed_values, dtype=np.uint32), worker_count)
    results_by_chunk: list[tuple[tuple[int, ...], ...] | None] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _sample_negative_chunk,
                chunk,
                context,
                dst_pool,
                num_negatives,
                hard_negative_ratio,
                popular_negative_ratio,
                test_candidate_negative_ratio,
                candidate_pool,
            )
            for chunk in chunks
        ]
        for future in as_completed(futures):
            chunk_index, chunk_results = future.result()
            results_by_chunk[chunk_index] = chunk_results

    results: list[tuple[int, ...]] = []
    for chunk_results in results_by_chunk:
        if chunk_results is None:
            raise RuntimeError("negative sampling worker did not return a chunk")
        results.extend(chunk_results)
    return results


def sample_mixed_negatives(
    src: int,
    positive_dst: int,
    query_time: int,
    context: NegativeSamplingContext,
    dst_pool: np.ndarray,
    num_negatives: int,
    rng: np.random.Generator,
    hard_negative_ratio: float = 0.5,
    popular_negative_ratio: float = 0.25,
    test_candidate_negative_ratio: float = 0.0,
    candidate_pool: Container[int] | None = None,
) -> tuple[int, ...]:
    positive_dst = int(positive_dst)
    used = {positive_dst}
    if candidate_pool is None:
        candidate_pool = build_candidate_pool(dst_pool)
    seen_history = context.index.source_view(src, query_time)
    source_history = {int(dst) for dst in seen_history.visible_dsts}

    negatives: list[int] = []
    hard_ratio = _clamp_ratio(hard_negative_ratio)
    popular_ratio = _clamp_ratio(popular_negative_ratio)
    test_candidate_ratio = _clamp_ratio(test_candidate_negative_ratio)
    test_candidate_count = min(num_negatives, round(num_negatives * test_candidate_ratio))
    if test_candidate_count > 0:
        _take_test_candidate_negatives(
            context,
            test_candidate_count,
            negatives,
            used,
            rng,
            blocked=source_history,
        )

    remaining = num_negatives - len(negatives)
    hard_count = min(remaining, round(num_negatives * hard_ratio))
    popular_count = min(remaining - hard_count, round(num_negatives * popular_ratio))

    hard_start_count = len(negatives)
    recent_count = (hard_count + 1) // 2
    if recent_count > 0:
        recent_candidates = _source_recent_negative_candidates(src, query_time, context)
        _take_candidates(recent_candidates, recent_count, negatives, used, rng, allowed=candidate_pool)

    structural_count = hard_count - (len(negatives) - hard_start_count)
    if structural_count > 0:
        structural_candidates = _structural_hard_negative_candidates(src, query_time, context)
        _take_candidates(
            structural_candidates,
            structural_count,
            negatives,
            used,
            rng,
            blocked=source_history,
            allowed=candidate_pool,
        )

    if popular_count > 0:
        popular_candidates = _popular_negative_candidates(dst_pool, context, query_time)
        _take_candidates(
            popular_candidates,
            popular_count,
            negatives,
            used,
            rng,
            blocked=source_history,
            allowed=candidate_pool,
        )

    random_target = num_negatives - len(negatives)
    if len(dst_pool) > 0:
        _take_random_candidates(dst_pool, random_target, negatives, used, rng, blocked=source_history)

    if len(negatives) < num_negatives:
        for value in dst_pool:
            dst = int(value)
            if dst in used:
                continue
            used.add(dst)
            negatives.append(dst)
            if len(negatives) >= num_negatives:
                break

    if len(negatives) < num_negatives:
        negatives.extend([positive_dst] * (num_negatives - len(negatives)))
    return tuple(negatives)


def _source_recent_negative_candidates(
    src: int,
    query_time: int,
    context: NegativeSamplingContext,
) -> list[int]:
    source_view = context.index.source_view(src, query_time)
    return [int(dst) for dst in reversed(source_view.visible_dsts)]


def _structural_hard_negative_candidates(
    src: int,
    query_time: int,
    context: NegativeSamplingContext,
) -> list[int]:
    index = context.index
    source_view = index.source_view(src, query_time)
    recent = [int(dst) for dst in reversed(source_view.visible_dsts)]
    candidates: list[int] = []
    seen_set: set[int] = set()
    for dst in recent:
        if dst in seen_set:
            continue
        seen_set.add(dst)

    if recent:
        last_dst = recent[0]
        transition_candidates = getattr(index, "transition_candidates", None)
        if callable(transition_candidates):
            candidates.extend(
                int(candidate)
                for candidate in transition_candidates(
                    last_dst,
                    query_time,
                    limit=STRUCTURAL_CANDIDATE_LIMIT,
                )
            )
        else:
            transitions = [
                (candidate, index.transition_count(last_dst, candidate, query_time))
                for candidate in context.dst_values[:STRUCTURAL_CANDIDATE_LIMIT]
            ]
            transitions.sort(key=lambda item: (-item[1], item[0]))
            candidates.extend(candidate for candidate, count in transitions if count > 0)

    cooccur_candidates = getattr(index, "cooccur_candidates", None)
    if callable(cooccur_candidates):
        candidates.extend(
            int(candidate)
            for candidate in cooccur_candidates(
                src,
                query_time,
                limit=STRUCTURAL_CANDIDATE_LIMIT,
            )
        )
    else:
        cooccurs = [
            (candidate, index.cooccur_count(src, candidate, query_time))
            for candidate in context.dst_values[:STRUCTURAL_CANDIDATE_LIMIT]
        ]
        cooccurs.sort(key=lambda item: (-item[1], item[0]))
        candidates.extend(candidate for candidate, count in cooccurs if count > 0)
    return _dedupe(candidates)


def _popular_negative_candidates(
    dst_pool: np.ndarray,
    context: NegativeSamplingContext,
    query_time: int,
) -> list[int]:
    popular_destinations = getattr(context.index, "popular_destinations", None)
    if callable(popular_destinations):
        return [
            int(dst)
            for dst in popular_destinations(
                query_time,
                limit=POPULAR_CANDIDATE_LIMIT,
            )
        ]

    candidates: list[tuple[int, int]] = []
    for value in dst_pool[:POPULAR_CANDIDATE_LIMIT]:
        dst = int(value)
        count = context.index.destination_view(dst, query_time).cutoff
        if count > 0:
            candidates.append((dst, count))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [dst for dst, _ in candidates]


def _take_candidates(
    candidates: list[int],
    target_count: int,
    negatives: list[int],
    used: set[int],
    rng: np.random.Generator,
    blocked: set[int] | None = None,
    allowed: Container[int] | None = None,
) -> None:
    if target_count <= 0 or not candidates:
        return
    start_count = len(negatives)
    order = np.arange(len(candidates))
    rng.shuffle(order)
    for idx in order:
        dst = int(candidates[int(idx)])
        if dst in used or (blocked is not None and dst in blocked) or (allowed is not None and dst not in allowed):
            continue
        used.add(dst)
        negatives.append(dst)
        if len(negatives) - start_count >= target_count:
            break


def _take_random_candidates(
    dst_pool: np.ndarray,
    target_count: int,
    negatives: list[int],
    used: set[int],
    rng: np.random.Generator,
    blocked: set[int] | None = None,
) -> None:
    attempts = 0
    start_count = len(negatives)
    while len(negatives) - start_count < target_count and attempts < 20:
        attempts += 1
        draw_size = max((target_count - (len(negatives) - start_count)) * 3, 16)
        sampled = rng.choice(dst_pool, size=draw_size, replace=len(dst_pool) < draw_size)
        for value in sampled:
            dst = int(value)
            if dst in used or (blocked is not None and dst in blocked):
                continue
            used.add(dst)
            negatives.append(dst)
            if len(negatives) - start_count >= target_count:
                break


def _take_test_candidate_negatives(
    context: NegativeSamplingContext,
    target_count: int,
    negatives: list[int],
    used: set[int],
    rng: np.random.Generator,
    blocked: set[int] | None = None,
) -> None:
    values = context.test_candidate_values
    if target_count <= 0 or values is None or len(values) == 0:
        return

    weights = context.test_candidate_weights
    draw_size = min(max(target_count * 8, 32), max(len(values), 1))
    replace = len(values) < draw_size
    start_count = len(negatives)
    attempts = 0
    while len(negatives) - start_count < target_count and attempts < 5:
        attempts += 1
        sampled = rng.choice(values, size=draw_size, replace=replace, p=weights)
        for value in sampled:
            dst = int(value)
            if dst in used or (blocked is not None and dst in blocked):
                continue
            used.add(dst)
            negatives.append(dst)
            if len(negatives) - start_count >= target_count:
                break


def _clamp_ratio(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _dedupe(values: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _resolve_worker_count(workers: int, job_count: int) -> int:
    if workers <= 1 or job_count <= 1:
        return 1
    cpu_limit = max((cpu_count() or workers) - 1, 1)
    return max(1, min(int(workers), int(job_count), cpu_limit))


def _negative_sampling_chunks(
    jobs: list[NegativeSamplingJob],
    seeds: np.ndarray,
    worker_count: int,
) -> list[tuple[int, tuple[NegativeSamplingJob, ...], tuple[int, ...]]]:
    target_chunks = max(worker_count * 8, 1)
    chunk_size = max(1, (len(jobs) + target_chunks - 1) // target_chunks)
    if len(jobs) >= 1024:
        chunk_size = max(256, min(2048, chunk_size))

    chunks: list[tuple[int, tuple[NegativeSamplingJob, ...], tuple[int, ...]]] = []
    for chunk_index, start in enumerate(range(0, len(jobs), chunk_size)):
        end = min(start + chunk_size, len(jobs))
        chunks.append(
            (
                chunk_index,
                tuple(jobs[start:end]),
                tuple(int(seed) for seed in seeds[start:end]),
            )
        )
    return chunks


def _sample_negative_chunk(
    payload: tuple[int, tuple[NegativeSamplingJob, ...], tuple[int, ...]],
    context: NegativeSamplingContext,
    dst_pool: np.ndarray,
    num_negatives: int,
    hard_negative_ratio: float,
    popular_negative_ratio: float,
    test_candidate_negative_ratio: float,
    candidate_pool: Container[int] | None,
) -> tuple[int, tuple[tuple[int, ...], ...]]:
    chunk_index, jobs, seeds = payload
    results: list[tuple[int, ...]] = []
    for job, seed in zip(jobs, seeds, strict=True):
        results.append(
            sample_mixed_negatives(
                src=job.src,
                positive_dst=job.positive_dst,
                query_time=job.query_time,
                context=context,
                dst_pool=dst_pool,
                num_negatives=num_negatives,
                rng=np.random.default_rng(seed),
                hard_negative_ratio=hard_negative_ratio,
                popular_negative_ratio=popular_negative_ratio,
                test_candidate_negative_ratio=test_candidate_negative_ratio,
                candidate_pool=candidate_pool,
            )
        )
    return chunk_index, tuple(results)
