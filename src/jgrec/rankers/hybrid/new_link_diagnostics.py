from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentRankingMetrics:
    rows: int
    top1_errors: int
    reciprocal_rank_sum: float
    regret_sum: float
    mrr: float | None
    top1_error_rate: float | None


@dataclass(frozen=True)
class NewLinkSliceReport:
    start: int
    stop: int
    repeat: SegmentRankingMetrics
    new: SegmentRankingMetrics
    repeat_minus_new_mrr: float | None


@dataclass(frozen=True)
class NewLinkErrorReport:
    rows: int
    repeat: SegmentRankingMetrics
    new: SegmentRankingMetrics
    new_row_share: float
    new_regret_share: float | None
    repeat_minus_new_mrr: float | None
    slices: tuple[NewLinkSliceReport, ...]


def historical_pair_mask(
    context_src: np.ndarray,
    context_dst: np.ndarray,
    query_src: np.ndarray,
    positive_dst: np.ndarray,
) -> np.ndarray:
    """Mark query positives whose source-target pair occurs in the fixed context."""
    context_src_values = np.asarray(context_src, dtype=np.int64)
    context_dst_values = np.asarray(context_dst, dtype=np.int64)
    query_src_values = np.asarray(query_src, dtype=np.int64)
    positive_dst_values = np.asarray(positive_dst, dtype=np.int64)
    if context_src_values.ndim != 1 or context_dst_values.ndim != 1:
        raise ValueError("context source and target arrays must be one-dimensional")
    if query_src_values.ndim != 1 or positive_dst_values.ndim != 1:
        raise ValueError("query source and positive target arrays must be one-dimensional")
    if context_src_values.shape != context_dst_values.shape:
        raise ValueError("context source and target arrays must align")
    if query_src_values.shape != positive_dst_values.shape:
        raise ValueError("query source and positive target arrays must align")
    if query_src_values.size == 0:
        return np.empty(0, dtype=bool)
    if context_src_values.size == 0:
        return np.zeros(query_src_values.size, dtype=bool)

    context_keys = np.unique(_pair_keys(context_src_values, context_dst_values))
    query_keys = _pair_keys(query_src_values, positive_dst_values)
    return np.isin(query_keys, context_keys, assume_unique=False)


def new_link_error_report(
    scores: np.ndarray,
    repeat_mask: np.ndarray,
    *,
    slices: tuple[slice, ...],
) -> NewLinkErrorReport:
    values = np.asarray(scores)
    repeated = np.asarray(repeat_mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("scores must contain query groups with at least two candidates")
    if repeated.ndim != 1 or repeated.shape[0] != values.shape[0]:
        raise ValueError("scores and repeat mask must align")
    if values.shape[0] == 0:
        raise ValueError("scores must contain at least one query")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")

    ranks = 1 + (values[:, 1:] > values[:, 0:1]).sum(axis=1)
    reciprocal_ranks = 1.0 / ranks
    repeat_metrics = _segment_metrics(reciprocal_ranks, repeated)
    new_metrics = _segment_metrics(reciprocal_ranks, ~repeated)
    total_regret = repeat_metrics.regret_sum + new_metrics.regret_sum
    slice_reports = tuple(
        _slice_report(reciprocal_ranks, repeated, part)
        for part in slices
    )
    return NewLinkErrorReport(
        rows=int(values.shape[0]),
        repeat=repeat_metrics,
        new=new_metrics,
        new_row_share=float(new_metrics.rows / values.shape[0]),
        new_regret_share=(float(new_metrics.regret_sum / total_regret) if total_regret > 0.0 else None),
        repeat_minus_new_mrr=_mrr_gap(repeat_metrics, new_metrics),
        slices=slice_reports,
    )


def passes_new_link_concentration_gate(
    report: NewLinkErrorReport,
    *,
    min_rows_per_segment: int = 100,
    min_new_row_share: float = 0.50,
    min_new_regret_share: float = 0.65,
    min_full_mrr_gap: float = 0.03,
    min_slice_mrr_gap: float = 0.02,
) -> bool:
    thresholds = np.asarray(
        [min_new_row_share, min_new_regret_share, min_full_mrr_gap, min_slice_mrr_gap],
        dtype=np.float64,
    )
    if min_rows_per_segment < 1 or not np.all(np.isfinite(thresholds)) or np.any(thresholds < 0.0):
        raise ValueError("new-link concentration thresholds are invalid")
    if report.repeat.rows < min_rows_per_segment or report.new.rows < min_rows_per_segment:
        return False
    if any(
        part.repeat.rows < min_rows_per_segment or part.new.rows < min_rows_per_segment
        for part in report.slices
    ):
        return False
    if report.new_regret_share is None or report.repeat_minus_new_mrr is None:
        return False
    return bool(
        report.new_row_share >= min_new_row_share
        and report.new_regret_share >= min_new_regret_share
        and report.repeat_minus_new_mrr >= min_full_mrr_gap
        and all(
            part.repeat_minus_new_mrr is not None
            and part.repeat_minus_new_mrr >= min_slice_mrr_gap
            for part in report.slices
        )
    )


def _pair_keys(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    pairs = np.empty((src.size, 2), dtype=np.int64)
    pairs[:, 0] = src
    pairs[:, 1] = dst
    return pairs.view(np.dtype((np.void, pairs.dtype.itemsize * 2))).reshape(-1)


def _segment_metrics(reciprocal_ranks: np.ndarray, mask: np.ndarray) -> SegmentRankingMetrics:
    values = reciprocal_ranks[mask]
    rows = int(values.size)
    if rows == 0:
        return SegmentRankingMetrics(
            rows=0,
            top1_errors=0,
            reciprocal_rank_sum=0.0,
            regret_sum=0.0,
            mrr=None,
            top1_error_rate=None,
        )
    top1_errors = int(np.count_nonzero(values < 1.0))
    reciprocal_rank_sum = float(values.sum())
    return SegmentRankingMetrics(
        rows=rows,
        top1_errors=top1_errors,
        reciprocal_rank_sum=reciprocal_rank_sum,
        regret_sum=float(np.sum(1.0 - values)),
        mrr=float(reciprocal_rank_sum / rows),
        top1_error_rate=float(top1_errors / rows),
    )


def _slice_report(
    reciprocal_ranks: np.ndarray,
    repeat_mask: np.ndarray,
    part: slice,
) -> NewLinkSliceReport:
    start = 0 if part.start is None else int(part.start)
    stop = reciprocal_ranks.size if part.stop is None else int(part.stop)
    step = 1 if part.step is None else int(part.step)
    if step != 1 or start < 0 or stop <= start or stop > reciprocal_ranks.size:
        raise ValueError("diagnostic slices must be non-empty contiguous ranges within scores")
    repeat_metrics = _segment_metrics(reciprocal_ranks[start:stop], repeat_mask[start:stop])
    new_metrics = _segment_metrics(reciprocal_ranks[start:stop], ~repeat_mask[start:stop])
    return NewLinkSliceReport(
        start=start,
        stop=stop,
        repeat=repeat_metrics,
        new=new_metrics,
        repeat_minus_new_mrr=_mrr_gap(repeat_metrics, new_metrics),
    )


def _mrr_gap(repeat: SegmentRankingMetrics, new: SegmentRankingMetrics) -> float | None:
    if repeat.mrr is None or new.mrr is None:
        return None
    return float(repeat.mrr - new.mrr)
