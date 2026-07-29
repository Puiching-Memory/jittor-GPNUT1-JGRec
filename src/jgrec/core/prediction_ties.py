from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TieBreakReport:
    rows_with_ties: int = 0
    tied_groups: int = 0
    tied_candidates: int = 0
    rank_fallback_rows: int = 0


def break_prediction_ties(
    scores: np.ndarray,
    *,
    priorities: np.ndarray | None = None,
    candidate_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, TieBreakReport]:
    """Give every candidate a stable float64 score without reordering non-ties.

    Exact-score groups are ordered by descending priority, ascending candidate
    id, then their original column. Adjacent float64 values are used so the
    change is as small as the output representation permits.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("prediction scores must be a finite two-dimensional matrix")
    shape = values.shape
    if priorities is None:
        priority_values = np.zeros(shape, dtype=np.float64)
    else:
        priority_values = np.asarray(priorities, dtype=np.float64)
        if priority_values.shape != shape or not np.all(np.isfinite(priority_values)):
            raise ValueError("tie-break priorities must match prediction scores and be finite")
    if candidate_ids is None:
        id_values = np.broadcast_to(np.arange(shape[1], dtype=np.int64), shape)
    else:
        id_values = np.asarray(candidate_ids)
        if id_values.shape != shape:
            raise ValueError("candidate ids must match prediction scores")

    output = values.copy()
    rows_with_ties = 0
    tied_groups = 0
    tied_candidates = 0
    rank_fallback_rows = 0
    for row_index in range(shape[0]):
        row = values[row_index]
        unique_values, inverse, counts = np.unique(
            row,
            return_inverse=True,
            return_counts=True,
        )
        duplicate_groups = np.flatnonzero(counts > 1)
        if duplicate_groups.size == 0:
            continue
        rows_with_ties += 1
        tied_groups += int(duplicate_groups.size)
        tied_candidates += int(counts[duplicate_groups].sum())
        try:
            for group_index in duplicate_groups:
                columns = np.flatnonzero(inverse == group_index)
                ordered = columns[
                    np.lexsort(
                        (
                            columns,
                            id_values[row_index, columns],
                            -priority_values[row_index, columns],
                        )
                    )
                ]
                value = float(unique_values[group_index])
                lower = (
                    float(unique_values[group_index - 1])
                    if group_index > 0
                    else -np.inf
                )
                upper = (
                    float(unique_values[group_index + 1])
                    if group_index + 1 < unique_values.size
                    else np.inf
                )
                replacements = _descending_float64_values(
                    value,
                    len(ordered),
                    lower=lower,
                    upper=upper,
                )
                output[row_index, ordered] = replacements
            row_output = output[row_index]
            contains_subnormal = np.any(
                (row_output > 0.0)
                & (row_output < np.finfo(np.float64).tiny)
            )
            if (
                np.unique(row_output).size != shape[1]
                or contains_subnormal
            ):
                raise FloatingPointError(
                    "tie perturbation is not stable under float64 output"
                )
        except FloatingPointError:
            output[row_index] = _strict_rank_scores(
                row,
                priority_values[row_index],
                id_values[row_index],
            )
            rank_fallback_rows += 1

    return output, TieBreakReport(
        rows_with_ties=rows_with_ties,
        tied_groups=tied_groups,
        tied_candidates=tied_candidates,
        rank_fallback_rows=rank_fallback_rows,
    )


def _descending_float64_values(
    value: float,
    count: int,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    # Some CUDA/Jittor process environments enable flush-to-zero. In that
    # mode nextafter(0, +inf) collapses back to zero, so use normal float64
    # values explicitly for the clipped lower boundary.
    if value == 0.0:
        step = np.finfo(np.float64).tiny
        upward = (
            np.arange(count - 1, -1, -1, dtype=np.float64) * step
        )
        if upward[0] < upper:
            return upward

    downward = np.empty(count, dtype=np.float64)
    downward[0] = value
    for index in range(1, count):
        downward[index] = np.nextafter(downward[index - 1], -np.inf)
    if downward[-1] > lower and downward[-1] >= 0.0:
        return downward

    upward = np.empty(count, dtype=np.float64)
    upward[-1] = value
    for index in range(count - 2, -1, -1):
        upward[index] = np.nextafter(upward[index + 1], np.inf)
    if upward[0] < upper and upward[0] <= 1.0:
        return upward

    raise FloatingPointError(
        "cannot break an exact prediction tie without crossing a neighboring score"
    )


def _strict_rank_scores(
    scores: np.ndarray,
    priorities: np.ndarray,
    candidate_ids: np.ndarray,
) -> np.ndarray:
    columns = np.arange(scores.size)
    order = np.lexsort(
        (
            columns,
            candidate_ids,
            -priorities,
            -scores,
        )
    )
    ordered_scores = np.linspace(1.0, 0.0, num=scores.size)
    output = np.empty(scores.size, dtype=np.float64)
    output[order] = ordered_scores
    return output
