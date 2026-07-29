from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jittor as jt
import numpy as np


@dataclass(frozen=True)
class TemporalOOFFold:
    train_rows: tuple[int, int]
    score_rows: tuple[int, int]


def expanding_oof_folds(
    *,
    row_count: int,
    burn_in: int,
    fold_size: int,
) -> tuple[TemporalOOFFold, ...]:
    total = int(row_count)
    initial = int(burn_in)
    step = int(fold_size)
    if total <= 0:
        raise ValueError("OOF row count must be positive")
    if initial <= 0 or initial >= total:
        raise ValueError("OOF burn-in must be between zero and row count")
    if step <= 0:
        raise ValueError("OOF fold size must be positive")

    folds: list[TemporalOOFFold] = []
    score_start = initial
    while score_start < total:
        score_stop = min(score_start + step, total)
        folds.append(
            TemporalOOFFold(
                train_rows=(0, score_start),
                score_rows=(score_start, score_stop),
            )
        )
        score_start = score_stop
    return tuple(folds)


def validate_candidate_groups(
    src: np.ndarray,
    dst: np.ndarray,
    candidates: np.ndarray,
    *,
    width: int,
) -> None:
    source_values = np.asarray(src)
    positive_values = np.asarray(dst)
    candidate_values = np.asarray(candidates)
    if source_values.ndim != 1 or positive_values.ndim != 1:
        raise ValueError("source and positive arrays must be one-dimensional")
    if candidate_values.ndim != 2:
        raise ValueError("candidate groups must be a two-dimensional matrix")
    if candidate_values.shape[1] != int(width):
        raise ValueError(f"candidate groups must contain exactly {width} candidates")
    if not (
        source_values.shape[0]
        == positive_values.shape[0]
        == candidate_values.shape[0]
    ):
        raise ValueError("candidate group row counts do not align")
    if not np.array_equal(candidate_values[:, 0], positive_values):
        raise ValueError("the positive candidate must be stored in column 0")


def listwise_positive_loss(logits: jt.Var) -> jt.Var:
    if len(logits.shape) != 2 or logits.shape[1] <= 1:
        raise ValueError(
            "listwise logits must have shape [queries, candidates] "
            "with at least two candidates"
        )
    return -jt.nn.log_softmax(logits, dim=1)[:, 0].mean()


def full_candidate_mrr(scores: np.ndarray) -> float:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 1:
        raise ValueError(
            "candidate scores must have shape [queries, candidates] "
            "with at least two candidates"
        )
    ranks = 1 + np.sum(values[:, 1:] > values[:, :1], axis=1)
    return float(np.mean(1.0 / ranks))


def graph_candidate_logits(
    model,
    user_ids: np.ndarray,
    candidate_ids: np.ndarray,
) -> jt.Var:
    users = np.asarray(user_ids, dtype=np.int32)
    candidates = np.asarray(candidate_ids, dtype=np.int32)
    if users.ndim != 1 or candidates.ndim != 2:
        raise ValueError("mapped users/candidates must have shapes [B] and [B, C]")
    if users.shape[0] != candidates.shape[0]:
        raise ValueError("mapped user and candidate rows do not align")
    if np.any(users < 0):
        raise ValueError("positive training sources must exist in the graph ID map")

    valid = candidates >= 0
    safe_candidates = candidates.clip(min=0)
    user_all, item_all = model.get_all_embeddings()
    user_embeddings = user_all[jt.array(users, dtype=jt.int32)]
    item_embeddings = item_all[
        jt.array(safe_candidates, dtype=jt.int32)
    ]
    logits = (
        item_embeddings
        * user_embeddings.unsqueeze(1)
    ).sum(dim=-1)
    if not np.all(valid):
        logits = logits * jt.array(valid.astype(np.float32, copy=False))
    return logits


def replace_feature_column(
    source: np.ndarray,
    replacement: np.ndarray,
    *,
    column: int,
    output_path: Path,
    batch_rows: int = 256,
) -> dict[str, object]:
    source_values = np.asarray(source)
    replacement_values = np.asarray(replacement)
    if source_values.ndim != 3:
        raise ValueError("source features must have shape [queries, candidates, features]")
    if replacement_values.shape != source_values.shape[:2]:
        raise ValueError("replacement scores must match source query/candidate shape")
    feature_column = int(column)
    if not 0 <= feature_column < source_values.shape[2]:
        raise ValueError("replacement feature column is out of range")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite feature cache: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary feature cache already exists: {temporary}")

    rows_per_batch = max(int(batch_rows), 1)
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=source_values.dtype,
        shape=source_values.shape,
    )
    try:
        for start in range(0, source_values.shape[0], rows_per_batch):
            end = min(start + rows_per_batch, source_values.shape[0])
            output[start:end] = source_values[start:end]
            output[start:end, :, feature_column] = replacement_values[start:end]
        output.flush()
        unchanged = True
        for start in range(0, source_values.shape[0], rows_per_batch):
            end = min(start + rows_per_batch, source_values.shape[0])
            if feature_column > 0 and not np.array_equal(
                output[start:end, :, :feature_column],
                source_values[start:end, :, :feature_column],
            ):
                unchanged = False
                break
            if feature_column + 1 < source_values.shape[2] and not np.array_equal(
                output[start:end, :, feature_column + 1 :],
                source_values[start:end, :, feature_column + 1 :],
            ):
                unchanged = False
                break
        if not unchanged:
            raise RuntimeError("non-target feature columns changed during replacement")
    finally:
        del output
    temporary.replace(destination)
    return {
        "shape": list(source_values.shape),
        "replaced_column": feature_column,
        "unchanged_columns_equal": True,
    }
