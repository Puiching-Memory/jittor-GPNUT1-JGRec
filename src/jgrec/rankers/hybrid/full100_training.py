from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from jgrec.core.types import TestQueryArray


def validate_joint_cache_reports(
    train_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that train and validation caches came from one live build."""
    if train_report.get("status") != "complete":
        raise ValueError("joint training cache report is incomplete")
    if validation_report.get("status") != "complete":
        raise ValueError("joint validation cache report is incomplete")

    train_dataset = train_report.get("dataset_name")
    validation_dataset = validation_report.get("dataset_name")
    if (
        train_dataset is not None
        or validation_dataset is not None
    ) and train_dataset != validation_dataset:
        raise ValueError("joint cache reports refer to different datasets")

    train_joint = train_report.get("joint_build")
    validation_joint = validation_report.get("joint_build")
    if not isinstance(train_joint, Mapping) or not isinstance(
        validation_joint,
        Mapping,
    ):
        raise ValueError("joint cache reports lack same-process provenance")
    build_id = str(train_joint.get("id", ""))
    if not build_id or build_id != str(validation_joint.get("id", "")):
        raise ValueError("joint cache reports do not share one build ID")
    train_pid = int(train_joint.get("pid", -1))
    validation_pid = int(validation_joint.get("pid", -2))
    if train_pid <= 0 or validation_pid <= 0:
        raise ValueError("joint cache reports have an invalid process ID")
    if train_joint.get("role") != "train":
        raise ValueError("joint training report has the wrong role")
    if validation_joint.get("role") != "validation":
        raise ValueError("joint validation report has the wrong role")

    train_artifacts = train_report.get("artifacts")
    if not isinstance(train_artifacts, Mapping):
        raise ValueError("joint training report lacks artifacts")
    feature_artifact = train_artifacts.get("features")
    if not isinstance(feature_artifact, Mapping):
        raise ValueError("joint training report lacks a feature artifact")
    train_feature_sha256 = str(feature_artifact.get("sha256", ""))
    if (
        not train_feature_sha256
        or validation_report.get("train_feature_sha256")
        != train_feature_sha256
    ):
        raise ValueError("joint validation report does not bind the training feature hash")
    validation_recovery = train_pid != validation_pid
    if validation_recovery:
        _validate_frozen_candidate_validation_recovery(
            validation_report=validation_report,
            build_id=build_id,
            train_pid=train_pid,
            validation_pid=validation_pid,
            train_feature_sha256=train_feature_sha256,
        )
    result = {
        "matched": True,
        "joint_build_id": build_id,
        "process_id": train_pid,
        "train_feature_sha256": train_feature_sha256,
    }
    if validation_recovery:
        result.update(
            {
                "validation_recovery": True,
                "validation_process_id": validation_pid,
            }
        )
    return result


def _validate_frozen_candidate_validation_recovery(
    *,
    validation_report: Mapping[str, Any],
    build_id: str,
    train_pid: int,
    validation_pid: int,
    train_feature_sha256: str,
) -> None:
    recovery = validation_report.get("recovery")
    if not isinstance(recovery, Mapping):
        raise ValueError(
            "joint cache reports were not produced by the same process "
            "and lack frozen-candidate recovery evidence"
        )
    if recovery.get("protocol") != (
        "frozen_candidate_validation_recovery_v1"
    ):
        raise ValueError("validation recovery protocol differs")
    expected_bindings = {
        "recovery_process_id": validation_pid,
        "train_joint_build_id": build_id,
        "train_joint_process_id": train_pid,
        "train_feature_sha256": train_feature_sha256,
    }
    for key, expected in expected_bindings.items():
        if recovery.get(key) != expected:
            raise ValueError(f"validation recovery {key} differs")
    if recovery.get("candidate_sampling_performed") is not False:
        raise ValueError(
            "validation recovery candidate sampling must be disabled"
        )
    if recovery.get("frozen_query_alignment_exact") is not True:
        raise ValueError(
            "validation recovery lacks exact frozen query alignment"
        )
    sidecars = recovery.get("reference_sidecars_exact")
    required_sidecars = (
        "candidates",
        "src",
        "dst",
        "time",
        "row_indices",
    )
    if (
        not isinstance(sidecars, Mapping)
        or any(sidecars.get(name) is not True for name in required_sidecars)
    ):
        raise ValueError(
            "validation recovery reference sidecars are not exact"
        )


def build_frozen_candidate_queries(
    positives: Any,
    frozen_candidates: np.ndarray,
) -> TestQueryArray:
    """Build queries from a caller-owned candidate contract without sampling."""
    candidate_values = np.asarray(frozen_candidates)
    if candidate_values.ndim != 2:
        raise ValueError(
            "frozen candidate IDs must be a two-dimensional matrix"
        )
    validate_candidate_matrix(
        positives.dst,
        candidate_values,
        expected_candidate_count=int(candidate_values.shape[1]),
    )
    return TestQueryArray(
        src=np.asarray(positives.src, dtype=np.int32).copy(),
        time=np.asarray(positives.time, dtype=np.int32).copy(),
        candidates=np.asarray(
            candidate_values,
            dtype=np.int32,
        ).copy(),
    )


def validate_frozen_validation_alignment(
    *,
    positives: Any,
    row_indices: np.ndarray,
    candidates: np.ndarray,
    reference_src: np.ndarray,
    reference_dst: np.ndarray,
    reference_time: np.ndarray,
    reference_row_indices: np.ndarray,
    reference_candidates: np.ndarray,
) -> dict[str, bool]:
    """Require sampled validation rows and candidates to equal the frozen contract."""
    actual = {
        "candidates": np.asarray(candidates),
        "src": np.asarray(positives.src),
        "dst": np.asarray(positives.dst),
        "time": np.asarray(positives.time),
        "row_indices": np.asarray(row_indices),
    }
    reference = {
        "candidates": np.asarray(reference_candidates),
        "src": np.asarray(reference_src),
        "dst": np.asarray(reference_dst),
        "time": np.asarray(reference_time),
        "row_indices": np.asarray(reference_row_indices),
    }
    for name in ("candidates", "src", "dst", "time", "row_indices"):
        current = actual[name]
        frozen = reference[name]
        if (
            current.shape != frozen.shape
            or not np.array_equal(current, frozen)
        ):
            raise ValueError(
                f"frozen validation {name} differs from reference"
            )
    validate_candidate_matrix(
        actual["dst"],
        actual["candidates"],
        expected_candidate_count=int(actual["candidates"].shape[1]),
    )
    return dict.fromkeys(actual, True)


def matched_cache_replay_report(
    *,
    expected_candidates: np.ndarray,
    actual_candidates: np.ndarray,
    expected_features: np.ndarray,
    actual_features: np.ndarray,
    rtol: float = 2e-5,
    atol: float = 2e-6,
) -> dict[str, Any]:
    expected_candidate_values = np.asarray(expected_candidates)
    actual_candidate_values = np.asarray(actual_candidates)
    if expected_candidate_values.shape != actual_candidate_values.shape:
        raise ValueError("replay and cached candidate tensors must have the same shape")
    candidate_mismatches = expected_candidate_values != actual_candidate_values
    feature_report = replay_feature_report(
        expected_features,
        actual_features,
        rtol=rtol,
        atol=atol,
    )
    candidate_mismatched_values = int(np.count_nonzero(candidate_mismatches))
    candidate_mismatched_rows = int(
        np.count_nonzero(np.any(candidate_mismatches, axis=1))
    )
    return {
        "matched": bool(
            candidate_mismatched_values == 0 and feature_report["matched"]
        ),
        "candidate_shape": list(expected_candidate_values.shape),
        "candidate_mismatched_rows": candidate_mismatched_rows,
        "candidate_mismatched_values": candidate_mismatched_values,
        "feature_report": feature_report,
    }


def sample_chronological_events(
    events: Any,
    *,
    max_events: int,
    rng: np.random.Generator,
    row_offset: int = 0,
) -> tuple[Any, np.ndarray]:
    """Sample rows without replacement while preserving chronological order."""
    if row_offset < 0:
        raise ValueError("row offset must be non-negative")
    if max_events <= 0 or len(events) <= max_events:
        relative_rows = np.arange(len(events), dtype=np.int64)
    else:
        relative_rows = np.sort(
            rng.choice(len(events), size=int(max_events), replace=False)
        )
    return events.take(relative_rows), relative_rows + int(row_offset)


def resolve_training_context_end(
    *,
    train_end: int,
    configured_context_ratio: float,
    requested_train_rows: int,
) -> int:
    """Keep the configured context unless a larger exact train pool needs less."""
    if train_end <= 1:
        raise ValueError("train end must leave at least one context row")
    if not 0.0 < configured_context_ratio < 1.0:
        raise ValueError("configured context ratio must be between zero and one")
    if requested_train_rows <= 0:
        raise ValueError("requested training rows must be positive")
    latest_context_end = train_end - requested_train_rows
    if latest_context_end < 1:
        raise ValueError(
            f"cannot fit {requested_train_rows} training rows before validation "
            f"with train_end={train_end}"
        )
    configured_context_end = max(
        1,
        min(train_end - 1, int(train_end * configured_context_ratio)),
    )
    return min(configured_context_end, latest_context_end)


def select_recent_events(
    events: Any,
    *,
    requested_rows: int,
) -> tuple[Any, np.ndarray]:
    """Select an exact tail window and return positions within the input pool."""
    if requested_rows <= 0:
        raise ValueError("requested recent event rows must be positive")
    available_rows = len(events)
    if available_rows < requested_rows:
        raise ValueError(
            f"recent event pool has only {available_rows} rows; "
            f"{requested_rows} were requested"
        )
    start = available_rows - requested_rows
    return events[start:], np.arange(start, available_rows, dtype=np.int64)


def validate_full100_cache_arrays(
    *,
    features: np.ndarray,
    candidates: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    time: np.ndarray,
    row_indices: np.ndarray,
    expected_train_rows: int,
    expected_candidate_count: int,
    expected_feature_count: int,
) -> dict[str, int | bool]:
    expected_feature_shape = (
        int(expected_train_rows),
        int(expected_candidate_count),
        int(expected_feature_count),
    )
    if tuple(features.shape) != expected_feature_shape:
        raise ValueError(
            "feature cache shape does not match the frozen train/candidate/feature contract"
        )
    expected_candidate_shape = expected_feature_shape[:2]
    if tuple(candidates.shape) != expected_candidate_shape:
        raise ValueError("candidate cache shape does not align with feature rows")

    expected_sidecar_shape = (int(expected_train_rows),)
    sidecars = {
        "src": src,
        "dst": dst,
        "time": time,
        "row_indices": row_indices,
    }
    mismatched = {
        name: tuple(np.shape(values))
        for name, values in sidecars.items()
        if tuple(np.shape(values)) != expected_sidecar_shape
    }
    if mismatched:
        raise ValueError(f"sidecar arrays do not align with train rows: {mismatched}")

    row_values = np.asarray(row_indices, dtype=np.int64)
    strictly_increasing = bool(
        len(row_values) <= 1 or np.all(row_values[1:] > row_values[:-1])
    )
    if not strictly_increasing:
        raise ValueError("row-index sidecar must be strictly increasing")
    return {
        "train_rows": int(expected_train_rows),
        "candidate_count": int(expected_candidate_count),
        "feature_count": int(expected_feature_count),
        "sidecar_rows": int(expected_train_rows),
        "row_indices_strictly_increasing": strictly_increasing,
    }


def validate_candidate_matrix(
    positive_dst: np.ndarray,
    candidates: np.ndarray,
    *,
    expected_candidate_count: int,
) -> dict[str, int]:
    positive_values = np.asarray(positive_dst)
    candidate_values = np.asarray(candidates)
    if candidate_values.ndim != 2:
        raise ValueError("candidate IDs must be a two-dimensional matrix")
    if candidate_values.dtype.kind not in "iu":
        raise ValueError("candidate IDs must use an integer dtype")
    if candidate_values.shape != (len(positive_values), int(expected_candidate_count)):
        raise ValueError("candidate IDs do not match the expected row count and candidate width")

    positive_mismatches = int(np.count_nonzero(candidate_values[:, 0] != positive_values))
    if positive_mismatches:
        raise ValueError(f"candidate position zero does not match {positive_mismatches} positives")

    ordered = np.sort(candidate_values, axis=1)
    duplicate_rows = int(np.count_nonzero(np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)))
    if duplicate_rows:
        raise ValueError(f"candidate IDs contain duplicate values in {duplicate_rows} rows")
    return {
        "rows": len(positive_values),
        "candidate_count": int(candidate_values.shape[1]),
        "positive_mismatches": positive_mismatches,
        "duplicate_rows": duplicate_rows,
    }


def replay_feature_report(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float = 2e-5,
    atol: float = 2e-6,
) -> dict[str, Any]:
    expected_values = np.asarray(expected, dtype=np.float32)
    actual_values = np.asarray(actual, dtype=np.float32)
    if expected_values.shape != actual_values.shape:
        raise ValueError("replay and cached feature tensors must have the same shape")
    close = np.isclose(expected_values, actual_values, rtol=rtol, atol=atol)
    absolute_error = np.abs(expected_values - actual_values)
    return {
        "shape": list(expected_values.shape),
        "rtol": float(rtol),
        "atol": float(atol),
        "matched": bool(np.all(close)),
        "mismatched_candidate_rows": int(np.count_nonzero(np.any(~close, axis=-1))),
        "mismatched_values": int(np.count_nonzero(~close)),
        "max_abs_error": float(np.max(absolute_error, initial=0.0)),
        "mean_abs_error": float(np.mean(absolute_error)) if absolute_error.size else 0.0,
    }


def passes_full100_gate(
    *,
    baseline_full_mrr: float,
    candidate_full_mrr: float,
    baseline_slice_mrrs: Sequence[float],
    candidate_slice_mrrs: Sequence[float],
    min_full_delta: float,
) -> bool:
    if len(baseline_slice_mrrs) != len(candidate_slice_mrrs):
        raise ValueError("baseline and candidate slices must have equal lengths")
    return bool(
        candidate_full_mrr - baseline_full_mrr + 1e-12 >= min_full_delta
        and all(
            candidate >= baseline
            for baseline, candidate in zip(
                baseline_slice_mrrs,
                candidate_slice_mrrs,
                strict=True,
            )
        )
    )


def passes_matched_control_gate(
    *,
    baseline_full_mrr: float,
    control_full_mrr: float,
    candidate_full_mrr: float,
    baseline_slice_mrrs: Sequence[float],
    candidate_slice_mrrs: Sequence[float],
    min_full_delta: float,
) -> bool:
    return bool(
        passes_full100_gate(
            baseline_full_mrr=baseline_full_mrr,
            candidate_full_mrr=candidate_full_mrr,
            baseline_slice_mrrs=baseline_slice_mrrs,
            candidate_slice_mrrs=candidate_slice_mrrs,
            min_full_delta=min_full_delta,
        )
        and candidate_full_mrr + 1e-12 >= control_full_mrr
    )
