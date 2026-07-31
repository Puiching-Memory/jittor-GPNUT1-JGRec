import numpy as np

from jgrec.core.types import InteractionTable
from jgrec.rankers.hybrid.full100_training import (
    build_frozen_candidate_queries,
    matched_cache_replay_report,
    passes_full100_gate,
    passes_matched_control_gate,
    replay_feature_report,
    resolve_training_context_end,
    sample_chronological_events,
    select_recent_events,
    validate_candidate_matrix,
    validate_frozen_validation_alignment,
    validate_full100_cache_arrays,
    validate_joint_cache_reports,
)


def test_frozen_candidate_queries_preserve_exact_matrix_without_rng():
    positives = InteractionTable(
        src=np.asarray([7, 8], dtype=np.int32),
        dst=np.asarray([70, 80], dtype=np.int32),
        time=np.asarray([700, 800], dtype=np.int32),
    )
    frozen = np.asarray(
        [[70, 71, 72], [80, 81, 82]],
        dtype=np.int32,
    )

    queries = build_frozen_candidate_queries(positives, frozen)

    np.testing.assert_array_equal(queries.src, positives.src)
    np.testing.assert_array_equal(queries.time, positives.time)
    np.testing.assert_array_equal(queries.candidates, frozen)
    assert not np.shares_memory(queries.candidates, frozen)

    mismatched = frozen.copy()
    mismatched[1, 0] = 999
    with np.testing.assert_raises_regex(ValueError, "position zero"):
        build_frozen_candidate_queries(positives, mismatched)


def test_frozen_validation_alignment_requires_every_sidecar_exact():
    positives = InteractionTable(
        src=np.asarray([7, 8], dtype=np.int32),
        dst=np.asarray([70, 80], dtype=np.int32),
        time=np.asarray([700, 800], dtype=np.int32),
    )
    row_indices = np.asarray([1007, 1008], dtype=np.int64)
    candidates = np.asarray(
        [[70, 71, 72], [80, 81, 82]],
        dtype=np.int32,
    )

    assert validate_frozen_validation_alignment(
        positives=positives,
        row_indices=row_indices,
        candidates=candidates,
        reference_src=positives.src.copy(),
        reference_dst=positives.dst.copy(),
        reference_time=positives.time.copy(),
        reference_row_indices=row_indices.copy(),
        reference_candidates=candidates.copy(),
    ) == {
        "candidates": True,
        "src": True,
        "dst": True,
        "time": True,
        "row_indices": True,
    }

    mismatched_rows = row_indices.copy()
    mismatched_rows[1] += 1
    with np.testing.assert_raises_regex(
        ValueError,
        "row_indices differs",
    ):
        validate_frozen_validation_alignment(
            positives=positives,
            row_indices=row_indices,
            candidates=candidates,
            reference_src=positives.src,
            reference_dst=positives.dst,
            reference_time=positives.time,
            reference_row_indices=mismatched_rows,
            reference_candidates=candidates,
        )


def test_training_context_backs_off_only_enough_to_fit_requested_recent_rows():
    assert resolve_training_context_end(
        train_end=1_000_000,
        configured_context_ratio=0.75,
        requested_train_rows=200_000,
    ) == 750_000
    assert resolve_training_context_end(
        train_end=587_221,
        configured_context_ratio=0.75,
        requested_train_rows=200_000,
    ) == 387_221

    with np.testing.assert_raises_regex(ValueError, "before validation"):
        resolve_training_context_end(
            train_end=200_000,
            configured_context_ratio=0.75,
            requested_train_rows=200_000,
        )


def test_validate_candidate_matrix_requires_positive_first_and_unique_candidates():
    positives = np.asarray([10, 20], dtype=np.int32)
    valid = np.asarray([[10, 11, 12], [20, 21, 22]], dtype=np.int32)

    report = validate_candidate_matrix(positives, valid, expected_candidate_count=3)

    assert report == {
        "rows": 2,
        "candidate_count": 3,
        "positive_mismatches": 0,
        "duplicate_rows": 0,
    }

    invalid = valid.copy()
    invalid[1] = np.asarray([20, 21, 20], dtype=np.int32)
    with np.testing.assert_raises_regex(ValueError, "duplicate"):
        validate_candidate_matrix(positives, invalid, expected_candidate_count=3)


def test_replay_feature_report_counts_candidate_rows_and_enforces_shape():
    expected = np.zeros((2, 3, 2), dtype=np.float32)
    actual = expected.copy()
    actual[1, 2, 1] = 1e-3

    report = replay_feature_report(expected, actual, rtol=1e-5, atol=1e-6)

    assert report["matched"] is False
    assert report["mismatched_candidate_rows"] == 1
    assert report["mismatched_values"] == 1
    assert report["max_abs_error"] == float(actual[1, 2, 1])

    with np.testing.assert_raises_regex(ValueError, "same shape"):
        replay_feature_report(expected, actual[:, :2])


def test_full100_gate_requires_declared_gain_and_every_slice_non_decreasing():
    assert passes_full100_gate(
        baseline_full_mrr=0.50,
        candidate_full_mrr=0.502,
        baseline_slice_mrrs=(0.48, 0.50, 0.52),
        candidate_slice_mrrs=(0.48, 0.501, 0.525),
        min_full_delta=0.002,
    )
    assert not passes_full100_gate(
        baseline_full_mrr=0.50,
        candidate_full_mrr=0.5019,
        baseline_slice_mrrs=(0.48, 0.50, 0.52),
        candidate_slice_mrrs=(0.49, 0.51, 0.53),
        min_full_delta=0.002,
    )
    assert not passes_full100_gate(
        baseline_full_mrr=0.50,
        candidate_full_mrr=0.503,
        baseline_slice_mrrs=(0.48, 0.50, 0.52),
        candidate_slice_mrrs=(0.4799, 0.51, 0.53),
        min_full_delta=0.002,
    )

    with np.testing.assert_raises_regex(ValueError, "equal lengths"):
        passes_full100_gate(
            baseline_full_mrr=0.50,
            candidate_full_mrr=0.503,
            baseline_slice_mrrs=(0.48, 0.50),
            candidate_slice_mrrs=(0.49,),
            min_full_delta=0.002,
        )


def test_matched_control_gate_also_requires_full100_to_beat_control():
    common = {
        "baseline_full_mrr": 0.50,
        "baseline_slice_mrrs": (0.48, 0.50, 0.52),
        "candidate_slice_mrrs": (0.481, 0.503, 0.524),
        "min_full_delta": 0.002,
    }
    assert passes_matched_control_gate(
        **common,
        candidate_full_mrr=0.503,
        control_full_mrr=0.5029,
    )
    assert not passes_matched_control_gate(
        **common,
        candidate_full_mrr=0.503,
        control_full_mrr=0.5031,
    )
    assert not passes_matched_control_gate(
        **common,
        candidate_full_mrr=0.501,
        control_full_mrr=0.5005,
    )


def test_select_recent_events_returns_exact_tail_and_rejects_short_pool():
    events = np.arange(12, dtype=np.int64)

    selected, row_indices = select_recent_events(events, requested_rows=5)

    np.testing.assert_array_equal(selected, np.asarray([7, 8, 9, 10, 11]))
    np.testing.assert_array_equal(row_indices, np.asarray([7, 8, 9, 10, 11]))

    with np.testing.assert_raises_regex(ValueError, "only 12"):
        select_recent_events(events, requested_rows=13)
    with np.testing.assert_raises_regex(ValueError, "positive"):
        select_recent_events(events, requested_rows=0)


def test_validate_full100_cache_arrays_requires_all_rows_and_sidecars_to_align():
    features = np.zeros((4, 100, 63), dtype=np.float32)
    candidates = np.tile(np.arange(100, dtype=np.int32), (4, 1))
    src = np.arange(4, dtype=np.int32)
    dst = np.zeros(4, dtype=np.int32)
    time = np.arange(4, dtype=np.int64)
    row_indices = np.arange(20, 24, dtype=np.int64)

    report = validate_full100_cache_arrays(
        features=features,
        candidates=candidates,
        src=src,
        dst=dst,
        time=time,
        row_indices=row_indices,
        expected_train_rows=4,
        expected_candidate_count=100,
        expected_feature_count=63,
    )

    assert report == {
        "train_rows": 4,
        "candidate_count": 100,
        "feature_count": 63,
        "sidecar_rows": 4,
        "row_indices_strictly_increasing": True,
    }

    with np.testing.assert_raises_regex(ValueError, "sidecar"):
        validate_full100_cache_arrays(
            features=features,
            candidates=candidates,
            src=src[:-1],
            dst=dst,
            time=time,
            row_indices=row_indices,
            expected_train_rows=4,
            expected_candidate_count=100,
            expected_feature_count=63,
        )


def test_matched_cache_replay_requires_exact_candidates_and_close_features():
    candidates = np.asarray([[10, 11, 12], [20, 21, 22]], dtype=np.int32)
    features = np.arange(12, dtype=np.float32).reshape(2, 3, 2)

    matched = matched_cache_replay_report(
        expected_candidates=candidates,
        actual_candidates=candidates.copy(),
        expected_features=features,
        actual_features=features.copy(),
    )

    assert matched["matched"] is True
    assert matched["candidate_mismatched_rows"] == 0
    assert matched["candidate_mismatched_values"] == 0
    assert matched["feature_report"]["matched"] is True

    changed_candidates = candidates.copy()
    changed_candidates[1, 2] = 99
    mismatched = matched_cache_replay_report(
        expected_candidates=candidates,
        actual_candidates=changed_candidates,
        expected_features=features,
        actual_features=features.copy(),
    )
    assert mismatched["matched"] is False
    assert mismatched["candidate_mismatched_rows"] == 1
    assert mismatched["candidate_mismatched_values"] == 1


def test_sample_chronological_events_preserves_sorted_global_row_identity():
    events = np.arange(20, dtype=np.int64)
    selected, global_rows = sample_chronological_events(
        events,
        max_events=6,
        rng=np.random.default_rng(60),
        row_offset=100,
    )

    assert len(selected) == 6
    assert np.all(global_rows[1:] > global_rows[:-1])
    np.testing.assert_array_equal(selected, global_rows - 100)

    all_events, all_rows = sample_chronological_events(
        events,
        max_events=0,
        rng=np.random.default_rng(60),
        row_offset=100,
    )
    np.testing.assert_array_equal(all_events, events)
    np.testing.assert_array_equal(all_rows, np.arange(100, 120))


def test_joint_cache_reports_require_one_process_and_exact_train_hash_binding():
    train_report = {
        "status": "complete",
        "joint_build": {
            "id": "joint-build-123",
            "pid": 4321,
            "role": "train",
        },
        "artifacts": {
            "features": {
                "sha256": "train-feature-sha",
            },
        },
    }
    validation_report = {
        "status": "complete",
        "joint_build": {
            "id": "joint-build-123",
            "pid": 4321,
            "role": "validation",
        },
        "train_feature_sha256": "train-feature-sha",
    }

    assert validate_joint_cache_reports(train_report, validation_report) == {
        "matched": True,
        "joint_build_id": "joint-build-123",
        "process_id": 4321,
        "train_feature_sha256": "train-feature-sha",
    }

    mismatched_process = {
        **validation_report,
        "joint_build": {
            **validation_report["joint_build"],
            "pid": 9876,
        },
    }
    with np.testing.assert_raises_regex(ValueError, "same process"):
        validate_joint_cache_reports(train_report, mismatched_process)

    mismatched_hash = {
        **validation_report,
        "train_feature_sha256": "different-sha",
    }
    with np.testing.assert_raises_regex(ValueError, "feature hash"):
        validate_joint_cache_reports(train_report, mismatched_hash)


def test_joint_cache_reports_accept_strict_frozen_query_recovery():
    train_report = {
        "status": "complete",
        "joint_build": {
            "id": "joint-build-123",
            "pid": 4321,
            "role": "train",
        },
        "artifacts": {
            "features": {
                "sha256": "train-feature-sha",
            },
        },
    }
    validation_report = {
        "status": "complete",
        "joint_build": {
            "id": "joint-build-123",
            "pid": 9876,
            "role": "validation",
        },
        "train_feature_sha256": "train-feature-sha",
        "recovery": {
            "protocol": "frozen_candidate_validation_recovery_v1",
            "recovery_process_id": 9876,
            "train_joint_build_id": "joint-build-123",
            "train_joint_process_id": 4321,
            "train_feature_sha256": "train-feature-sha",
            "candidate_sampling_performed": False,
            "frozen_query_alignment_exact": True,
            "reference_sidecars_exact": {
                "candidates": True,
                "src": True,
                "dst": True,
                "time": True,
                "row_indices": True,
            },
        },
    }

    assert validate_joint_cache_reports(
        train_report,
        validation_report,
    ) == {
        "matched": True,
        "joint_build_id": "joint-build-123",
        "process_id": 4321,
        "train_feature_sha256": "train-feature-sha",
        "validation_recovery": True,
        "validation_process_id": 9876,
    }

    unsafe = {
        **validation_report,
        "recovery": {
            **validation_report["recovery"],
            "candidate_sampling_performed": True,
        },
    }
    with np.testing.assert_raises_regex(
        ValueError,
        "candidate sampling",
    ):
        validate_joint_cache_reports(train_report, unsafe)


def test_joint_cache_reports_reject_different_dataset_names():
    train_report = {
        "status": "complete",
        "dataset_name": "dataset1",
        "joint_build": {
            "id": "joint-build-123",
            "pid": 4321,
            "role": "train",
        },
        "artifacts": {
            "features": {
                "sha256": "train-feature-sha",
            },
        },
    }
    validation_report = {
        "status": "complete",
        "dataset_name": "dataset2",
        "joint_build": {
            "id": "joint-build-123",
            "pid": 4321,
            "role": "validation",
        },
        "train_feature_sha256": "train-feature-sha",
    }

    with np.testing.assert_raises_regex(ValueError, "dataset"):
        validate_joint_cache_reports(train_report, validation_report)
