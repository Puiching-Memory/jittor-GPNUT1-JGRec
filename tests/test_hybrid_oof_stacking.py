from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from jgrec.rankers.hybrid.oof_stacking import (
    StableExpertLogitFeatureView,
    expanding_timestamp_oof_folds,
    oof_row_fold_assignments,
    stable_expert_logit_feature_names,
    stable_expert_logit_features,
    tie_neutral_mrr,
)


def test_expanding_oof_folds_align_boundaries_to_unseen_timestamps() -> None:
    times = np.repeat(np.arange(10, dtype=np.int64), 5)

    folds = expanding_timestamp_oof_folds(
        times,
        warmup_rows=12,
        fold_rows=10,
        fold_count=4,
        meta_train_fold_count=3,
    )

    assert [
        (fold.train_rows, fold.score_rows, fold.role)
        for fold in folds
    ] == [
        ((0, 10), (10, 20), "meta_train"),
        ((0, 20), (20, 30), "meta_train"),
        ((0, 30), (30, 40), "meta_train"),
        ((0, 40), (40, 50), "meta_validation"),
    ]
    scored = np.concatenate(
        [
            np.arange(*fold.score_rows, dtype=np.int64)
            for fold in folds
        ]
    )
    np.testing.assert_array_equal(
        scored,
        np.arange(10, 50, dtype=np.int64),
    )
    assert all(
        times[fold.train_rows[1] - 1] < times[fold.score_rows[0]]
        for fold in folds
    )


def test_stable_logit_features_are_affine_invariant_and_permutation_equivariant() -> None:
    logits = np.asarray(
        [
            [[2.0, 2.0, 0.0, 1.0], [3.0, -1.0, 0.0, 2.0]],
            [[-2.0, 4.0, 1.0, 1.0], [0.0, 2.0, 5.0, -3.0]],
        ],
        dtype=np.float32,
    )
    permutation = np.asarray([2, 0, 3, 1])

    features = stable_expert_logit_features(logits)
    affine = stable_expert_logit_features(logits * 3.5 + 17.0)
    permuted = stable_expert_logit_features(
        logits[:, :, permutation]
    )

    assert features.shape == (2, 4, 17)
    np.testing.assert_allclose(affine, features, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(
        permuted,
        features[:, permutation],
        rtol=0.0,
        atol=2e-6,
    )

    names = stable_expert_logit_feature_names(("left", "right"))
    left_rank = names.index("left__percentile_rank")
    left_support = names.index("left__top1_support")
    assert features[0, 0, left_rank] == features[0, 1, left_rank]
    assert features[0, 0, left_support] == 0.5
    assert features[0, 1, left_support] == 0.5


def test_oof_assignments_reject_overlap_and_missing_score_rows() -> None:
    times = np.arange(10, dtype=np.int64)
    folds = expanding_timestamp_oof_folds(
        times,
        warmup_rows=2,
        fold_rows=2,
        fold_count=4,
        meta_train_fold_count=3,
    )

    assignments = oof_row_fold_assignments(10, folds)

    np.testing.assert_array_equal(
        assignments,
        np.asarray([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int16),
    )
    overlapping = (
        folds[0],
        replace(folds[1], score_rows=(3, 6)),
        *folds[2:],
    )
    with pytest.raises(ValueError, match="overlap"):
        oof_row_fold_assignments(10, overlapping)
    missing = (
        folds[0],
        replace(folds[1], score_rows=(5, 6)),
        *folds[2:],
    )
    with pytest.raises(ValueError, match="gap"):
        oof_row_fold_assignments(10, missing)


def test_stable_logit_feature_view_matches_eager_transform() -> None:
    logits = np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4)
    logits[:, 3, 1] = logits[:, 3, 2]
    view = StableExpertLogitFeatureView(logits, row_start=1, row_stop=5)

    assert view.shape == (4, 4, 17)
    np.testing.assert_allclose(
        view[[2, 0]],
        stable_expert_logit_features(logits[:, [3, 1], :]),
    )
    np.testing.assert_allclose(
        view[1:3],
        stable_expert_logit_features(logits[:, 2:4, :]),
    )
    np.testing.assert_allclose(
        view[2],
        stable_expert_logit_features(logits[:, 3:4, :])[0],
    )


def test_stable_logit_features_treat_cuda_scale_noise_as_ties() -> None:
    logits = np.asarray(
        [
            [[1.0, 1.0, 0.0, -1.0]],
            [[0.5, 0.5, 0.2, -0.3]],
        ],
        dtype=np.float32,
    )
    noisy = logits.copy()
    noisy[0, 0, 0] += np.float32(3e-6)
    noisy[0, 0, 1] -= np.float32(2e-6)
    noisy[1, 0, 0] -= np.float32(2e-6)
    noisy[1, 0, 1] += np.float32(3e-6)

    clean_features = stable_expert_logit_features(logits)
    noisy_features = stable_expert_logit_features(noisy)

    np.testing.assert_allclose(
        noisy_features,
        clean_features,
        rtol=0.0,
        atol=1.1e-3,
    )
    names = stable_expert_logit_feature_names(("left", "right"))
    stable_indices = [
        index
        for index, name in enumerate(names)
        if name.endswith(("percentile_rank", "top1_support"))
    ]
    np.testing.assert_array_equal(
        noisy_features[..., stable_indices],
        clean_features[..., stable_indices],
    )


def test_tie_neutral_mrr_does_not_reward_positive_candidate_position() -> None:
    scores = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    first = tie_neutral_mrr(
        scores,
        np.asarray([0, 0], dtype=np.int32),
    )
    second = tie_neutral_mrr(
        scores,
        np.asarray([1, 1], dtype=np.int32),
    )

    assert first == pytest.approx(2.0 / 3.0)
    assert second == pytest.approx(first)
