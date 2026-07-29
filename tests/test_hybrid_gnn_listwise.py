from __future__ import annotations

import jittor as jt
import numpy as np
import pytest

from jgrec.rankers.hybrid.gnn_listwise import (
    expanding_oof_folds,
    full_candidate_mrr,
    listwise_positive_loss,
    replace_feature_column,
    validate_candidate_groups,
)


def test_gnn_listwise_positive_loss_matches_group_softmax_reference():
    logits = np.asarray(
        [[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    reference = float(
        np.mean(
            -shifted[:, 0]
            + np.log(np.exp(shifted).sum(axis=1))
        )
    )

    actual = float(
        listwise_positive_loss(
            jt.array(logits, dtype=jt.float32)
        ).item()
    )

    assert actual == pytest.approx(reference, rel=1e-6)


def test_gnn_listwise_contract_requires_100_candidates_and_positive_at_zero():
    src = np.asarray([10, 20], dtype=np.int32)
    dst = np.asarray([30, 40], dtype=np.int32)
    candidates = np.tile(np.arange(100, dtype=np.int32), (2, 1))
    candidates[:, 0] = dst

    validate_candidate_groups(src, dst, candidates, width=100)

    wrong_width = candidates[:, :99]
    with pytest.raises(ValueError, match="100 candidates"):
        validate_candidate_groups(src, dst, wrong_width, width=100)

    wrong_positive = candidates.copy()
    wrong_positive[1, 0] = 99
    with pytest.raises(ValueError, match="column 0"):
        validate_candidate_groups(src, dst, wrong_positive, width=100)


def test_gnn_full_candidate_mrr_uses_positive_column_zero():
    scores = np.asarray(
        [[3.0, 2.0, 1.0], [0.0, 2.0, 1.0]],
        dtype=np.float32,
    )

    assert full_candidate_mrr(scores) == pytest.approx(2.0 / 3.0)


def test_replace_gnn_feature_column_preserves_every_other_value(tmp_path):
    source = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    replacement = np.asarray(
        [[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]],
        dtype=np.float32,
    )
    output_path = tmp_path / "validation-with-new-gnn-short.npy"

    report = replace_feature_column(
        source,
        replacement,
        column=2,
        output_path=output_path,
        batch_rows=1,
    )

    actual = np.load(output_path, allow_pickle=False)
    assert np.array_equal(actual[..., 2], replacement)
    assert np.array_equal(actual[..., :2], source[..., :2])
    assert np.array_equal(actual[..., 3:], source[..., 3:])
    assert np.array_equal(source, np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    assert report == {
        "shape": [2, 3, 4],
        "replaced_column": 2,
        "unchanged_columns_equal": True,
    }


def test_expanding_oof_folds_cover_post_burn_in_rows_without_future_leakage():
    folds = expanding_oof_folds(
        row_count=200_000,
        burn_in=25_000,
        fold_size=25_000,
    )

    assert len(folds) == 7
    assert folds[0].train_rows == (0, 25_000)
    assert folds[0].score_rows == (25_000, 50_000)
    assert folds[-1].train_rows == (0, 175_000)
    assert folds[-1].score_rows == (175_000, 200_000)
    assert all(
        fold.train_rows[1] <= fold.score_rows[0]
        for fold in folds
    )
    scored_rows = np.concatenate(
        [
            np.arange(*fold.score_rows, dtype=np.int32)
            for fold in folds
        ]
    )
    assert np.array_equal(
        scored_rows,
        np.arange(25_000, 200_000, dtype=np.int32),
    )
