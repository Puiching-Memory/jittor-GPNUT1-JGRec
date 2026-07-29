import numpy as np
import pytest

from jgrec.rankers.hybrid.multi_horizon_oof import (
    HORIZON_NAMES,
    MultiHorizonOOFPiece,
    assemble_multi_horizon_oof,
    audit_multi_horizon_oof,
    canonical_multi_horizon_slices,
)
from jgrec.rankers.hybrid.source_sequence_cache import SourceConditionedFold


def _folds() -> tuple[SourceConditionedFold, ...]:
    return (
        SourceConditionedFold(
            index=0,
            train_rows=(0, 4),
            score_rows=(4, 7),
            role="selection",
            train_time_max=40,
            score_time_min=50,
            score_time_max=70,
        ),
        SourceConditionedFold(
            index=1,
            train_rows=(0, 7),
            score_rows=(7, 10),
            role="selection",
            train_time_max=70,
            score_time_min=80,
            score_time_max=100,
        ),
        SourceConditionedFold(
            index=2,
            train_rows=(0, 10),
            score_rows=(10, 13),
            role="gate",
            train_time_max=100,
            score_time_min=110,
            score_time_max=130,
        ),
    )


def _piece(
    *,
    horizon: str = "short",
    origin_index: int = 0,
    score_rows: tuple[int, int] = (4, 7),
    base_value: float = 1.0,
    correction: float = 0.05,
    train_time_max: int = 40,
    history_time_limit: int = 50,
) -> MultiHorizonOOFPiece:
    rows = score_rows[1] - score_rows[0]
    base = np.full((rows, 3), base_value, dtype=np.float32)
    corrected = base.copy()
    corrected[:, 0] += correction
    corrected[:, 1] -= correction
    return MultiHorizonOOFPiece(
        horizon=horizon,
        origin_index=origin_index,
        score_rows=score_rows,
        train_stop=score_rows[0],
        train_time_max=train_time_max,
        history_time_limit=history_time_limit,
        query_times=np.arange(
            history_time_limit,
            history_time_limit + rows,
            dtype=np.int64,
        ),
        base_logits=base,
        corrected_logits=corrected,
    )


def test_canonical_lattice_is_short_medium_long_without_overlap():
    slices = canonical_multi_horizon_slices(_folds())

    assert HORIZON_NAMES == ("short", "medium", "long")
    assert [
        (row.horizon, row.origin_index, row.score_rows)
        for row in slices
    ] == [
        ("short", 0, (4, 7)),
        ("short", 1, (7, 10)),
        ("short", 2, (10, 13)),
        ("medium", 0, (7, 10)),
        ("medium", 1, (10, 13)),
        ("long", 0, (10, 13)),
    ]
    assert all(row.train_stop <= row.score_rows[0] for row in slices)
    assert all(row.train_time_max < row.score_time_min for row in slices)


def test_assembly_marks_only_covered_rows_and_replays_residual():
    artifact = assemble_multi_horizon_oof(
        [_piece()],
        row_count=13,
        candidate_count=3,
    )

    assert artifact.residuals.shape == (3, 13, 3)
    assert artifact.valid_mask[0, 4:7].all()
    assert not artifact.valid_mask[0, :4].any()
    assert not artifact.valid_mask[1:].any()
    assert np.count_nonzero(artifact.residuals[:, :4]) == 0
    assert np.count_nonzero(artifact.base_logits[~artifact.valid_mask]) == 0
    assert np.count_nonzero(artifact.corrected_logits[~artifact.valid_mask]) == 0
    assert np.all(artifact.origin_index[0, 4:7] == 0)
    assert np.all(artifact.origin_index[~artifact.valid_mask] == -1)
    np.testing.assert_allclose(
        artifact.base_logits + artifact.residuals,
        artifact.corrected_logits,
        atol=1e-7,
    )


def test_assembly_rejects_same_horizon_overlap():
    with pytest.raises(ValueError, match="overlap"):
        assemble_multi_horizon_oof(
            [
                _piece(score_rows=(4, 7)),
                _piece(score_rows=(6, 9)),
            ],
            row_count=13,
            candidate_count=3,
        )


def test_assembly_rejects_non_oof_time_boundary():
    leaking = _piece(
        train_time_max=51,
        history_time_limit=50,
    )

    with pytest.raises(ValueError, match="strictly after"):
        assemble_multi_horizon_oof(
            [leaking],
            row_count=13,
            candidate_count=3,
        )


def test_audit_catches_cap_and_invalid_row_corruption():
    artifact = assemble_multi_horizon_oof(
        [_piece(correction=0.05)],
        row_count=13,
        candidate_count=3,
    )

    passed = audit_multi_horizon_oof(artifact, cap=0.10)
    assert passed["passed"]
    assert passed["coverage_rows"] == {
        "short": 3,
        "medium": 0,
        "long": 0,
    }

    artifact.residuals[0, 4, 0] = 0.2
    artifact.base_logits[1, 0, 0] = 1.0
    failed = audit_multi_horizon_oof(artifact, cap=0.10)
    assert not failed["passed"]
    assert failed["cap_passed"] is False
    assert failed["invalid_zero_passed"] is False

