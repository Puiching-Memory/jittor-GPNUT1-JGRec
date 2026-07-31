from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jgrec.rankers.hybrid.base_context_head import (
    BaseContextHeadArtifact,
    load_base_context_head,
    save_base_context_head,
)


def test_base_context_head_round_trips_exact_package_contract(
    tmp_path: Path,
) -> None:
    artifact = BaseContextHeadArtifact(
        context_transform_version=1,
        hidden_dim=4,
        feature_indices=(0, 2),
        mean=np.arange(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
        state={
            "linear1.weight": np.arange(24, dtype=np.float32).reshape(4, 6),
            "linear1.bias": np.zeros(4, dtype=np.float32),
        },
        best_val_ap=0.25,
        best_val_mrr=0.5,
        candidate_name="external_base_context_v1",
        fit_rows=(10, 30),
        tune_rows=(30, 40),
    )
    path = tmp_path / "head.npz"

    save_base_context_head(path, artifact)
    restored = load_base_context_head(
        path,
        expected_context_transform_version=1,
    )

    assert restored.context_transform_version == 1
    assert restored.hidden_dim == 4
    assert restored.feature_indices == (0, 2)
    assert restored.candidate_name == "external_base_context_v1"
    assert restored.fit_rows == (10, 30)
    assert restored.tune_rows == (30, 40)
    np.testing.assert_array_equal(restored.mean, artifact.mean)
    np.testing.assert_array_equal(
        restored.state["linear1.weight"],
        artifact.state["linear1.weight"],
    )


def test_base_context_head_rejects_wrong_width_or_context_version(
    tmp_path: Path,
) -> None:
    artifact = BaseContextHeadArtifact(
        context_transform_version=1,
        hidden_dim=4,
        feature_indices=(0, 2),
        mean=np.zeros(5, dtype=np.float32),
        std=np.ones(5, dtype=np.float32),
        state={
            "linear1.weight": np.zeros((4, 5), dtype=np.float32),
        },
        best_val_ap=0.25,
        best_val_mrr=0.5,
        candidate_name="bad",
        fit_rows=(10, 30),
        tune_rows=(30, 40),
    )

    with pytest.raises(ValueError, match="three context channels"):
        save_base_context_head(tmp_path / "bad-width.npz", artifact)

    valid = BaseContextHeadArtifact(
        **{
            **artifact.__dict__,
            "mean": np.zeros(6, dtype=np.float32),
            "std": np.ones(6, dtype=np.float32),
            "state": {
                "linear1.weight": np.zeros((4, 6), dtype=np.float32),
            },
        }
    )
    path = tmp_path / "valid.npz"
    save_base_context_head(path, valid)
    with pytest.raises(ValueError, match="context transform version"):
        load_base_context_head(
            path,
            expected_context_transform_version=0,
        )
