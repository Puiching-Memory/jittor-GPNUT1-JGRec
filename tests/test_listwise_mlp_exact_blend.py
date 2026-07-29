from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jgrec.listwise_mlp_exact_blend import (
    A2_ROLLING_FOLDS,
    A2_WEIGHTS,
    build_rolling_selection_manifest,
    materialize_fold_candidates,
    validate_frozen_a2_weights,
    validate_rolling_folds,
)


def test_a2_fold_protocol_is_three_monotonic_expanding_origins() -> None:
    folds = validate_rolling_folds(
        A2_ROLLING_FOLDS,
        row_count=200_000,
    )

    assert [fold.fold_id for fold in folds] == [
        "fold-0",
        "fold-1",
        "fold-2",
    ]
    assert [fold.train_rows for fold in folds] == [
        (0, 79_909),
        (0, 118_816),
        (0, 159_804),
    ]
    assert [fold.score_rows for fold in folds] == [
        (79_909, 118_816),
        (118_816, 159_804),
        (159_804, 200_000),
    ]


def test_a2_weights_must_match_config_frozen_before_online_result() -> None:
    frozen = {
        "status": "frozen_before_any_partial_blend_metric",
        "weights": list(A2_WEIGHTS),
    }

    assert validate_frozen_a2_weights(frozen) == A2_WEIGHTS

    frozen["weights"] = [*A2_WEIGHTS, 0.25]
    with pytest.raises(ValueError, match="predeclared A2 weights"):
        validate_frozen_a2_weights(frozen)


def test_materialized_fold_candidates_use_exact_final_blend_formula(
    tmp_path: Path,
) -> None:
    baseline = np.asarray(
        [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
        dtype=np.float64,
    )
    auxiliary = np.asarray(
        [[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]],
        dtype=np.float64,
    )
    candidates = np.asarray(
        [[11, 12, 13], [21, 22, 23]],
        dtype=np.int32,
    )

    entry = materialize_fold_candidates(
        output_dir=tmp_path / "fold-0",
        fold_id="fold-0",
        integration_id="listwise_mlp_exact_current_champion_v1",
        train_time_max=100,
        score_time_min=101,
        score_time_max=200,
        baseline_scores=baseline,
        auxiliary_scores=auxiliary,
        candidate_ids=candidates,
        weights=(0.1, 0.3),
    )

    assert entry["candidate_fingerprint"] == hashlib.sha256(candidates.tobytes(order="C")).hexdigest()
    assert entry["baseline"]["sha256"] == _sha256(tmp_path / "fold-0" / "baseline.npy")
    actual = np.load(tmp_path / "fold-0" / "candidate-w0.3.npy")
    np.testing.assert_allclose(
        actual,
        0.7 * baseline + 0.3 * auxiliary,
        atol=0.0,
        rtol=0.0,
    )
    descriptor = entry["candidates"]["0.3"]
    assert descriptor["integration_id"] == ("listwise_mlp_exact_current_champion_v1")
    assert descriptor["candidate_fingerprint"] == entry["candidate_fingerprint"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_fold_candidates(
            output_dir=tmp_path / "fold-0",
            fold_id="fold-0",
            integration_id="listwise_mlp_exact_current_champion_v1",
            train_time_max=100,
            score_time_min=101,
            score_time_max=200,
            baseline_scores=baseline,
            auxiliary_scores=auxiliary,
            candidate_ids=candidates,
            weights=(0.1, 0.3),
        )


def test_selection_manifest_preserves_fold_artifact_identity(
    tmp_path: Path,
) -> None:
    fold_entries = []
    for index in range(3):
        baseline = np.asarray(
            [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
            dtype=np.float64,
        )
        auxiliary = np.asarray(
            [[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]],
            dtype=np.float64,
        )
        fold_entries.append(
            materialize_fold_candidates(
                output_dir=tmp_path / f"fold-{index}",
                fold_id=f"fold-{index}",
                integration_id=("listwise_mlp_exact_current_champion_v1"),
                train_time_max=100 + index * 100,
                score_time_min=101 + index * 100,
                score_time_max=200 + index * 100,
                baseline_scores=baseline,
                auxiliary_scores=auxiliary,
                candidate_ids=np.asarray(
                    [[11, 12, 13], [21, 22, 23]],
                    dtype=np.int32,
                ),
                weights=(0.1, 0.3),
            )
        )
    output = tmp_path / "rolling-manifest.json"

    manifest = build_rolling_selection_manifest(
        integration_id="listwise_mlp_exact_current_champion_v1",
        fold_entries=fold_entries,
        output_path=output,
    )

    assert manifest["protocol"] == ("exact_integrated_rolling_weight_selection_v1")
    assert len(manifest["folds"]) == 3
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert manifest["folds"][2]["candidates"]["0.3"]["sha256"] == (fold_entries[2]["candidates"]["0.3"]["sha256"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
