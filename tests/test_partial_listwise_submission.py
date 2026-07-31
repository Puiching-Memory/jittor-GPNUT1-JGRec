from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from jgrec.partial_listwise_submission import (
    build_partial_listwise_submission,
    materialize_submission_member_scores,
    ranking_metric_panel,
)


def test_ranking_metric_panel_reports_top_k_and_rank_distribution() -> None:
    scores = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.2, 0.8, 0.1, 0.0],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=np.float64,
    )

    metrics = ranking_metric_panel(scores)

    assert metrics["mrr"] == pytest.approx((1.0 + 0.5 + 0.25) / 3.0)
    assert metrics["hit_at_1"] == pytest.approx(1.0 / 3.0)
    assert metrics["hit_at_3"] == pytest.approx(2.0 / 3.0)
    assert metrics["hit_at_5"] == pytest.approx(1.0)
    assert metrics["hit_at_10"] == pytest.approx(1.0)
    assert metrics["mean_rank"] == pytest.approx(7.0 / 3.0)
    assert metrics["median_rank"] == pytest.approx(2.0)


def test_build_partial_listwise_submission_preserves_dataset1_and_formula(
    tmp_path: Path,
) -> None:
    dataset1 = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
        dtype=np.float64,
    )
    champion2 = np.array(
        [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
        dtype=np.float64,
    )
    expert2 = np.array(
        [[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]],
        dtype=np.float64,
    )
    dataset1_path = tmp_path / "dataset1.csv"
    dataset2_path = tmp_path / "dataset2.csv"
    expert_path = tmp_path / "expert.npy"
    np.savetxt(dataset1_path, dataset1, delimiter=",", fmt="%.8f")
    np.savetxt(dataset2_path, champion2, delimiter=",", fmt="%.8f")
    np.save(expert_path, expert2, allow_pickle=False)
    champion_zip = tmp_path / "champion.zip"
    with zipfile.ZipFile(champion_zip, "w") as archive:
        archive.write(dataset1_path, arcname="dataset1.csv")
        archive.write(dataset2_path, arcname="dataset2.csv")

    output_dir = tmp_path / "candidate"
    report = build_partial_listwise_submission(
        champion_zip=champion_zip,
        expert_scores_path=expert_path,
        output_dir=output_dir,
        auxiliary_weight=0.20,
        expected_rows={"dataset1": 2, "dataset2": 2},
        expected_columns=3,
        expected_champion_zip_sha256=_sha256(champion_zip),
        expected_dataset1_sha256=_sha256(dataset1_path),
        expected_dataset2_sha256=_sha256(dataset2_path),
        expert_name="listwise_two_tower",
        expert_model_sha256="model-sha",
        candidate_manifest_sha256="manifest-sha",
        selection_lock_sha256="lock-sha",
        dataset2_mode="cooccur_lift_aux_expert_blend",
    )

    assert report["status"] == "online_candidate"
    assert report["submission_authorized_by_user"] is True
    assert report["promotion_authorized"] is False
    expected = 0.8 * champion2 + 0.2 * expert2
    actual = np.loadtxt(
        output_dir / "csv" / "dataset2.csv",
        delimiter=",",
        ndmin=2,
    )
    np.testing.assert_allclose(actual, expected, atol=5e-9, rtol=0.0)
    with zipfile.ZipFile(output_dir / "result.zip") as archive:
        assert archive.namelist() == ["dataset1.csv", "dataset2.csv"]
        assert archive.read("dataset1.csv") == dataset1_path.read_bytes()
    assert report["dataset1"]["sha256"] == _sha256(dataset1_path)
    assert report["dataset2"]["formula"] == (
        "candidate = 0.80 * champion + 0.20 * expert"
    )
    assert report["dataset2"]["mode"] == (
        "cooccur_lift_aux_expert_blend"
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_partial_listwise_submission(
            champion_zip=champion_zip,
            expert_scores_path=expert_path,
            output_dir=output_dir,
            auxiliary_weight=0.20,
            expected_rows={"dataset1": 2, "dataset2": 2},
            expected_columns=3,
            expected_champion_zip_sha256=_sha256(champion_zip),
            expected_dataset1_sha256=_sha256(dataset1_path),
            expected_dataset2_sha256=_sha256(dataset2_path),
            expert_name="listwise_two_tower",
            expert_model_sha256="model-sha",
            candidate_manifest_sha256="manifest-sha",
            selection_lock_sha256="lock-sha",
        )


def test_build_partial_listwise_submission_rejects_non_normalized_expert(
    tmp_path: Path,
) -> None:
    dataset = np.array([[0.7, 0.2, 0.1]], dtype=np.float64)
    csv_path = tmp_path / "dataset.csv"
    np.savetxt(csv_path, dataset, delimiter=",", fmt="%.8f")
    champion_zip = tmp_path / "champion.zip"
    with zipfile.ZipFile(champion_zip, "w") as archive:
        archive.write(csv_path, arcname="dataset1.csv")
        archive.write(csv_path, arcname="dataset2.csv")
    expert_path = tmp_path / "expert.npy"
    np.save(
        expert_path,
        np.array([[0.2, 0.2, 0.2]], dtype=np.float64),
        allow_pickle=False,
    )

    with pytest.raises(ValueError, match="row-normalized"):
        build_partial_listwise_submission(
            champion_zip=champion_zip,
            expert_scores_path=expert_path,
            output_dir=tmp_path / "candidate",
            auxiliary_weight=0.20,
            expected_rows={"dataset1": 1, "dataset2": 1},
            expected_columns=3,
            expected_champion_zip_sha256=_sha256(champion_zip),
            expected_dataset1_sha256=_sha256(csv_path),
            expected_dataset2_sha256=_sha256(csv_path),
            expert_name="listwise_two_tower",
            expert_model_sha256="model-sha",
            candidate_manifest_sha256="manifest-sha",
            selection_lock_sha256="lock-sha",
        )


def test_materialize_submission_member_scores_verifies_source_and_shape(
    tmp_path: Path,
) -> None:
    scores = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
        dtype=np.float64,
    )
    csv_path = tmp_path / "dataset2.csv"
    np.savetxt(csv_path, scores, delimiter=",", fmt="%.8f")
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.write(csv_path, arcname="dataset2.csv")
    output_path = tmp_path / "expert.npy"

    report = materialize_submission_member_scores(
        source_zip=source_zip,
        member_name="dataset2.csv",
        output_path=output_path,
        expected_zip_sha256=_sha256(source_zip),
        expected_member_sha256=_sha256(csv_path),
        expected_shape=(2, 3),
    )

    np.testing.assert_array_equal(np.load(output_path), scores)
    assert report["status"] == "passed"
    assert report["shape"] == [2, 3]
    assert report["source_member_sha256"] == _sha256(csv_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_submission_member_scores(
            source_zip=source_zip,
            member_name="dataset2.csv",
            output_path=output_path,
            expected_zip_sha256=_sha256(source_zip),
            expected_member_sha256=_sha256(csv_path),
            expected_shape=(2, 3),
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
