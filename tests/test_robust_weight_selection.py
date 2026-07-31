from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jgrec.robust_weight_selection import (
    evaluate_locked_external,
    ranking_metrics,
    select_rolling_origin_weight,
)


def test_ranking_metrics_reports_requested_panel_and_query_movements() -> None:
    baseline = _scores_for_ranks([2, 2, 4, 11], candidate_count=12)
    candidate = _scores_for_ranks([1, 3, 4, 10], candidate_count=12)

    panel = ranking_metrics(candidate, baseline_scores=baseline)

    assert panel["mrr"] == pytest.approx((1.0 + 1.0 / 3.0 + 1.0 / 4.0 + 1.0 / 10.0) / 4.0)
    assert panel["hit_at_1"] == pytest.approx(0.25)
    assert panel["hit_at_3"] == pytest.approx(0.5)
    assert panel["hit_at_10"] == pytest.approx(1.0)
    assert panel["ndcg_at_10"] == pytest.approx(
        (1.0 + 1.0 / np.log2(4.0) + 1.0 / np.log2(5.0) + 1.0 / np.log2(11.0)) / 4.0
    )
    assert panel["mean_rank"] == pytest.approx(4.5)
    assert panel["query_movements"] == {
        "improved": 2,
        "unchanged": 1,
        "worsened": 1,
    }


def test_selection_prefers_cross_fold_stability_over_single_fold_peak(
    tmp_path: Path,
) -> None:
    manifest_path = _write_selection_fixture(tmp_path)

    report = select_rolling_origin_weight(
        manifest_path=manifest_path,
        output_dir=tmp_path / "selection",
    )

    assert report["status"] == "selected"
    assert report["selected_weight"] == pytest.approx(0.2)
    unstable = report["weights"]["0.1"]
    stable = report["weights"]["0.2"]
    assert unstable["eligible"] is False
    assert "all_folds_mrr_non_decreasing" in unstable["failed_gates"]
    assert stable["eligible"] is True
    assert stable["failed_gates"] == []
    assert stable["stability"]["worst_fold_mrr_delta"] > 0.0
    assert set(stable["pooled"]["candidate"]) == {
        "hit_at_1",
        "hit_at_3",
        "hit_at_10",
        "mean_rank",
        "mrr",
        "ndcg_at_10",
        "query_movements",
    }
    lock = json.loads((tmp_path / "selection" / "selection-lock.json").read_text(encoding="utf-8"))
    assert lock["selected_weight"] == pytest.approx(0.2)
    assert lock["integration_id"] == "two_tower_full_reranker_v1"


def test_selection_rejects_cross_expert_candidate_identity(
    tmp_path: Path,
) -> None:
    manifest_path = _write_selection_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["folds"][1]["candidates"]["0.2"]["integration_id"] = "standalone_two_tower"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integration_id"):
        select_rolling_origin_weight(
            manifest_path=manifest_path,
            output_dir=tmp_path / "selection",
        )


def test_selection_rejects_non_chronological_or_too_few_folds(
    tmp_path: Path,
) -> None:
    manifest_path = _write_selection_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["folds"] = manifest["folds"][:2]
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least three"):
        select_rolling_origin_weight(
            manifest_path=manifest_path,
            output_dir=tmp_path / "selection-a",
        )

    manifest_path = _write_selection_fixture(tmp_path / "other")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["folds"][1]["score_time_min"] = 15
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly after"):
        select_rolling_origin_weight(
            manifest_path=manifest_path,
            output_dir=tmp_path / "selection-b",
        )


def test_external_requires_lock_and_can_only_be_opened_once(
    tmp_path: Path,
) -> None:
    selection_manifest = _write_selection_fixture(tmp_path)
    select_rolling_origin_weight(
        manifest_path=selection_manifest,
        output_dir=tmp_path / "selection",
    )
    lock_path = tmp_path / "selection" / "selection-lock.json"
    external_manifest = _write_external_fixture(
        tmp_path,
        selection_lock=lock_path,
    )
    state_dir = tmp_path / "external-state"

    report = evaluate_locked_external(
        manifest_path=external_manifest,
        selection_lock_path=lock_path,
        state_dir=state_dir,
    )

    assert report["status"] == "accepted"
    assert report["selected_weight"] == pytest.approx(0.2)
    assert report["candidate"]["query_movements"]["improved"] == 2
    assert report["candidate"]["query_movements"]["worsened"] == 0
    assert (state_dir / "external-open-receipt.json").is_file()
    assert (state_dir / "external-evaluation-report.json").is_file()

    with pytest.raises(FileExistsError, match="already opened"):
        evaluate_locked_external(
            manifest_path=external_manifest,
            selection_lock_path=lock_path,
            state_dir=state_dir,
        )


def test_external_lock_mismatch_fails_before_consuming_holdout(
    tmp_path: Path,
) -> None:
    selection_manifest = _write_selection_fixture(tmp_path)
    select_rolling_origin_weight(
        manifest_path=selection_manifest,
        output_dir=tmp_path / "selection",
    )
    lock_path = tmp_path / "selection" / "selection-lock.json"
    external_manifest = _write_external_fixture(
        tmp_path,
        selection_lock=lock_path,
    )
    payload = json.loads(external_manifest.read_text(encoding="utf-8"))
    payload["selection_lock_sha256"] = "0" * 64
    external_manifest.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    state_dir = tmp_path / "external-state"

    with pytest.raises(ValueError, match="selection lock hash"):
        evaluate_locked_external(
            manifest_path=external_manifest,
            selection_lock_path=lock_path,
            state_dir=state_dir,
        )

    assert not (state_dir / "external-open-receipt.json").exists()


def test_external_candidate_weight_must_match_selection_lock(
    tmp_path: Path,
) -> None:
    selection_manifest = _write_selection_fixture(tmp_path)
    select_rolling_origin_weight(
        manifest_path=selection_manifest,
        output_dir=tmp_path / "selection",
    )
    lock_path = tmp_path / "selection" / "selection-lock.json"
    external_manifest = _write_external_fixture(
        tmp_path,
        selection_lock=lock_path,
    )
    payload = json.loads(external_manifest.read_text(encoding="utf-8"))
    payload["candidate"]["weight"] = 0.1
    external_manifest.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    state_dir = tmp_path / "external-state"

    with pytest.raises(ValueError, match="candidate weight"):
        evaluate_locked_external(
            manifest_path=external_manifest,
            selection_lock_path=lock_path,
            state_dir=state_dir,
        )

    assert not (state_dir / "external-open-receipt.json").exists()


def test_external_requires_strict_mrr_improvement(
    tmp_path: Path,
) -> None:
    selection_manifest = _write_selection_fixture(tmp_path)
    select_rolling_origin_weight(
        manifest_path=selection_manifest,
        output_dir=tmp_path / "selection",
    )
    lock_path = tmp_path / "selection" / "selection-lock.json"
    external_manifest = _write_external_fixture(
        tmp_path,
        selection_lock=lock_path,
    )
    payload = json.loads(external_manifest.read_text(encoding="utf-8"))
    baseline_path = tmp_path / payload["baseline"]["path"]
    candidate_path = tmp_path / payload["candidate"]["path"]
    np.save(
        candidate_path,
        np.load(baseline_path, allow_pickle=False),
        allow_pickle=False,
    )
    payload["candidate"]["sha256"] = _sha256(candidate_path)
    external_manifest.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    report = evaluate_locked_external(
        manifest_path=external_manifest,
        selection_lock_path=lock_path,
        state_dir=tmp_path / "external-state",
    )

    assert report["status"] == "rejected"
    assert report["gates"]["mrr_strictly_increasing"] is False
    assert report["delta_candidate_minus_baseline"]["mrr"] == pytest.approx(0.0)


def _write_selection_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    integration_id = "two_tower_full_reranker_v1"
    candidate_fingerprint = "candidate-order-sha256"
    folds = []
    intervals = [(0, 9, 10, 19), (0, 19, 20, 29), (0, 29, 30, 39)]
    for index, (train_min, train_max, score_min, score_max) in enumerate(intervals):
        baseline = _scores_for_ranks([2, 2, 2, 2])
        unstable_ranks = [1, 1, 1, 1] if index == 0 else ([3, 3, 3, 3] if index == 1 else [2, 2, 2, 2])
        stable = _scores_for_ranks([1, 2, 2, 2])
        baseline_path = root / f"fold-{index}-baseline.npy"
        unstable_path = root / f"fold-{index}-w010.npy"
        stable_path = root / f"fold-{index}-w020.npy"
        np.save(baseline_path, baseline, allow_pickle=False)
        np.save(
            unstable_path,
            _scores_for_ranks(unstable_ranks),
            allow_pickle=False,
        )
        np.save(stable_path, stable, allow_pickle=False)
        folds.append(
            {
                "fold_id": f"fold-{index}",
                "train_time_min": train_min,
                "train_time_max": train_max,
                "score_time_min": score_min,
                "score_time_max": score_max,
                "candidate_fingerprint": candidate_fingerprint,
                "baseline": _artifact(root, baseline_path),
                "candidates": {
                    "0.1": {
                        **_artifact(root, unstable_path),
                        "integration_id": integration_id,
                        "candidate_fingerprint": candidate_fingerprint,
                    },
                    "0.2": {
                        **_artifact(root, stable_path),
                        "integration_id": integration_id,
                        "candidate_fingerprint": candidate_fingerprint,
                    },
                },
            }
        )
    manifest = {
        "protocol": "exact_integrated_rolling_weight_selection_v1",
        "integration_id": integration_id,
        "positive_candidate_column": 0,
        "folds": folds,
    }
    manifest_path = root / "rolling-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _write_external_fixture(
    root: Path,
    *,
    selection_lock: Path,
) -> Path:
    baseline_path = root / "external-baseline.npy"
    candidate_path = root / "external-candidate-w020.npy"
    np.save(
        baseline_path,
        _scores_for_ranks([2, 2, 3, 3]),
        allow_pickle=False,
    )
    np.save(
        candidate_path,
        _scores_for_ranks([1, 2, 2, 3]),
        allow_pickle=False,
    )
    manifest = {
        "protocol": "exact_integrated_external_holdout_v1",
        "integration_id": "two_tower_full_reranker_v1",
        "selected_weight": 0.2,
        "selection_lock_sha256": _sha256(selection_lock),
        "candidate_fingerprint": "candidate-order-sha256",
        "training_time_max": 100,
        "score_time_min": 200,
        "score_time_max": 300,
        "minimum_train_to_score_gap": 100,
        "baseline": _artifact(root, baseline_path),
        "candidate": {
            **_artifact(root, candidate_path),
            "integration_id": "two_tower_full_reranker_v1",
            "candidate_fingerprint": "candidate-order-sha256",
            "weight": 0.2,
        },
    }
    manifest_path = root / "external-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _artifact(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
    }


def _scores_for_ranks(
    ranks: list[int],
    *,
    candidate_count: int = 5,
) -> np.ndarray:
    scores = np.full((len(ranks), candidate_count), 0.1, dtype=np.float64)
    scores[:, 0] = 0.5
    for row, rank in enumerate(ranks):
        if rank < 1 or rank > candidate_count:
            raise ValueError("rank outside candidate count")
        scores[row, 1:rank] = 0.6
    return scores


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
