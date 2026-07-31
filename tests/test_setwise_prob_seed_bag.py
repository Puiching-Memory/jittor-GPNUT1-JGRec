from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jgrec.setwise_prob_seed_bag import (
    FrozenSeedBagConfigError,
    load_frozen_seed_bag_config,
    load_verified_source_baseline,
    mean_seed_probabilities,
    training_seeds,
    validate_source_rolling_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG_PATH = ROOT / "docs" / "experiments" / "setwise-prob-seed-bag-v1.frozen.json"


def _load_frozen_payload() -> dict:
    return json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))


def _source_manifest() -> dict:
    return {
        "protocol": "exact_integrated_rolling_weight_selection_v1",
        "integration_id": "listwise_mlp_exact_current_champion_v1",
        "positive_candidate_column": 0,
        "folds": [
            {
                "fold_id": "fold-0",
                "train_rows": [0, 79909],
                "score_rows": [79909, 118816],
                "baseline": {
                    "path": "/tmp/fold0-baseline.npy",
                    "sha256": "0" * 64,
                },
                "candidate_fingerprint": "1" * 64,
            },
            {
                "fold_id": "fold-1",
                "train_rows": [0, 118816],
                "score_rows": [118816, 159804],
                "baseline": {
                    "path": "/tmp/fold1-baseline.npy",
                    "sha256": "2" * 64,
                },
                "candidate_fingerprint": "3" * 64,
            },
            {
                "fold_id": "fold-2",
                "train_rows": [0, 159804],
                "score_rows": [159804, 200000],
                "baseline": {
                    "path": "/tmp/fold2-baseline.npy",
                    "sha256": "4" * 64,
                },
                "candidate_fingerprint": "5" * 64,
            },
        ],
    }


def test_frozen_config_has_exact_precommitted_search_space() -> None:
    config = load_frozen_seed_bag_config(FROZEN_CONFIG_PATH)

    assert config.integration_id == "setwise_prob_seed_bag_v1"
    assert config.weights == (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    assert config.fold_boundaries == (
        (0, 79909, 118816),
        (0, 118816, 159804),
        (0, 159804, 200000),
    )
    assert config.seed_salts == (10007, 20011)
    assert config.epochs == 4
    assert training_seeds(config, 0) == (10067, 20071)
    assert training_seeds(config, 1) == (11076, 21080)
    assert training_seeds(config, 2) == (12085, 22089)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.update(weights=[0.05, 0.1]), "weights"),
        (lambda p: p.update(seed_salts=[1, 2]), "seed_salts"),
        (lambda p: p.update(setwise_epochs=5), "setwise_epochs"),
        (lambda p: p.update(status="draft"), "status"),
        (
            lambda p: p["folds"][0].update(score_rows=[79909, 118815]),
            "folds",
        ),
        (
            lambda p: p.update(auxiliary_formula="mean(candidate ranks)"),
            "probability",
        ),
    ],
)
def test_frozen_config_rejects_drift(mutate, match: str) -> None:
    payload = _load_frozen_payload()
    mutate(payload)

    with pytest.raises(FrozenSeedBagConfigError, match=match):
        load_frozen_seed_bag_config(payload)


def test_auxiliary_expert_is_exactly_two_seed_probability_mean() -> None:
    seed_a = np.array([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]], dtype=np.float32)
    seed_b = np.array([[0.5, 0.3, 0.2], [0.3, 0.6, 0.1]], dtype=np.float32)

    actual = mean_seed_probabilities([seed_a, seed_b])

    np.testing.assert_allclose(actual, (seed_a + seed_b) / 2.0, atol=1e-7)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=1e-7)


@pytest.mark.parametrize(
    "probabilities",
    [
        [np.ones((2, 3), dtype=np.float32)],
        [np.ones((2, 3), dtype=np.float32)] * 3,
        [
            np.full((2, 3), 1 / 3, dtype=np.float32),
            np.full((2, 4), 1 / 4, dtype=np.float32),
        ],
        [
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            np.array([[3.0, 2.0, 1.0]], dtype=np.float32),
        ],
    ],
)
def test_auxiliary_expert_rejects_non_probability_or_non_two_seed_inputs(
    probabilities,
) -> None:
    with pytest.raises(ValueError):
        mean_seed_probabilities(probabilities)


def test_source_manifest_requires_today_exact_rolling_identity_and_folds() -> None:
    config = load_frozen_seed_bag_config(FROZEN_CONFIG_PATH)

    folds = validate_source_rolling_manifest(_source_manifest(), config)

    assert tuple(fold["fold_id"] for fold in folds) == (
        "fold-0",
        "fold-1",
        "fold-2",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(integration_id="some_other_baseline"),
        lambda m: m["folds"][1].update(train_rows=[0, 118815]),
        lambda m: m["folds"][2]["baseline"].update(sha256="not-a-hash"),
        lambda m: m["folds"][0].update(candidate_fingerprint="not-a-hash"),
    ],
)
def test_source_manifest_rejects_wrong_baseline_or_fold_structure(mutation) -> None:
    config = load_frozen_seed_bag_config(FROZEN_CONFIG_PATH)
    manifest = copy.deepcopy(_source_manifest())
    mutation(manifest)

    with pytest.raises(ValueError):
        validate_source_rolling_manifest(manifest, config)


def test_source_baseline_is_bound_to_hash_shape_and_candidate_ids(
    tmp_path: Path,
) -> None:
    candidates = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64)
    baseline = np.array([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3]], dtype=np.float64)
    path = tmp_path / "baseline.npy"
    np.save(path, baseline, allow_pickle=False)
    fold = {
        "fold_id": "fold-0",
        "baseline": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "candidate_fingerprint": hashlib.sha256(np.ascontiguousarray(candidates).tobytes(order="C")).hexdigest(),
    }

    actual = load_verified_source_baseline(fold, candidates)

    np.testing.assert_array_equal(actual, baseline)


def test_source_baseline_rejects_hash_or_candidate_drift(tmp_path: Path) -> None:
    candidates = np.array([[7, 8, 9]], dtype=np.int64)
    path = tmp_path / "baseline.npy"
    np.save(path, np.array([[0.7, 0.2, 0.1]]), allow_pickle=False)
    fold = {
        "fold_id": "fold-0",
        "baseline": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "candidate_fingerprint": hashlib.sha256(np.ascontiguousarray(candidates).tobytes(order="C")).hexdigest(),
    }

    changed_candidates = candidates.copy()
    changed_candidates[0, 1] = 99
    with pytest.raises(ValueError, match="candidate fingerprint"):
        load_verified_source_baseline(fold, changed_candidates)

    fold["baseline"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="sha256"):
        load_verified_source_baseline(fold, candidates)
