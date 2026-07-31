from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

INTEGRATION_ID = "setwise_prob_seed_bag_v1"
SOURCE_INTEGRATION_ID = "listwise_mlp_exact_current_champion_v1"
FROZEN_STATUS = "frozen_before_any_setwise_prob_seed_bag_metric"
FROZEN_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
FROZEN_FOLDS = (
    ("fold-0", (0, 79_909), (79_909, 118_816)),
    ("fold-1", (0, 118_816), (118_816, 159_804)),
    ("fold-2", (0, 159_804), (159_804, 200_000)),
)
BASE_SEED = 60
SEED_SALTS = (10_007, 20_011)
SEED_STRIDE = 1_009
SETWISE_EPOCHS = 4
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 0
SEED_FORMULA = "base_seed + fold_index * 1009 + seed_salt"
AUXILIARY_FORMULA = "0.5 * setwise_probability(seed_salt=10007) + 0.5 * setwise_probability(seed_salt=20011)"
SOURCE_PROTOCOL = "exact_integrated_rolling_weight_selection_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class FrozenSeedBagConfigError(ValueError):
    """Raised when the precommitted seed-bag contract has drifted."""


@dataclass(frozen=True)
class FrozenSeedBagConfig:
    integration_id: str
    weights: tuple[float, ...]
    folds: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]
    base_seed: int
    seed_salts: tuple[int, int]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stop_patience: int

    @property
    def fold_boundaries(self) -> tuple[tuple[int, int, int], ...]:
        return tuple((train_rows[0], train_rows[1], score_rows[1]) for _, train_rows, score_rows in self.folds)


def load_frozen_seed_bag_config(
    source: Path | str | Mapping[str, Any],
) -> FrozenSeedBagConfig:
    payload = _load_payload(source)
    _require_equal(payload, "status", FROZEN_STATUS)
    _require_equal(payload, "integration_id", INTEGRATION_ID)
    _require_equal(payload, "weights", list(FROZEN_WEIGHTS))
    _require_equal(payload, "base_seed", BASE_SEED)
    _require_equal(payload, "seed_salts", list(SEED_SALTS))
    _require_equal(payload, "training_seed_formula", SEED_FORMULA)
    _require_equal(payload, "setwise_epochs", SETWISE_EPOCHS)
    _require_equal(payload, "batch_size", BATCH_SIZE)
    _require_equal(payload, "learning_rate", LEARNING_RATE)
    _require_equal(payload, "weight_decay", WEIGHT_DECAY)
    _require_equal(payload, "early_stop_patience", EARLY_STOP_PATIENCE)
    _require_equal(payload, "auxiliary_formula", AUXILIARY_FORMULA)
    _require_equal(payload, "external_open_limit", 1)

    expected_folds = [
        {
            "fold_id": fold_id,
            "train_rows": list(train_rows),
            "score_rows": list(score_rows),
        }
        for fold_id, train_rows, score_rows in FROZEN_FOLDS
    ]
    if payload.get("folds") != expected_folds:
        raise FrozenSeedBagConfigError(f"folds differ from the frozen exact-rolling boundaries {expected_folds}")
    prohibited = payload.get("prohibited")
    if not isinstance(prohibited, list) or "rank averaging" not in prohibited:
        raise FrozenSeedBagConfigError("probability-only contract is missing")

    return FrozenSeedBagConfig(
        integration_id=INTEGRATION_ID,
        weights=FROZEN_WEIGHTS,
        folds=FROZEN_FOLDS,
        base_seed=BASE_SEED,
        seed_salts=SEED_SALTS,
        epochs=SETWISE_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )


def training_seeds(
    config: FrozenSeedBagConfig,
    fold_index: int,
) -> tuple[int, int]:
    if not 0 <= fold_index < len(config.folds):
        raise IndexError(f"fold_index out of range: {fold_index}")
    return tuple(config.base_seed + fold_index * SEED_STRIDE + salt for salt in config.seed_salts)


def mean_seed_probabilities(
    probabilities: Sequence[np.ndarray],
) -> np.ndarray:
    if len(probabilities) != 2:
        raise ValueError("auxiliary expert requires exactly two seed probabilities")
    seed_a = _validated_probabilities(probabilities[0], label="seed probability 0")
    seed_b = _validated_probabilities(probabilities[1], label="seed probability 1")
    if seed_a.shape != seed_b.shape:
        raise ValueError("seed probability matrices must have identical shapes")
    return 0.5 * seed_a + 0.5 * seed_b


def validate_source_rolling_manifest(
    manifest: Mapping[str, Any],
    config: FrozenSeedBagConfig,
) -> tuple[dict[str, Any], ...]:
    if manifest.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"source protocol must be {SOURCE_PROTOCOL}")
    if manifest.get("integration_id") != SOURCE_INTEGRATION_ID:
        raise ValueError(f"source integration_id must be {SOURCE_INTEGRATION_ID}")
    if manifest.get("positive_candidate_column") != 0:
        raise ValueError("source positive_candidate_column must be 0")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != len(config.folds):
        raise ValueError("source manifest must contain the three frozen folds")

    validated: list[dict[str, Any]] = []
    for source_fold, expected in zip(folds, config.folds, strict=True):
        fold_id, train_rows, score_rows = expected
        if not isinstance(source_fold, dict):
            raise ValueError(f"{fold_id} source entry must be an object")
        if source_fold.get("fold_id") != fold_id:
            raise ValueError(f"source fold identity differs for {fold_id}")
        if source_fold.get("train_rows") != list(train_rows):
            raise ValueError(f"source train_rows differ for {fold_id}")
        if source_fold.get("score_rows") != list(score_rows):
            raise ValueError(f"source score_rows differ for {fold_id}")
        _validate_artifact_descriptor(source_fold.get("baseline"), fold_id)
        fingerprint = source_fold.get("candidate_fingerprint")
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            raise ValueError(f"{fold_id} candidate fingerprint must be sha256")
        validated.append(source_fold)
    return tuple(validated)


def load_verified_source_baseline(
    source_fold: Mapping[str, Any],
    candidate_ids: np.ndarray,
) -> np.ndarray:
    fold_id = str(source_fold.get("fold_id", "<unknown>"))
    descriptor = source_fold.get("baseline")
    _validate_artifact_descriptor(descriptor, fold_id)
    assert isinstance(descriptor, dict)
    path = Path(descriptor["path"])
    if not path.is_file():
        raise ValueError(f"{fold_id} baseline is missing: {path}")
    actual_digest = _sha256(path)
    if actual_digest != descriptor["sha256"]:
        raise ValueError(f"{fold_id} baseline sha256 differs from its manifest")

    candidates = np.asarray(candidate_ids)
    if candidates.ndim != 2 or not np.issubdtype(candidates.dtype, np.integer):
        raise ValueError(f"{fold_id} candidate ids must be a 2D integer matrix")
    fingerprint = hashlib.sha256(np.ascontiguousarray(candidates).tobytes(order="C")).hexdigest()
    if fingerprint != source_fold.get("candidate_fingerprint"):
        raise ValueError(f"{fold_id} candidate fingerprint differs from source")

    baseline = np.load(path, mmap_mode="r", allow_pickle=False)
    if baseline.shape != candidates.shape:
        raise ValueError(f"{fold_id} baseline shape {baseline.shape} differs from candidate shape {candidates.shape}")
    return _validated_probabilities(baseline, label=f"{fold_id} baseline")


def _load_payload(source: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenSeedBagConfigError(f"cannot read frozen config: {path}") from error
    if not isinstance(payload, dict):
        raise FrozenSeedBagConfigError("frozen config must be a JSON object")
    return payload


def _require_equal(
    payload: Mapping[str, Any],
    key: str,
    expected: Any,
) -> None:
    if payload.get(key) != expected:
        label = "probability formula" if key == "auxiliary_formula" else key
        raise FrozenSeedBagConfigError(f"{label} differs from the frozen value {expected!r}")


def _validated_probabilities(values: np.ndarray, *, label: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] < 1 or probabilities.shape[1] < 2:
        raise ValueError(f"{label} must be a non-empty 2D matrix")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{label} must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"{label} must be probabilities in [0, 1]")
    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{label} must be row-normalized probabilities")
    return probabilities


def _validate_artifact_descriptor(descriptor: Any, fold_id: str) -> None:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{fold_id} baseline descriptor must be an object")
    if not isinstance(descriptor.get("path"), str) or not descriptor["path"]:
        raise ValueError(f"{fold_id} baseline path must be non-empty")
    digest = descriptor.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{fold_id} baseline sha256 is invalid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
