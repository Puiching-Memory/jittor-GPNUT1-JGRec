from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jgrec.rankers.hybrid.cooccur_lift import (
    SEED_STRIDE,
    load_frozen_cooccur_lift_config,
)


@dataclass(frozen=True)
class LockedExternalSetup:
    integration_id: str
    selected_weight: float
    full_origin_seed: int
    selection_lock_path: Path
    selection_lock_sha256: str


def validate_locked_external_setup(
    *,
    frozen_config_path: Path,
    selection_lock_path: Path,
) -> LockedExternalSetup:
    config = load_frozen_cooccur_lift_config(frozen_config_path)
    lock_path = Path(selection_lock_path)
    lock = _read_json(lock_path)
    if lock.get("protocol") != "exact_integrated_weight_selection_lock_v1":
        raise ValueError("selection lock protocol is invalid")
    if lock.get("integration_id") != config.integration_id:
        raise ValueError("selection lock integration_id differs from frozen config")
    if lock.get("external_holdout_read") is not False:
        raise ValueError("selection lock external_holdout_read must be false")
    selected_weight = float(lock.get("selected_weight"))
    if selected_weight not in config.weights:
        raise ValueError("selection lock selected_weight is outside frozen weights")
    selection_manifest_hash = lock.get("selection_manifest_sha256")
    if not _is_sha256(selection_manifest_hash):
        raise ValueError("selection lock selection_manifest_sha256 is invalid")
    return LockedExternalSetup(
        integration_id=config.integration_id,
        selected_weight=selected_weight,
        full_origin_seed=(
            config.base_seed
            + len(config.folds) * SEED_STRIDE
            + config.seed_salt
        ),
        selection_lock_path=lock_path.resolve(),
        selection_lock_sha256=_sha256(lock_path),
    )


def build_external_manifest(
    *,
    contract: LockedExternalSetup,
    candidate_fingerprint: str,
    training_time_max: int,
    score_time_min: int,
    score_time_max: int,
    baseline_path: Path,
    baseline_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
) -> dict[str, Any]:
    for label, value in (
        ("candidate_fingerprint", candidate_fingerprint),
        ("baseline_sha256", baseline_sha256),
        ("candidate_sha256", candidate_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{label} is not a lowercase SHA-256")
    if not training_time_max < score_time_min <= score_time_max:
        raise ValueError("external score interval must follow full-origin training")
    return {
        "schema_version": 1,
        "protocol": "exact_integrated_external_holdout_v1",
        "integration_id": contract.integration_id,
        "selected_weight": contract.selected_weight,
        "selection_lock_sha256": contract.selection_lock_sha256,
        "positive_candidate_column": 0,
        "candidate_fingerprint": candidate_fingerprint,
        "training_time_max": int(training_time_max),
        "score_time_min": int(score_time_min),
        "score_time_max": int(score_time_max),
        "minimum_train_to_score_gap": 1,
        "baseline": {
            "path": str(Path(baseline_path).resolve()),
            "sha256": baseline_sha256,
        },
        "candidate": {
            "path": str(Path(candidate_path).resolve()),
            "sha256": candidate_sha256,
            "integration_id": contract.integration_id,
            "weight": contract.selected_weight,
            "candidate_fingerprint": candidate_fingerprint,
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
