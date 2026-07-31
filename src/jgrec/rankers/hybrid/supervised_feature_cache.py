from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from jgrec.core.types import InteractionTable

from .config import TrainingConfig

if TYPE_CHECKING:
    from .auto_strategy import DatasetProfile


SUPERVISED_FEATURE_CACHE_VERSION = 1


@dataclass(frozen=True)
class SupervisedFeatureCache:
    root: Path

    def __init__(self, root: Path | str) -> None:
        object.__setattr__(self, "root", Path(root))

    def load(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        train_path, val_path, manifest_path = self._paths(key)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != SUPERVISED_FEATURE_CACHE_VERSION or manifest.get("key") != key:
                return None
            train = np.load(train_path, mmap_mode="r", allow_pickle=False)
            val = np.load(val_path, mmap_mode="r", allow_pickle=False)
            if not _matches_manifest(train, manifest["train"]) or not _matches_manifest(val, manifest["val"]):
                return None
            return train, val
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def load_fusion_rng_state(self, key: str) -> dict[str, Any] | None:
        _, _, manifest_path = self._paths(key)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != SUPERVISED_FEATURE_CACHE_VERSION or manifest.get("key") != key:
                return None
            state = manifest.get("fusion_rng_state")
            return state if isinstance(state, dict) else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def load_candidate_ids(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        _, _, manifest_path = self._paths(key)
        train_path, val_path = self._candidate_paths(key)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != SUPERVISED_FEATURE_CACHE_VERSION or manifest.get("key") != key:
                return None
            train_descriptor = manifest.get("train_candidates")
            val_descriptor = manifest.get("val_candidates")
            if not isinstance(train_descriptor, dict) or not isinstance(val_descriptor, dict):
                return None
            train = np.load(train_path, mmap_mode="r", allow_pickle=False)
            val = np.load(val_path, mmap_mode="r", allow_pickle=False)
            if not _matches_manifest(train, train_descriptor) or not _matches_manifest(val, val_descriptor):
                return None
            return train, val
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def save(
        self,
        key: str,
        train: np.ndarray,
        val: np.ndarray,
        *,
        fusion_rng_state: dict[str, Any] | None = None,
        train_candidates: np.ndarray | None = None,
        val_candidates: np.ndarray | None = None,
    ) -> None:
        train_path, val_path, manifest_path = self._paths(key)
        candidate_arrays = _validated_candidate_ids(
            train,
            val,
            train_candidates=train_candidates,
            val_candidates=val_candidates,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path.unlink(missing_ok=True)
        _save_npy_atomic(train_path, train)
        _save_npy_atomic(val_path, val)
        manifest = {
            "version": SUPERVISED_FEATURE_CACHE_VERSION,
            "key": key,
            "train": _array_descriptor(train),
            "val": _array_descriptor(val),
            "fusion_rng_state": _jsonable(fusion_rng_state),
        }
        if candidate_arrays is not None:
            train_candidate_ids, val_candidate_ids = candidate_arrays
            train_candidate_path, val_candidate_path = self._candidate_paths(key)
            _save_npy_atomic(train_candidate_path, train_candidate_ids)
            _save_npy_atomic(val_candidate_path, val_candidate_ids)
            manifest["train_candidates"] = _array_descriptor(train_candidate_ids)
            manifest["val_candidates"] = _array_descriptor(val_candidate_ids)
        _write_text_atomic(
            manifest_path,
            json.dumps(manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        )

    def _paths(self, key: str) -> tuple[Path, Path, Path]:
        if not key or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in key):
            raise ValueError("supervised feature cache key must contain only letters, digits, '-' or '_'")
        return (
            self.root / f"{key}.train.npy",
            self.root / f"{key}.val.npy",
            self.root / f"{key}.json",
        )

    def _candidate_paths(self, key: str) -> tuple[Path, Path]:
        self._paths(key)
        return (
            self.root / f"{key}.train-candidates.npy",
            self.root / f"{key}.val-candidates.npy",
        )


def supervised_feature_cache_key(
    interactions: InteractionTable,
    config: TrainingConfig,
    *,
    recent_window: int,
    feature_names: tuple[str, ...],
    dataset_profile: DatasetProfile | None = None,
) -> str:
    """Return a content key for everything that can change supervised feature values."""
    payload = {
        "version": SUPERVISED_FEATURE_CACHE_VERSION,
        "recent_window": int(recent_window),
        "feature_names": list(feature_names),
        "split_and_sampling": {
            "val_ratio": config.val_ratio,
            "context_ratio": config.context_ratio,
            "max_train_events": config.max_train_events,
            "max_val_events": config.max_val_events,
            "train_num_negatives": config.resolved_train_num_negatives(),
            "val_num_negatives": config.resolved_val_num_negatives(),
            "hard_negative_ratio": config.hard_negative_ratio,
            "popular_negative_ratio": config.popular_negative_ratio,
            "test_candidate_negative_ratio": config.test_candidate_negative_ratio,
            "seed": config.seed,
        },
        "towers": {
            "candidate_prior": config.candidate_prior_config(),
            "target_window": config.target_window_config(),
            "structure": config.structure_config(),
            "source_profile": config.source_profile_config(),
            "graph": config.graph_config(),
            "sequence": config.sequence_config(),
            "two_tower": config.two_tower_config(),
        },
        "dataset_profile": dataset_profile,
    }
    digest = hashlib.blake2b(digest_size=20)
    digest.update(
        json.dumps(_jsonable(payload), ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    for name, values in (("src", interactions.src), ("dst", interactions.dst), ("time", interactions.time)):
        array = np.ascontiguousarray(values)
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _array_descriptor(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    return {"shape": list(values.shape), "dtype": values.dtype.str}


def _matches_manifest(array: np.ndarray, descriptor: dict[str, Any]) -> bool:
    return list(array.shape) == descriptor.get("shape") and array.dtype.str == descriptor.get("dtype")


def _validated_candidate_ids(
    train: np.ndarray,
    val: np.ndarray,
    *,
    train_candidates: np.ndarray | None,
    val_candidates: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if train_candidates is None and val_candidates is None:
        return None
    if train_candidates is None or val_candidates is None:
        raise ValueError("train and validation candidate IDs must be provided together")
    train_ids = np.asarray(train_candidates)
    val_ids = np.asarray(val_candidates)
    if train_ids.shape != np.asarray(train).shape[:2]:
        raise ValueError("train candidate IDs must match train feature rows and candidate width")
    if val_ids.shape != np.asarray(val).shape[:2]:
        raise ValueError("validation candidate IDs must match validation feature rows and candidate width")
    if train_ids.dtype.kind not in "iu" or val_ids.dtype.kind not in "iu":
        raise ValueError("candidate IDs must use an integer dtype")
    return train_ids, val_ids


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        Path(temp_name).unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, value: str) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        with suppress(OSError):
            os.close(fd)
        Path(temp_name).unlink(missing_ok=True)
        raise


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return [[int(key), int(count)] for key, count in sorted(value.items())]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
