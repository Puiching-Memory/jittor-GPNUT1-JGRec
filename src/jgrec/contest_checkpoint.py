from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_FORMAT = "jgrec-contest-checkpoint"
CHECKPOINT_VERSION = 1
_RESERVED_METADATA_KEYS = {"format", "version", "model_name", "datasets"}


def get_model_state(model: Any) -> dict[str, np.ndarray]:
    return {key: np.asarray(value.numpy(), dtype=np.float32).copy() for key, value in model.state_dict().items()}


def set_model_state(model: Any, state: dict[str, np.ndarray]) -> None:
    import jittor as jt  # noqa: PLC0415

    model.load_state_dict({key: jt.array(value, dtype=jt.float32) for key, value in state.items()})


class ContestCheckpointWriter:
    """Stream dataset snapshots into one checkpoint and publish it atomically."""

    def __init__(
        self,
        path: Path,
        *,
        model_name: str,
        expected_datasets: tuple[str, ...],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.expected_datasets = tuple(expected_datasets)
        self._written: set[str] = set()
        self._written_order: list[str] = []
        self._closed = False
        self._temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        extra = dict(metadata or {})
        overlap = _RESERVED_METADATA_KEYS.intersection(extra)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"checkpoint metadata uses reserved keys: {names}")
        self.metadata = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "model_name": model_name,
            "datasets": self.expected_datasets,
            **extra,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"checkpoint already exists: {self.path}")
        if self._temp_path.exists():
            self._resume_partial()
        else:
            self._file = self._temp_path.open("wb")
            pickle.dump(self.metadata, self._file, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    def written_datasets(self) -> tuple[str, ...]:
        return tuple(self._written_order)

    @property
    def is_complete(self) -> bool:
        return all(name in self._written for name in self.expected_datasets)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def add_dataset(self, name: str, state: Any) -> None:
        self._ensure_open()
        if name not in self.expected_datasets:
            raise KeyError(f"unexpected checkpoint dataset: {name}")
        if name in self._written:
            raise ValueError(f"checkpoint dataset already written: {name}")
        pickle.dump(("dataset", name, state), self._file, protocol=pickle.HIGHEST_PROTOCOL)
        self._file.flush()
        self._written.add(name)
        self._written_order.append(name)

    def finalize(self) -> None:
        self._ensure_open()
        missing = tuple(name for name in self.expected_datasets if name not in self._written)
        if missing:
            raise RuntimeError(f"missing datasets: {', '.join(missing)}")
        pickle.dump(("complete", self.expected_datasets), self._file, protocol=pickle.HIGHEST_PROTOCOL)
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._closed = True
        os.replace(self._temp_path, self.path)

    def close_partial(self) -> None:
        self._ensure_open()
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._closed = True

    def abort(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True
        self._temp_path.unlink(missing_ok=True)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("checkpoint writer is closed")

    def _resume_partial(self) -> None:
        self._file = self._temp_path.open("r+b")
        existing_metadata = pickle.load(self._file)
        _validate_metadata(existing_metadata)
        if existing_metadata != self.metadata:
            self._file.close()
            self._closed = True
            raise ValueError("partial checkpoint metadata does not match this run")
        last_good_offset = self._file.tell()
        while True:
            try:
                record = pickle.load(self._file)
            except (EOFError, pickle.UnpicklingError):
                self._file.seek(last_good_offset)
                self._file.truncate()
                break
            if not isinstance(record, tuple) or len(record) != 3 or record[0] != "dataset":
                self._file.close()
                self._closed = True
                raise ValueError("partial checkpoint contains an invalid record")
            name = str(record[1])
            self._written.add(name)
            self._written_order.append(name)
            last_good_offset = self._file.tell()
        self._file.seek(0, os.SEEK_END)


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as checkpoint_file:
        metadata = pickle.load(checkpoint_file)
    _validate_metadata(metadata)
    return metadata


def load_checkpoint_dataset(path: Path, dataset_name: str) -> Any:
    with Path(path).open("rb") as checkpoint_file:
        metadata = pickle.load(checkpoint_file)
        _validate_metadata(metadata)
        if dataset_name not in metadata["datasets"]:
            raise KeyError(f"checkpoint does not contain dataset: {dataset_name}")

        while True:
            try:
                record = pickle.load(checkpoint_file)
            except EOFError:
                break
            if not isinstance(record, tuple) or not record:
                raise ValueError("invalid checkpoint record")
            if record[0] == "dataset":
                _, name, state = record
                if name == dataset_name:
                    return state
            elif record[0] == "complete":
                break

    raise KeyError(f"checkpoint dataset state is missing: {dataset_name}")


def _validate_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("invalid checkpoint metadata")
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported checkpoint format")
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {metadata.get('version')}")
    if not isinstance(metadata.get("datasets"), tuple):
        raise ValueError("checkpoint datasets metadata must be a tuple")
