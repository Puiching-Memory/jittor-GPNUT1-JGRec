from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def get_model_state(model: Any) -> dict[str, np.ndarray]:
    """Extract a numpy snapshot of a Jittor module's state_dict."""
    return {
        key: np.asarray(value.numpy(), dtype=np.float32).copy()
        for key, value in model.state_dict().items()
    }


def set_model_state(model: Any, state: dict[str, np.ndarray]) -> None:
    """Load a numpy state dictionary into a Jittor module."""
    import jittor as jt  # noqa: PLC0415

    model.load_state_dict({key: jt.array(value, dtype=jt.float32) for key, value in state.items()})


def save_model_state(path: Path, state: dict[str, np.ndarray]) -> None:
    """Persist a model state dictionary to a compressed ``.npz`` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **state)


def load_model_state(path: Path) -> dict[str, np.ndarray]:
    """Load a model state dictionary from a ``.npz`` file."""
    with np.load(path) as data:
        return {key: np.array(value) for key, value in data.items()}
