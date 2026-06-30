from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LossEarlyStopper:
    patience: int
    best_loss: float = float("inf")
    best_epoch: int = 0
    counter: int = 0
    best_state: dict[str, np.ndarray] | None = None

    def update(self, epoch: int, loss: float, model: Any | None = None) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.counter = 0
            if model is not None:
                self.best_state = _snapshot_state(model)
            return False
        self.counter += 1
        return self.patience > 0 and self.counter >= self.patience

    def restore_best(self, model: Any) -> None:
        if self.best_state is None:
            return
        import jittor as jt  # noqa: PLC0415

        model.load_state_dict({key: jt.array(value) for key, value in self.best_state.items()})


def _snapshot_state(model: Any) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(value.numpy(), dtype=np.float32).copy()
        for key, value in model.state_dict().items()
    }
