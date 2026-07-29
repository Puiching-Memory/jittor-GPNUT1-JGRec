from __future__ import annotations

import math
from typing import Any


def tower_epoch_learning_rate(
    *,
    initial_lr: float,
    epoch: int,
    total_epochs: int,
    schedule: str,
    min_lr_ratio: float,
) -> float:
    initial = float(initial_lr)
    epoch_index = int(epoch)
    epoch_count = int(total_epochs)
    minimum_ratio = float(min_lr_ratio)
    normalized_schedule = str(schedule).strip().lower()

    if initial <= 0.0 or not math.isfinite(initial):
        raise ValueError("initial_lr must be positive and finite")
    if epoch_count < 1:
        raise ValueError("total_epochs must be at least 1")
    if not 1 <= epoch_index <= epoch_count:
        raise ValueError("epoch must be between 1 and total_epochs")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    if normalized_schedule == "constant" or epoch_count == 1:
        return initial
    if normalized_schedule != "cosine":
        raise ValueError(f"unsupported tower learning-rate schedule: {schedule}")

    progress = (epoch_index - 1) / (epoch_count - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return initial * (
        minimum_ratio + (1.0 - minimum_ratio) * cosine
    )


def set_tower_optimizer_learning_rate(
    optimizer: Any,
    *,
    initial_lr: float,
    epoch: int,
    total_epochs: int,
    schedule: str,
    min_lr_ratio: float,
) -> float:
    learning_rate = tower_epoch_learning_rate(
        initial_lr=initial_lr,
        epoch=epoch,
        total_epochs=total_epochs,
        schedule=schedule,
        min_lr_ratio=min_lr_ratio,
    )
    optimizer.lr = learning_rate
    return learning_rate
