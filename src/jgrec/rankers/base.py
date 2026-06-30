from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from jgrec.core.types import FitContext, InteractionTable, TestQueryArray, TrainingReport


class Ranker(Protocol):
    name: str

    def fit(self, interactions: InteractionTable, context: FitContext) -> TrainingReport:
        ...

    def predict_batch(self, queries: TestQueryArray) -> np.ndarray:
        ...

    def save_checkpoint(self, path: Path) -> None:
        """Persist the ranker's learned weights to ``path``."""
        ...

    def load_checkpoint(self, path: Path) -> None:
        """Load learned weights from ``path`` into the ranker."""
        ...
