from __future__ import annotations

from typing import Protocol

import numpy as np

from jgrec.core.types import FitContext, InteractionTable, TestQueryArray, TrainingReport


class Ranker(Protocol):
    name: str

    def fit(self, interactions: InteractionTable, context: FitContext) -> TrainingReport:
        ...

    def predict_batch(self, queries: TestQueryArray) -> np.ndarray:
        ...
