from __future__ import annotations

from typing import Protocol

import numpy as np

from jgrec.core.types import FitContext, Interaction, TestQuery, TrainingReport


class Ranker(Protocol):
    name: str

    def fit(self, interactions: list[Interaction], context: FitContext) -> TrainingReport:
        ...

    def predict_batch(self, queries: list[TestQuery]) -> np.ndarray:
        ...

