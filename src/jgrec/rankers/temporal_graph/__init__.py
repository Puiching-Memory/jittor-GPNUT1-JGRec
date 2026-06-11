from __future__ import annotations

from .config import TemporalGraphTrainingConfig


def __getattr__(name: str):
    if name in {"TemporalGraphRanker", "TemporalGraphRankerAdapter"}:
        from .ranker import TemporalGraphRanker, TemporalGraphRankerAdapter  # noqa: PLC0415

        return {
            "TemporalGraphRanker": TemporalGraphRanker,
            "TemporalGraphRankerAdapter": TemporalGraphRankerAdapter,
        }[name]
    raise AttributeError(name)

__all__ = ["TemporalGraphRanker", "TemporalGraphRankerAdapter", "TemporalGraphTrainingConfig"]
