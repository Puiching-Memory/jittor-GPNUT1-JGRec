from .config import TrainingConfig

__all__ = ["HybridRankerAdapter", "TemporalHybridRanker", "TrainingConfig"]


def __getattr__(name: str):
    if name in {"HybridRankerAdapter", "TemporalHybridRanker"}:
        from .ranker import HybridRankerAdapter, TemporalHybridRanker

        return {
            "HybridRankerAdapter": HybridRankerAdapter,
            "TemporalHybridRanker": TemporalHybridRanker,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

