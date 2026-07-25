from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .base import Ranker

RankerFactory = Callable[[Any], Ranker]


@dataclass
class RankerRegistry:
    _factories: dict[str, RankerFactory]

    def __init__(self) -> None:
        self._factories = {}

    def register(self, name: str, factory: RankerFactory) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("ranker name cannot be empty")
        self._factories[key] = factory

    def create(self, name: str, config: Any) -> Ranker:
        key = name.strip().lower()
        try:
            return self._factories[key](config)
        except KeyError as exc:
            available = ", ".join(self.names())
            raise ValueError(f"unknown ranker '{name}', available: {available}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


registry = RankerRegistry()


def create_ranker(name: str, config: Any) -> Ranker:
    ensure_builtin_rankers()
    return registry.create(name, config)


def available_rankers() -> tuple[str, ...]:
    ensure_builtin_rankers()
    return registry.names()


def ensure_builtin_rankers() -> None:
    registered = set(registry.names())

    def hybrid_factory(config: Any) -> Ranker:
        from .hybrid.config import TrainingConfig  # noqa: PLC0415
        from .hybrid.ranker import HybridRankerAdapter  # noqa: PLC0415

        return HybridRankerAdapter(config if isinstance(config, TrainingConfig) else TrainingConfig())

    def craft_factory(config: Any) -> Ranker:
        from .craft.config import CRAFTBaselineConfig  # noqa: PLC0415
        from .craft.ranker import CRAFTBaselineRanker  # noqa: PLC0415

        return CRAFTBaselineRanker(config if isinstance(config, CRAFTBaselineConfig) else CRAFTBaselineConfig())

    def temporal_graph_factory(config: Any) -> Ranker:
        from .temporal_graph.config import TemporalGraphTrainingConfig  # noqa: PLC0415
        from .temporal_graph.ranker import TemporalGraphRankerAdapter  # noqa: PLC0415

        return TemporalGraphRankerAdapter(
            config if isinstance(config, TemporalGraphTrainingConfig) else TemporalGraphTrainingConfig()
        )

    def hybrid_heuristic_factory(config: Any) -> Ranker:
        from .hybrid_heuristic.config import TrainingConfig  # noqa: PLC0415
        from .hybrid_heuristic.ranker import HybridRankerAdapter  # noqa: PLC0415

        return HybridRankerAdapter(config if isinstance(config, TrainingConfig) else TrainingConfig())

    if "hybrid" not in registered:
        registry.register("hybrid", hybrid_factory)
    if "hybrid-heuristic" not in registered:
        registry.register("hybrid-heuristic", hybrid_heuristic_factory)
    if "craft" not in registered:
        registry.register("craft", craft_factory)
    if "temporal-graph" not in registered:
        registry.register("temporal-graph", temporal_graph_factory)
