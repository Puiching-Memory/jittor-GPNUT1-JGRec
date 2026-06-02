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
    if registry.names():
        return

    def temporal_graph_factory(config: Any) -> Ranker:
        from .temporal_graph.ranker import TemporalGraphRankerAdapter, TemporalGraphTrainingConfig

        return TemporalGraphRankerAdapter(
            config if isinstance(config, TemporalGraphTrainingConfig) else TemporalGraphTrainingConfig()
        )

    def craft_factory(config: Any) -> Ranker:
        from .craft.config import CRAFTBaselineConfig
        from .craft.ranker import CRAFTBaselineRanker

        return CRAFTBaselineRanker(config if isinstance(config, CRAFTBaselineConfig) else CRAFTBaselineConfig())

    registry.register("temporal-graph", temporal_graph_factory)
    registry.register("craft", craft_factory)
