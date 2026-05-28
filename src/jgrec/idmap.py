from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core.types import Interaction


@dataclass(frozen=True)
class NodeIdMap:
    src_to_id: dict[int, int]
    dst_to_id: dict[int, int]
    src_values: tuple[int, ...]
    dst_values: tuple[int, ...]

    @classmethod
    def from_interactions(cls, interactions: list[Interaction]) -> NodeIdMap:
        src_values = tuple(sorted({item.src for item in interactions}))
        dst_values = tuple(sorted({item.dst for item in interactions}))
        return cls(
            src_to_id={value: idx for idx, value in enumerate(src_values)},
            dst_to_id={value: idx for idx, value in enumerate(dst_values)},
            src_values=src_values,
            dst_values=dst_values,
        )

    @property
    def num_src(self) -> int:
        return len(self.src_values)

    @property
    def num_dst(self) -> int:
        return len(self.dst_values)

    def src_id(self, raw_id: int) -> int:
        return self.src_to_id.get(raw_id, -1)

    def dst_id(self, raw_id: int) -> int:
        return self.dst_to_id.get(raw_id, -1)

    def dst_ids(self, raw_ids: tuple[int, ...]) -> np.ndarray:
        return np.asarray([self.dst_to_id.get(value, -1) for value in raw_ids], dtype=np.int32)
