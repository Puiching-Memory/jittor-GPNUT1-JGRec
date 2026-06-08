from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core.types import InteractionTable


@dataclass(frozen=True)
class NodeIdMap:
    src_to_id: dict[int, int]
    dst_to_id: dict[int, int]
    src_values: tuple[int, ...]
    dst_values: tuple[int, ...]

    @classmethod
    def from_interactions(cls, interactions: InteractionTable) -> NodeIdMap:
        src_values = tuple(np.unique(interactions.src).astype(int).tolist())
        dst_values = tuple(np.unique(interactions.dst).astype(int).tolist())
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

    def src_ids(self, raw_ids) -> np.ndarray:
        values = np.asarray(raw_ids)
        mapped = np.empty(values.shape, dtype=np.int32)
        for index, value in np.ndenumerate(values):
            mapped[index] = self.src_to_id.get(int(value), -1)
        return mapped

    def dst_ids(self, raw_ids) -> np.ndarray:
        values = np.asarray(raw_ids)
        mapped = np.empty(values.shape, dtype=np.int32)
        for index, value in np.ndenumerate(values):
            mapped[index] = self.dst_to_id.get(int(value), -1)
        return mapped
