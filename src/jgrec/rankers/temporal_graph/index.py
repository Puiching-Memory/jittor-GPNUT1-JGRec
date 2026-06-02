from __future__ import annotations

import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jittor as jt
import jittor_geometric
import numpy as np
from jittor_geometric.data import TemporalData

from jgrec.core.io import read_test_queries
from jgrec.core.types import Interaction

PAD_NODE_ID = 0


def temporal_loader_api() -> tuple[type, Any]:
    root = Path(jittor_geometric.__file__).resolve().parent
    module_path = root / "dataloader" / "temporal_dataloader.py"
    spec = importlib.util.spec_from_file_location("_jgrec_temporal_dataloader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load temporal dataloader from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TemporalDataLoader, module.get_neighbor_sampler


class SafeTemporalNeighborSampler:
    def __init__(self, sampler: Any) -> None:
        self.sampler = sampler
        self.num_nodes = len(getattr(sampler, "nodes_neighbor_times", ()))

    def get_historical_neighbors_left(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        node_interact_times = np.asarray(node_interact_times)
        neighbors = np.zeros((len(node_ids), num_neighbors), dtype=np.int64)
        edge_ids = np.zeros((len(node_ids), num_neighbors), dtype=np.int64)
        times = np.zeros((len(node_ids), num_neighbors), dtype=np.float32)
        valid = (node_ids >= 0) & (node_ids < self.num_nodes)
        if not np.any(valid):
            return neighbors, edge_ids, times

        valid_neighbors, valid_edges, valid_times = self.sampler.get_historical_neighbors_left(
            node_ids=node_ids[valid],
            node_interact_times=node_interact_times[valid],
            num_neighbors=num_neighbors,
        )
        neighbors[valid] = valid_neighbors
        edge_ids[valid] = valid_edges
        times[valid] = valid_times
        return neighbors, edge_ids, times


@dataclass(frozen=True)
class TemporalNodeMap:
    src_to_id: dict[int, int]
    dst_to_id: dict[int, int]
    src_values: tuple[int, ...]
    dst_values: tuple[int, ...]

    @classmethod
    def from_interactions_and_test(cls, interactions: list[Interaction], test_path: Path | None) -> TemporalNodeMap:
        src_values = {item.src for item in interactions}
        dst_values = {item.dst for item in interactions}
        if test_path is not None and test_path.exists():
            for query in read_test_queries(test_path):
                src_values.add(query.src)
                dst_values.update(query.candidates)

        ordered_src = tuple(sorted(src_values))
        ordered_dst = tuple(sorted(dst_values))
        src_to_id = {value: idx + 1 for idx, value in enumerate(ordered_src)}
        dst_offset = len(ordered_src) + 1
        dst_to_id = {value: dst_offset + idx for idx, value in enumerate(ordered_dst)}
        return cls(src_to_id=src_to_id, dst_to_id=dst_to_id, src_values=ordered_src, dst_values=ordered_dst)

    @property
    def num_src(self) -> int:
        return len(self.src_values)

    @property
    def num_dst(self) -> int:
        return len(self.dst_values)

    @property
    def num_nodes(self) -> int:
        return 1 + self.num_src + self.num_dst

    @property
    def dst_ids_array(self) -> np.ndarray:
        return np.asarray([self.dst_to_id[value] for value in self.dst_values], dtype=np.int32)

    def src_id(self, raw_id: int) -> int:
        return self.src_to_id.get(raw_id, PAD_NODE_ID)

    def dst_id(self, raw_id: int) -> int:
        return self.dst_to_id.get(raw_id, PAD_NODE_ID)

    def dst_ids(self, raw_ids: tuple[int, ...]) -> np.ndarray:
        return np.asarray([self.dst_id(value) for value in raw_ids], dtype=np.int32)


def temporal_data_from_interactions(interactions: list[Interaction], node_map: TemporalNodeMap) -> TemporalData:
    src = np.asarray([node_map.src_id(item.src) for item in interactions], dtype=np.int32)
    dst = np.asarray([node_map.dst_id(item.dst) for item in interactions], dtype=np.int32)
    times = np.asarray([item.time for item in interactions], dtype=np.int32)
    edge_ids = np.arange(len(interactions), dtype=np.int32) + 1
    return TemporalData(
        src=jt.array(src, dtype=jt.int32),
        dst=jt.array(dst, dtype=jt.int32),
        t=jt.array(times, dtype=jt.int32),
        edge_ids=jt.array(edge_ids, dtype=jt.int32),
    )


def safe_neighbor_sampler(sampler: Any) -> SafeTemporalNeighborSampler:
    return SafeTemporalNeighborSampler(sampler)


def scan_test_nodes_csv(path: Path) -> tuple[set[int], set[int]]:
    sources: set[int] = set()
    candidates: set[int] = set()
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            sources.add(int(row[0]))
            candidates.update(int(value) for value in row[2:])
    return sources, candidates
