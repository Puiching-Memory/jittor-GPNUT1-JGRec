from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jgrec.core.io import read_test_queries
from jgrec.core.types import InteractionTable

PAD_NODE_ID = 0


def temporal_loader_api() -> tuple[type, Any]:
    import jittor_geometric  # noqa: PLC0415

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
        self.sample_neighbor_strategy = getattr(sampler, "sample_neighbor_strategy", None)
        self._recent_offsets: np.ndarray | None = None
        self._recent_neighbor_ids: np.ndarray | None = None
        self._recent_edge_ids: np.ndarray | None = None
        self._recent_neighbor_times: np.ndarray | None = None
        if self.sample_neighbor_strategy == "recent":
            self._build_recent_index()

    def get_historical_neighbors_left(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        if node_interact_times is None:
            node_interact_times = np.ones_like(node_ids) * np.finfo(np.float32).max
        else:
            node_interact_times = np.asarray(node_interact_times)
        if self._recent_offsets is not None:
            return self._get_recent_neighbors_left(node_ids, node_interact_times, num_neighbors)

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

    def _build_recent_index(self) -> None:
        neighbor_ids = getattr(self.sampler, "nodes_neighbor_ids", ())
        edge_ids = getattr(self.sampler, "nodes_edge_ids", ())
        neighbor_times = getattr(self.sampler, "nodes_neighbor_times", ())
        lengths = np.fromiter((len(values) for values in neighbor_times), dtype=np.int64, count=self.num_nodes)
        offsets = np.empty(self.num_nodes + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        self._recent_offsets = offsets
        total = int(offsets[-1])
        if total == 0:
            self._recent_neighbor_ids = np.empty(0, dtype=np.int64)
            self._recent_edge_ids = np.empty(0, dtype=np.int64)
            self._recent_neighbor_times = np.empty(0, dtype=np.int64)
            return
        self._recent_neighbor_ids = np.concatenate(neighbor_ids).astype(np.int64, copy=False)
        self._recent_edge_ids = np.concatenate(edge_ids).astype(np.int64, copy=False)
        self._recent_neighbor_times = np.concatenate(neighbor_times)

    def _get_recent_neighbors_left(
        self,
        node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offsets = self._recent_offsets
        neighbor_ids = self._recent_neighbor_ids
        edge_ids_flat = self._recent_edge_ids
        neighbor_times = self._recent_neighbor_times
        if offsets is None or neighbor_ids is None or edge_ids_flat is None or neighbor_times is None:
            raise RuntimeError("recent neighbor index is not initialized")

        row_count = len(node_ids)
        neighbors = np.zeros((row_count, num_neighbors), dtype=np.int64)
        edge_ids = np.zeros((row_count, num_neighbors), dtype=np.int64)
        times = np.zeros((row_count, num_neighbors), dtype=np.float32)
        valid = (node_ids >= 0) & (node_ids < self.num_nodes)
        if not np.any(valid) or neighbor_times.size == 0:
            return neighbors, edge_ids, times

        positions = np.flatnonzero(valid)
        valid_nodes = node_ids[positions]
        starts = offsets[valid_nodes]
        ends = offsets[valid_nodes + 1]
        lo = starts.copy()
        hi = ends.copy()
        query_times = node_interact_times[positions]

        while True:
            active = lo < hi
            if not np.any(active):
                break
            active_indices = np.flatnonzero(active)
            mid = (lo[active_indices] + hi[active_indices]) // 2
            go_right = neighbor_times[mid] < query_times[active_indices]
            lo[active_indices[go_right]] = mid[go_right] + 1
            hi[active_indices[~go_right]] = mid[~go_right]

        available = lo - starts
        row_counts = np.minimum(available, num_neighbors)
        populated = row_counts > 0
        if not np.any(populated):
            return neighbors, edge_ids, times

        row_positions = positions[populated]
        row_counts = row_counts[populated]
        end_positions = lo[populated]
        cols = np.arange(num_neighbors, dtype=np.int64)
        take_indices = end_positions[:, np.newaxis] - row_counts[:, np.newaxis] + cols[np.newaxis, :]
        mask = cols[np.newaxis, :] < row_counts[:, np.newaxis]
        rows = np.broadcast_to(row_positions[:, np.newaxis], take_indices.shape)
        output_cols = np.broadcast_to(cols[np.newaxis, :], take_indices.shape)

        neighbors[rows[mask], output_cols[mask]] = neighbor_ids[take_indices[mask]]
        edge_ids[rows[mask], output_cols[mask]] = edge_ids_flat[take_indices[mask]]
        times[rows[mask], output_cols[mask]] = neighbor_times[take_indices[mask]]
        return neighbors, edge_ids, times


@dataclass(frozen=True)
class TemporalNodeMap:
    src_to_id: dict[int, int]
    dst_to_id: dict[int, int]
    src_values: tuple[int, ...]
    dst_values: tuple[int, ...]
    src_raw_ids: np.ndarray
    dst_raw_ids: np.ndarray
    src_compact_ids: np.ndarray
    dst_compact_ids: np.ndarray

    @classmethod
    def from_interactions_and_test(cls, interactions: InteractionTable, test_path: Path | None) -> TemporalNodeMap:
        src_values = np.unique(interactions.src.astype(np.int64, copy=False))
        dst_values = np.unique(interactions.dst.astype(np.int64, copy=False))
        if test_path is not None and test_path.exists():
            queries = read_test_queries(test_path)
            src_values = np.union1d(src_values, queries.src.astype(np.int64, copy=False))
            dst_values = np.union1d(dst_values, queries.candidates.astype(np.int64, copy=False).reshape(-1))

        ordered_src = tuple(src_values.astype(int).tolist())
        ordered_dst = tuple(dst_values.astype(int).tolist())
        src_raw_ids = np.asarray(ordered_src, dtype=np.int64)
        dst_raw_ids = np.asarray(ordered_dst, dtype=np.int64)
        src_compact_ids = np.arange(1, len(ordered_src) + 1, dtype=np.int32)
        dst_offset = len(ordered_src) + 1
        dst_compact_ids = np.arange(dst_offset, dst_offset + len(ordered_dst), dtype=np.int32)
        src_to_id = {value: int(src_compact_ids[idx]) for idx, value in enumerate(ordered_src)}
        dst_to_id = {value: int(dst_compact_ids[idx]) for idx, value in enumerate(ordered_dst)}
        return cls(
            src_to_id=src_to_id,
            dst_to_id=dst_to_id,
            src_values=ordered_src,
            dst_values=ordered_dst,
            src_raw_ids=src_raw_ids,
            dst_raw_ids=dst_raw_ids,
            src_compact_ids=src_compact_ids,
            dst_compact_ids=dst_compact_ids,
        )

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
        return self.dst_compact_ids

    def src_id(self, raw_id: int) -> int:
        return self.src_to_id.get(raw_id, PAD_NODE_ID)

    def dst_id(self, raw_id: int) -> int:
        return self.dst_to_id.get(raw_id, PAD_NODE_ID)

    def src_ids(self, raw_ids: np.ndarray) -> np.ndarray:
        return _map_sorted_ids(raw_ids, self.src_raw_ids, self.src_compact_ids)

    def dst_ids(self, raw_ids: np.ndarray) -> np.ndarray:
        return _map_sorted_ids(raw_ids, self.dst_raw_ids, self.dst_compact_ids)


def temporal_data_from_interactions(interactions: InteractionTable, node_map: TemporalNodeMap):
    import jittor as jt  # noqa: PLC0415
    from jittor_geometric.data import TemporalData  # noqa: PLC0415

    src = node_map.src_ids(interactions.src)
    dst = node_map.dst_ids(interactions.dst)
    times = interactions.time.astype(np.int32, copy=False)
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
    queries = read_test_queries(path)
    src_values = np.unique(queries.src).astype(int).tolist()
    dst_values = np.unique(queries.candidates).astype(int).tolist()
    return set(src_values), set(dst_values)


def _map_sorted_ids(raw_ids: np.ndarray, raw_values: np.ndarray, compact_values: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_ids, dtype=np.int64)
    flat = raw.reshape(-1)
    output = np.full(flat.shape, PAD_NODE_ID, dtype=np.int32)
    if flat.size == 0 or raw_values.size == 0:
        return output.reshape(raw.shape)

    positions = np.searchsorted(raw_values, flat)
    in_bounds = positions < raw_values.size
    if not np.any(in_bounds):
        return output.reshape(raw.shape)

    checked_positions = positions[in_bounds]
    matches = raw_values[checked_positions] == flat[in_bounds]
    if not np.any(matches):
        return output.reshape(raw.shape)

    flat_indices = np.flatnonzero(in_bounds)[matches]
    output[flat_indices] = compact_values[checked_positions[matches]]
    return output.reshape(raw.shape)
