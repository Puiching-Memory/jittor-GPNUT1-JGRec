from __future__ import annotations

import numpy as np

EMPTY_I32 = np.empty(0, dtype=np.int32)
EMPTY_I64 = np.empty(0, dtype=np.int64)
EMPTY_F32 = np.empty(0, dtype=np.float32)


class SparseCountMap:
    """CSR-based sparse map replacing dict[int, dict[int, int]].

    ~8 bytes/entry vs ~106 bytes for nested Python dicts (~13x saving).
    """

    __slots__ = ("col_indices", "row_keys", "row_offsets", "values")

    def __init__(
        self,
        row_keys: np.ndarray,
        row_offsets: np.ndarray,
        col_indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.row_keys = row_keys
        self.row_offsets = row_offsets
        self.col_indices = col_indices
        self.values = values

    @classmethod
    def empty(cls) -> SparseCountMap:
        return cls(EMPTY_I32, np.zeros(1, dtype=np.int32), EMPTY_I32, EMPTY_I32)

    @classmethod
    def from_nested_dict(cls, data: dict[int, dict[int, int]]) -> SparseCountMap:
        if not data:
            return cls.empty()
        sorted_keys = sorted(data.keys())
        all_cols: list[int] = []
        all_vals: list[int] = []
        offsets = [0]
        for key in sorted_keys:
            inner = data[key]
            if inner:
                for col, val in sorted(inner.items()):
                    all_cols.append(col)
                    all_vals.append(val)
            offsets.append(len(all_cols))
        return cls(
            np.asarray(sorted_keys, dtype=np.int32),
            np.asarray(offsets, dtype=np.int32),
            np.asarray(all_cols, dtype=np.int32) if all_cols else EMPTY_I32.copy(),
            np.asarray(all_vals, dtype=np.int32) if all_vals else EMPTY_I32.copy(),
        )

    @classmethod
    def from_snapshot(cls, data: dict) -> SparseCountMap:
        if data.get("format") != "csr-v1":
            return cls.from_nested_dict(data)
        row_keys = np.asarray(data["row_keys"], dtype=np.int32)
        row_offsets = np.asarray(data["row_offsets"], dtype=np.int32)
        col_indices = np.asarray(data["col_indices"], dtype=np.int32)
        values = np.asarray(data["values"], dtype=np.int32)
        if row_offsets.shape != (len(row_keys) + 1,):
            raise ValueError("invalid sparse count row offsets")
        if col_indices.shape != values.shape:
            raise ValueError("invalid sparse count columns and values")
        if row_offsets.size == 0 or row_offsets[0] != 0 or row_offsets[-1] != len(values):
            raise ValueError("invalid sparse count offset bounds")
        return cls(row_keys, row_offsets, col_indices, values)

    def snapshot(self) -> dict[str, object]:
        return {
            "format": "csr-v1",
            "row_keys": self.row_keys,
            "row_offsets": self.row_offsets,
            "col_indices": self.col_indices,
            "values": self.values,
        }

    def get_count(self, left: int, right: int) -> int:
        idx = int(np.searchsorted(self.row_keys, left))
        if idx >= len(self.row_keys) or self.row_keys[idx] != left:
            return 0
        start, end = int(self.row_offsets[idx]), int(self.row_offsets[idx + 1])
        if start == end:
            return 0
        pos = int(np.searchsorted(self.col_indices[start:end], right))
        abs_pos = start + pos
        if abs_pos < end and self.col_indices[abs_pos] == right:
            return int(self.values[abs_pos])
        return 0

    def get_row(self, left: int) -> tuple[np.ndarray, np.ndarray] | None:
        idx = int(np.searchsorted(self.row_keys, left))
        if idx >= len(self.row_keys) or self.row_keys[idx] != left:
            return None
        start, end = int(self.row_offsets[idx]), int(self.row_offsets[idx + 1])
        if start == end:
            return None
        return self.col_indices[start:end], self.values[start:end]

    def to_nested_dict(self) -> dict[int, dict[int, int]]:
        result: dict[int, dict[int, int]] = {}
        for i, key in enumerate(self.row_keys.tolist()):
            start, end = int(self.row_offsets[i]), int(self.row_offsets[i + 1])
            if start < end:
                cols = self.col_indices[start:end].tolist()
                vals = self.values[start:end].tolist()
                result[key] = dict(zip(cols, vals, strict=True))
        return result

    def copy(self) -> SparseCountMap:
        return SparseCountMap(
            self.row_keys.copy(),
            self.row_offsets.copy(),
            self.col_indices.copy(),
            self.values.copy(),
        )

    def sum_rows(self, left_keys: np.ndarray) -> dict[int, int]:
        cols, values = self.sum_rows_arrays(left_keys)
        return dict(zip(cols.tolist(), values.tolist(), strict=True))

    def sum_rows_arrays(self, left_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if left_keys.size == 0 or len(self.row_keys) == 0:
            return EMPTY_I32, EMPTY_I64
        indices = np.searchsorted(self.row_keys, left_keys)
        valid = indices < len(self.row_keys)
        valid_positions = np.flatnonzero(valid)
        valid[valid_positions] &= self.row_keys[indices[valid_positions]] == left_keys[valid_positions]
        valid_indices = indices[valid]
        if valid_indices.size == 0:
            return EMPTY_I32, EMPTY_I64
        starts = self.row_offsets[valid_indices]
        ends = self.row_offsets[valid_indices + 1]
        mask = starts < ends
        if not np.any(mask):
            return EMPTY_I32, EMPTY_I64
        starts, ends = starts[mask], ends[mask]
        cols = np.concatenate([self.col_indices[s:e] for s, e in zip(starts, ends, strict=True)])
        vals = np.concatenate([self.values[s:e] for s, e in zip(starts, ends, strict=True)])
        unique_cols, inverse = np.unique(cols, return_inverse=True)
        summed = np.zeros(len(unique_cols), dtype=np.int64)
        np.add.at(summed, inverse, vals)
        return unique_cols, summed

    def batch_get_counts(self, left_keys: np.ndarray, right_key: int) -> np.ndarray:
        result = np.zeros(len(left_keys), dtype=np.int32)
        if left_keys.size == 0 or len(self.row_keys) == 0:
            return result
        indices = np.searchsorted(self.row_keys, left_keys)
        valid = indices < len(self.row_keys)
        valid_positions = np.flatnonzero(valid)
        valid[valid_positions] &= (
            self.row_keys[indices[valid_positions]] == left_keys[valid_positions]
        )
        for i in np.flatnonzero(valid):
            idx = indices[i]
            start, end = int(self.row_offsets[idx]), int(self.row_offsets[idx + 1])
            if start == end:
                continue
            pos = int(np.searchsorted(self.col_indices[start:end], right_key))
            abs_pos = start + pos
            if abs_pos < end and self.col_indices[abs_pos] == right_key:
                result[i] = int(self.values[abs_pos])
        return result

    def sum_row_counts_for_candidates(
        self,
        left_keys: np.ndarray,
        candidate_ids: np.ndarray,
        *,
        exclude_equal: bool = False,
    ) -> np.ndarray:
        result = np.zeros(len(candidate_ids), dtype=np.int64)
        if left_keys.size == 0 or candidate_ids.size == 0 or len(self.row_keys) == 0:
            return result

        row_indices = np.searchsorted(self.row_keys, left_keys)
        valid_rows = np.flatnonzero(row_indices < len(self.row_keys))
        valid_rows = valid_rows[self.row_keys[row_indices[valid_rows]] == left_keys[valid_rows]]
        for left_row in valid_rows:
            row_idx = row_indices[left_row]
            start, end = int(self.row_offsets[row_idx]), int(self.row_offsets[row_idx + 1])
            columns = self.col_indices[start:end]
            positions = np.searchsorted(columns, candidate_ids)
            matched = np.flatnonzero(positions < len(columns))
            matched = matched[columns[positions[matched]] == candidate_ids[matched]]
            if exclude_equal:
                matched = matched[candidate_ids[matched] != left_keys[left_row]]
            result[matched] += self.values[start + positions[matched]]
        return result

    def __bool__(self) -> bool:
        return len(self.row_keys) > 0

    def nnz(self) -> int:
        return len(self.values)


class SparseFloatMap:
    """CSR-based sparse map for compact float-valued graph aggregates."""

    __slots__ = ("col_indices", "row_keys", "row_offsets", "values")

    def __init__(
        self,
        row_keys: np.ndarray,
        row_offsets: np.ndarray,
        col_indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.row_keys = np.asarray(row_keys, dtype=np.int32)
        self.row_offsets = np.asarray(row_offsets, dtype=np.int32)
        self.col_indices = np.asarray(col_indices, dtype=np.int32)
        self.values = np.asarray(values, dtype=np.float32)

    @classmethod
    def empty(cls) -> SparseFloatMap:
        return cls(EMPTY_I32, np.zeros(1, dtype=np.int32), EMPTY_I32, EMPTY_F32)

    @classmethod
    def from_nested_dict(cls, data: dict[int, dict[int, float]]) -> SparseFloatMap:
        if not data:
            return cls.empty()
        sorted_keys = sorted(data)
        all_cols: list[int] = []
        all_values: list[float] = []
        offsets = [0]
        for key in sorted_keys:
            for col, value in sorted(data[key].items()):
                all_cols.append(col)
                all_values.append(value)
            offsets.append(len(all_cols))
        return cls(
            np.asarray(sorted_keys, dtype=np.int32),
            np.asarray(offsets, dtype=np.int32),
            np.asarray(all_cols, dtype=np.int32),
            np.asarray(all_values, dtype=np.float32),
        )

    @classmethod
    def from_snapshot(cls, data: dict) -> SparseFloatMap:
        if data.get("format") != "csr-float-v1":
            return cls.from_nested_dict(data)
        result = cls(
            data["row_keys"],
            data["row_offsets"],
            data["col_indices"],
            data["values"],
        )
        if result.row_offsets.shape != (len(result.row_keys) + 1,):
            raise ValueError("invalid sparse float row offsets")
        if result.col_indices.shape != result.values.shape:
            raise ValueError("invalid sparse float columns and values")
        if (
            result.row_offsets.size == 0
            or result.row_offsets[0] != 0
            or result.row_offsets[-1] != len(result.values)
        ):
            raise ValueError("invalid sparse float offset bounds")
        return result

    def snapshot(self) -> dict[str, object]:
        return {
            "format": "csr-float-v1",
            "row_keys": self.row_keys,
            "row_offsets": self.row_offsets,
            "col_indices": self.col_indices,
            "values": self.values,
        }

    def copy(self) -> SparseFloatMap:
        return SparseFloatMap(
            self.row_keys.copy(),
            self.row_offsets.copy(),
            self.col_indices.copy(),
            self.values.copy(),
        )

    def sum_row_values_for_candidates(
        self,
        left_keys: np.ndarray,
        candidate_ids: np.ndarray,
        *,
        exclude_equal: bool = False,
    ) -> np.ndarray:
        result = np.zeros(len(candidate_ids), dtype=np.float64)
        if left_keys.size == 0 or candidate_ids.size == 0 or len(self.row_keys) == 0:
            return result

        row_indices = np.searchsorted(self.row_keys, left_keys)
        valid_rows = np.flatnonzero(row_indices < len(self.row_keys))
        valid_rows = valid_rows[
            self.row_keys[row_indices[valid_rows]] == left_keys[valid_rows]
        ]
        for left_row in valid_rows:
            row_idx = row_indices[left_row]
            start, end = int(self.row_offsets[row_idx]), int(self.row_offsets[row_idx + 1])
            columns = self.col_indices[start:end]
            positions = np.searchsorted(columns, candidate_ids)
            matched = np.flatnonzero(positions < len(columns))
            matched = matched[columns[positions[matched]] == candidate_ids[matched]]
            if exclude_equal:
                matched = matched[candidate_ids[matched] != left_keys[left_row]]
            result[matched] += self.values[start + positions[matched]]
        return result

    def __bool__(self) -> bool:
        return len(self.row_keys) > 0

    def nnz(self) -> int:
        return len(self.values)
