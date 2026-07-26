from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class ByteBudgetLRU(Generic[K, V]):  # noqa: UP046 - Jittor validation runs on Python 3.10.
    def __init__(self, *, max_bytes: int, max_entries: int) -> None:
        self.max_bytes = max(int(max_bytes), 0)
        self.max_entries = max(int(max_entries), 0)
        self.current_bytes = 0
        self._entries: OrderedDict[K, tuple[V, int]] = OrderedDict()

    def get(self, key: K) -> V | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key: K, value: V, *, size_bytes: int) -> None:
        size_bytes = int(size_bytes)
        if size_bytes < 0:
            raise ValueError("cache value size cannot be negative")

        previous = self._entries.pop(key, None)
        if previous is not None:
            self.current_bytes -= previous[1]
        if self.max_entries == 0 or self.max_bytes == 0 or size_bytes > self.max_bytes:
            return

        self._entries[key] = (value, size_bytes)
        self.current_bytes += size_bytes
        while self.current_bytes > self.max_bytes or len(self._entries) > self.max_entries:
            _, (_, evicted_size) = self._entries.popitem(last=False)
            self.current_bytes -= evicted_size

    def clear(self) -> None:
        self._entries.clear()
        self.current_bytes = 0

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)
