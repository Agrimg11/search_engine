"""
lru_cache.py — Generic, capacity-bounded Least Recently Used (LRU) cache.

Ports: src/core/lru_cache.hpp

Design — O(1) for every operation:
  Recency order : collections.OrderedDict
                  move_to_end() is O(1) — implemented as a doubly-linked list
                  internally, same as the C++ std::list + splice pattern.

On a cache HIT  → entry is moved to the MRU end.
On a cache MISS (not full) → new entry appended at MRU end.
On a cache MISS (full)    → LRU entry (oldest) evicted first.

Special case: capacity == 0 → get() always returns None, put() is no-op.

Thread safety: NOT thread-safe. Single-threaded (asyncio) use only.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Optional, TypeVar

K = TypeVar('K')
V = TypeVar('V')


class LRUCache(Generic[K, V]):
    """
    Generic LRU cache with hit/miss statistics.

    Args:
        capacity: Maximum number of entries. 0 means disabled (always miss).
    """

    def __init__(self, capacity: int = 100) -> None:
        self._capacity = capacity
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._hits:   int = 0
        self._misses: int = 0

    # ── Core API ──────────────────────────────────────────────────────────────

    def get(self, key: K) -> Optional[V]:
        """
        Look up a key.

        Returns the cached value on HIT (and promotes to MRU),
        or None on MISS.

        Time: O(1) average.
        """
        if self._capacity == 0:
            self._misses += 1
            return None

        if key not in self._cache:
            self._misses += 1
            return None

        # Promote to MRU position (end).
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]

    def put(self, key: K, value: V) -> None:
        """
        Insert or update a key-value pair.

        Existing key: update value, promote to MRU.
        Cache full:   evict LRU entry (beginning), then insert.
        capacity == 0: no-op.

        Time: O(1) average.
        """
        if self._capacity == 0:
            return

        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return

        # Evict LRU if at capacity.
        if len(self._cache) >= self._capacity:
            self._cache.popitem(last=False)   # pop from LRU end (beginning)

        self._cache[key] = value

    def contains(self, key: K) -> bool:
        """Check if key exists WITHOUT updating recency or counters."""
        if self._capacity == 0:
            return False
        return key in self._cache

    def clear(self) -> None:
        """Remove all entries. Does NOT reset hit/miss statistics."""
        self._cache.clear()

    # ── Capacity / Size ───────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def empty(self) -> bool:
        return len(self._cache) == 0

    def set_capacity(self, new_cap: int) -> None:
        """Change the maximum capacity, evicting LRU entries if needed."""
        self._capacity = new_cap
        while new_cap > 0 and len(self._cache) > new_cap:
            self._cache.popitem(last=False)
        if new_cap == 0:
            self._cache.clear()

    def get_keys(self) -> list[K]:
        """Return all keys ordered from MRU (end) to LRU (beginning)."""
        return list(reversed(self._cache.keys()))

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
