"""In-memory cache store with LRU eviction."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

from stateloom.cache.base import CacheEntry


class MemoryCacheStore:
    """Thread-safe in-memory cache with LRU eviction.

    Supports both global and per-session scoping via the session_id parameter.
    Uses time.monotonic() for timestamps.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._lock = threading.Lock()
        # Global cache: hash -> CacheEntry
        self._global_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        # Per-session index: session_id -> set of hashes
        self._session_index: dict[str, set[str]] = {}

    def get(self, request_hash: str, session_id: str | None = None) -> CacheEntry | None:
        with self._lock:
            if session_id is not None:
                # Scoped lookup: only hit if entry belongs to this session
                session_hashes = self._session_index.get(session_id)
                if not session_hashes or request_hash not in session_hashes:
                    return None

            entry = self._global_cache.get(request_hash)
            if entry is not None:
                self._global_cache.move_to_end(request_hash)
            return entry

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self._global_cache[entry.request_hash] = entry
            self._global_cache.move_to_end(entry.request_hash)

            # Update session index
            if entry.session_id not in self._session_index:
                self._session_index[entry.session_id] = set()
            self._session_index[entry.session_id].add(entry.request_hash)

            # LRU eviction
            while len(self._global_cache) > self._max_size:
                evicted_hash, _ = self._global_cache.popitem(last=False)
                # Clean up session index
                for hashes in self._session_index.values():
                    hashes.discard(evicted_hash)

    def get_all_entries(self, session_id: str | None = None) -> list[CacheEntry]:
        with self._lock:
            if session_id is not None:
                hashes = self._session_index.get(session_id, set())
                return [self._global_cache[h] for h in hashes if h in self._global_cache]
            return list(self._global_cache.values())

    def evict_expired(self, ttl: float) -> int:
        if ttl <= 0:
            return 0
        now = time.time()
        evicted = 0
        with self._lock:
            expired = [h for h, e in self._global_cache.items() if now - e.created_at > ttl]
            for h in expired:
                del self._global_cache[h]
                for hashes in self._session_index.values():
                    hashes.discard(h)
                evicted += 1
        return evicted

    def size(self) -> int:
        with self._lock:
            return len(self._global_cache)

    def clear(self) -> None:
        with self._lock:
            self._global_cache.clear()
            self._session_index.clear()

    def purge_by_content(self, identifier: str) -> int:
        """Delete cache entries containing the identifier in response_json."""
        evicted = 0
        with self._lock:
            to_remove = [h for h, e in self._global_cache.items() if identifier in e.response_json]
            for h in to_remove:
                del self._global_cache[h]
                for hashes in self._session_index.values():
                    hashes.discard(h)
                evicted += 1
        return evicted

    def close(self) -> None:
        self.clear()
