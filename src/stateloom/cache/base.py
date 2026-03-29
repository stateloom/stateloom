"""Cache store protocol and entry model for StateLoom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class CacheEntry:
    """A single cached LLM response."""

    request_hash: str
    session_id: str
    response_json: str
    model: str
    provider: str
    cost: float
    created_at: float  # epoch seconds (time.time for sqlite, time.monotonic for memory)
    embedding: list[float] | None = None


@runtime_checkable
class CacheStore(Protocol):
    """Protocol for pluggable cache storage backends."""

    def get(self, request_hash: str, session_id: str | None = None) -> CacheEntry | None:
        """Look up a cache entry by exact hash. If session_id is given, scope to that session."""
        ...

    def put(self, entry: CacheEntry) -> None:
        """Store a cache entry."""
        ...

    def get_all_entries(self, session_id: str | None = None) -> list[CacheEntry]:
        """Return all entries, optionally filtered by session_id."""
        ...

    def evict_expired(self, ttl: float) -> int:
        """Remove entries older than ttl seconds. Returns count of evicted entries."""
        ...

    def size(self) -> int:
        """Return the total number of cached entries."""
        ...

    def clear(self) -> None:
        """Remove all entries."""
        ...

    def purge_by_content(self, identifier: str) -> int:
        """Delete cache entries containing the identifier. Returns count."""
        ...

    def close(self) -> None:
        """Release any resources held by the store."""
        ...
