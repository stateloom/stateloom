"""StateLoom cache package — pluggable storage backends and semantic matching."""

from stateloom.cache.base import CacheEntry, CacheStore
from stateloom.cache.memory_store import MemoryCacheStore
from stateloom.cache.sqlite_store import SQLiteCacheStore

__all__ = [
    "CacheEntry",
    "CacheStore",
    "MemoryCacheStore",
    "SQLiteCacheStore",
]

# Conditional exports for vector backend
try:
    from stateloom.cache.vector_backend import (  # noqa: F401
        FaissBackend,
        VectorBackend,
    )

    __all__.extend(["VectorBackend", "FaissBackend"])
except ImportError:
    pass

# Conditional exports for semantic matching
try:
    from stateloom.cache.semantic import (  # noqa: F401
        SemanticMatcher,
        is_semantic_available,
    )

    __all__.extend(["SemanticMatcher", "is_semantic_available"])
except ImportError:
    pass
