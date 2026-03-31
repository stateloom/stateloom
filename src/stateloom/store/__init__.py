"""Persistence backends for StateLoom."""

from stateloom.store.base import Store
from stateloom.store.memory_store import MemoryStore
from stateloom.store.sqlite_store import SQLiteStore

__all__ = ["MemoryStore", "SQLiteStore", "Store"]

try:
    from stateloom.store.postgres_store import PostgresStore  # noqa: F401

    __all__.append("PostgresStore")
except ImportError:
    pass
