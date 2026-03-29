"""Clean teardown — restore original methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("stateloom.intercept")

# Registry of all patches applied
_patch_registry: list[PatchRecord] = []


@dataclass
class PatchRecord:
    """Record of a single monkey-patch applied."""

    target: Any  # The class/module that was patched
    method_name: str  # The method name that was replaced
    original: Any  # The original method
    description: str = ""  # Human-readable description


def register_patch(target: Any, method_name: str, original: Any, description: str = "") -> None:
    """Register a patch for later teardown."""
    _patch_registry.append(
        PatchRecord(
            target=target,
            method_name=method_name,
            original=original,
            description=description,
        )
    )


def unpatch_all() -> None:
    """Restore all original methods. Used in shutdown and test teardown."""
    for record in _patch_registry:
        try:
            setattr(record.target, record.method_name, record.original)
            logger.debug(f"[StateLoom] Unpatched {record.description}")
        except Exception as e:
            logger.warning(f"[StateLoom] Failed to unpatch {record.description}: {e}")
    _patch_registry.clear()


def get_original(target_class: Any, method_name: str) -> Any | None:
    """Get the original method for a patched class method, or None if not patched."""
    for record in _patch_registry:
        if record.target is target_class and record.method_name == method_name:
            return record.original
    return None


def get_patch_count() -> int:
    """Get the number of active patches."""
    return len(_patch_registry)
