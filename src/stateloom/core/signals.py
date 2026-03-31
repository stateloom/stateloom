"""Lightweight typed signal dispatcher.

Replaces ad-hoc ``list[Any]`` hook lists with a generic ``Signal[T]``
that supports connect, disconnect, emit, and introspection.  Any
extension (not just EE) can register receivers.

Usage::

    from stateloom.core.signals import Signal

    on_startup = Signal[Gate]("startup")
    on_startup.connect(my_callback)   # my_callback(gate) -> None
    on_startup.emit(gate)             # fires all receivers
    on_startup.disconnect(my_callback)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Generic, TypeVar

logger = logging.getLogger("stateloom.core.signals")

T = TypeVar("T")


class Signal(Generic[T]):
    """Typed signal that receivers can connect to."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._receivers: list[Callable[[T], None]] = []

    def connect(self, receiver: Callable[[T], None]) -> None:
        """Register a receiver. Duplicates are ignored."""
        if receiver not in self._receivers:
            self._receivers.append(receiver)

    def disconnect(self, receiver: Callable[[T], None]) -> None:
        """Remove a receiver."""
        self._receivers = [r for r in self._receivers if r is not receiver]

    def emit(self, sender: T) -> None:
        """Fire the signal, calling all receivers in order.

        Exceptions from individual receivers are logged and swallowed
        so one bad receiver never breaks others.
        """
        for receiver in self._receivers:
            try:
                receiver(sender)
            except Exception:
                logger.debug("Signal %s receiver failed", self.name, exc_info=True)

    def clear(self) -> None:
        """Remove all receivers."""
        self._receivers.clear()

    @property
    def receivers(self) -> list[Callable[[T], None]]:
        """Return copy of receivers list (for introspection)."""
        return list(self._receivers)

    def __len__(self) -> int:
        return len(self._receivers)

    def __bool__(self) -> bool:
        return bool(self._receivers)

    def __repr__(self) -> str:
        return f"Signal({self.name!r}, receivers={len(self._receivers)})"
