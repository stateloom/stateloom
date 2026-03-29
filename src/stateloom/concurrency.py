"""Concurrency utilities — propagate StateLoom session context to child threads."""

from __future__ import annotations

import contextvars
import functools
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from stateloom.intercept.unpatch import register_patch


def patch_threading() -> None:
    """Patch threading.Thread and ThreadPoolExecutor to propagate StateLoom session context.

    Python's ``ContextVar`` does not automatically propagate to child threads.
    This function monkey-patches ``threading.Thread`` so that the context
    (including the current StateLoom session) is copied at thread creation
    time and restored when the thread runs.

    Also patches ``ThreadPoolExecutor.submit`` to copy context at submit time,
    so callables executed in the pool inherit the caller's session context.

    Call ``unpatch_all()`` to restore the originals.
    """
    # --- threading.Thread ---
    _original_init = threading.Thread.__init__
    _original_run = threading.Thread.run

    def _patched_init(self: threading.Thread, *args: Any, **kwargs: Any) -> None:
        # Capture the current context at thread creation time
        self._stateloom_ctx = contextvars.copy_context()  # type: ignore[attr-defined]  # Dynamic attr on Thread for context propagation
        _original_init(self, *args, **kwargs)

    def _patched_run(self: threading.Thread) -> None:
        # Run the thread's target inside the captured context
        self._stateloom_ctx.run(_original_run, self)  # type: ignore[attr-defined]  # Dynamic attr on Thread for context propagation

    threading.Thread.__init__ = _patched_init  # type: ignore[assignment]  # Monkey-patching Thread for context propagation
    threading.Thread.run = _patched_run  # type: ignore[assignment]  # Monkey-patching Thread for context propagation
    register_patch(threading.Thread, "__init__", _original_init, "threading.Thread.__init__")
    register_patch(threading.Thread, "run", _original_run, "threading.Thread.run")

    # --- ThreadPoolExecutor ---
    _original_submit = ThreadPoolExecutor.submit

    def _patched_submit(
        self: ThreadPoolExecutor, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        ctx = contextvars.copy_context()

        @functools.wraps(fn)
        def _wrapper(*a: Any, **kw: Any) -> Any:
            return ctx.run(fn, *a, **kw)

        return _original_submit(self, _wrapper, *args, **kwargs)

    ThreadPoolExecutor.submit = _patched_submit  # type: ignore[assignment]  # Monkey-patching ThreadPoolExecutor for context propagation
    register_patch(ThreadPoolExecutor, "submit", _original_submit, "ThreadPoolExecutor.submit")
