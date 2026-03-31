"""LangGraph adapter — automatically wraps LangGraph tool calls with StateLoom.

Usage:
    from stateloom.ext.langgraph import patch_langgraph_tools
    patch_langgraph_tools()
"""

from __future__ import annotations

import functools
import logging
import time
from typing import TYPE_CHECKING, Any, cast

from stateloom.core.event import ToolCallEvent
from stateloom.intercept.unpatch import register_patch

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.ext.langgraph")


def _extract_tool_name(request: Any) -> str:
    """Extract tool name from a ToolCallRequest or legacy dict/object."""
    # New API: ToolCallRequest with .tool_call dict
    tool_call = getattr(request, "tool_call", request)
    if isinstance(tool_call, dict):
        return cast(str, tool_call.get("name", "unknown"))
    return cast(str, getattr(tool_call, "name", "unknown"))


def patch_langgraph_tools(gate: Gate | None = None) -> None:
    """Patch LangGraph's ToolNode to wrap tool invocations with StateLoom.

    This allows LangGraph users to get tool-level observability without
    decorating each tool function with @gate.tool().

    Supports both the current LangGraph API (``_execute_tool_sync`` /
    ``_execute_tool_async``) and the legacy ``_execute_tool`` method.

    Args:
        gate: The Gate instance. If None, uses the global instance.
    """
    if gate is None:
        import stateloom

        gate = stateloom.get_gate()

    if gate is None:
        logger.warning("[StateLoom] No Gate instance available — LangGraph patch skipped.")
        return

    try:
        from langgraph.prebuilt import ToolNode
    except ImportError:
        logger.warning(
            "[StateLoom] LangGraph not installed. Install with: pip install stateloom[langgraph]"
        )
        return

    def _record_tool(tool_name: str, elapsed_ms: float) -> None:
        """Record a ToolCallEvent for the current session."""
        assert gate is not None
        session = gate.get_or_create_session()
        step = session.next_step()
        event = ToolCallEvent(
            session_id=session.id,
            step=step,
            tool_name=tool_name,
            mutates_state=False,
            latency_ms=elapsed_ms,
        )
        session.call_count += 1
        gate.store.save_event(event)

    # --- Current API: _execute_tool_sync + _execute_tool_async ---
    if hasattr(ToolNode, "_execute_tool_sync"):
        original_sync = ToolNode._execute_tool_sync

        @functools.wraps(original_sync)
        def patched_sync(self_node: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            tool_name = _extract_tool_name(request)
            start = time.perf_counter()
            try:
                return original_sync(self_node, request, *args, **kwargs)
            finally:
                _record_tool(tool_name, (time.perf_counter() - start) * 1000)

        ToolNode._execute_tool_sync = patched_sync  # type: ignore[assignment]
        register_patch(
            ToolNode,
            "_execute_tool_sync",
            original_sync,
            "langgraph.ToolNode._execute_tool_sync (stateloom adapter)",
        )
        logger.info("[StateLoom] LangGraph ToolNode._execute_tool_sync patched")

    if hasattr(ToolNode, "_execute_tool_async"):
        original_async = ToolNode._execute_tool_async

        @functools.wraps(original_async)
        async def patched_async(self_node: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            tool_name = _extract_tool_name(request)
            start = time.perf_counter()
            try:
                return await original_async(self_node, request, *args, **kwargs)
            finally:
                _record_tool(tool_name, (time.perf_counter() - start) * 1000)

        ToolNode._execute_tool_async = patched_async  # type: ignore[assignment]
        register_patch(
            ToolNode,
            "_execute_tool_async",
            original_async,
            "langgraph.ToolNode._execute_tool_async (stateloom adapter)",
        )
        logger.info("[StateLoom] LangGraph ToolNode._execute_tool_async patched")

    # --- Legacy API: _execute_tool (older LangGraph versions) ---
    elif hasattr(ToolNode, "_execute_tool"):
        original_legacy = ToolNode._execute_tool

        @functools.wraps(original_legacy)
        async def patched_legacy(self_node: Any, tool_call: Any, *args: Any, **kwargs: Any) -> Any:
            tool_name = _extract_tool_name(tool_call)
            start = time.perf_counter()
            try:
                return await original_legacy(self_node, tool_call, *args, **kwargs)
            finally:
                _record_tool(tool_name, (time.perf_counter() - start) * 1000)

        ToolNode._execute_tool = patched_legacy
        register_patch(
            ToolNode,
            "_execute_tool",
            original_legacy,
            "langgraph.ToolNode._execute_tool (stateloom adapter)",
        )
        logger.info("[StateLoom] LangGraph ToolNode._execute_tool patched (legacy)")
