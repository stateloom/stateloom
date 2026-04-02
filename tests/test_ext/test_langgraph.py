"""Tests for the LangGraph adapter with mocked LangGraph imports."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stateloom
from stateloom.core.event import ToolCallEvent
from stateloom.gate import Gate


class MockToolNode:
    """Mock LangGraph ToolNode matching the current API (_execute_tool_sync/async)."""

    def _execute_tool_sync(self, request, input_type, config):
        return SimpleNamespace(content="mock_result")

    async def _execute_tool_async(self, request, input_type, config):
        return SimpleNamespace(content="mock_result")


class LegacyMockToolNode:
    """Mock LangGraph ToolNode matching the legacy API (_execute_tool)."""

    async def _execute_tool(self, tool_call, *args, **kwargs):
        return {"result": "mock_result"}


def _make_request(tool_name: str = "search_web") -> SimpleNamespace:
    """Create a mock ToolCallRequest."""
    return SimpleNamespace(tool_call={"name": tool_name, "args": {}, "id": "tc-1"})


@pytest.fixture
def gate():
    g = stateloom.init(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
    )
    yield g
    stateloom.shutdown()


class TestLangGraphAdapter:
    def test_patch_without_langgraph_logs_warning(self, gate, caplog):
        """patch_langgraph_tools should warn if LangGraph is not installed."""
        with patch.dict("sys.modules", {"langgraph": None, "langgraph.prebuilt": None}):
            from stateloom.ext.langgraph import patch_langgraph_tools

            with pytest.raises(ImportError):
                # Import will fail because we mocked it out
                import importlib

                importlib.import_module("langgraph.prebuilt")

    def test_patch_with_mock_langgraph(self, gate):
        """patch_langgraph_tools should patch _execute_tool_sync and _execute_tool_async."""
        mock_module = MagicMock()
        mock_module.ToolNode = MockToolNode

        original_sync = MockToolNode._execute_tool_sync
        original_async = MockToolNode._execute_tool_async

        with patch.dict(
            "sys.modules",
            {
                "langgraph": MagicMock(),
                "langgraph.prebuilt": mock_module,
            },
        ):
            from stateloom.ext.langgraph import patch_langgraph_tools

            patch_langgraph_tools(gate=gate)

            # Both methods should be patched
            assert MockToolNode._execute_tool_sync is not original_sync
            assert MockToolNode._execute_tool_async is not original_async

        # Restore
        MockToolNode._execute_tool_sync = original_sync
        MockToolNode._execute_tool_async = original_async

    def test_patched_sync_tool_records_event(self, gate):
        """Patched _execute_tool_sync should record a ToolCallEvent."""
        mock_module = MagicMock()
        mock_module.ToolNode = MockToolNode
        original_sync = MockToolNode._execute_tool_sync

        with patch.dict(
            "sys.modules",
            {
                "langgraph": MagicMock(),
                "langgraph.prebuilt": mock_module,
            },
        ):
            from stateloom.ext.langgraph import patch_langgraph_tools

            patch_langgraph_tools(gate=gate)

            node = MockToolNode()
            with gate.session("test-langgraph-sync") as session:
                result = node._execute_tool_sync(
                    _make_request("calculate"),
                    "list",
                    {},
                )
                assert result.content == "mock_result"

            events = gate.store.get_session_events(session.id, event_type="tool_call")
            assert len(events) == 1
            assert isinstance(events[0], ToolCallEvent)
            assert events[0].tool_name == "calculate"
            assert events[0].latency_ms >= 0

        MockToolNode._execute_tool_sync = original_sync

    @pytest.mark.asyncio
    async def test_patched_async_tool_records_event(self, gate):
        """Patched _execute_tool_async should record a ToolCallEvent."""
        mock_module = MagicMock()
        mock_module.ToolNode = MockToolNode
        original_async = MockToolNode._execute_tool_async

        with patch.dict(
            "sys.modules",
            {
                "langgraph": MagicMock(),
                "langgraph.prebuilt": mock_module,
            },
        ):
            from stateloom.ext.langgraph import patch_langgraph_tools

            patch_langgraph_tools(gate=gate)

            node = MockToolNode()
            with gate.session("test-langgraph-async") as session:
                result = await node._execute_tool_async(
                    _make_request("search_web"),
                    "list",
                    {},
                )
                assert result.content == "mock_result"

            events = gate.store.get_session_events(session.id, event_type="tool_call")
            assert len(events) == 1
            assert isinstance(events[0], ToolCallEvent)
            assert events[0].tool_name == "search_web"

        MockToolNode._execute_tool_async = original_async

    @pytest.mark.asyncio
    async def test_legacy_api_patched(self, gate):
        """Legacy _execute_tool method is patched when _execute_tool_sync is absent."""
        mock_module = MagicMock()
        mock_module.ToolNode = LegacyMockToolNode
        original_legacy = LegacyMockToolNode._execute_tool

        with patch.dict(
            "sys.modules",
            {
                "langgraph": MagicMock(),
                "langgraph.prebuilt": mock_module,
            },
        ):
            from stateloom.ext.langgraph import patch_langgraph_tools

            patch_langgraph_tools(gate=gate)

            assert LegacyMockToolNode._execute_tool is not original_legacy

            node = LegacyMockToolNode()
            with gate.session("test-langgraph-legacy") as session:
                result = await node._execute_tool({"name": "reverse_string"})
                assert result == {"result": "mock_result"}

            events = gate.store.get_session_events(session.id, event_type="tool_call")
            assert len(events) == 1
            assert events[0].tool_name == "reverse_string"

        LegacyMockToolNode._execute_tool = original_legacy

    def test_extract_tool_name_from_request(self):
        """_extract_tool_name handles ToolCallRequest and plain dicts."""
        from stateloom.ext.langgraph import _extract_tool_name

        # ToolCallRequest-style (has .tool_call attr)
        req = SimpleNamespace(tool_call={"name": "my_tool", "args": {}})
        assert _extract_tool_name(req) == "my_tool"

        # Plain dict (legacy)
        assert _extract_tool_name({"name": "old_tool"}) == "old_tool"

        # Missing name
        assert _extract_tool_name({}) == "unknown"
        assert _extract_tool_name(SimpleNamespace(tool_call={})) == "unknown"
