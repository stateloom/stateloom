"""Tests for framework context bridge in the middleware pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

import stateloom
from stateloom.core.context import clear_framework_context, set_framework_context
from stateloom.core.event import LLMCallEvent
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.cost_tracker import CostTracker
from stateloom.middleware.pipeline import Pipeline
from stateloom.pricing.registry import PricingRegistry


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


class TestPipelineFrameworkContext:
    def test_pipeline_reads_framework_context_sync(self, gate):
        """execute_sync populates ctx._framework_context from the ContextVar."""
        pipeline = Pipeline()

        captured_ctx = {}

        async def capture_middleware(ctx, call_next):
            captured_ctx["fw"] = ctx._framework_context
            return await call_next(ctx)

        # Create a simple middleware that captures the context
        mw = MagicMock()
        mw.process = capture_middleware
        pipeline.add(mw)

        set_framework_context(
            {
                "langchain": {"run_id": "abc-123", "tags": ["test"]},
            }
        )

        try:
            with gate.session("test-pipeline-fw") as session:
                pipeline.execute_sync(
                    provider="openai",
                    method="chat.completions.create",
                    model="gpt-4o",
                    request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
                    session=session,
                    config=gate.config,
                    llm_call=lambda: {"choices": [{"message": {"content": "hello"}}]},
                )
        finally:
            clear_framework_context()

        assert captured_ctx["fw"] == {
            "langchain": {"run_id": "abc-123", "tags": ["test"]},
        }

    @pytest.mark.asyncio
    async def test_pipeline_reads_framework_context_async(self, gate):
        """execute_async populates ctx._framework_context from the ContextVar."""
        pipeline = Pipeline()

        captured_ctx = {}

        async def capture_middleware(ctx, call_next):
            captured_ctx["fw"] = ctx._framework_context
            return await call_next(ctx)

        mw = MagicMock()
        mw.process = capture_middleware
        pipeline.add(mw)

        set_framework_context(
            {
                "langgraph": {"node": "agent_1"},
            }
        )

        try:
            async with gate.async_session("test-pipeline-fw-async") as session:
                await pipeline.execute_async(
                    provider="openai",
                    method="chat.completions.create",
                    model="gpt-4o",
                    request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
                    session=session,
                    config=gate.config,
                    llm_call=lambda: {"choices": [{"message": {"content": "hello"}}]},
                )
        finally:
            clear_framework_context()

        assert captured_ctx["fw"] == {"langgraph": {"node": "agent_1"}}

    def test_pipeline_empty_framework_context(self, gate):
        """Empty ContextVar results in empty _framework_context on ctx."""
        pipeline = Pipeline()

        captured_ctx = {}

        async def capture_middleware(ctx, call_next):
            captured_ctx["fw"] = ctx._framework_context
            return await call_next(ctx)

        mw = MagicMock()
        mw.process = capture_middleware
        pipeline.add(mw)

        clear_framework_context()

        with gate.session("test-pipeline-empty-fw") as session:
            pipeline.execute_sync(
                provider="openai",
                method="chat.completions.create",
                model="gpt-4o",
                request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
                session=session,
                config=gate.config,
                llm_call=lambda: {"choices": [{"message": {"content": "hello"}}]},
            )

        assert captured_ctx["fw"] == {}

    def test_pipeline_context_is_shallow_copy(self, gate):
        """Pipeline shallow-copies the ContextVar to prevent mutation."""
        pipeline = Pipeline()

        captured_ctx = {}

        async def capture_middleware(ctx, call_next):
            captured_ctx["fw"] = ctx._framework_context
            # Mutate the copy — should not affect the ContextVar
            ctx._framework_context["mutated"] = True
            return await call_next(ctx)

        mw = MagicMock()
        mw.process = capture_middleware
        pipeline.add(mw)

        original = {"langchain": {"run_id": "xyz"}}
        set_framework_context(original)

        try:
            with gate.session("test-pipeline-copy") as session:
                pipeline.execute_sync(
                    provider="openai",
                    method="chat.completions.create",
                    model="gpt-4o",
                    request_kwargs={"messages": [{"role": "user", "content": "hi"}]},
                    session=session,
                    config=gate.config,
                    llm_call=lambda: {"choices": [{"message": {"content": "hello"}}]},
                )
        finally:
            clear_framework_context()

        # The ContextVar's original dict should not have been mutated
        assert "mutated" not in original


class TestCostTrackerFrameworkContext:
    def test_framework_context_flows_to_event_metadata(self, gate):
        """CostTracker merges _framework_context into event.metadata."""
        pricing = PricingRegistry()
        tracker = CostTracker(pricing)

        with gate.session("test-cost-fw") as session:
            ctx = MiddlewareContext(session=session, config=gate.config)
            ctx.provider = "openai"
            ctx.model = "gpt-4o"
            ctx.prompt_tokens = 100
            ctx.completion_tokens = 50
            ctx.latency_ms = 500.0
            ctx.request_kwargs = {"messages": [{"role": "user", "content": "hello"}]}
            ctx._framework_context = {
                "langchain": {
                    "run_id": "run-abc",
                    "chain_name": "RetrievalQA",
                    "tags": ["prod", "v2"],
                }
            }

            tracker._track_cost(ctx)

        assert len(ctx.events) == 1
        event = ctx.events[0]
        assert isinstance(event, LLMCallEvent)
        assert "langchain" in event.metadata
        assert event.metadata["langchain"]["run_id"] == "run-abc"
        assert event.metadata["langchain"]["chain_name"] == "RetrievalQA"
        assert event.metadata["langchain"]["tags"] == ["prod", "v2"]

    def test_no_framework_context_no_metadata_pollution(self, gate):
        """Empty _framework_context doesn't add keys to event.metadata."""
        pricing = PricingRegistry()
        tracker = CostTracker(pricing)

        with gate.session("test-cost-no-fw") as session:
            ctx = MiddlewareContext(session=session, config=gate.config)
            ctx.provider = "openai"
            ctx.model = "gpt-4o"
            ctx.prompt_tokens = 50
            ctx.completion_tokens = 25
            ctx.latency_ms = 200.0
            ctx.request_kwargs = {"messages": [{"role": "user", "content": "hi"}]}
            # _framework_context is empty by default

            tracker._track_cost(ctx)

        event = ctx.events[0]
        assert "langchain" not in event.metadata
        assert "langgraph" not in event.metadata
