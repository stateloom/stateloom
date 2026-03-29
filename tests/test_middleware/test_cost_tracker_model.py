"""Tests for CostTracker per-model cost breakdown population."""

from __future__ import annotations

import types
from dataclasses import field
from unittest.mock import MagicMock

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.cost_tracker import CostTracker
from stateloom.pricing.registry import PricingRegistry


def _make_ctx(model: str, prompt_tokens: int, completion_tokens: int) -> MiddlewareContext:
    session = Session(id="ct-model-test")
    config = StateLoomConfig()
    ctx = MiddlewareContext(session=session, config=config)
    ctx.model = model
    ctx.provider = "openai"
    ctx.prompt_tokens = prompt_tokens
    ctx.completion_tokens = completion_tokens
    ctx.request_kwargs = {"messages": [{"role": "user", "content": "hi"}]}
    # Simulate non-streaming response
    ctx.response = types.SimpleNamespace(
        usage=types.SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hello"))],
    )
    return ctx


@pytest.mark.asyncio
async def test_cost_tracker_populates_cost_by_model():
    pricing = PricingRegistry()
    pricing.register("gpt-4o-mini", input_per_token=0.00001, output_per_token=0.00002)
    tracker = CostTracker(pricing)

    ctx = _make_ctx("gpt-4o-mini", 100, 50)

    async def call_next(c):
        return c.response

    await tracker.process(ctx, call_next)

    assert "gpt-4o-mini" in ctx.session.cost_by_model
    assert ctx.session.cost_by_model["gpt-4o-mini"] > 0
    assert "gpt-4o-mini" in ctx.session.tokens_by_model
    assert ctx.session.tokens_by_model["gpt-4o-mini"]["prompt_tokens"] == 100
    assert ctx.session.tokens_by_model["gpt-4o-mini"]["completion_tokens"] == 50
    assert ctx.session.tokens_by_model["gpt-4o-mini"]["total_tokens"] == 150


@pytest.mark.asyncio
async def test_cost_tracker_populates_tokens_by_model_multi():
    pricing = PricingRegistry()
    pricing.register("gpt-4o-mini", input_per_token=0.00001, output_per_token=0.00002)
    pricing.register("gpt-4o", input_per_token=0.0001, output_per_token=0.0002)
    tracker = CostTracker(pricing)

    # First call — cheap model
    ctx1 = _make_ctx("gpt-4o-mini", 100, 50)
    session = ctx1.session

    async def call_next(c):
        return c.response

    await tracker.process(ctx1, call_next)

    # Second call — expensive model (reuse same session)
    ctx2 = MiddlewareContext(session=session, config=StateLoomConfig())
    ctx2.model = "gpt-4o"
    ctx2.provider = "openai"
    ctx2.prompt_tokens = 200
    ctx2.completion_tokens = 100
    ctx2.request_kwargs = {"messages": [{"role": "user", "content": "hi"}]}
    ctx2.response = types.SimpleNamespace(
        usage=types.SimpleNamespace(prompt_tokens=200, completion_tokens=100),
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hello"))],
    )

    await tracker.process(ctx2, call_next)

    assert len(session.cost_by_model) == 2
    assert "gpt-4o-mini" in session.cost_by_model
    assert "gpt-4o" in session.cost_by_model
    assert session.cost_by_model["gpt-4o"] > session.cost_by_model["gpt-4o-mini"]
    assert session.tokens_by_model["gpt-4o"]["prompt_tokens"] == 200
    assert session.tokens_by_model["gpt-4o-mini"]["prompt_tokens"] == 100
