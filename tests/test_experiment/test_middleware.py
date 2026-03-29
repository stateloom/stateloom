"""Tests for the experiment middleware."""

import asyncio

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.session import Session
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.experiment import ExperimentMiddleware


@pytest.fixture
def middleware():
    return ExperimentMiddleware(store=None)


def _make_ctx(
    provider="openai",
    model="gpt-4o",
    metadata=None,
    request_kwargs=None,
):
    session = Session(id="test-session")
    session.metadata = metadata or {}
    return MiddlewareContext(
        session=session,
        config=StateLoomConfig(dashboard=False, console_output=False),
        provider=provider,
        model=model,
        request_kwargs=request_kwargs or {"model": model, "messages": []},
    )


async def _passthrough(ctx):
    return ctx.response


class TestModelOverride:
    def test_override_model(self, middleware):
        ctx = _make_ctx(
            metadata={
                "experiment_variant_config": {
                    "name": "fast",
                    "model": "gpt-4o-mini",
                }
            }
        )
        result = asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.model == "gpt-4o-mini"
        assert ctx.request_kwargs["model"] == "gpt-4o-mini"

    def test_no_model_override(self, middleware):
        ctx = _make_ctx(
            metadata={
                "experiment_variant_config": {
                    "name": "control",
                }
            }
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.model == "gpt-4o"


class TestRequestOverrides:
    def test_merge_overrides(self, middleware):
        ctx = _make_ctx(
            request_kwargs={"model": "gpt-4o", "messages": [], "temperature": 0.7},
            metadata={
                "experiment_variant_config": {
                    "name": "cold",
                    "request_overrides": {"temperature": 0.1, "max_tokens": 500},
                }
            },
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.request_kwargs["temperature"] == 0.1
        assert ctx.request_kwargs["max_tokens"] == 500


class TestSystemPromptOverride:
    def test_openai_replace_existing(self, middleware):
        ctx = _make_ctx(
            provider="openai",
            request_kwargs={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "old prompt"},
                    {"role": "user", "content": "hello"},
                ],
            },
            metadata={
                "experiment_variant_config": {
                    "name": "new-prompt",
                    "request_overrides": {"system_prompt": "new prompt"},
                }
            },
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.request_kwargs["messages"][0]["content"] == "new prompt"
        assert ctx.request_kwargs["messages"][1]["role"] == "user"

    def test_openai_insert_system(self, middleware):
        ctx = _make_ctx(
            provider="openai",
            request_kwargs={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
            },
            metadata={
                "experiment_variant_config": {
                    "name": "new-prompt",
                    "request_overrides": {"system_prompt": "injected prompt"},
                }
            },
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.request_kwargs["messages"][0]["role"] == "system"
        assert ctx.request_kwargs["messages"][0]["content"] == "injected prompt"
        assert ctx.request_kwargs["messages"][1]["role"] == "user"

    def test_anthropic_system_kwarg(self, middleware):
        ctx = _make_ctx(
            provider="anthropic",
            request_kwargs={
                "model": "claude-3",
                "messages": [{"role": "user", "content": "hello"}],
            },
            metadata={
                "experiment_variant_config": {
                    "name": "new-prompt",
                    "request_overrides": {"system_prompt": "anthropic system"},
                }
            },
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.request_kwargs["system"] == "anthropic system"
        # Messages should not have a system message inserted
        assert ctx.request_kwargs["messages"][0]["role"] == "user"

    def test_gemini_system_instruction(self, middleware):
        ctx = _make_ctx(
            provider="gemini",
            request_kwargs={
                "model": "gemini-pro",
                "messages": [{"role": "user", "content": "hello"}],
            },
            metadata={
                "experiment_variant_config": {
                    "name": "new-prompt",
                    "request_overrides": {"system_prompt": "gemini system"},
                }
            },
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.request_kwargs["system_instruction"] == "gemini system"

    def test_fallback_provider(self, middleware):
        ctx = _make_ctx(
            provider="unknown",
            request_kwargs={
                "model": "custom-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
            metadata={
                "experiment_variant_config": {
                    "name": "test",
                    "request_overrides": {"system_prompt": "fallback system"},
                }
            },
        )
        asyncio.run(middleware.process(ctx, _passthrough))
        # Falls back to OpenAI-style
        assert ctx.request_kwargs["messages"][0]["role"] == "system"


class TestNoOp:
    def test_no_experiment_config(self, middleware):
        ctx = _make_ctx()
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.model == "gpt-4o"

    def test_empty_metadata(self, middleware):
        ctx = _make_ctx(metadata={})
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.model == "gpt-4o"


class TestFailOpen:
    def test_error_in_override_does_not_crash(self, middleware):
        ctx = _make_ctx(
            metadata={
                "experiment_variant_config": "invalid-not-a-dict",
            }
        )
        # Should not raise — fail-open
        asyncio.run(middleware.process(ctx, _passthrough))
        assert ctx.model == "gpt-4o"

    def test_error_in_system_prompt_does_not_crash(self, middleware):
        ctx = _make_ctx(
            provider="openai",
            request_kwargs={"model": "gpt-4o"},  # No messages key
            metadata={
                "experiment_variant_config": {
                    "name": "test",
                    "request_overrides": {"system_prompt": "crash?"},
                }
            },
        )
        # Should not raise
        asyncio.run(middleware.process(ctx, _passthrough))
