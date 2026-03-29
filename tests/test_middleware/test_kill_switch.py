"""Tests for the kill switch middleware."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from stateloom.core.config import KillSwitchRule, StateLoomConfig
from stateloom.core.errors import StateLoomKillSwitchError
from stateloom.core.event import KillSwitchEvent
from stateloom.core.session import Session
from stateloom.core.types import EventType
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.kill_switch import KillSwitchMiddleware
from stateloom.store.memory_store import MemoryStore


def _make_ctx(
    provider: str = "openai",
    model: str = "gpt-4",
    **config_overrides,
) -> MiddlewareContext:
    defaults = {"console_output": False}
    defaults.update(config_overrides)
    return MiddlewareContext(
        session=Session(id="test-session"),
        config=StateLoomConfig(**defaults),
        provider=provider,
        model=model,
        request_kwargs={"messages": [{"role": "user", "content": "hello"}]},
    )


def _make_config(**overrides) -> StateLoomConfig:
    defaults = {"console_output": False}
    defaults.update(overrides)
    return StateLoomConfig(**defaults)


class TestKillSwitchPassThrough:
    """Kill switch is inactive and no rules — traffic passes through."""

    async def test_inactive_no_rules(self):
        config = _make_config(kill_switch_active=False)
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx()
        called = False

        async def call_next(c):
            nonlocal called
            called = True
            return "response"

        result = await mw.process(ctx, call_next)
        assert result == "response"
        assert called
        assert len(ctx.events) == 0

    async def test_inactive_non_matching_rules(self):
        """Rules exist but none match the request."""
        config = _make_config(
            kill_switch_active=False,
            kill_switch_rules=[KillSwitchRule(model="claude-*")],
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx(model="gpt-4")

        async def call_next(c):
            return "ok"

        result = await mw.process(ctx, call_next)
        assert result == "ok"


class TestGlobalKillSwitch:
    """Global kill switch active — blocks everything."""

    async def test_raises_error(self):
        config = _make_config(kill_switch_active=True)
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError):
            await mw.process(ctx, call_next)

    async def test_custom_message(self):
        msg = "Maintenance in progress"
        config = _make_config(kill_switch_active=True, kill_switch_message=msg)
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError, match="Maintenance"):
            await mw.process(ctx, call_next)

    async def test_event_created(self):
        config = _make_config(kill_switch_active=True)
        store = MemoryStore()
        mw = KillSwitchMiddleware(config, store)
        ctx = _make_ctx()

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError):
            await mw.process(ctx, call_next)

        assert len(ctx.events) == 1
        event = ctx.events[0]
        assert isinstance(event, KillSwitchEvent)
        assert event.event_type == EventType.KILL_SWITCH
        assert event.reason == "kill_switch_active"
        assert event.blocked_model == "gpt-4"
        assert event.blocked_provider == "openai"

    async def test_events_persisted_to_store(self):
        config = _make_config(kill_switch_active=True)
        store = MemoryStore()
        mw = KillSwitchMiddleware(config, store)
        ctx = _make_ctx()

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError):
            await mw.process(ctx, call_next)

        events = store.get_session_events("test-session")
        assert len(events) == 1
        assert isinstance(events[0], KillSwitchEvent)


class TestResponseMode:
    """kill_switch_response_mode == 'response' returns static response."""

    async def test_response_mode_skips_call(self):
        config = _make_config(
            kill_switch_active=True,
            kill_switch_response_mode="response",
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            # Should be called because response mode goes through pipeline
            assert c.skip_call is True
            return c.cached_response

        result = await mw.process(ctx, call_next)
        assert result == {"kill_switch": True, "message": config.kill_switch_message}
        assert ctx.skip_call is True

    async def test_response_mode_does_not_raise(self):
        config = _make_config(
            kill_switch_active=True,
            kill_switch_response_mode="response",
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return c.cached_response

        # Should NOT raise
        result = await mw.process(ctx, call_next)
        assert result is not None


class TestRuleMatching:
    """Granular rule matching with glob patterns."""

    async def test_exact_model_match(self):
        config = _make_config(
            kill_switch_rules=[KillSwitchRule(model="gpt-4")],
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx(model="gpt-4")

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError):
            await mw.process(ctx, call_next)

    async def test_glob_model_match(self):
        config = _make_config(
            kill_switch_rules=[KillSwitchRule(model="gpt-*")],
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx(model="gpt-4-turbo")

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError):
            await mw.process(ctx, call_next)

    async def test_glob_model_no_match(self):
        config = _make_config(
            kill_switch_rules=[KillSwitchRule(model="gpt-*")],
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx(model="claude-3-opus")

        async def call_next(c):
            return "ok"

        result = await mw.process(ctx, call_next)
        assert result == "ok"

    async def test_provider_match(self):
        config = _make_config(
            kill_switch_rules=[KillSwitchRule(provider="anthropic")],
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx(provider="anthropic", model="claude-3-opus")

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError):
            await mw.process(ctx, call_next)

    async def test_environment_match(self):
        config = _make_config(
            kill_switch_rules=[KillSwitchRule(environment="production")],
            kill_switch_environment="production",
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError):
            await mw.process(ctx, call_next)

    async def test_environment_no_match(self):
        config = _make_config(
            kill_switch_rules=[KillSwitchRule(environment="production")],
            kill_switch_environment="staging",
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return "ok"

        result = await mw.process(ctx, call_next)
        assert result == "ok"

    async def test_agent_version_match(self):
        config = _make_config(
            kill_switch_rules=[KillSwitchRule(agent_version="v2.1.0")],
            kill_switch_agent_version="v2.1.0",
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError):
            await mw.process(ctx, call_next)

    async def test_empty_rule_matches_nothing(self):
        """A rule with no filters should not match anything."""
        config = _make_config(
            kill_switch_rules=[KillSwitchRule()],
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return "ok"

        result = await mw.process(ctx, call_next)
        assert result == "ok"

    async def test_multiple_rules_first_match_wins(self):
        """First matching rule should be used."""
        config = _make_config(
            kill_switch_rules=[
                KillSwitchRule(model="gpt-4", message="GPT-4 blocked"),
                KillSwitchRule(provider="openai", message="OpenAI blocked"),
            ],
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx(model="gpt-4", provider="openai")

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError, match="GPT-4 blocked"):
            await mw.process(ctx, call_next)

    async def test_rule_specific_message_override(self):
        config = _make_config(
            kill_switch_message="Global message",
            kill_switch_rules=[KillSwitchRule(model="gpt-4", message="Rule-specific message")],
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx(model="gpt-4")

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError, match="Rule-specific message"):
            await mw.process(ctx, call_next)

    async def test_rule_matched_event_has_rule_dict(self):
        rule = KillSwitchRule(model="gpt-*", reason="cost overrun")
        config = _make_config(kill_switch_rules=[rule])
        store = MemoryStore()
        mw = KillSwitchMiddleware(config, store)
        ctx = _make_ctx(model="gpt-4")

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError):
            await mw.process(ctx, call_next)

        event = ctx.events[0]
        assert isinstance(event, KillSwitchEvent)
        assert event.reason == "cost overrun"
        assert event.matched_rule["model"] == "gpt-*"

    async def test_rule_with_response_mode(self):
        """Rule match with response mode should return static response."""
        config = _make_config(
            kill_switch_response_mode="response",
            kill_switch_rules=[KillSwitchRule(model="gpt-4", message="Blocked by rule")],
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx(model="gpt-4")

        async def call_next(c):
            return c.cached_response

        result = await mw.process(ctx, call_next)
        assert result == {"kill_switch": True, "message": "Blocked by rule"}


class TestKillSwitchWebhook:
    """Kill switch webhook firing tests."""

    async def test_webhook_fired_on_global_kill_switch(self):
        """Webhook fires when global kill switch is active and URL is configured."""
        config = _make_config(
            kill_switch_active=True,
            kill_switch_webhook_url="https://hooks.example.com/ks",
        )
        store = MemoryStore()
        mw = KillSwitchMiddleware(config, store)
        ctx = _make_ctx()

        async def call_next(c):
            raise AssertionError("Should not be called")

        with patch.object(KillSwitchMiddleware, "_fire_webhook") as mock_fire:
            with pytest.raises(StateLoomKillSwitchError):
                await mw.process(ctx, call_next)

            mock_fire.assert_called_once_with(
                "https://hooks.example.com/ks",
                "test-session",
                "kill_switch_active",
                config.kill_switch_message,
                "gpt-4",
                "openai",
                {},
            )

    async def test_webhook_not_fired_when_no_url(self):
        """No webhook thread spawned when URL is empty."""
        config = _make_config(kill_switch_active=True)
        store = MemoryStore()
        mw = KillSwitchMiddleware(config, store)
        ctx = _make_ctx()

        async def call_next(c):
            raise AssertionError("Should not be called")

        with patch("stateloom.middleware.kill_switch.threading.Thread") as mock_thread:
            with pytest.raises(StateLoomKillSwitchError):
                await mw.process(ctx, call_next)

            mock_thread.assert_not_called()

    async def test_webhook_fired_on_rule_match(self):
        """Webhook fires for granular rule match."""
        rule = KillSwitchRule(model="gpt-*", reason="cost_overrun", message="GPT-4 suspended")
        config = _make_config(
            kill_switch_rules=[rule],
            kill_switch_webhook_url="https://hooks.example.com/ks",
        )
        store = MemoryStore()
        mw = KillSwitchMiddleware(config, store)
        ctx = _make_ctx(model="gpt-4")

        async def call_next(c):
            raise AssertionError("Should not be called")

        with patch.object(KillSwitchMiddleware, "_fire_webhook") as mock_fire:
            with pytest.raises(StateLoomKillSwitchError):
                await mw.process(ctx, call_next)

            args = mock_fire.call_args[0]
            assert args[0] == "https://hooks.example.com/ks"
            assert args[2] == "cost_overrun"  # reason
            assert args[3] == "GPT-4 suspended"  # message
            assert args[6]["model"] == "gpt-*"  # matched_rule dict

    async def test_event_records_webhook_metadata(self):
        """Event has webhook_fired=True and webhook_url when URL is configured."""
        config = _make_config(
            kill_switch_active=True,
            kill_switch_webhook_url="https://hooks.example.com/ks",
        )
        store = MemoryStore()
        mw = KillSwitchMiddleware(config, store)
        ctx = _make_ctx()

        async def call_next(c):
            raise AssertionError("Should not be called")

        with patch.object(KillSwitchMiddleware, "_fire_webhook"):
            with pytest.raises(StateLoomKillSwitchError):
                await mw.process(ctx, call_next)

        event = ctx.events[0]
        assert isinstance(event, KillSwitchEvent)
        assert event.webhook_fired is True
        assert event.webhook_url == "https://hooks.example.com/ks"

    async def test_event_no_webhook_metadata_when_no_url(self):
        """Event has webhook_fired=False when no URL is configured."""
        config = _make_config(kill_switch_active=True)
        store = MemoryStore()
        mw = KillSwitchMiddleware(config, store)
        ctx = _make_ctx()

        async def call_next(c):
            raise AssertionError("Should not be called")

        with pytest.raises(StateLoomKillSwitchError):
            await mw.process(ctx, call_next)

        event = ctx.events[0]
        assert isinstance(event, KillSwitchEvent)
        assert event.webhook_fired is False
        assert event.webhook_url == ""

    async def test_webhook_fired_in_response_mode(self):
        """Webhook fires even in response mode."""
        config = _make_config(
            kill_switch_active=True,
            kill_switch_response_mode="response",
            kill_switch_webhook_url="https://hooks.example.com/ks",
        )
        mw = KillSwitchMiddleware(config)
        ctx = _make_ctx()

        async def call_next(c):
            return c.cached_response

        with patch.object(KillSwitchMiddleware, "_fire_webhook") as mock_fire:
            result = await mw.process(ctx, call_next)

        assert result is not None
        mock_fire.assert_called_once()
        assert mock_fire.call_args[0][0] == "https://hooks.example.com/ks"
