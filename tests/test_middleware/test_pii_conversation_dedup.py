"""Tests for PII scanner conversation history handling.

Validates that the PII scanner handles CLI-style flows where the full
conversation history is resent on every turn.

- BLOCK mode (new PII): blocks the entire request, creates event.
- BLOCK mode (old PII): strips the offending message from the request
  entirely — PII never reaches the LLM, conversation isn't stuck.
- REDACT mode: re-redacts content and creates events on every call.
- AUDIT mode: logs events on every call.
"""

from __future__ import annotations

import pytest
from stateloom.core.config import PIIRule, StateLoomConfig
from stateloom.core.errors import StateLoomPIIBlockedError
from stateloom.core.session import Session
from stateloom.core.types import FailureAction, PIIMode
from stateloom.middleware.base import MiddlewareContext
from stateloom.middleware.pii_scanner import PIIScannerMiddleware
from stateloom.store.memory_store import MemoryStore


def _make_ctx(
    messages: list[dict],
    session: Session | None = None,
    system: str = "",
    *,
    proxy: bool = True,
) -> MiddlewareContext:
    """Create a MiddlewareContext with the given messages.

    *proxy* defaults to ``True`` because the existing tests exercise the
    CLI/proxy path (Phase 1 strip, active-turn detection, cooldown timers).
    Set ``proxy=False`` to exercise the SDK path.
    """
    if session is None:
        session = Session(id="test-session")
    kwargs: dict = {"messages": messages}
    if system:
        kwargs["system"] = system
    if proxy:
        kwargs["_proxy"] = True
    return MiddlewareContext(
        session=session,
        config=StateLoomConfig(),
        provider="openai",
        model="gpt-4",
        request_kwargs=kwargs,
    )


async def _passthrough(ctx: MiddlewareContext):
    """No-op call_next that returns a sentinel."""
    return "LLM_RESPONSE"


# ---------------------------------------------------------------------------
# BLOCK mode — new PII blocks, old PII stripped
# ---------------------------------------------------------------------------


class TestBlockNewPII:
    """New BLOCK-mode PII blocks the entire request."""

    async def test_first_ssn_blocked(self):
        """New SSN content → blocked with event."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 1

    async def test_new_ssn_at_any_position_blocks(self):
        """New SSN anywhere in message list → blocks."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "my ssn is 123-45-6789"},
            ],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 1


class TestBlockOldPIIStripped:
    """Old BLOCK-mode PII → message stripped, request continues."""

    async def test_old_ssn_stripped_new_message_passes(self):
        """Turn 2: SSN in history stripped, clean new message passes."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        # Turn 1: SSN blocked
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Turn 2: SSN in history + new clean message
        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "my ssn is 123-45-6789"},
                {"role": "user", "content": "what's the weather like?"},
            ],
            session=session,
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        # SSN message was stripped
        assert len(ctx2.request_kwargs["messages"]) == 1
        assert ctx2.request_kwargs["messages"][0]["content"] == "what's the weather like?"
        # No new events (SSN was already evented on turn 1)
        assert len(ctx2.events) == 0

    async def test_old_ssn_stripped_preserves_other_messages(self):
        """Multiple messages with old SSN in the middle → only SSN msg stripped."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        # Turn 1: block
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Turn 2: old SSN at index 1, clean messages around it
        ctx2 = _make_ctx(
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "my ssn is 123-45-6789"},
                {"role": "assistant", "content": "I can't help with that"},
                {"role": "user", "content": "tell me a joke"},
            ],
            session=session,
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        remaining = ctx2.request_kwargs["messages"]
        assert len(remaining) == 3
        assert remaining[0]["content"] == "You are helpful"
        assert remaining[1]["content"] == "I can't help with that"
        assert remaining[2]["content"] == "tell me a joke"

    async def test_retry_same_messages_stripped_during_cooldown(self):
        """CLI retry (< 2s after block) → all messages stripped → blocked."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        # Turn 1: blocked (sets cooldown timestamp)
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Retry (immediately, within cooldown) → all messages stripped → blocked
        # (empty messages would crash the provider SDK, so we raise instead)
        ctx2 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx2, _passthrough)

    async def test_retry_after_cooldown_stripped(self):
        """Retry after cooldown expires → all messages stripped → blocked.

        Phase 1 strips messages containing previously-blocked PII. When all
        messages are stripped, we raise StateLoomPIIBlockedError instead of
        forwarding an empty request that would crash the provider SDK.
        """
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Simulate cooldown expired (set timestamp to 3s ago)
        session.metadata["_pii_last_block_ts"] = session.metadata["_pii_last_block_ts"] - 3.0

        ctx2 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx2, _passthrough)

    async def test_assistant_prefill_with_ssn_stripped_during_cooldown(self):
        """CLI sends SSN + empty assistant prefill during cooldown → all stripped → blocked."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # SSN message stripped + orphaned assistant cleaned up → empty → blocked
        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "my ssn is 123-45-6789"},
                {"role": "assistant", "content": ""},
            ],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx2, _passthrough)

    async def test_assistant_prefill_list_content_stripped_during_cooldown(self):
        """CLI sends SSN + list-format assistant prefill during cooldown."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # SSN stripped + orphaned list-format assistant cleaned up → empty → blocked
        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "my ssn is 123-45-6789"},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx2, _passthrough)

    async def test_nonempty_assistant_preserved_after_strip(self):
        """After stripping SSN at index 0, orphaned leading assistant is also
        stripped (it becomes invalid as a conversation opener).  The clean
        user message at the end is preserved."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # History: SSN + real assistant reply + new user message
        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "my ssn is 123-45-6789"},
                {"role": "assistant", "content": "I cannot process that."},
                {"role": "user", "content": "what time is it?"},
            ],
            session=session,
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        remaining = ctx2.request_kwargs["messages"]
        # SSN stripped + orphaned leading assistant stripped → only the clean user message
        assert len(remaining) == 1
        assert remaining[0]["content"] == "what time is it?"

    async def test_old_ssn_at_different_index_stripped(self):
        """Same SSN shifted to different array position → still stripped."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Same SSN at index 2 now
        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "my ssn is 123-45-6789"},
                {"role": "user", "content": "what time is it?"},
            ],
            session=session,
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        remaining = ctx2.request_kwargs["messages"]
        assert len(remaining) == 3
        contents = [m["content"] for m in remaining]
        assert "my ssn is 123-45-6789" not in contents
        assert "what time is it?" in contents


# ---------------------------------------------------------------------------
# REDACT mode dedup
# ---------------------------------------------------------------------------


class TestRedactDedup:
    """REDACT mode creates events on every call and always redacts."""

    async def test_first_redact_creates_event(self):
        """New content with REDACT PII → event created, content redacted."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.REDACT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx = _make_ctx(
            [{"role": "user", "content": "My email is test@secret.com"}],
            session=session,
        )
        await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 1
        assert "test@secret.com" not in ctx.request_kwargs["messages"][0]["content"]

    async def test_old_redact_suppresses_duplicate_event(self):
        """Old content with REDACT PII → re-redacted silently (no dup event)."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.REDACT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "My email is test@secret.com"}],
            session=session,
        )
        await scanner._do_process(ctx1, _passthrough)
        assert len(ctx1.events) == 1

        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "My email is test@secret.com"},
                {"role": "user", "content": "Thanks!"},
            ],
            session=session,
        )
        await scanner._do_process(ctx2, _passthrough)
        # Duplicate event suppressed — same value already seen
        assert len(ctx2.events) == 0
        # Content is still redacted even without an event
        assert "test@secret.com" not in ctx2.request_kwargs["messages"][0]["content"]

    async def test_new_redact_value_creates_event(self):
        """A NEW email address in history still creates an event."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.REDACT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "My email is test@secret.com"}],
            session=session,
        )
        await scanner._do_process(ctx1, _passthrough)
        assert len(ctx1.events) == 1

        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "My email is test@secret.com"},
                {"role": "user", "content": "Also: other@secret.com"},
            ],
            session=session,
        )
        await scanner._do_process(ctx2, _passthrough)
        # Old value suppressed, but new value (other@secret.com) creates event
        assert len(ctx2.events) == 1
        assert ctx2.events[0].metadata.get("redacted_preview", "").startswith("o")


# ---------------------------------------------------------------------------
# AUDIT mode dedup
# ---------------------------------------------------------------------------


class TestAuditDedup:
    """AUDIT mode deduplicates events for already-seen PII values."""

    async def test_first_audit_creates_event(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx = _make_ctx(
            [{"role": "user", "content": "Email: user@example.com"}],
            session=session,
        )
        await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 1

    async def test_old_audit_suppresses_duplicate_event(self):
        """Same PII in history on second call → no duplicate event."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "Email: user@example.com"}],
            session=session,
        )
        await scanner._do_process(ctx1, _passthrough)

        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "Email: user@example.com"},
                {"role": "user", "content": "Hello!"},
            ],
            session=session,
        )
        await scanner._do_process(ctx2, _passthrough)
        # Duplicate event suppressed — same value already seen in history
        assert len(ctx2.events) == 0

    async def test_new_audit_value_in_active_turn_creates_event(self):
        """Same PII typed again in the active turn → event emitted."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "Email: user@example.com"}],
            session=session,
        )
        await scanner._do_process(ctx1, _passthrough)
        assert len(ctx1.events) == 1

        # Same email in both history AND active turn
        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "Email: user@example.com"},
                {"role": "user", "content": "Repeat: user@example.com"},
            ],
            session=session,
        )
        await scanner._do_process(ctx2, _passthrough)
        # History dup suppressed, but active turn always emits
        assert len(ctx2.events) == 1

    async def test_multiple_pii_values_create_events(self):
        """Multiple distinct PII values across messages all create events."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "Email: first@example.com"}],
            session=session,
        )
        await scanner._do_process(ctx1, _passthrough)

        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "Email: first@example.com"},
                {"role": "user", "content": "Another: second@example.com"},
            ],
            session=session,
        )
        await scanner._do_process(ctx2, _passthrough)
        # first@ suppressed (already seen), second@ is new → 1 event
        assert len(ctx2.events) == 1


# ---------------------------------------------------------------------------
# List content format (Anthropic multi-part)
# ---------------------------------------------------------------------------


class TestListContentFormat:
    """Anthropic multi-part content (list-of-blocks) is handled correctly."""

    async def test_list_content_pii_detected(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [{"role": "user", "content": [{"type": "text", "text": "SSN: 123-45-6789"}]}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 1

    async def test_list_content_old_ssn_stripped(self):
        """Old list-format SSN → stripped from request."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": [{"type": "text", "text": "SSN: 123-45-6789"}]}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        ctx2 = _make_ctx(
            [
                {"role": "user", "content": [{"type": "text", "text": "SSN: 123-45-6789"}]},
                {"role": "user", "content": "hello"},
            ],
            session=session,
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        assert len(ctx2.request_kwargs["messages"]) == 1
        assert ctx2.request_kwargs["messages"][0]["content"] == "hello"


# ---------------------------------------------------------------------------
# cache_control ignored in hashing
# ---------------------------------------------------------------------------


class TestCacheControlIgnored:
    """Claude CLI adds cache_control to messages — hash must ignore it."""

    async def test_cache_control_added_later_still_recognized_as_old(self):
        """Same text with cache_control added → still recognized as old, stripped."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": [{"type": "text", "text": "SSN: 123-45-6789"}]}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Same text with cache_control added → old → stripped
        ctx2 = _make_ctx(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "SSN: 123-45-6789",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": "hello"},
            ],
            session=session,
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        assert len(ctx2.request_kwargs["messages"]) == 1
        assert ctx2.request_kwargs["messages"][0]["content"] == "hello"
        assert len(ctx2.events) == 0


# ---------------------------------------------------------------------------
# System prompt dedup
# ---------------------------------------------------------------------------


class TestSystemPromptDedup:
    """System prompt scanning deduplicates events for already-seen PII."""

    async def test_same_system_prompt_suppressed_on_second_call(self):
        """Same system prompt PII → event on first call, suppressed on second."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "Hello"}],
            session=session,
            system="Contact support@company.com for help",
        )
        await scanner._do_process(ctx1, _passthrough)
        assert len(ctx1.events) == 1

        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "What time is it?"},
            ],
            session=session,
            system="Contact support@company.com for help",
        )
        await scanner._do_process(ctx2, _passthrough)
        # Duplicate suppressed — same system prompt PII already seen
        assert len(ctx2.events) == 0

    async def test_changed_system_prompt_rescanned(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "Hello"}],
            session=session,
            system="Contact support@company.com for help",
        )
        await scanner._do_process(ctx1, _passthrough)

        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "Hi"},
            ],
            session=session,
            system="Contact admin@newdomain.com for help",
        )
        await scanner._do_process(ctx2, _passthrough)
        assert len(ctx2.events) == 1
        assert ctx2.events[0].pii_field == "system"

    async def test_old_blocked_system_prompt_stripped(self):
        """Old system prompt with BLOCK PII → stripped from request."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "hello"}],
            session=session,
            system="Admin SSN: 123-45-6789",
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Same system prompt on next turn → stripped
        ctx2 = _make_ctx(
            [{"role": "user", "content": "what time is it?"}],
            session=session,
            system="Admin SSN: 123-45-6789",
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        assert "system" not in ctx2.request_kwargs


# ---------------------------------------------------------------------------
# Duplicate content in one request
# ---------------------------------------------------------------------------


class TestDuplicateContent:
    """Same content at multiple indices in one request."""

    async def test_duplicate_ssn_one_event(self):
        """SSN at 2 indices (same content) → 1 event, blocks."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [
                {"role": "user", "content": "my ssn is 123-45-6789"},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": "my ssn is 123-45-6789"},
            ],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx, _passthrough)
        # First occurrence (history) → strip, second (active turn) → block with event
        assert len(ctx.events) == 1


# ---------------------------------------------------------------------------
# Metadata restoration
# ---------------------------------------------------------------------------


class TestMetadataRestoration:
    """Scanned hashes survive session resume via gate.py."""

    def test_metadata_restored_on_session_resume(self):
        import stateloom

        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )
        with gate.session(session_id="resume-test") as session:
            session.metadata["_pii_blocked_hashes"] = ["abc", "def"]
        with gate.session(session_id="resume-test") as session:
            assert session.metadata.get("_pii_blocked_hashes") == ["abc", "def"]

    def test_estimated_api_cost_restored(self):
        import stateloom

        gate = stateloom.init(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            local_model=None,
        )
        with gate.session(session_id="cost-test") as session:
            session.estimated_api_cost = 0.05
        with gate.session(session_id="cost-test") as session:
            assert session.estimated_api_cost == 0.05


# ---------------------------------------------------------------------------
# Independent API calls (SDK flow)
# ---------------------------------------------------------------------------


class TestIndependentAPICalls:
    """API/SDK flow: each call has independent messages → full enforcement."""

    async def test_same_count_different_content_scanned(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "Hello"}],
            session=session,
        )
        await scanner._do_process(ctx1, _passthrough)

        ctx2 = _make_ctx(
            [{"role": "user", "content": "Email: test@example.com"}],
            session=session,
        )
        await scanner._do_process(ctx2, _passthrough)
        assert len(ctx2.events) == 1


# ---------------------------------------------------------------------------
# Batch API (all new)
# ---------------------------------------------------------------------------


class TestBatchAPIFullEnforcement:
    """First call: all content is new → full enforcement."""

    async def test_all_messages_scanned_on_first_call(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        ctx = _make_ctx(
            [
                {"role": "user", "content": "Email: a@b.com"},
                {"role": "assistant", "content": "Got it"},
                {"role": "user", "content": "Also c@d.com"},
            ],
            session=session,
        )
        await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 2


# ---------------------------------------------------------------------------
# Mixed old BLOCK + REDACT in same request
# ---------------------------------------------------------------------------


class TestMixedBlockAndRedact:
    """Old blocked PII stripped + new REDACT PII redacted in same request."""

    async def test_old_block_stripped_new_redact_applied(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                ),
                PIIRule(pattern="email", mode=PIIMode.REDACT),
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        # Turn 1: SSN blocked
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Turn 2: old SSN + new email
        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "my ssn is 123-45-6789"},
                {"role": "user", "content": "my email is test@secret.com"},
            ],
            session=session,
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        # SSN message stripped
        assert len(ctx2.request_kwargs["messages"]) == 1
        # Email redacted
        assert "test@secret.com" not in ctx2.request_kwargs["messages"][0]["content"]
        # 1 event for new email
        assert len(ctx2.events) == 1
        assert ctx2.events[0].pii_type == "email"


# ---------------------------------------------------------------------------
# Format-change robustness (PII value-based dedup)
# ---------------------------------------------------------------------------


class TestFormatChangeRobustness:
    """PII value-based dedup is robust against CLI format changes."""

    async def test_string_to_list_format_change_still_strips(self):
        """SSN blocked in string content, resent as list-of-blocks → stripped."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        # Turn 1: SSN blocked in string format
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Turn 2: same SSN resent as list-of-blocks (CLI format change)
        ctx2 = _make_ctx(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "my ssn is 123-45-6789"},
                    ],
                },
                {"role": "user", "content": "what's the weather?"},
            ],
            session=session,
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        assert len(ctx2.request_kwargs["messages"]) == 1
        assert ctx2.request_kwargs["messages"][0]["content"] == "what's the weather?"
        assert len(ctx2.events) == 0

    async def test_same_pii_different_surrounding_text_strips(self):
        """SSN in 'my ssn is X' on turn 1, 'SSN: X' on turn 2 → stripped."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        # Turn 1: SSN blocked
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Turn 2: same SSN value in different surrounding text
        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "SSN: 123-45-6789"},
                {"role": "user", "content": "hello"},
            ],
            session=session,
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        assert len(ctx2.request_kwargs["messages"]) == 1
        assert ctx2.request_kwargs["messages"][0]["content"] == "hello"
        assert len(ctx2.events) == 0

    async def test_redact_across_format_change(self):
        """Email in string then list-of-blocks → dup suppressed (same hash)."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.REDACT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="s1")

        # Turn 1: email in string format
        ctx1 = _make_ctx(
            [{"role": "user", "content": "My email is test@secret.com"}],
            session=session,
        )
        await scanner._do_process(ctx1, _passthrough)
        assert len(ctx1.events) == 1

        # Turn 2: same email in list-of-blocks format (history message)
        ctx2 = _make_ctx(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "My email is test@secret.com"},
                    ],
                },
                {"role": "user", "content": "Thanks!"},
            ],
            session=session,
        )
        await scanner._do_process(ctx2, _passthrough)
        # Same value hash → duplicate event suppressed
        assert len(ctx2.events) == 0

    async def test_no_raw_pii_in_metadata(self):
        """Verify only SHA-256 hex digests in metadata, no raw PII."""

        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                ),
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx, _passthrough)

        # Check metadata keys
        blocked = session.metadata.get("_pii_blocked_hashes", [])
        assert len(blocked) >= 1

        # Verify they're SHA-256 hex digests (64 hex chars)
        for h in blocked:
            assert len(h) == 64
            int(h, 16)  # must be valid hex

        # No raw PII in metadata
        metadata_str = str(session.metadata)
        assert "123-45-6789" not in metadata_str

    async def test_phase1_strips_before_phase2_blocks(self):
        """Old blocked SSN + new BLOCK email → SSN stripped, email blocked."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                ),
                PIIRule(
                    pattern="email", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                ),
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        # Turn 1: SSN blocked
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)
        assert len(ctx1.events) == 1

        # Turn 2: old SSN (should be stripped by Phase 1) + new email (should block in Phase 2)
        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "my ssn is 123-45-6789"},
                {"role": "user", "content": "email me at admin@secret.com"},
            ],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx2, _passthrough)
        # 1 event for new email (SSN was stripped by Phase 1, no event)
        assert len(ctx2.events) == 1
        assert ctx2.events[0].pii_type == "email"
        # SSN message was stripped (only email message remains before block)
        assert len(ctx2.request_kwargs["messages"]) == 1


# ---------------------------------------------------------------------------
# System-reminder filtering (CLI-injected context)
# ---------------------------------------------------------------------------


class TestSystemReminderFiltering:
    """PII scanner ignores <system-reminder> blocks in user messages.

    CLI tools (Claude CLI, Gemini CLI) inject file content, tool results,
    and other context in <system-reminder>...</system-reminder> blocks
    within user messages.  These must not trigger PII detection.
    """

    async def test_ssn_in_system_reminder_ignored(self):
        """SSN inside <system-reminder> in a user message → not detected."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [
                {
                    "role": "user",
                    "content": (
                        "hello\n"
                        "<system-reminder>server.log: ssn detected 123-45-6789</system-reminder>\n"
                        "what's the weather?"
                    ),
                }
            ],
            session=session,
        )
        result = await scanner._do_process(ctx, _passthrough)
        assert result == "LLM_RESPONSE"
        assert len(ctx.events) == 0

    async def test_ssn_outside_system_reminder_still_detected(self):
        """SSN outside <system-reminder> in same message → still detected."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [
                {
                    "role": "user",
                    "content": (
                        "my ssn is 123-45-6789\n"
                        "<system-reminder>some context here</system-reminder>"
                    ),
                }
            ],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 1

    async def test_system_reminder_in_list_content_ignored(self):
        """SSN inside <system-reminder> in list-of-blocks content → not detected."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>SSN in log: 123-45-6789</system-reminder>",
                        },
                        {"type": "text", "text": "hello"},
                    ],
                }
            ],
            session=session,
        )
        result = await scanner._do_process(ctx, _passthrough)
        assert result == "LLM_RESPONSE"
        assert len(ctx.events) == 0

    async def test_system_reminder_not_stripped_for_assistant_messages(self):
        """<system-reminder> in assistant messages is still scanned (not CLI-injected)."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "<system-reminder>SSN: 123-45-6789</system-reminder>",
                },
                {"role": "user", "content": "thanks"},
            ],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx, _passthrough)

    async def test_multiple_system_reminders_all_ignored(self):
        """Multiple <system-reminder> blocks in one message → all ignored."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [
                {
                    "role": "user",
                    "content": (
                        "<system-reminder>File: 123-45-6789</system-reminder>\n"
                        "what time is it?\n"
                        "<system-reminder>Log: SSN 987-65-4321</system-reminder>"
                    ),
                }
            ],
            session=session,
        )
        result = await scanner._do_process(ctx, _passthrough)
        assert result == "LLM_RESPONSE"
        assert len(ctx.events) == 0

    async def test_system_reminder_pii_not_stripped_in_phase1(self):
        """Previously-blocked SSN inside <system-reminder> doesn't trigger Phase 1 strip."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        # Turn 1: SSN blocked
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Turn 2: SSN appears inside system-reminder (e.g. from server.log)
        # Phase 1 should NOT strip this message (SSN is inside system-reminder)
        # Phase 2 should NOT detect it (system-reminder is blanked)
        ctx2 = _make_ctx(
            [
                {
                    "role": "user",
                    "content": (
                        "<system-reminder>PII blocked: 123-45-6789</system-reminder>\nhello"
                    ),
                }
            ],
            session=session,
        )
        result = await scanner._do_process(ctx2, _passthrough)
        assert result == "LLM_RESPONSE"
        # Message was NOT stripped/blocked — system-reminder invisible to PII scanner
        assert len(ctx2.request_kwargs["messages"]) == 1
        assert len(ctx2.events) == 0


# ---------------------------------------------------------------------------
# CLI concatenation (old PII in current user message)
# ---------------------------------------------------------------------------


class TestCLIConcatenation:
    """CLI tools concatenate failed request text into the current turn.

    When a previous request was blocked (e.g. SSN), CLI tools like Claude
    CLI embed that text directly into the next user message WITHOUT
    system-reminder tags.  Phase 1 strips the entire active turn message
    when it contains previously-blocked PII, preventing the PII-contaminated
    text from ever reaching the LLM.
    """

    async def test_old_ssn_concatenated_into_new_message_stripped(self):
        """Old SSN + new text in last user message → entire message stripped.

        CLI concatenation of old PII into the active turn is handled by
        Phase 1 which strips the entire message (not just the PII value)
        to prevent any PII-contaminated text from reaching the LLM.
        """
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        # Turn 1: SSN blocked
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Expire cooldown
        session.metadata["_pii_last_block_ts"] = session.metadata["_pii_last_block_ts"] - 3.0

        # Turn 2: CLI concatenates old SSN + new text into last user message
        ctx2 = _make_ctx(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "my ssn is 123-45-6789\nhi there"},
            ],
            session=session,
        )
        # Phase 1 strips the active turn; earlier messages kept
        await scanner._do_process(ctx2, _passthrough)
        msgs = ctx2.request_kwargs["messages"]
        assert all("123-45-6789" not in str(m.get("content", "")) for m in msgs)
        # The history messages (hello/hi there) are preserved
        assert any(m.get("content") == "hello" for m in msgs)

    async def test_old_ssn_in_list_content_last_msg_stripped(self):
        """Old SSN in list-of-blocks last user message → all stripped → blocked."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Expire cooldown
        session.metadata["_pii_last_block_ts"] = session.metadata["_pii_last_block_ts"] - 3.0

        # CLI sends old SSN in list-of-blocks format in the current message
        ctx2 = _make_ctx(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "my ssn is 123-45-6789"},
                        {"type": "text", "text": "hello"},
                    ],
                }
            ],
            session=session,
        )
        # Phase 1 strips the only message → empty → blocked
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx2, _passthrough)

    async def test_new_ssn_in_last_msg_still_blocks(self):
        """New (never-seen) SSN in last user message → still blocked."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        ctx = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789\nhello"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 1

    async def test_old_ssn_strips_message_even_with_new_email(self):
        """Old SSN + new email in same message → entire message stripped.

        Phase 1 strips the entire active turn when it contains previously-
        blocked PII.  Even though the message also contains new PII (email),
        the message is removed to prevent the old SSN from reaching the LLM.
        The new email in that message is a collateral loss — the user can
        re-send it without the SSN.
        """
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                ),
                PIIRule(
                    pattern="email", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                ),
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="s1")

        # Turn 1: SSN blocked
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)

        # Expire cooldown
        session.metadata["_pii_last_block_ts"] = session.metadata["_pii_last_block_ts"] - 3.0

        # Turn 2: old SSN + new email in last message → stripped → blocked
        ctx2 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789\nemail: admin@secret.com"}],
            session=session,
        )
        # Phase 1 strips the only message → empty → blocked
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx2, _passthrough)


# ---------------------------------------------------------------------------
# SDK flow — no Phase 1, no dedup, every call treated independently
# ---------------------------------------------------------------------------


class TestSDKFlowBlock:
    """SDK (non-proxy) calls: BLOCK creates event every time, no Phase 1."""

    async def test_sdk_block_creates_event(self):
        """SDK block mode → event on first call."""
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="sdk-s1")

        ctx = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
            proxy=False,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 1
        assert ctx.events[0].action_taken == "blocked"

    async def test_sdk_block_rerun_same_session_still_blocks_with_event(self):
        """SDK re-run with same session ID → still blocks with new event.

        This is the critical difference from proxy: proxy Phase 1 would
        silently strip the message, SDK always blocks.
        """
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        store = MemoryStore()
        scanner = PIIScannerMiddleware(config, store=store)
        session = Session(id="sdk-s1")

        # Call 1
        ctx1 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
            proxy=False,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx1, _passthrough)
        assert len(ctx1.events) == 1

        # Call 2 — same session, same content → still blocked with event
        ctx2 = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
            proxy=False,
        )
        with pytest.raises(StateLoomPIIBlockedError):
            await scanner._do_process(ctx2, _passthrough)
        assert len(ctx2.events) == 1


class TestSDKFlowRedact:
    """SDK (non-proxy) calls: REDACT creates event every time."""

    async def test_sdk_redact_creates_event(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.REDACT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="sdk-s1")

        ctx = _make_ctx(
            [{"role": "user", "content": "My email is test@secret.com"}],
            session=session,
            proxy=False,
        )
        await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 1
        assert "test@secret.com" not in ctx.request_kwargs["messages"][0]["content"]

    async def test_sdk_redact_rerun_creates_event_again(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.REDACT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="sdk-s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "My email is test@secret.com"}],
            session=session,
            proxy=False,
        )
        await scanner._do_process(ctx1, _passthrough)
        assert len(ctx1.events) == 1

        ctx2 = _make_ctx(
            [{"role": "user", "content": "My email is test@secret.com"}],
            session=session,
            proxy=False,
        )
        await scanner._do_process(ctx2, _passthrough)
        assert len(ctx2.events) == 1


class TestSDKFlowAudit:
    """SDK (non-proxy) calls: AUDIT creates event every time."""

    async def test_sdk_audit_creates_event(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="sdk-s1")

        ctx = _make_ctx(
            [{"role": "user", "content": "Email: user@example.com"}],
            session=session,
            proxy=False,
        )
        await scanner._do_process(ctx, _passthrough)
        assert len(ctx.events) == 1

    async def test_sdk_audit_rerun_creates_event_again(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[PIIRule(pattern="email", mode=PIIMode.AUDIT)],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="sdk-s1")

        ctx1 = _make_ctx(
            [{"role": "user", "content": "Email: user@example.com"}],
            session=session,
            proxy=False,
        )
        await scanner._do_process(ctx1, _passthrough)

        ctx2 = _make_ctx(
            [{"role": "user", "content": "Email: user@example.com"}],
            session=session,
            proxy=False,
        )
        await scanner._do_process(ctx2, _passthrough)
        assert len(ctx2.events) == 1


class TestSDKCliInternalBypass:
    """_cli_internal still skips PII scanning entirely (both paths)."""

    async def test_cli_internal_skips_scanning(self):
        config = StateLoomConfig(
            pii_enabled=True,
            pii_rules=[
                PIIRule(
                    pattern="ssn", mode=PIIMode.BLOCK, on_middleware_failure=FailureAction.BLOCK
                )
            ],
        )
        scanner = PIIScannerMiddleware(config)
        session = Session(id="sdk-s1")

        ctx = _make_ctx(
            [{"role": "user", "content": "my ssn is 123-45-6789"}],
            session=session,
            proxy=False,
        )
        ctx.request_kwargs["_cli_internal"] = True
        result = await scanner._do_process(ctx, _passthrough)
        assert result == "LLM_RESPONSE"
        assert len(ctx.events) == 0
