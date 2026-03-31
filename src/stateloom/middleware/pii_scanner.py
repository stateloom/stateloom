"""PII scanner middleware — detect, audit, redact, or block PII in LLM calls."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from stateloom.core.config import PIIRule, StateLoomConfig
from stateloom.core.errors import StateLoomPIIBlockedError
from stateloom.core.event import PIIDetectionEvent
from stateloom.core.types import ActionTaken, FailureAction, PIIMode
from stateloom.middleware.base import MiddlewareContext
from stateloom.pii.rehydrator import PIIRehydrator
from stateloom.pii.scanner import PIIMatch, PIIScanner

if TYPE_CHECKING:
    from stateloom.pii.stream_buffer import StreamPIIBuffer

logger = logging.getLogger("stateloom.middleware.pii")

# Map PIIMode → ActionTaken so event.action_taken is always the canonical enum value.
_MODE_TO_ACTION: dict[PIIMode, ActionTaken] = {
    PIIMode.AUDIT: ActionTaken.LOGGED,
    PIIMode.REDACT: ActionTaken.REDACTED,
    PIIMode.BLOCK: ActionTaken.BLOCKED,
}

# Store key for persisted PII config state
_STORE_KEY_PII_CONFIG = "pii_config_json"

# How often to poll the store for cross-process state (seconds)
_STORE_POLL_INTERVAL = 2.0


def _mask_pii_text(text: str, pii_type: str) -> str:
    """Create a safe masked preview of detected PII for dashboard display."""
    if not text or len(text) <= 2:
        return "*" * len(text)

    if pii_type == "email" and "@" in text:
        local, domain = text.split("@", 1)
        return f"{local[0]}***@{domain}"

    if pii_type in ("ssn", "social_security"):
        return f"***-**-{text[-4:]}" if len(text) >= 4 else "***"

    if pii_type in ("credit_card", "credit_card_number"):
        return f"****-****-****-{text[-4:]}" if len(text) >= 4 else "****"

    if pii_type in ("phone", "phone_number"):
        return f"(***) ***-{text[-4:]}" if len(text) >= 4 else "***"

    if pii_type.startswith("api_key"):
        return f"{text[:5]}...{text[-3:]}" if len(text) > 8 else f"{text[:3]}***"

    # Generic: first char + masked middle + last char
    return f"{text[0]}{'*' * min(len(text) - 2, 10)}{text[-1]}"


def _extract_text(content: str | list[Any]) -> str:
    """Extract plain text from string or Anthropic list-of-blocks content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return " ".join(parts)
    return ""


def _pii_mode_severity(mode: PIIMode) -> int:
    """Return severity level for PII mode comparison (strictest wins)."""
    return {PIIMode.AUDIT: 0, PIIMode.REDACT: 1, PIIMode.BLOCK: 2}.get(mode, 0)


class PIIScannerMiddleware:
    """Scans LLM requests for PII and applies configured actions.

    Streaming limitation: PII scanning operates on request messages before the LLM
    call. Streaming responses are not scanned. This is a deliberate trade-off —
    real-time PII scanning of token streams would add per-token latency.
    """

    def __init__(
        self,
        config: StateLoomConfig,
        store: Any = None,
        org_rules_fn: Callable[[str], list[PIIRule]] | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._org_rules_fn = org_rules_fn

        # NER detector (optional — lazy-loads GLiNER model)
        ner = None
        if config.pii.ner_enabled:
            from stateloom.pii.ner_detector import NERDetector

            ner = NERDetector(
                model_name=config.pii.ner_model,
                labels=config.pii.ner_labels,
                threshold=config.pii.ner_threshold,
            )

        self._scanner = PIIScanner(config, ner_detector=ner)
        # Build mode lookup from rules
        self._modes: dict[str, PIIMode] = {}
        self._rules: dict[str, PIIRule] = {}
        for rule in config.pii.rules:
            self._modes[rule.pattern] = rule.mode
            self._rules[rule.pattern] = rule
        self._default_mode = config.pii.default_mode
        self._last_store_poll: float = 0.0
        # When init() provides explicit rules, don't let store sync override them
        self._rules_from_init = len(config.pii.rules) > 0

    def reload_rules(self) -> None:
        """Reload rules from config (called after runtime rule changes)."""
        self._modes = {}
        self._rules = {}
        for rule in self._config.pii.rules:
            self._modes[rule.pattern] = rule.mode
            self._rules[rule.pattern] = rule
        self._default_mode = self._config.pii.default_mode

    def _sync_from_store(self) -> None:
        """Poll persisted PII config from the store (cross-process sync).

        Skipped when ``init()`` provided explicit PII rules — code-level config
        takes precedence over dashboard/API-persisted config.
        """
        if not self._store or self._rules_from_init:
            return
        now = time.monotonic()
        if now - self._last_store_poll < _STORE_POLL_INTERVAL:
            return
        self._last_store_poll = now
        try:
            blob = self._store.get_secret(_STORE_KEY_PII_CONFIG)
            if not blob:
                return
            data = json.loads(blob)
            if "enabled" in data:
                self._config.pii_enabled = bool(data["enabled"])
            if "default_mode" in data:
                self._config.pii_default_mode = PIIMode(data["default_mode"])
            if "rules" in data:
                old_patterns = {r.pattern: r.mode.value for r in self._config.pii_rules}
                new_rules = [PIIRule(**r) for r in data["rules"]]
                new_patterns = {r.pattern: r.mode.value for r in new_rules}
                if old_patterns != new_patterns:
                    self._config.pii_rules = new_rules
                    self.reload_rules()
        except Exception:
            logger.debug("Failed to sync PII config from store", exc_info=True)

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        self._sync_from_store()
        if not self._config.pii_enabled:
            return await call_next(ctx)
        try:
            return await self._do_process(ctx, call_next)
        except (StateLoomPIIBlockedError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            # Internal PII scanner failure — check per-rule on_middleware_failure
            failure_action = self._get_strictest_failure_action()
            if failure_action == FailureAction.BLOCK:
                logger.error(f"[StateLoom] PII scanner failed, blocking: {exc}")
                raise
            logger.warning(
                "[StateLoom] PII scanner INTERNAL ERROR, passing through: %s",
                exc,
                exc_info=True,
            )
            return await call_next(ctx)

    # Regex to match <system-reminder>...</system-reminder> blocks.
    # CLI tools (Claude CLI, Gemini CLI) embed file content, tool results,
    # and context in these blocks within user messages.  They are injected
    # by the CLI — not typed by the user — and must not be scanned for PII.
    _SYSTEM_REMINDER_RE = re.compile(
        r"<system-reminder>.*?</system-reminder>",
        re.DOTALL,
    )

    @staticmethod
    def _blank_system_reminders(text: str) -> str:
        """Replace ``<system-reminder>`` blocks with same-length whitespace.

        Preserves character positions so that any PII matches found in the
        remaining (user-typed) text have correct ``start``/``end`` offsets.
        """
        return PIIScannerMiddleware._SYSTEM_REMINDER_RE.sub(
            lambda m: " " * len(m.group(0)),
            text,
        )

    @staticmethod
    def _pii_value_hash(value: str) -> str:
        """SHA-256 hash of a PII value for deduplication.

        Hashes the actual PII text (e.g. ``"123-45-6789"``), not the
        surrounding message.  This is robust against CLI format changes
        (string → list-of-blocks, added ``cache_control``, etc.).
        """
        return hashlib.sha256(value.strip().encode()).hexdigest()

    def _scan_content(
        self,
        content: Any,
        field: str,
        *,
        strip_reminders: bool = False,
    ) -> list[PIIMatch]:
        """Scan message content for PII, handling str and list-of-blocks.

        When *strip_reminders* is True, ``<system-reminder>`` blocks are
        blanked (replaced with spaces) before scanning so that CLI-injected
        context (file content, tool results) does not trigger false PII
        detections.  Character positions are preserved.
        """
        if isinstance(content, str):
            text = self._blank_system_reminders(content) if strip_reminders else content
            return self._scanner.scan(text, field=field)
        if isinstance(content, list):
            matches: list[PIIMatch] = []
            for j, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    text = (
                        self._blank_system_reminders(part["text"])
                        if strip_reminders
                        else part["text"]
                    )
                    matches.extend(
                        self._scanner.scan(text, field=f"{field}[{j}].text"),
                    )
            return matches
        return []

    @staticmethod
    def _blank_pii_in_content(
        content: Any,
        matches: list[PIIMatch],
        field_prefix: str,
    ) -> Any:
        """Replace matched PII text with ``[REDACTED]`` in message content.

        Handles both ``str`` and ``list[dict]`` (Anthropic multi-part) content.
        Returns the modified content (new object for str, mutated for list).
        """
        if isinstance(content, str):
            # Apply replacements right-to-left so positions stay valid.
            relevant = [m for m in matches if m.field == field_prefix]
            relevant.sort(key=lambda m: m.start, reverse=True)
            for m in relevant:
                content = content[: m.start] + "[REDACTED]" + content[m.end :]
            return content
        if isinstance(content, list):
            for j, part in enumerate(content):
                if not (isinstance(part, dict) and part.get("type") == "text"):
                    continue
                part_field = f"{field_prefix}[{j}].text"
                relevant = [m for m in matches if m.field == part_field]
                if not relevant:
                    continue
                relevant.sort(key=lambda m: m.start, reverse=True)
                text = part["text"]
                for m in relevant:
                    text = text[: m.start] + "[REDACTED]" + text[m.end :]
                part["text"] = text
            return content
        return content

    def _strip_known_blocked_pii(
        self,
        messages: list[dict[str, Any]],
        session: Any,
        system: str,
    ) -> tuple[set[int], bool]:
        """Phase 1: remove previously-blocked PII from the request.

        For **history messages** (not the last user message): strips the
        entire message if it contains previously-blocked PII.

        For the **last user message** (the active turn): blanks (redacts)
        individual PII values in-place.  CLI tools concatenate prior
        failed messages into the current turn without ``<system-reminder>``
        tags, so we must clean the PII out of the content rather than
        stripping the whole message.

        Returns ``(strip_indices, strip_system)`` — indices of messages to
        remove and whether the system prompt should be stripped.

        Fast no-op when no PII has been blocked before.
        """
        blocked_hashes: set[str] = set(
            session.metadata.get("_pii_blocked_hashes", []),
        )
        if not blocked_hashes:
            return set(), False

        # Find the last user message.
        last_user_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        strip_indices: set[int] = set()
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue
            msg_matches = self._scan_content(
                content,
                field=f"messages[{i}].content",
                strip_reminders=(msg.get("role") == "user"),
            )
            # Collect matches whose PII hash is in the blocked set.
            blocked_in_msg: list[PIIMatch] = []
            for match in msg_matches:
                mode = self._get_mode(match.pattern_name, org_id=session.org_id)
                if mode == PIIMode.BLOCK:
                    vh = self._pii_value_hash(match.matched_text)
                    if vh in blocked_hashes:
                        blocked_in_msg.append(match)

            if not blocked_in_msg:
                continue

            if i == last_user_idx:
                # Active turn contains previously-blocked PII — strip the
                # entire message.  CLIs retry/concatenate blocked messages
                # into new turns; leaving the PII (even redacted) would let
                # it reach the LLM.  Stripping silently removes the message
                # so the request proceeds without the PII-contaminated text.
                # Unlike re-blocking (which raises an error the CLI retries
                # forever), stripping lets the pipeline continue.
                #
                # We set a flag so the proxy handler can return a visible
                # content-policy response instead of forwarding a request
                # with the user's input missing.
                logger.info(
                    "[PII] Phase 1: stripping active turn msg[%d] "
                    "containing %d previously-blocked PII value(s)",
                    i,
                    len(blocked_in_msg),
                )
                strip_indices.add(i)
                session.metadata["_pii_active_turn_stripped"] = True
            else:
                # History message — strip the entire message.
                strip_indices.add(i)

        strip_system = False
        if isinstance(system, str) and system:
            sys_matches = self._scanner.scan(system, field="system")
            for match in sys_matches:
                mode = self._get_mode(match.pattern_name, org_id=session.org_id)
                if mode == PIIMode.BLOCK:
                    vh = self._pii_value_hash(match.matched_text)
                    if vh in blocked_hashes:
                        strip_system = True
                        break

        return strip_indices, strip_system

    def _make_pii_event(
        self,
        ctx: MiddlewareContext,
        match: PIIMatch,
        mode: PIIMode,
        extra_meta: dict[str, Any] | None = None,
    ) -> PIIDetectionEvent:
        """Build a ``PIIDetectionEvent`` for a single PII match."""
        meta: dict[str, Any] = {
            "redacted_preview": _mask_pii_text(match.matched_text, match.pattern_name),
            "match_length": len(match.matched_text),
        }
        if extra_meta:
            meta.update(extra_meta)
        return PIIDetectionEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            pii_type=match.pattern_name,
            mode=mode.value,
            pii_field=match.field,
            action_taken=_MODE_TO_ACTION[mode],
            metadata=meta,
        )

    async def _do_process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        # Skip PII scanning for CLI-internal requests (e.g. session-summary-generation).
        # These carry old conversation history that was already PII-scanned in its
        # original session context — re-scanning would create phantom PII detections
        # in unrelated sessions.
        if ctx.request_kwargs.get("_cli_internal"):
            return await call_next(ctx)

        # Dispatch: proxy requests use the stateful CLI-aware path;
        # SDK requests use the simple per-call path.
        if ctx.request_kwargs.get("_proxy"):
            return await self._do_process_proxy(ctx, call_next)
        return await self._do_process_sdk(ctx, call_next)

    # ------------------------------------------------------------------
    # SDK path — simple, stateless per-call scanning
    # ------------------------------------------------------------------

    async def _do_process_sdk(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        """PII scanning for SDK / ``stateloom.chat()`` calls.

        Every call is independent — no Phase 1, no active-turn vs history
        distinction, no ``_pii_blocked_hashes`` tracking.  Each message is
        scanned equally and the configured action (block/redact/audit)
        applies unconditionally.
        """
        messages = ctx.request_kwargs.get("messages", [])
        system = ctx.request_kwargs.get("system", "")

        rehydrator = PIIRehydrator()
        blocked_matches: list[PIIMatch] = []
        redact_matches: list[PIIMatch] = []

        # Scan all messages
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue
            msg_matches = self._scan_content(
                content,
                field=f"messages[{i}].content",
            )
            for match in msg_matches:
                mode = self._get_mode(match.pattern_name, org_id=ctx.session.org_id)
                if mode == PIIMode.BLOCK:
                    event = self._make_pii_event(ctx, match, mode)
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()
                    blocked_matches.append(match)
                elif mode == PIIMode.REDACT:
                    event = self._make_pii_event(ctx, match, mode)
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()
                    redact_matches.append(match)
                elif mode == PIIMode.AUDIT:
                    event = self._make_pii_event(ctx, match, mode)
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()

        # Scan system prompt
        if isinstance(system, str) and system:
            sys_matches = self._scanner.scan(system, field="system")
            for match in sys_matches:
                mode = self._get_mode(match.pattern_name, org_id=ctx.session.org_id)
                if mode == PIIMode.BLOCK:
                    event = self._make_pii_event(ctx, match, mode)
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()
                    blocked_matches.append(match)
                elif mode == PIIMode.REDACT:
                    event = self._make_pii_event(ctx, match, mode)
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()
                    redact_matches.append(match)
                elif mode == PIIMode.AUDIT:
                    event = self._make_pii_event(ctx, match, mode)
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()

        if not blocked_matches and not redact_matches and not ctx.events:
            return await call_next(ctx)

        # Block
        if blocked_matches:
            self._save_events_directly(ctx)
            pii_types = ", ".join(set(m.pattern_name for m in blocked_matches))
            raise StateLoomPIIBlockedError(
                pii_type=pii_types,
                session_id=ctx.session.id,
            )

        # Redact
        if redact_matches:
            self._redact_messages(ctx, redact_matches, rehydrator)
            system = ctx.request_kwargs.get("system", "")
            if isinstance(system, str) and system:
                sys_redact = [m for m in redact_matches if m.field == "system"]
                if sys_redact:
                    ctx.request_kwargs["system"] = rehydrator.redact(system, sys_redact)

        result = await call_next(ctx)

        if redact_matches and result is not None:
            self._rehydrate_response(result, rehydrator, provider=ctx.provider)

        return result

    # ------------------------------------------------------------------
    # Proxy path — stateful, CLI-aware scanning
    # ------------------------------------------------------------------

    async def _do_process_proxy(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        """PII scanning for proxy / CLI flows.

        Includes Phase 1 (strip previously-blocked PII), active-turn vs
        history distinction, cooldown timers, and hash tracking — all the
        logic needed for CLI tools that resend full conversation history.
        """

        messages = ctx.request_kwargs.get("messages", [])
        system = ctx.request_kwargs.get("system", "")

        # Clear per-call flag before Phase 1 (prevents stale flag from
        # a previous call in the same session).
        ctx.session.metadata.pop("_pii_active_turn_stripped", None)

        # ── Phase 1: strip messages containing previously-blocked PII ──
        strip_indices, strip_system = self._strip_known_blocked_pii(
            messages,
            ctx.session,
            system,
        )

        if strip_indices:
            logger.info(
                "[PII] Stripping %d previously-blocked message(s) from request",
                len(strip_indices),
            )
            messages = [msg for i, msg in enumerate(messages) if i not in strip_indices]
            # Ensure the conversation ends with a 'user' message. If stripping
            # left an 'assistant' message at the end, we must remove it to
            # avoid Anthropic "assistant prefill" 400 errors.
            while messages and messages[-1].get("role") == "assistant":
                logger.info("[PII] Stripping trailing assistant message orphaned by PII removal")
                messages.pop()
            # Also strip leading assistant messages (if the first user
            # message was the one with blocked PII and got stripped).
            while messages and messages[0].get("role") == "assistant":
                logger.info("[PII] Stripping leading assistant message orphaned by PII removal")
                messages.pop(0)

            ctx.request_kwargs["messages"] = messages

            # If all messages were stripped (entire request was previously-
            # blocked PII, e.g. a retry), raise instead of forwarding an
            # empty request that would crash the provider SDK.
            if not messages:
                logger.warning(
                    "[PII] All messages stripped by Phase 1 — request contains "
                    "only previously-blocked PII"
                )
                self._save_events_directly(ctx)
                raise StateLoomPIIBlockedError(
                    pii_type="previously blocked",
                    session_id=ctx.session.id,
                )

        if strip_system:
            logger.info("[PII] Stripping previously-blocked system prompt from request")
            ctx.request_kwargs.pop("system", None)
            system = ""

        # ── Phase 2: scan remaining messages for PII ──
        blocked_hashes: set[str] = set(
            ctx.session.metadata.get("_pii_blocked_hashes", []),
        )

        # Snapshot of PII blocked in *previous* requests — used to
        # distinguish cross-request re-blocks from same-request duplicates.
        initial_blocked_hashes = frozenset(blocked_hashes)

        rehydrator = PIIRehydrator()
        blocked_matches: list[PIIMatch] = []
        redact_matches: list[PIIMatch] = []
        # History messages with new BLOCK-mode PII — strip instead of blocking.
        phase2_strip_indices: set[int] = set()

        # Find the last user message index — PII here was deliberately typed
        # by the user (active turn), not re-sent as history by the CLI.
        last_user_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue

            msg_matches = self._scan_content(
                content,
                field=f"messages[{i}].content",
                strip_reminders=(msg.get("role") == "user"),
            )
            if not msg_matches:
                continue

            is_active_turn = i == last_user_idx

            for match in msg_matches:
                mode = self._get_mode(match.pattern_name, org_id=ctx.session.org_id)
                vh = self._pii_value_hash(match.matched_text)

                if mode == PIIMode.BLOCK:
                    if is_active_turn:
                        # PII in the active turn — always block the request.
                        is_reblock = vh in initial_blocked_hashes
                        emit_event = True
                        if is_reblock and vh in initial_blocked_hashes:
                            # PII was blocked in a PREVIOUS request.
                            # Suppress event during cooldown (CLI retry).
                            last_ts = ctx.session.metadata.get(
                                "_pii_last_block_ts",
                                0,
                            )
                            if time.time() - last_ts < 2.0:
                                emit_event = False

                        if emit_event:
                            event = PIIDetectionEvent(
                                session_id=ctx.session.id,
                                step=ctx.session.step_counter,
                                pii_type=match.pattern_name,
                                mode=mode.value,
                                pii_field=match.field,
                                action_taken=_MODE_TO_ACTION[mode],
                                metadata={
                                    "redacted_preview": _mask_pii_text(
                                        match.matched_text,
                                        match.pattern_name,
                                    ),
                                    "match_length": len(match.matched_text),
                                    **({"reblock": True} if is_reblock else {}),
                                },
                            )
                            ctx.events.append(event)
                            ctx.session.add_pii_detection()
                        blocked_matches.append(match)
                        blocked_hashes.add(vh)
                    elif vh not in initial_blocked_hashes and msg.get("role") == "user":
                        # New PII in a history USER message (not the active
                        # turn).  Strip instead of blocking — the user didn't
                        # type this now; the CLI included old conversation
                        # history.  This prevents phantom sessions on CLI
                        # restart when history contains previously-blocked PII.
                        phase2_strip_indices.add(i)
                        blocked_hashes.add(vh)
                    elif vh not in initial_blocked_hashes:
                        # New PII in a non-user history message (assistant/
                        # model) — block like normal (possible model PII leak).
                        event = PIIDetectionEvent(
                            session_id=ctx.session.id,
                            step=ctx.session.step_counter,
                            pii_type=match.pattern_name,
                            mode=mode.value,
                            pii_field=match.field,
                            action_taken=_MODE_TO_ACTION[mode],
                            metadata={
                                "redacted_preview": _mask_pii_text(
                                    match.matched_text,
                                    match.pattern_name,
                                ),
                                "match_length": len(match.matched_text),
                            },
                        )
                        ctx.events.append(event)
                        ctx.session.add_pii_detection()
                        blocked_matches.append(match)
                        blocked_hashes.add(vh)
                    else:
                        # Previously-blocked PII in a history message.
                        # During cooldown (< 2s after block), re-block so
                        # multi-request CLI turns are handled consistently.
                        # Outside cooldown, Phase 1 already blanked it —
                        # skip silently.
                        last_block_ts = ctx.session.metadata.get(
                            "_pii_last_block_ts",
                            0,
                        )
                        if time.time() - last_block_ts < 2.0:
                            blocked_matches.append(match)

                elif mode == PIIMode.REDACT:
                    event = PIIDetectionEvent(
                        session_id=ctx.session.id,
                        step=ctx.session.step_counter,
                        pii_type=match.pattern_name,
                        mode=mode.value,
                        pii_field=match.field,
                        action_taken=_MODE_TO_ACTION[mode],
                        metadata={
                            "redacted_preview": _mask_pii_text(
                                match.matched_text,
                                match.pattern_name,
                            ),
                            "match_length": len(match.matched_text),
                        },
                    )
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()
                    redact_matches.append(match)

                elif mode == PIIMode.AUDIT:
                    event = PIIDetectionEvent(
                        session_id=ctx.session.id,
                        step=ctx.session.step_counter,
                        pii_type=match.pattern_name,
                        mode=mode.value,
                        pii_field=match.field,
                        action_taken=_MODE_TO_ACTION[mode],
                        metadata={
                            "redacted_preview": _mask_pii_text(
                                match.matched_text,
                                match.pattern_name,
                            ),
                            "match_length": len(match.matched_text),
                        },
                    )
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()

        # --- System prompt ---
        if isinstance(system, str) and system:
            sys_matches = self._scanner.scan(system, field="system")
            for match in sys_matches:
                mode = self._get_mode(match.pattern_name, org_id=ctx.session.org_id)
                vh = self._pii_value_hash(match.matched_text)

                if mode == PIIMode.BLOCK:
                    event = PIIDetectionEvent(
                        session_id=ctx.session.id,
                        step=ctx.session.step_counter,
                        pii_type=match.pattern_name,
                        mode=mode.value,
                        pii_field=match.field,
                        action_taken=_MODE_TO_ACTION[mode],
                        metadata={
                            "redacted_preview": _mask_pii_text(
                                match.matched_text,
                                match.pattern_name,
                            ),
                            "match_length": len(match.matched_text),
                        },
                    )
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()
                    blocked_matches.append(match)
                    blocked_hashes.add(vh)
                elif mode == PIIMode.REDACT:
                    event = PIIDetectionEvent(
                        session_id=ctx.session.id,
                        step=ctx.session.step_counter,
                        pii_type=match.pattern_name,
                        mode=mode.value,
                        pii_field=match.field,
                        action_taken=_MODE_TO_ACTION[mode],
                        metadata={
                            "redacted_preview": _mask_pii_text(
                                match.matched_text,
                                match.pattern_name,
                            ),
                            "match_length": len(match.matched_text),
                        },
                    )
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()
                    redact_matches.append(match)
                elif mode == PIIMode.AUDIT:
                    event = PIIDetectionEvent(
                        session_id=ctx.session.id,
                        step=ctx.session.step_counter,
                        pii_type=match.pattern_name,
                        mode=mode.value,
                        pii_field=match.field,
                        action_taken=_MODE_TO_ACTION[mode],
                        metadata={
                            "redacted_preview": _mask_pii_text(
                                match.matched_text,
                                match.pattern_name,
                            ),
                            "match_length": len(match.matched_text),
                        },
                    )
                    ctx.events.append(event)
                    ctx.session.add_pii_detection()

        # Persist blocked hashes immediately so they survive block errors
        # and upstream failures.
        ctx.session.metadata["_pii_blocked_hashes"] = list(blocked_hashes)
        if self._store:
            try:
                self._store.save_session(ctx.session)
            except Exception:
                logger.debug("Failed to persist PII hashes early", exc_info=True)

        # Phase 2 strip: new BLOCK-mode PII found only in history messages.
        # Strip those messages so the PII never reaches the LLM, but don't
        # block the request (the user didn't type it in the active turn).
        if phase2_strip_indices:
            logger.info(
                "[PII] Stripping %d history message(s) with new PII (not in active turn)",
                len(phase2_strip_indices),
            )
            messages = [msg for i, msg in enumerate(messages) if i not in phase2_strip_indices]
            while messages and messages[-1].get("role") == "assistant":
                messages.pop()
            while messages and messages[0].get("role") == "assistant":
                messages.pop(0)
            ctx.request_kwargs["messages"] = messages

        has_actions = (
            blocked_matches
            or redact_matches
            or strip_indices
            or strip_system
            or phase2_strip_indices
            or ctx.events
        )
        if not has_actions:
            return await call_next(ctx)

        # Handle new blocks — block the entire request
        if blocked_matches:
            # Set cooldown timestamp whenever a block event was emitted
            # (new or re-block).  This ensures CLI auto-retries within 2s
            # are suppressed.  Re-blocks during cooldown never create
            # events, so they never reach here — no lockout risk.
            has_new_blocks = any(
                isinstance(e, PIIDetectionEvent) and e.action_taken == ActionTaken.BLOCKED
                for e in ctx.events
            )
            if has_new_blocks:
                ctx.session.metadata["_pii_last_block_ts"] = time.time()
            logger.warning(
                "[PII] BLOCKING request — %d new PII matches (types: %s)",
                len(blocked_matches),
                ", ".join(set(m.pattern_name for m in blocked_matches)),
            )
            self._save_events_directly(ctx)
            pii_types = ", ".join(set(m.pattern_name for m in blocked_matches))
            raise StateLoomPIIBlockedError(
                pii_type=pii_types,
                session_id=ctx.session.id,
            )

        # Handle redactions — modify request in place
        if redact_matches:
            self._redact_messages(ctx, redact_matches, rehydrator)
            system = ctx.request_kwargs.get("system", "")
            if isinstance(system, str) and system:
                sys_redact = [m for m in redact_matches if m.field == "system"]
                if sys_redact:
                    ctx.request_kwargs["system"] = rehydrator.redact(system, sys_redact)

        # Recalculate request_hash after modifying request_kwargs so
        # downstream middleware (e.g. loop detector) sees the post-strip
        # request, not the original.
        if strip_indices or strip_system or redact_matches:
            try:
                serialized = json.dumps(ctx.request_kwargs, sort_keys=True, default=str)
                ctx.request_hash = hashlib.sha256(serialized.encode()).hexdigest()[:16]
            except (TypeError, ValueError):
                pass

        # Make the call
        result = await call_next(ctx)

        # Rehydrate response if we redacted anything
        if redact_matches and result is not None:
            self._rehydrate_response(result, rehydrator, provider=ctx.provider)

        return result

    def _save_events_directly(self, ctx: MiddlewareContext) -> None:
        """Persist PII events directly to the store (bypass EventRecorder).

        When PII is blocked, the exception short-circuits the pipeline before
        EventRecorder runs. Save events here so they appear in the dashboard.
        """
        if not self._store:
            return
        for event in ctx.events:
            try:
                self._store.save_event(event)
            except Exception:
                logger.debug("Failed to persist PII event directly", exc_info=True)
        try:
            self._store.save_session(ctx.session)
        except Exception:
            logger.debug("Failed to persist session after PII block", exc_info=True)

    def _get_strictest_failure_action(self) -> FailureAction:
        """Get the strictest on_middleware_failure from block-mode rules."""
        for rule in self._config.pii.rules:
            if rule.mode == PIIMode.BLOCK and rule.on_middleware_failure == FailureAction.BLOCK:
                return FailureAction.BLOCK
        return FailureAction.PASS

    def _get_mode(self, pattern_name: str, org_id: str = "") -> PIIMode:
        """Get the configured mode for a pattern, merging org-level rules."""
        # Direct lookup
        if pattern_name in self._modes:
            mode = self._modes[pattern_name]
        else:
            # Check group names (e.g. "api_key" for "api_key_openai")
            mode = self._default_mode
            for key, m in self._modes.items():
                if pattern_name.startswith(key):
                    mode = m
                    break

        # Merge org-level rules (strictest wins)
        if org_id and self._org_rules_fn:
            org_rules = self._org_rules_fn(org_id)
            for rule in org_rules:
                if pattern_name == rule.pattern or pattern_name.startswith(rule.pattern):
                    if _pii_mode_severity(rule.mode) > _pii_mode_severity(mode):
                        mode = rule.mode

        return mode

    @staticmethod
    def _reindex_matches(
        matches: list[PIIMatch],
        stripped: set[int],
    ) -> None:
        """Update match field references after messages have been stripped.

        When messages at certain indices are removed, remaining messages
        shift down.  This updates ``match.field`` strings like
        ``messages[5].content`` → ``messages[3].content`` in-place.
        """
        import re as _re

        # Build old-index → new-index mapping
        index_map: dict[int, int] = {}
        offset = 0
        for old_idx in range(max(stripped) + 50):  # generous upper bound
            if old_idx in stripped:
                offset += 1
            else:
                index_map[old_idx] = old_idx - offset

        pattern = _re.compile(r"messages\[(\d+)\]")
        for match in matches:

            def _replace(m: _re.Match[str]) -> str:
                old = int(m.group(1))
                new = index_map.get(old, old)
                return f"messages[{new}]"

            match.field = pattern.sub(_replace, match.field)

    def _redact_messages(
        self,
        ctx: MiddlewareContext,
        matches: list[PIIMatch],
        rehydrator: PIIRehydrator,
    ) -> None:
        """Redact PII in request messages (handles both str and list content)."""
        messages = ctx.request_kwargs.get("messages", [])
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                field_prefix = f"messages[{i}].content"
                field_matches = [m for m in matches if m.field == field_prefix]
                if field_matches:
                    messages[i]["content"] = rehydrator.redact(content, field_matches)
            elif isinstance(content, list):
                for j, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        field = f"messages[{i}].content[{j}].text"
                        field_matches = [m for m in matches if m.field == field]
                        if field_matches:
                            part["text"] = rehydrator.redact(
                                part["text"],
                                field_matches,
                            )

    def _rehydrate_response(
        self, response: Any, rehydrator: PIIRehydrator, provider: str = ""
    ) -> None:
        """Rehydrate PII placeholders in the LLM response."""
        try:
            from stateloom.intercept.provider_registry import get_adapter

            adapter = get_adapter(provider)
            if adapter is not None:
                adapter.modify_response_text(response, rehydrator.rehydrate)
                return

            # Fallback: try all registered adapters (first one that has text wins)
            from stateloom.intercept.provider_registry import get_all_adapters

            for adapter in get_all_adapters().values():
                content = adapter.extract_content(response)
                if content:
                    adapter.modify_response_text(response, rehydrator.rehydrate)
                    return
        except Exception as e:
            logger.warning(f"[StateLoom] Failed to rehydrate response: {e}")

    @property
    def stream_buffer_enabled(self) -> bool:
        """Whether stream PII buffering is active."""
        return self._config.pii.enabled and self._config.pii.stream_buffer_enabled

    def create_stream_buffer(self) -> StreamPIIBuffer | None:
        """Create a StreamPIIBuffer for scanning streaming responses.

        Returns None if stream buffering is disabled.
        Uses regex-only scanning (NER is too slow for per-chunk use).
        """
        if not self.stream_buffer_enabled:
            return None
        from stateloom.pii.stream_buffer import StreamPIIBuffer

        return StreamPIIBuffer(
            scanner=PIIScanner(self._config),  # regex-only for streaming
            mode=self._default_mode,
            buffer_size=self._config.pii.stream_buffer_size,
        )
