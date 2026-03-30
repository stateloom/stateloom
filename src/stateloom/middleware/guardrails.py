"""Guardrail middleware — prompt injection, jailbreak, and system prompt leak detection."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import StateLoomGuardrailError
from stateloom.core.event import GuardrailEvent
from stateloom.core.types import ActionTaken, GuardrailMode
from stateloom.guardrails.output_scanner import SystemPromptLeakScanner
from stateloom.guardrails.patterns import GUARDRAIL_PATTERNS, scan_text
from stateloom.middleware.base import MiddlewareContext

logger = logging.getLogger("stateloom.middleware.guardrails")

# Store key for persisted guardrails config state
_STORE_KEY_GUARDRAILS_CONFIG = "guardrails_config_json"

# How often to poll the store for cross-process state (seconds)
_STORE_POLL_INTERVAL = 2.0


class GuardrailMiddleware:
    """Scans LLM requests for prompt injection/jailbreak and responses for system prompt leaks."""

    def __init__(
        self,
        config: StateLoomConfig,
        store: Any = None,
        metrics: Any = None,
        registry: Any = None,
    ) -> None:
        self._config = config
        self._store = store
        self._metrics = metrics
        self._registry = registry
        self._heuristic_patterns = GUARDRAIL_PATTERNS
        self._local_validator: Any | None = None
        self._local_validator_lock = threading.Lock()
        self._nli_classifier: Any | None = None
        self._nli_classifier_lock = threading.Lock()
        self._output_scanner = SystemPromptLeakScanner(
            config.guardrails.system_prompt_leak_threshold,
        )
        self._last_store_poll: float = 0.0

    def _sync_from_store(self) -> None:
        """Poll persisted guardrails config from the store (cross-process sync)."""
        if not self._store:
            return
        now = time.monotonic()
        if now - self._last_store_poll < _STORE_POLL_INTERVAL:
            return
        self._last_store_poll = now
        try:
            blob = self._store.get_secret(_STORE_KEY_GUARDRAILS_CONFIG)
            if not blob:
                return
            data = json.loads(blob)
            if "enabled" in data:
                self._config.guardrails_enabled = bool(data["enabled"])
            if "mode" in data:
                self._config.guardrails_mode = GuardrailMode(data["mode"])
            if "heuristic_enabled" in data:
                self._config.guardrails_heuristic_enabled = bool(data["heuristic_enabled"])
            if "nli_enabled" in data:
                self._config.guardrails_nli_enabled = bool(data["nli_enabled"])
            if "nli_threshold" in data:
                self._config.guardrails_nli_threshold = float(data["nli_threshold"])
            if "local_model_enabled" in data:
                self._config.guardrails_local_model_enabled = bool(data["local_model_enabled"])
            if "output_scanning_enabled" in data:
                self._config.guardrails_output_scanning_enabled = bool(
                    data["output_scanning_enabled"]
                )
            if "disabled_rules" in data:
                self._config.guardrails_disabled_rules = list(data["disabled_rules"])
        except Exception:
            logger.debug("Failed to sync guardrails config from store", exc_info=True)

    def _get_local_validator(self) -> Any:
        """Lazy-init the local Llama-Guard validator (enterprise feature)."""
        if self._local_validator is not None:
            return self._local_validator

        # Enterprise gate: skip if guardrails_local feature is not available
        if self._registry is not None and not self._registry.is_available("guardrails_local"):
            return None

        with self._local_validator_lock:
            if self._local_validator is not None:
                return self._local_validator

            from stateloom.guardrails.local_validator import LocalGuardrailValidator

            self._local_validator = LocalGuardrailValidator(
                model=self._config.guardrails.local_model,
                timeout=self._config.guardrails.local_model_timeout,
            )
            return self._local_validator

    def _get_nli_classifier(self) -> Any:
        """Lazy-init the NLI injection classifier."""
        if self._nli_classifier is not None:
            return self._nli_classifier

        with self._nli_classifier_lock:
            if self._nli_classifier is not None:
                return self._nli_classifier

            from stateloom.guardrails.nli_classifier import NLIInjectionClassifier

            self._nli_classifier = NLIInjectionClassifier(
                model_name=self._config.guardrails.nli_model,
            )
            return self._nli_classifier

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        """Run guardrail checks before and after the LLM call."""
        self._sync_from_store()
        if not self._config.guardrails_enabled:
            return await call_next(ctx)

        # Skip guardrails for CLI internal overhead calls (e.g. haiku/flash
        # quota checks) — these are system-generated, not user content.
        if ctx.request_kwargs.get("_cli_internal"):
            return await call_next(ctx)

        # === PRE-CALL: Input validation ===
        try:
            messages = ctx.request_kwargs.get("messages", [])
            user_text = self._extract_user_text(messages)

            if user_text:
                # Level 1: Heuristic scan (~0ms)
                if self._config.guardrails.heuristic_enabled:
                    blocked = self._scan_heuristic(user_text, ctx)
                    if blocked:
                        return None  # _scan_heuristic raises on block

                # Level 2: NLI classifier (~5-20ms, opt-in via runtime toggle)
                if self._config.guardrails.nli_enabled:
                    blocked = self._scan_nli(user_text, ctx)
                    if blocked:
                        return None

                # Level 3: Local model scan
                if self._config.guardrails.local_model_enabled:
                    blocked = self._scan_local_model(messages, ctx)
                    if blocked:
                        return None

        except StateLoomGuardrailError:
            raise
        except Exception:
            logger.debug("Guardrail input scan error (fail-open)", exc_info=True)

        # === LLM CALL ===
        result = await call_next(ctx)

        # === POST-CALL: Output validation ===
        if self._config.guardrails.output_scanning_enabled:
            try:
                self._scan_output(ctx, result)
            except Exception:
                logger.debug("Guardrail output scan error (fail-open)", exc_info=True)

        return result

    @staticmethod
    def _extract_user_text(messages: list[dict]) -> str:
        """Extract text from the last few user messages for scanning."""
        parts: list[str] = []
        count = 0
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part["text"])
            count += 1
            if count >= 3:
                break
        return "\n".join(reversed(parts))

    def _scan_heuristic(self, text: str, ctx: MiddlewareContext) -> bool:
        """Run heuristic regex patterns. Returns True if blocked."""
        matches = scan_text(text, disabled_rules=self._config.guardrails.disabled_rules)
        if not matches:
            return False

        logger.info(
            "Guardrail heuristic: %d match(es) in session=%s (rules: %s)",
            len(matches), ctx.session.id,
            ", ".join(m.pattern_name for m in matches),
        )
        enforce = self._config.guardrails.mode == GuardrailMode.ENFORCE

        for match in matches:
            event = GuardrailEvent(
                session_id=ctx.session.id,
                step=ctx.session.step_counter,
                rule_name=match.pattern_name,
                category=match.category,
                severity=match.severity,
                score=1.0,
                action_taken=ActionTaken.BLOCKED
                if enforce and match.severity in ("high", "critical")
                else ActionTaken.LOGGED,
                violation_text=match.matched_text[:200],
                scan_phase="input",
                validator_type="heuristic",
            )
            ctx.events.append(event)
            ctx.session.add_guardrail_detection()

            self._fire_webhook(event)

        # Block in enforce mode for high/critical severity
        if enforce:
            critical_match = next(
                (m for m in matches if m.severity in ("high", "critical")),
                None,
            )
            if critical_match:
                self._save_events_directly(ctx)
                raise StateLoomGuardrailError(
                    rule_name=critical_match.pattern_name,
                    category=critical_match.category,
                    session_id=ctx.session.id,
                )

        return False

    def _scan_nli(self, text: str, ctx: MiddlewareContext) -> bool:
        """Run NLI injection classification. Returns True if blocked."""
        classifier = self._get_nli_classifier()
        if classifier is None:
            return False

        score = classifier.classify(text)
        if score is None:
            return False  # fail-open: NLI unavailable or error

        threshold = self._config.guardrails.nli_threshold
        if score < threshold:
            return False

        # Map score to severity: >0.9 → critical, >0.8 → high, else → medium
        if score > 0.9:
            severity = "critical"
        elif score > 0.8:
            severity = "high"
        else:
            severity = "medium"

        logger.info(
            "Guardrail NLI: injection detected in session=%s score=%.3f severity=%s",
            ctx.session.id,
            score,
            severity,
        )
        enforce = self._config.guardrails.mode == GuardrailMode.ENFORCE
        event = GuardrailEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            rule_name="nli_injection",
            category="injection",
            severity=severity,
            score=round(score, 4),
            action_taken=ActionTaken.BLOCKED if enforce and severity in ("high", "critical") else ActionTaken.LOGGED,
            violation_text=text[:200],
            scan_phase="input",
            validator_type="nli",
        )
        ctx.events.append(event)
        ctx.session.add_guardrail_detection()
        self._fire_webhook(event)

        # Block in enforce mode for high/critical severity
        if enforce and severity in ("high", "critical"):
            self._save_events_directly(ctx)
            raise StateLoomGuardrailError(
                rule_name="nli_injection",
                category="injection",
                session_id=ctx.session.id,
            )

        return False

    def _scan_local_model(self, messages: list[dict], ctx: MiddlewareContext) -> bool:
        """Run Llama-Guard validation. Returns True if blocked."""
        validator = self._get_local_validator()
        if validator is None:
            return False
        result = validator.validate(messages)

        if result.safe:
            return False

        logger.info(
            "Guardrail local model: unsafe detected in session=%s category=%s severity=%s",
            ctx.session.id, result.category, result.severity,
        )
        enforce = self._config.guardrails.mode == GuardrailMode.ENFORCE
        event = GuardrailEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            rule_name="llama_guard",
            category=result.category,
            severity=result.severity,
            score=result.score,
            action_taken=ActionTaken.BLOCKED if enforce else ActionTaken.LOGGED,
            violation_text=result.raw_output[:200],
            scan_phase="input",
            validator_type="local_model",
        )
        ctx.events.append(event)
        ctx.session.add_guardrail_detection()
        self._fire_webhook(event)

        if enforce:
            self._save_events_directly(ctx)
            raise StateLoomGuardrailError(
                rule_name="llama_guard",
                category=result.category,
                session_id=ctx.session.id,
            )

        return False

    def _scan_output(self, ctx: MiddlewareContext, result: Any) -> None:
        """Scan LLM response for system prompt leaks."""
        # Extract system prompt from request
        system_prompt = ""
        messages = ctx.request_kwargs.get("messages", [])
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_prompt = content
                break
        if not system_prompt:
            system_prompt = ctx.request_kwargs.get("system", "")

        if not system_prompt:
            return

        # Extract response text
        response_text = self._extract_response_text(result)
        if not response_text:
            return

        scan_result = self._output_scanner.scan(response_text, system_prompt)
        if scan_result.safe:
            return

        logger.warning(
            "Guardrail output: system prompt leak detected in session=%s score=%.3f",
            ctx.session.id, scan_result.score,
        )

        event = GuardrailEvent(
            session_id=ctx.session.id,
            step=ctx.session.step_counter,
            rule_name="system_prompt_leak",
            category="system_prompt_leak",
            severity="high",
            score=scan_result.score,
            action_taken=ActionTaken.LOGGED,  # output leaks are always logged, never blocked
            violation_text=f"leak_score={scan_result.score:.3f}",
            scan_phase="output",
            validator_type="output_scanner",
        )
        ctx.events.append(event)
        ctx.session.add_guardrail_detection()
        self._fire_webhook(event)

    @staticmethod
    def _extract_response_text(result: Any) -> str:
        """Extract plain text from an LLM response (any provider format)."""
        if result is None:
            return ""

        # Dict response (passthrough proxy)
        if isinstance(result, dict):
            # OpenAI format
            choices = result.get("choices", [])
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message", {})
                if isinstance(msg, dict):
                    return msg.get("content", "") or ""

            # Anthropic format
            content = result.get("content", [])
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                if parts:
                    return "\n".join(parts)

            return ""

        # OpenAI SDK response
        if hasattr(result, "choices") and result.choices:
            choice = result.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content or ""

        # Anthropic SDK response
        if hasattr(result, "content") and isinstance(result.content, list):
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            if parts:
                return "\n".join(parts)

        return ""

    def _save_events_directly(self, ctx: MiddlewareContext) -> None:
        """Persist events directly to the store (bypass EventRecorder).

        Same pattern as PIIScannerMiddleware — when blocking before EventRecorder
        runs, save events here so they appear in the dashboard.
        """
        if not self._store:
            return
        for event in ctx.events:
            try:
                self._store.save_event(event)
            except Exception:
                logger.debug("Failed to persist guardrail event", exc_info=True)
        try:
            self._store.save_session(ctx.session)
        except Exception:
            logger.debug("Failed to persist session after guardrail block", exc_info=True)

    def _fire_webhook(self, event: GuardrailEvent) -> None:
        """Fire webhook notification on violation (non-blocking, daemon thread)."""
        url = self._config.guardrails.webhook_url
        if not url:
            return

        def _send() -> None:
            try:
                import httpx

                payload = {
                    "event_type": "guardrail_violation",
                    "rule_name": event.rule_name,
                    "category": event.category,
                    "severity": event.severity,
                    "score": event.score,
                    "action_taken": event.action_taken,
                    "session_id": event.session_id,
                    "scan_phase": event.scan_phase,
                    "validator_type": event.validator_type,
                }
                httpx.post(url, json=payload, timeout=10.0)
            except Exception:
                logger.debug("Guardrail webhook failed", exc_info=True)

        t = threading.Thread(target=_send, daemon=True)
        t.start()
