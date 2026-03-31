"""Local guardrail validation using Llama-Guard via Ollama."""

from __future__ import annotations

import logging
import threading
from typing import Any

from stateloom.guardrails.validators import GuardrailResult

logger = logging.getLogger("stateloom.guardrails.local")

# Llama-Guard 3 safety categories
_LLAMA_GUARD_CATEGORIES: dict[str, tuple[str, str]] = {
    "S1": ("Violent Crimes", "critical"),
    "S2": ("Non-Violent Crimes", "high"),
    "S3": ("Sex-Related Crimes", "critical"),
    "S4": ("Child Sexual Exploitation", "critical"),
    "S5": ("Defamation", "medium"),
    "S6": ("Specialized Advice", "medium"),
    "S7": ("Privacy", "high"),
    "S8": ("Intellectual Property", "medium"),
    "S9": ("Indiscriminate Weapons", "critical"),
    "S10": ("Hate", "high"),
    "S11": ("Suicide & Self-Harm", "critical"),
    "S12": ("Sexual Content", "medium"),
    "S13": ("Elections", "medium"),
}

_SAFE_RESULT = GuardrailResult(safe=True, score=0.0, rule_name="llama_guard")


class LocalGuardrailValidator:
    """Validates prompts using Llama-Guard via Ollama."""

    def __init__(self, model: str = "llama-guard3:1b", timeout: float = 10.0) -> None:
        self._model = model
        self._timeout = timeout
        self._client: Any | None = None
        self._lock = threading.Lock()
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        """Whether Ollama + Llama-Guard is available."""
        if self._available is None:
            self._lazy_init()
        return self._available is True

    def _lazy_init(self) -> bool:
        """Check Ollama availability and auto-pull model if needed."""
        if self._available is not None:
            return self._available

        with self._lock:
            if self._available is not None:
                return self._available

            try:
                from stateloom.local.client import OllamaClient

                client = OllamaClient(timeout=self._timeout)

                # Check if Ollama is reachable
                models = client.list_models()
                model_names = [m.get("name", "").split(":")[0] for m in models]
                base_model = self._model.split(":")[0]

                if base_model not in model_names:
                    # Try to pull the model
                    logger.info("Pulling Llama-Guard model: %s", self._model)
                    try:
                        client.pull_model(self._model)
                    except Exception as e:
                        logger.warning("Failed to pull %s: %s", self._model, e)
                        self._available = False
                        return False

                self._client = client
                self._available = True
                return True
            except Exception as e:
                logger.debug("Ollama not available for guardrails: %s", e)
                self._available = False
                return False

    def validate(self, messages: list[dict[str, Any]]) -> GuardrailResult:
        """Classify messages as safe/unsafe using Llama-Guard.

        Returns safe result on any error (fail-open).
        """
        if not self._lazy_init() or self._client is None:
            return _SAFE_RESULT

        try:
            # Build Llama-Guard conversation prompt
            conversation_parts: list[str] = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                if not content:
                    continue
                label = "User" if role in ("user", "system") else "Agent"
                conversation_parts.append(f"{label}: {content[:1000]}")

            if not conversation_parts:
                return _SAFE_RESULT

            conversation = "\n".join(conversation_parts)

            # Llama-Guard expects a single user message with the conversation
            guard_messages = [
                {
                    "role": "user",
                    "content": (
                        "Task: Check if there is unsafe content in 'User' messages "
                        "in conversations according to our safety policy.\n\n"
                        "<BEGIN CONVERSATION>\n"
                        f"{conversation}\n"
                        "<END CONVERSATION>\n\n"
                        "Provide your safety assessment for the user message in the "
                        "above conversation. First line must be 'safe' or 'unsafe'."
                    ),
                }
            ]

            response = self._client.chat(self._model, guard_messages)
            raw_output = response.content.strip().lower()

            return self._parse_response(raw_output)

        except Exception as e:
            logger.debug("Llama-Guard validation error (fail-open): %s", e)
            return _SAFE_RESULT

    def _parse_response(self, raw_output: str) -> GuardrailResult:
        """Parse Llama-Guard output into a GuardrailResult."""
        lines = raw_output.strip().split("\n")
        first_line = lines[0].strip() if lines else ""

        if first_line == "safe":
            return GuardrailResult(
                safe=True,
                score=0.0,
                raw_output=raw_output,
                rule_name="llama_guard",
            )

        if first_line.startswith("unsafe"):
            category_code = ""
            if len(lines) > 1:
                category_code = lines[1].strip().upper()
            elif " " in first_line:
                # Handle "unsafe S7" (space-separated) format
                parts = first_line.split(None, 1)
                if len(parts) > 1:
                    category_code = parts[1].strip().upper()

            cat_info = _LLAMA_GUARD_CATEGORIES.get(category_code, ("Unknown", "medium"))
            category_name = f"{category_code}: {cat_info[0]}" if category_code else "unsafe"
            severity = cat_info[1]

            return GuardrailResult(
                safe=False,
                category=category_name,
                score=1.0,
                raw_output=raw_output,
                rule_name="llama_guard",
                severity=severity,
            )

        # Unrecognized output — treat as safe (fail-open)
        logger.debug("Unrecognized Llama-Guard output: %s", raw_output[:200])
        return GuardrailResult(
            safe=True,
            score=0.0,
            raw_output=raw_output,
            rule_name="llama_guard",
        )
