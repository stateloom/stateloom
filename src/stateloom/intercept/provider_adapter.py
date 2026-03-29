"""Provider adapter protocol for extensible LLM provider support."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from stateloom.middleware.base import StreamChunkInfo

logger = logging.getLogger("stateloom.intercept.provider_adapter")

_MAX_UNWRAP_DEPTH = 10


@dataclass
class PatchTarget:
    """Describes a single method to monkey-patch."""

    target_class: Any
    method_name: str
    is_async: bool = False
    description: str = ""


@dataclass
class TokenFieldMap:
    """Declarative field-name mapping for token extraction."""

    usage_attr: str = "usage"
    input_field: str = "prompt_tokens"
    output_field: str = "completion_tokens"
    total_field: str = "total_tokens"  # empty string = compute input+output
    nested_attr: str = ""  # e.g. "tokens" for Cohere's usage.tokens.*


@runtime_checkable
class ProviderAdapter(Protocol):
    """Protocol that any LLM provider adapter must satisfy.

    Implement this protocol (or subclass ``BaseProviderAdapter``) to plug a
    new LLM provider into StateLoom's middleware pipeline.
    """

    @property
    def name(self) -> str:
        """Short provider identifier, e.g. 'mistral', 'cohere'."""
        ...

    @property
    def method_label(self) -> str:
        """Method label for logs/events, e.g. 'chat.completions.create'."""
        ...

    def get_patch_targets(self) -> list[PatchTarget]:
        """Return the class methods to monkey-patch.

        Return an empty list if the provider's SDK is not installed.
        """
        ...

    def extract_model(self, instance: Any, args: tuple, kwargs: dict[str, Any]) -> str:
        """Extract the model name from the call arguments."""
        ...

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        """Extract (prompt_tokens, completion_tokens, total_tokens) from a response."""
        ...

    def is_streaming(self, kwargs: dict[str, Any]) -> bool:
        """Return True if the call is a streaming request."""
        ...

    def extract_stream_tokens(self, chunk: Any, accumulated: dict[str, int]) -> dict[str, int]:
        """Update accumulated token counts from a stream chunk.

        Returns the updated accumulated dict.
        """
        ...

    def extract_chunk_info(self, chunk: Any) -> StreamChunkInfo:
        """Extract provider-agnostic metadata from a single stream chunk."""
        ...

    def apply_system_prompt(self, kwargs: dict[str, Any], prompt: str) -> None:
        """Apply a system prompt to the request kwargs in a provider-specific way."""
        ...

    def get_instance_targets(self, client: Any) -> list[tuple[Any, str]]:
        """Return (sub_object, method_name) pairs for instance wrapping via gate.wrap()."""
        ...

    def normalize_request(self, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Normalize positional args into kwargs for the middleware pipeline.

        Some providers (e.g. Gemini) pass the prompt as a positional arg.
        This method merges it into kwargs so the pipeline can access it.
        Default: return kwargs unchanged.
        """
        ...

    def extract_base_url(self, instance: Any) -> str:
        """Extract the provider API base URL from the SDK client instance.

        Used for compliance endpoint checks (data residency enforcement).
        Returns empty string if the URL cannot be determined.
        """
        ...

    @property
    def model_patterns(self) -> list[re.Pattern[str]]:
        """Regex patterns that match model names belonging to this provider.

        Used by ``resolve_provider()`` to map a model name to a provider
        without hardcoded patterns in the Client.
        """
        ...

    @property
    def default_base_url(self) -> str:
        """Default API base URL for this provider (used by Client for MiddlewareContext)."""
        ...

    def extract_content(self, response: Any) -> str:
        """Extract primary text content from a response. Returns '' on failure."""
        ...

    def modify_response_text(self, response: Any, modifier: Callable[[str], str]) -> None:
        """Apply text transformation to all text in the response (in-place).

        Used for PII rehydration. Fails silently on error.
        """
        ...

    def to_openai_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        """Convert response to OpenAI ChatCompletion dict format.

        Used by the proxy for OpenAI-compatible API responses.
        """
        ...

    def modify_chunk_text(self, chunk: Any, new_text: str) -> Any:
        """Return a copy of chunk with its text delta replaced by new_text."""
        ...

    def confidence_instruction(self) -> str:
        """Provider-preferred instruction for confidence reporting."""
        ...

    def extract_confidence(self, text: str) -> float | None:
        """Extract confidence from response text. Returns None to use default regex."""
        ...

    def prepare_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        provider_keys: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Callable[[], Any]]:
        """Build provider-specific request kwargs and an llm_call callable.

        Returns ``(request_kwargs, llm_call)`` — the same contract as
        ``Client._prepare_call()``.  Adapters that don't implement this
        raise ``NotImplementedError`` so the Client falls back to its
        legacy ``_prepare_*`` methods.
        """
        ...


class BaseProviderAdapter:
    """Convenience base class with sensible defaults.

    Subclass this and override only what differs for your provider.
    """

    @staticmethod
    def _unwrap_client(instance: Any) -> Any:
        """Walk the ``_client`` chain to reach the innermost SDK client.

        Bounded to ``_MAX_UNWRAP_DEPTH`` to prevent infinite loops from
        circular references.
        """
        client = instance
        for _ in range(_MAX_UNWRAP_DEPTH):
            if not hasattr(client, "_client"):
                break
            client = client._client
        else:
            logger.warning(
                "Client unwrap depth exceeded %d — possible circular reference",
                _MAX_UNWRAP_DEPTH,
            )
        return client

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def method_label(self) -> str:
        raise NotImplementedError

    def get_patch_targets(self) -> list[PatchTarget]:
        return []

    def extract_model(self, instance: Any, args: tuple, kwargs: dict[str, Any]) -> str:
        return kwargs.get("model", "unknown")

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        return (0, 0, 0)

    def _extract_tokens_from_fields(
        self, response: Any, fields: TokenFieldMap
    ) -> tuple[int, int, int]:
        """Extract token counts using a declarative field-name mapping."""
        try:
            usage = getattr(response, fields.usage_attr, None)
            if usage is None:
                return (0, 0, 0)
            if fields.nested_attr:
                usage = getattr(usage, fields.nested_attr, None)
                if usage is None:
                    return (0, 0, 0)
            prompt = int(getattr(usage, fields.input_field, 0) or 0)
            completion = int(getattr(usage, fields.output_field, 0) or 0)
            total = (
                int(getattr(usage, fields.total_field, 0) or 0)
                if fields.total_field
                else prompt + completion
            )
            return (prompt, completion, total)
        except (AttributeError, TypeError, ValueError):
            return (0, 0, 0)

    def is_streaming(self, kwargs: dict[str, Any]) -> bool:
        return kwargs.get("stream", False)

    def extract_stream_tokens(self, chunk: Any, accumulated: dict[str, int]) -> dict[str, int]:
        return accumulated

    def extract_chunk_info(self, chunk: Any) -> StreamChunkInfo:
        """Default: return empty StreamChunkInfo."""
        from stateloom.middleware.base import StreamChunkInfo

        return StreamChunkInfo()

    def apply_system_prompt(self, kwargs: dict[str, Any], prompt: str) -> None:
        """Default: OpenAI-style messages list."""
        messages = kwargs.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = prompt
        else:
            messages.insert(0, {"role": "system", "content": prompt})
        kwargs["messages"] = messages

    def get_instance_targets(self, client: Any) -> list[tuple[Any, str]]:
        return []

    def normalize_request(self, args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def extract_content(self, response: Any) -> str:
        """Default: try OpenAI-like choices[0].message.content."""
        try:
            if hasattr(response, "choices") and response.choices:
                msg = getattr(response.choices[0], "message", None)
                if msg:
                    return getattr(msg, "content", "") or ""
        except Exception:
            pass
        return ""

    def modify_response_text(self, response: Any, modifier: Callable[[str], str]) -> None:
        """Default: try OpenAI-like choices[].message.content."""
        try:
            if hasattr(response, "choices"):
                for choice in response.choices:
                    if hasattr(choice, "message") and hasattr(choice.message, "content"):
                        if choice.message.content:
                            choice.message.content = modifier(choice.message.content)
        except Exception:
            pass

    def to_openai_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        """Default: wrap extract_content() result in a minimal completion dict."""
        from stateloom.core.response_helpers import _make_completion

        content = self.extract_content(response)
        return _make_completion(
            content=content,
            model=model,
            request_id=request_id,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def _openai_style_to_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        """Convert an OpenAI-style response object (with .choices + .usage) to dict.

        Handles optional .data wrapper (Mistral). Shared by OpenAI and Mistral
        adapters for their manual fallback paths.
        """
        import logging
        import time

        _logger = logging.getLogger("stateloom.intercept.provider_adapter")

        data = response.data if hasattr(response, "data") else response
        choices: list[dict[str, Any]] = []
        try:
            for choice in data.choices:
                msg = getattr(choice, "message", None)
                choice_dict: dict[str, Any] = {
                    "index": getattr(choice, "index", 0),
                    "message": {
                        "role": getattr(msg, "role", "assistant") if msg else "assistant",
                        "content": getattr(msg, "content", "") if msg else "",
                    },
                    "finish_reason": getattr(choice, "finish_reason", "stop"),
                }
                if msg and getattr(msg, "tool_calls", None):
                    choice_dict["message"]["tool_calls"] = [
                        tc.model_dump() if hasattr(tc, "model_dump") else tc
                        for tc in msg.tool_calls
                    ]
                choices.append(choice_dict)
        except Exception:
            _logger.debug("OpenAI-style dict conversion failed", exc_info=True)

        usage = getattr(data, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": getattr(data, "model", model),
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    def modify_chunk_text(self, chunk: Any, new_text: str) -> Any:
        """Default: return chunk unchanged (no-op for unknown providers)."""
        return chunk

    def extract_base_url(self, instance: Any) -> str:
        return ""

    @property
    def model_patterns(self) -> list[re.Pattern[str]]:
        return []

    @property
    def default_base_url(self) -> str:
        return ""

    def confidence_instruction(self) -> str:
        """Provider-preferred instruction for confidence reporting.

        Override per-adapter for provider-specific formatting.
        """
        return "End your response with your confidence level: [Confidence: X.XX]"

    def extract_confidence(self, text: str) -> float | None:
        """Extract confidence from response text. Returns None to use default regex.

        Override per-adapter for provider-specific parsing.
        """
        return None  # fall back to default regex in consensus/confidence.py

    def prepare_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        provider_keys: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Callable[[], Any]]:
        raise NotImplementedError
