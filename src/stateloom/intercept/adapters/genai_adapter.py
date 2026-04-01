"""Google GenAI SDK (google-genai) provider adapter.

The new ``google-genai`` SDK uses per-client API keys and keyword-only
arguments.  Response objects share the same field names as the legacy
``google-generativeai`` SDK, so token/content extraction logic is identical.

This adapter is self-contained — it does NOT register in the adapter
registry (which would overwrite the existing ``GeminiAdapter`` for the
legacy SDK).  Instead, ``patch_genai()`` is called directly from
``auto_patch()`` following the same pattern as ``patch_litellm()``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from stateloom.core.types import Provider
from stateloom.intercept.provider_adapter import BaseProviderAdapter, PatchTarget, TokenFieldMap
from stateloom.intercept.unpatch import register_patch

if TYPE_CHECKING:
    from stateloom.gate import Gate
    from stateloom.middleware.base import StreamChunkInfo

logger = logging.getLogger("stateloom.intercept.genai")

_TOKEN_FIELDS = TokenFieldMap(
    usage_attr="usage_metadata",
    input_field="prompt_token_count",
    output_field="candidates_token_count",
    total_field="total_token_count",
)


class GenaiAdapter(BaseProviderAdapter):
    """Adapter for the ``google-genai`` Python SDK.

    The new SDK exposes ``google.genai.models.Models`` (sync) and
    ``google.genai.models.AsyncModels`` (async), each with
    ``generate_content`` and ``generate_content_stream`` methods.
    Streaming uses a separate method instead of a ``stream=True`` kwarg,
    so stream targets set ``always_streaming=True`` on their
    ``PatchTarget``.
    """

    @property
    def name(self) -> str:
        return Provider.GEMINI

    @property
    def method_label(self) -> str:
        return "generate_content"

    def get_patch_targets(self) -> list[PatchTarget]:
        try:
            from google.genai import models as genai_models
        except ImportError:
            return []

        models_cls = genai_models.Models
        async_models_cls = genai_models.AsyncModels

        return [
            PatchTarget(
                target_class=models_cls,
                method_name="generate_content",
                is_async=False,
                description="genai.Models.generate_content",
            ),
            PatchTarget(
                target_class=models_cls,
                method_name="generate_content_stream",
                is_async=False,
                description="genai.Models.generate_content_stream",
                always_streaming=True,
            ),
            PatchTarget(
                target_class=async_models_cls,
                method_name="generate_content",
                is_async=True,
                description="genai.AsyncModels.generate_content",
            ),
            PatchTarget(
                target_class=async_models_cls,
                method_name="generate_content_stream",
                is_async=True,
                description="genai.AsyncModels.generate_content_stream",
                always_streaming=True,
            ),
        ]

    def extract_model(self, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        # google-genai uses keyword-only: generate_content(*, model=..., contents=...)
        model = kwargs.get("model", "unknown")
        if isinstance(model, str):
            return model
        return str(model)

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        return self._extract_tokens_from_fields(response, _TOKEN_FIELDS)

    def extract_content(self, response: Any) -> str:
        # Same response structure as legacy SDK
        try:
            text = response.text
            if isinstance(text, str) and text:
                return text
        except Exception:
            pass
        try:
            candidates = getattr(response, "candidates", None)
            if candidates:
                content = getattr(candidates[0], "content", None)
                if content:
                    parts = getattr(content, "parts", None)
                    if parts:
                        text = getattr(parts[0], "text", None)
                        if isinstance(text, str):
                            return text
        except Exception:
            pass
        return ""

    def modify_response_text(self, response: Any, modifier: Callable[[str], str]) -> None:
        try:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            part.text = modifier(part.text)
        except Exception:
            pass

    def to_openai_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        from stateloom.core.response_helpers import _make_completion

        content = self.extract_content(response)
        prompt_tokens = 0
        completion_tokens = 0
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        return _make_completion(
            content=content,
            model=model,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def is_streaming(self, kwargs: dict[str, Any]) -> bool:
        # Streaming is handled by separate methods, not a kwarg
        return False

    def extract_stream_tokens(self, chunk: Any, accumulated: dict[str, int]) -> dict[str, int]:
        try:
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                accumulated["prompt_tokens"] = usage.prompt_token_count or 0
                accumulated["completion_tokens"] = usage.candidates_token_count or 0
        except AttributeError:
            pass
        return accumulated

    def extract_chunk_info(self, chunk: Any) -> StreamChunkInfo:
        from stateloom.middleware.base import StreamChunkInfo

        info = StreamChunkInfo()
        try:
            if hasattr(chunk, "text") and chunk.text:
                info.text_delta = chunk.text
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                info.prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                info.completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
                info.has_usage = True
            candidates = getattr(chunk, "candidates", None)
            if candidates:
                fr = getattr(candidates[0], "finish_reason", None)
                if fr:
                    info.finish_reason = str(fr)
        except (AttributeError, IndexError):
            pass
        return info

    def modify_chunk_text(self, chunk: Any, new_text: str) -> Any:
        try:
            if chunk.candidates and chunk.candidates[0].content.parts:
                chunk.candidates[0].content.parts[0].text = new_text
        except (AttributeError, IndexError):
            pass
        return chunk

    def apply_system_prompt(self, kwargs: dict[str, Any], prompt: str) -> None:
        kwargs["system_instruction"] = prompt

    def normalize_request(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Convert ``contents`` kwarg to OpenAI-style ``messages``."""
        result = dict(kwargs)
        contents = result.pop("contents", None)
        if contents is None:
            return result

        if isinstance(contents, str):
            result["messages"] = [{"role": "user", "content": contents}]
        elif isinstance(contents, list):
            messages: list[dict[str, Any]] = []
            for item in contents:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    # Gemini Content dict: {"role": ..., "parts": [...]}
                    role = item.get("role", "user")
                    parts = item.get("parts", [])
                    text_parts = []
                    for p in parts:
                        if isinstance(p, str):
                            text_parts.append(p)
                        elif isinstance(p, dict) and "text" in p:
                            text_parts.append(p["text"])
                    openai_role = "assistant" if role == "model" else role
                    messages.append({"role": openai_role, "content": " ".join(text_parts)})
                elif hasattr(item, "parts"):
                    text = " ".join(getattr(p, "text", str(p)) for p in item.parts)
                    role = getattr(item, "role", "user") or "user"
                    messages.append({"role": role, "content": text})
            result["messages"] = messages
        return result

    def rebuild_call_args(
        self,
        normalized_kwargs: dict[str, Any],
        original_args: tuple[Any, ...],
        original_kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Rebuild ``contents`` kwarg from middleware-modified ``messages``."""
        messages = normalized_kwargs.get("messages")
        if messages is None:
            return original_args, original_kwargs

        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": str(msg.get("content", ""))}]})

        rebuilt_kwargs = dict(original_kwargs)
        rebuilt_kwargs["contents"] = contents
        # google-genai SDK uses keyword-only args and never accepts "stream"
        rebuilt_kwargs.pop("stream", None)
        return original_args, rebuilt_kwargs

    def extract_base_url(self, instance: Any) -> str:
        return "https://generativelanguage.googleapis.com"

    @property
    def model_patterns(self) -> list[re.Pattern[str]]:
        return [re.compile(r"^gemini-")]

    @property
    def default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com"


def patch_genai(gate: Gate) -> list[str]:
    """Patch ``google.genai`` Models and AsyncModels.

    Self-contained patching function (like ``patch_litellm``).
    Returns a list of human-readable descriptions of what was patched.
    """
    try:
        from google.genai import models as genai_models
    except ImportError:
        return []

    from stateloom.intercept.generic_interceptor import _build_async_wrapper, _build_sync_wrapper

    adapter = GenaiAdapter()
    patched: list[str] = []

    models_cls = genai_models.Models
    async_models_cls = genai_models.AsyncModels

    # --- Sync targets ---
    for method_name, always_streaming in [
        ("generate_content", False),
        ("generate_content_stream", True),
    ]:
        original = getattr(models_cls, method_name)
        wrapper = _build_sync_wrapper(gate, adapter, original, always_streaming=always_streaming)
        setattr(models_cls, method_name, wrapper)
        desc = f"genai.Models.{method_name}"
        register_patch(models_cls, method_name, original, desc)
        kind = "sync+stream" if always_streaming else "sync"
        patched.append(f"{desc} ({kind})")

    # --- Async targets ---
    for method_name, always_streaming in [
        ("generate_content", False),
        ("generate_content_stream", True),
    ]:
        original = getattr(async_models_cls, method_name)
        wrapper = _build_async_wrapper(gate, adapter, original, always_streaming=always_streaming)
        setattr(async_models_cls, method_name, wrapper)
        desc = f"genai.AsyncModels.{method_name}"
        register_patch(async_models_cls, method_name, original, desc)
        kind = "async+stream" if always_streaming else "async"
        patched.append(f"{desc} ({kind})")

    logger.info("[StateLoom] Patched google-genai Models and AsyncModels")
    return patched
