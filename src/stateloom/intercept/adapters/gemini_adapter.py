"""Google Gemini provider adapter."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from stateloom.core.types import Provider

if TYPE_CHECKING:
    from stateloom.middleware.base import StreamChunkInfo
from stateloom.intercept.provider_adapter import BaseProviderAdapter, PatchTarget, TokenFieldMap

_TOKEN_FIELDS = TokenFieldMap(
    usage_attr="usage_metadata",
    input_field="prompt_token_count",
    output_field="candidates_token_count",
    total_field="total_token_count",
)


class GeminiAdapter(BaseProviderAdapter):
    """Adapter for the Google Generative AI (Gemini) Python SDK."""

    @property
    def name(self) -> str:
        return Provider.GEMINI

    @property
    def method_label(self) -> str:
        return "generate_content"

    def get_patch_targets(self) -> list[PatchTarget]:
        try:
            from google.generativeai import GenerativeModel  # type: ignore[attr-defined]
        except ImportError:
            return []

        return [
            PatchTarget(
                target_class=GenerativeModel,
                method_name="generate_content",
                is_async=False,
                description="gemini.GenerativeModel.generate_content",
            ),
            PatchTarget(
                target_class=GenerativeModel,
                method_name="generate_content_async",
                is_async=True,
                description="gemini.GenerativeModel.generate_content_async",
            ),
        ]

    def extract_model(self, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        return cast(str, getattr(instance, "model_name", None) or kwargs.get("model", "unknown"))

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        return self._extract_tokens_from_fields(response, _TOKEN_FIELDS)

    def extract_content(self, response: Any) -> str:
        # Primary: .text property
        try:
            text = response.text
            if isinstance(text, str) and text:
                return text
        except Exception:
            pass
        # Fallback: candidates[0].content.parts[0].text
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
        """Convert Gemini response to OpenAI ChatCompletion dict."""
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
            # Gemini doesn't have a standard finish_reason on chunks
            candidates = getattr(chunk, "candidates", None)
            if candidates:
                fr = getattr(candidates[0], "finish_reason", None)
                if fr:
                    info.finish_reason = str(fr)
        except (AttributeError, IndexError):
            pass
        return info

    def modify_chunk_text(self, chunk: Any, new_text: str) -> Any:
        """Replace the text in the first part of a Gemini stream chunk."""
        try:
            if chunk.candidates and chunk.candidates[0].content.parts:
                chunk.candidates[0].content.parts[0].text = new_text
        except (AttributeError, IndexError):
            pass
        return chunk

    def apply_system_prompt(self, kwargs: dict[str, Any], prompt: str) -> None:
        kwargs["system_instruction"] = prompt

    def normalize_request(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Move Gemini's positional `contents` arg into kwargs as `messages`."""
        result = dict(kwargs)
        # Gemini's generate_content(contents, ...) — contents is args[0]
        if args:
            contents = args[0]
            if isinstance(contents, str):
                result["messages"] = [{"role": "user", "content": contents}]
            elif isinstance(contents, list):
                messages = []
                for item in contents:
                    if isinstance(item, str):
                        messages.append({"role": "user", "content": item})
                    elif isinstance(item, dict):
                        messages.append(item)
                    elif hasattr(item, "parts"):
                        # Gemini Content object
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
        """Rebuild Gemini's positional ``contents`` arg from modified messages."""
        messages = normalized_kwargs.get("messages")
        if messages is None or not original_args:
            return original_args, original_kwargs

        # Single-message string shorthand — keep it as a plain string
        if (
            len(messages) == 1
            and isinstance(original_args[0], str)
            and messages[0].get("role") == "user"
        ):
            return (messages[0].get("content", ""),), original_kwargs

        # Multi-message or complex — rebuild as Content dicts
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": str(msg.get("content", ""))}]})
        return (contents,), original_kwargs

    def extract_base_url(self, instance: Any) -> str:
        return "https://generativelanguage.googleapis.com"

    def get_instance_targets(self, client: Any) -> list[tuple[Any, str]]:
        if hasattr(client, "generate_content"):
            return [(client, "generate_content")]
        return []

    @property
    def model_patterns(self) -> list[re.Pattern[str]]:
        return [re.compile(r"^gemini-")]

    @property
    def default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com"

    def prepare_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        provider_keys: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Callable[[], Any]]:
        contents: list[dict[str, Any]] = []
        system_instruction: str | None = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content if isinstance(content, str) else str(content)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": str(content)}]})

        gen_config: dict[str, Any] = {}
        for k in ("temperature", "top_p", "top_k"):
            if k in kwargs:
                gen_config[k] = kwargs.pop(k)
        # Translate OpenAI-style max_tokens to Gemini's max_output_tokens
        if "max_tokens" in kwargs:
            gen_config["max_output_tokens"] = kwargs.pop("max_tokens")
        if "max_output_tokens" in kwargs:
            gen_config["max_output_tokens"] = kwargs.pop("max_output_tokens")

        # Include both formats: "contents" for Gemini-aware code, "messages"
        # for middleware that expects OpenAI format (PII, cache, etc.)
        request_kwargs: dict[str, Any] = {
            "contents": contents,
            "messages": messages,
            **kwargs,
        }
        if system_instruction:
            request_kwargs["system_instruction"] = system_instruction

        _gen_config = gen_config
        _rk = request_kwargs  # closure reference to the SAME dict middleware modifies
        keys = provider_keys or {}

        def llm_call() -> Any:
            try:
                from google.generativeai import GenerativeModel  # type: ignore[attr-defined]
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required for Gemini models. "
                    "Install with: pip install google-generativeai"
                )

            from stateloom.intercept.unpatch import get_original

            if keys.get("google"):
                import google.generativeai as genai

                genai.configure(api_key=keys["google"])  # type: ignore[attr-defined]

            # Rebuild contents from messages at call time so middleware
            # modifications (e.g. PII redaction) are reflected.
            live_messages = _rk.get("messages", [])
            live_contents: list[dict[str, Any]] = []
            live_system: str | None = None
            for msg in live_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    live_system = content if isinstance(content, str) else str(content)
                    continue
                gemini_role = "model" if role == "assistant" else "user"
                live_contents.append({"role": gemini_role, "parts": [{"text": str(content)}]})

            gen_model = GenerativeModel(
                model,
                system_instruction=live_system,
            )

            original = get_original(GenerativeModel, "generate_content")
            if original:
                return original(
                    gen_model,
                    live_contents,
                    generation_config=_gen_config or None,
                )
            return gen_model.generate_content(
                live_contents,  # type: ignore[arg-type]
                generation_config=_gen_config or None,  # type: ignore[arg-type]
            )

        return request_kwargs, llm_call
