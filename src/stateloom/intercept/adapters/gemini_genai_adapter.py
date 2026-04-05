"""Google GenAI SDK (google-genai) adapter with full tool/function-calling support.

Built for the new ``google-genai`` SDK which supports ``parameters_json_schema``
on tool definitions (raw JSON Schema passthrough, no protobuf issues).

This adapter is a proxy-path adapter — ``get_patch_targets()`` returns ``[]``.
Auto-patching remains handled by the existing ``GenaiAdapter`` + ``patch_genai()``.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from stateloom.core.types import Provider
from stateloom.intercept.provider_adapter import BaseProviderAdapter, TokenFieldMap

if TYPE_CHECKING:
    from stateloom.middleware.base import StreamChunkInfo

logger = logging.getLogger("stateloom.intercept.gemini_genai")

_TOKEN_FIELDS = TokenFieldMap(
    usage_attr="usage_metadata",
    input_field="prompt_token_count",
    output_field="candidates_token_count",
    total_field="total_token_count",
)


def _convert_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions to Gemini function declarations.

    Uses ``parameters_json_schema`` for raw JSON Schema passthrough (new SDK).
    """
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        decl: dict[str, Any] = {"name": func.get("name", "")}
        if func.get("description"):
            decl["description"] = func["description"]
        params = func.get("parameters")
        if params:
            decl["parameters_json_schema"] = params
        declarations.append(decl)
    return declarations


def _convert_tool_choice(tool_choice: str | dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI ``tool_choice`` to Gemini ``tool_config``."""
    if isinstance(tool_choice, str):
        mode_map = {
            "auto": "AUTO",
            "none": "NONE",
            "required": "ANY",
        }
        mode = mode_map.get(tool_choice, "AUTO")
        return {"function_calling_config": {"mode": mode}}

    # {"type": "function", "function": {"name": "..."}}
    if isinstance(tool_choice, dict):
        func = tool_choice.get("function", {})
        name = func.get("name", "")
        if name:
            return {
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [name],
                }
            }
    return {"function_calling_config": {"mode": "AUTO"}}


def _convert_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Convert OpenAI messages to Gemini contents + system_instruction.

    Handles:
    - ``role: "system"`` → extracted as ``system_instruction``
    - ``tool_calls`` on assistant messages → ``functionCall`` parts
    - ``role: "tool"`` messages → ``functionResponse`` parts
    """
    contents: list[dict[str, Any]] = []
    system_instruction: str | None = None

    # Build a map of tool_call_id → function name from assistant messages
    # so we can resolve names for tool result messages that lack "name".
    tc_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                tc_id = tc.get("id", "")
                func_name = tc.get("function", {}).get("name", "")
                if tc_id and func_name:
                    tc_id_to_name[tc_id] = func_name

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_instruction = content if isinstance(content, str) else str(content)
            continue

        if role == "tool":
            # Tool result → functionResponse part
            tool_call_id = msg.get("tool_call_id", "")
            # Try to parse content as JSON; fall back to wrapping in {"result": ...}
            try:
                response_data = json.loads(content) if isinstance(content, str) else content
            except (json.JSONDecodeError, TypeError):
                response_data = {"result": content}
            # Gemini requires response to be a dict
            if not isinstance(response_data, dict):
                response_data = {"result": response_data}

            # Resolve function name: explicit > lookup from prior tool_calls > fallback
            func_name = msg.get("name") or tc_id_to_name.get(tool_call_id) or tool_call_id

            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": func_name,
                                "response": response_data,
                            }
                        }
                    ],
                }
            )
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []

            # Text content
            if content:
                parts.append({"text": str(content)})

            # Tool calls → functionCall parts
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {}
                parts.append(
                    {
                        "functionCall": {
                            "name": func.get("name", ""),
                            "args": args,
                        }
                    }
                )

            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        # user or any other role
        gemini_role = "user"
        contents.append(
            {
                "role": gemini_role,
                "parts": [{"text": str(content)}],
            }
        )

    return contents, system_instruction


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract OpenAI-format tool call dicts from a Gemini response.

    Checks ``candidates[0].content.parts`` for ``function_call`` attributes.
    """
    tool_calls: list[dict[str, Any]] = []
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return tool_calls
        content = getattr(candidates[0], "content", None)
        if not content:
            return tool_calls
        parts = getattr(content, "parts", None)
        if not parts:
            return tool_calls

        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            name = getattr(fc, "name", "")
            args = getattr(fc, "args", {})
            if isinstance(args, dict):
                args_str = json.dumps(args)
            else:
                args_str = json.dumps(dict(args)) if args else "{}"
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str,
                    },
                }
            )
    except Exception:
        logger.debug("Failed to extract tool calls from Gemini response", exc_info=True)
    return tool_calls


class GeminiGenaiAdapter(BaseProviderAdapter):
    """Adapter for the ``google-genai`` SDK with full tool/function-calling support.

    This is a proxy-path adapter — it implements ``prepare_chat()`` for the
    ``Client`` but does not register patch targets.  Auto-patching is handled
    by the existing ``GenaiAdapter`` + ``patch_genai()``.
    """

    # ---- Core properties ----

    @property
    def name(self) -> str:
        return Provider.GEMINI

    @property
    def method_label(self) -> str:
        return "generate_content"

    def get_patch_targets(self) -> list[Any]:
        return []

    # ---- Model extraction ----

    def extract_model(self, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        model = kwargs.get("model", "unknown")
        return model if isinstance(model, str) else str(model)

    # ---- Token extraction ----

    def extract_tokens(self, response: Any) -> tuple[int, int, int]:
        return self._extract_tokens_from_fields(response, _TOKEN_FIELDS)

    # ---- Content extraction ----

    def extract_content(self, response: Any) -> str:
        # Check for function calls first — if response is tool-call-only, return ""
        tool_calls = _extract_tool_calls(response)
        if tool_calls:
            # Still extract text if mixed (text + function calls)
            try:
                candidates = getattr(response, "candidates", None)
                if candidates:
                    content = getattr(candidates[0], "content", None)
                    if content:
                        parts = getattr(content, "parts", None)
                        if parts:
                            texts = []
                            for part in parts:
                                if getattr(part, "function_call", None) is None:
                                    text = getattr(part, "text", None)
                                    if isinstance(text, str) and text:
                                        texts.append(text)
                            return " ".join(texts) if texts else ""
            except Exception:
                pass
            return ""

        # Standard text extraction (same as GenaiAdapter)
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

    # ---- Response modification ----

    def modify_response_text(self, response: Any, modifier: Callable[[str], str]) -> None:
        try:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            part.text = modifier(part.text)
        except Exception:
            pass

    # ---- OpenAI dict conversion (with tool call support) ----

    def to_openai_dict(self, response: Any, model: str, request_id: str) -> dict[str, Any]:
        """Convert Gemini response to OpenAI ChatCompletion dict.

        Checks for function_call parts in the response; if present, emits
        ``finish_reason="tool_calls"`` and a ``tool_calls`` array.
        """
        tool_calls = _extract_tool_calls(response)
        content = self.extract_content(response)

        prompt_tokens = 0
        completion_tokens = 0
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        finish_reason = "tool_calls" if tool_calls else "stop"

        message_dict: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message_dict["tool_calls"] = tool_calls

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    # ---- Streaming ----

    def is_streaming(self, kwargs: dict[str, Any]) -> bool:
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

    # ---- System prompt / request normalization ----

    def apply_system_prompt(self, kwargs: dict[str, Any], prompt: str) -> None:
        kwargs["system_instruction"] = prompt

    # ---- URL / model patterns ----

    def extract_base_url(self, instance: Any) -> str:
        return "https://generativelanguage.googleapis.com"

    @property
    def model_patterns(self) -> list[re.Pattern[str]]:
        return [re.compile(r"^gemini-")]

    @property
    def default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com"

    # ---- prepare_chat (proxy path) ----

    def prepare_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        provider_keys: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Callable[[], Any]]:
        contents, system_instruction = _convert_messages(messages)

        gen_config: dict[str, Any] = {}
        for k in ("temperature", "top_p", "top_k"):
            if k in kwargs:
                gen_config[k] = kwargs.pop(k)
        if "max_tokens" in kwargs:
            gen_config["max_output_tokens"] = kwargs.pop("max_tokens")
        if "max_output_tokens" in kwargs:
            gen_config["max_output_tokens"] = kwargs.pop("max_output_tokens")

        # Build Gemini tools from OpenAI tool definitions
        openai_tools = kwargs.pop("tools", None)
        gemini_tools: list[dict[str, Any]] | None = None
        if openai_tools:
            declarations = _convert_openai_tools(openai_tools)
            if declarations:
                gemini_tools = [{"function_declarations": declarations}]

        # Convert tool_choice
        openai_tool_choice = kwargs.pop("tool_choice", None)
        gemini_tool_config: dict[str, Any] | None = None
        if openai_tool_choice is not None:
            gemini_tool_config = _convert_tool_choice(openai_tool_choice)

        # request_kwargs includes both formats for middleware compatibility
        request_kwargs: dict[str, Any] = {
            "contents": contents,
            "messages": messages,
            **kwargs,
        }
        if system_instruction:
            request_kwargs["system_instruction"] = system_instruction

        _gen_config = gen_config
        _gemini_tools = gemini_tools
        _gemini_tool_config = gemini_tool_config
        _rk = request_kwargs
        keys = provider_keys or {}

        def llm_call() -> Any:
            try:
                from google.genai import Client as GenaiClient
                from google.genai import models as genai_models
            except ImportError:
                raise ImportError(
                    "google-genai package is required for Gemini models. "
                    "Install with: pip install google-genai"
                )

            from stateloom.intercept.unpatch import get_original

            # Rebuild contents from messages at call time so middleware
            # modifications (e.g. PII redaction) are reflected.
            live_messages = _rk.get("messages", [])
            live_contents, live_system = _convert_messages(live_messages)

            client_kwargs: dict[str, Any] = {}
            if keys.get("google"):
                client_kwargs["api_key"] = keys["google"]
            client = GenaiClient(**client_kwargs)

            call_kwargs: dict[str, Any] = {
                "model": model,
                "contents": live_contents,
            }
            if live_system:
                call_kwargs["config"] = {}
                call_kwargs["config"]["system_instruction"] = live_system
            if _gen_config:
                config = call_kwargs.setdefault("config", {})
                config.update(_gen_config)
            if _gemini_tools:
                call_kwargs["config"] = call_kwargs.get("config", {})
                call_kwargs["config"]["tools"] = _gemini_tools
            if _gemini_tool_config:
                call_kwargs["config"] = call_kwargs.get("config", {})
                call_kwargs["config"]["tool_config"] = _gemini_tool_config

            # Use the unpatched method to avoid double middleware execution —
            # patch_genai() may have monkey-patched Models.generate_content.
            original = get_original(genai_models.Models, "generate_content")
            if original:
                return original(client.models, **call_kwargs)
            return client.models.generate_content(**call_kwargs)

        return request_kwargs, llm_call
