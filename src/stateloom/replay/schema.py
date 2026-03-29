"""JSON serialization for replay — no pickle, versioned schema."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger("stateloom.replay.schema")

from pydantic import BaseModel, Field

SESSION_SCHEMA_VERSION = "v1"


class CachedStreamChunks:
    """Replays cached stream chunks with optional inter-chunk delay.

    Supports both sync iteration (``for chunk in cached``) and async
    iteration (``async for chunk in cached``), so the same object can be
    returned from both ``_intercept_sync`` and ``_intercept_async``.
    """

    def __init__(self, chunks: list[Any], delay_ms: float = 0) -> None:
        self.chunks = chunks
        self.delay_ms = delay_ms

    def __iter__(self):
        for chunk in self.chunks:
            if self.delay_ms > 0:
                time.sleep(self.delay_ms / 1000)
            yield chunk

    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay_ms > 0:
                await asyncio.sleep(self.delay_ms / 1000)
            yield chunk


class StepRecordSchema(BaseModel):
    """Serializable step record for replay sessions."""

    step: int
    event_type: str
    request_hash: str = ""
    cached_response_json: str | None = None
    tool_name: str | None = None
    mutates_state: bool = False
    provider: str | None = None
    model: str | None = None


class SessionRecord(BaseModel):
    """Versioned session record for replay persistence."""

    version: str = SESSION_SCHEMA_VERSION
    session_id: str
    steps: list[StepRecordSchema] = Field(default_factory=list)


def serialize_response(response: Any) -> str:
    """Serialize an LLM response to JSON string.

    Uses model_dump_json() for Pydantic models, json.dumps() for plain dicts.
    For Gemini proxy objects, extracts text and token counts into a portable dict.
    """
    if response is None:
        return json.dumps(None)

    # Handle Gemini proxy objects (from response_converter or raw Gemini SDK)
    if hasattr(response, "text") and hasattr(response, "usage_metadata"):
        data: dict[str, Any] = {"_type": "gemini", "text": response.text}
        usage = response.usage_metadata
        if usage:
            data["prompt_tokens"] = getattr(usage, "prompt_token_count", 0)
            data["completion_tokens"] = getattr(usage, "candidates_token_count", 0)
        return json.dumps(data)

    if hasattr(response, "model_dump_json"):
        return response.model_dump_json()
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump())
    return json.dumps(response, default=str)


def serialize_stream_chunks(chunks: list[Any], provider: str) -> str:
    """Serialize a list of stream chunks to a JSON string.

    Each chunk is serialized via model_dump() / dict() fallback.
    The result is a JSON object with ``_type: "stream"`` marker so
    ``deserialize_response()`` can detect and reconstruct it.
    """
    serialized: list[Any] = []
    for chunk in chunks:
        try:
            if hasattr(chunk, "model_dump"):
                serialized.append(chunk.model_dump())
            elif hasattr(chunk, "dict"):
                serialized.append(chunk.dict())
            elif isinstance(chunk, dict):
                serialized.append(chunk)
            else:
                serialized.append(json.loads(json.dumps(chunk, default=str)))
        except Exception:
            logger.debug(
                "Stream chunk serialization fallback to str: %s",
                type(chunk).__name__,
            )
            serialized.append(str(chunk))
    return json.dumps({"_type": "stream", "provider": provider, "chunks": serialized})


def deserialize_stream_chunks(data: dict[str, Any], delay_ms: float = 0) -> CachedStreamChunks:
    """Reconstruct a CachedStreamChunks from a deserialized stream envelope.

    Attempts provider-specific chunk reconstruction; falls back to raw dicts.
    """
    provider = data.get("provider", "")
    raw_chunks = data.get("chunks", [])
    reconstructed: list[Any] = []

    for raw in raw_chunks:
        if not isinstance(raw, dict):
            reconstructed.append(raw)
            continue

        chunk_obj: Any = raw  # fallback

        if provider == "openai":
            try:
                from openai.types.chat import ChatCompletionChunk

                chunk_obj = ChatCompletionChunk.model_validate(raw)
            except Exception:
                logger.debug("OpenAI stream chunk reconstruction failed, using raw dict")

        elif provider == "anthropic":
            try:
                chunk_obj = _reconstruct_anthropic_stream_event(raw)
            except Exception:
                logger.debug("Anthropic stream chunk reconstruction failed, using raw dict")

        reconstructed.append(chunk_obj)

    return CachedStreamChunks(reconstructed, delay_ms=delay_ms)


def _reconstruct_anthropic_stream_event(raw: dict[str, Any]) -> Any:
    """Reconstruct an Anthropic streaming event from a dict.

    Dispatches on the ``type`` field to the correct Pydantic model.
    Returns the raw dict unchanged if reconstruction fails.
    """
    event_type = raw.get("type", "")

    # Map event types to their Anthropic SDK classes
    anthropic_event_map: dict[str, str] = {
        "message_start": "RawMessageStartEvent",
        "content_block_start": "RawContentBlockStartEvent",
        "content_block_delta": "RawContentBlockDeltaEvent",
        "content_block_stop": "RawContentBlockStopEvent",
        "message_delta": "RawMessageDeltaEvent",
        "message_stop": "RawMessageStopEvent",
    }

    class_name = anthropic_event_map.get(event_type)
    if not class_name:
        return raw

    import anthropic.types

    cls = getattr(anthropic.types, class_name, None)
    if cls is None:
        return raw

    return cls.model_validate(raw)


def deserialize_response(
    json_str: str | None, provider: str | None = None, delay_ms: float = 0
) -> Any:
    """Deserialize a JSON string back to a provider-native response object."""
    from stateloom.core.types import Provider

    if json_str is None:
        return None
    data = json.loads(json_str)

    # Stream replay: return CachedStreamChunks
    if isinstance(data, dict) and data.get("_type") == "stream":
        return deserialize_stream_chunks(data, delay_ms=delay_ms)

    # Reconstruct Gemini proxy
    if isinstance(data, dict) and data.get("_type") == "gemini":
        from stateloom.middleware.response_converter import _GeminiResponseProxy

        return _GeminiResponseProxy(
            content=data.get("text", ""),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
        )

    # Reconstruct OpenAI ChatCompletion
    is_openai_chat = (
        provider == Provider.OPENAI
        and isinstance(data, dict)
        and data.get("object") == "chat.completion"
    )
    if is_openai_chat:
        try:
            from openai.types.chat import ChatCompletion

            return ChatCompletion.model_validate(data)
        except Exception:
            logger.debug("OpenAI ChatCompletion reconstruction failed, returning raw dict")

    # Reconstruct Anthropic Message
    if provider == Provider.ANTHROPIC and isinstance(data, dict) and data.get("type") == "message":
        try:
            from anthropic.types import Message

            return Message.model_validate(data)
        except Exception:
            logger.debug("Anthropic Message reconstruction failed, returning raw dict")

    # Reconstruct LiteLLM ModelResponse
    is_litellm_chat = (
        provider == Provider.LITELLM
        and isinstance(data, dict)
        and data.get("object") == "chat.completion"
    )
    if is_litellm_chat:
        try:
            from litellm import ModelResponse

            return ModelResponse(**data)
        except Exception:
            logger.debug("LiteLLM ModelResponse reconstruction failed, returning raw dict")

    return data
