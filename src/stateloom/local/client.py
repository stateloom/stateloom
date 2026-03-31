"""Ollama HTTP client for local model inference."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx

from stateloom.core.types import Provider

logger = logging.getLogger("stateloom.local.client")


class OllamaModelNotFoundError(Exception):
    """Raised when Ollama returns 404 — model not pulled locally."""


@dataclass
class OllamaResponse:
    """Structured response from an Ollama chat call."""

    model: str = ""
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None


class RequestTranslator:
    """Translates provider request kwargs to Ollama chat format."""

    # Keys that Ollama doesn't support
    _UNSUPPORTED_KEYS = frozenset(
        {
            "response_format",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "logprobs",
            "top_logprobs",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "stream_options",
            "parallel_tool_calls",
            "service_tier",
            "store",
            "metadata",
        }
    )

    @staticmethod
    def translate(
        provider: str,
        model: str,
        request_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert any provider's request kwargs to Ollama /api/chat format."""
        messages = RequestTranslator._extract_messages(provider, request_kwargs)
        options = RequestTranslator._extract_options(request_kwargs)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options

        return payload

    @staticmethod
    def _extract_messages(provider: str, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract messages list, handling provider-specific formats."""
        messages: list[dict[str, Any]] = []

        if provider == Provider.ANTHROPIC:
            # Anthropic uses top-level 'system' param
            system = kwargs.get("system")
            if system:
                if isinstance(system, str):
                    messages.append({"role": "system", "content": system})
                elif isinstance(system, list):
                    # Anthropic system can be a list of content blocks
                    text_parts = []
                    for block in system:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif isinstance(block, str):
                            text_parts.append(block)
                    if text_parts:
                        messages.append({"role": "system", "content": "\n".join(text_parts)})

        elif provider in (Provider.GEMINI, "google"):
            # Gemini uses 'contents' with {role, parts} format and
            # optional top-level 'system_instruction'
            system_instruction = kwargs.get("system_instruction")
            if system_instruction:
                messages.append({"role": "system", "content": str(system_instruction)})
            contents = kwargs.get("contents", [])
            for entry in contents:
                if isinstance(entry, dict):
                    gemini_role = entry.get("role", "user")
                    role = "assistant" if gemini_role == "model" else gemini_role
                    parts = entry.get("parts", [])
                    text_parts = []
                    for part in parts:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    messages.append({"role": role, "content": "\n".join(text_parts)})
            if messages:
                return messages

        # Copy messages from kwargs (OpenAI format, also fallback for other providers)
        raw_messages = kwargs.get("messages", [])
        for msg in raw_messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # Handle content that's a list of content blocks
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = "\n".join(text_parts) if text_parts else ""
                messages.append({"role": role, "content": content})

        return messages

    @staticmethod
    def _extract_options(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract Ollama-compatible options from request kwargs."""
        options: dict[str, Any] = {}

        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            options["top_p"] = kwargs["top_p"]
        if "seed" in kwargs:
            options["seed"] = kwargs["seed"]

        # max_tokens / max_completion_tokens → num_predict
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        return options


class OllamaClient:
    """HTTP client for the Ollama API.

    Supports use as a context manager for automatic cleanup::

        with OllamaClient() as client:
            models = client.list_models()
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            # Suppress httpx request logging (noisy "HTTP Request: POST ..." lines)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            self._sync_client = httpx.Client(
                base_url=self._host,
                timeout=httpx.Timeout(
                    connect=30.0,
                    read=self._timeout,
                    write=30.0,
                    pool=30.0,
                ),
            )
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            logging.getLogger("httpx").setLevel(logging.WARNING)
            self._async_client = httpx.AsyncClient(
                base_url=self._host,
                timeout=httpx.Timeout(
                    connect=30.0,
                    read=self._timeout,
                    write=30.0,
                    pool=30.0,
                ),
            )
        return self._async_client

    def chat(
        self,
        provider: str,
        model: str,
        request_kwargs: dict[str, Any],
    ) -> OllamaResponse:
        """Send a chat request to Ollama (sync)."""
        payload = RequestTranslator.translate(provider, model, request_kwargs)
        client = self._get_sync_client()

        start = time.perf_counter()
        resp = client.post("/api/chat", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._check_response(resp, model)
        data = resp.json()
        return self._parse_response(data, elapsed_ms)

    async def achat(
        self,
        provider: str,
        model: str,
        request_kwargs: dict[str, Any],
    ) -> OllamaResponse:
        """Send a chat request to Ollama (async)."""
        payload = RequestTranslator.translate(provider, model, request_kwargs)
        client = self._get_async_client()

        start = time.perf_counter()
        resp = await client.post("/api/chat", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self._check_response(resp, model)
        data = resp.json()
        return self._parse_response(data, elapsed_ms)

    @staticmethod
    def _check_response(resp: httpx.Response, model: str) -> None:
        """Check HTTP response, raising descriptive errors for common failures."""
        if resp.status_code == 404:
            raise OllamaModelNotFoundError(f"model '{model}' not found — run: ollama pull {model}")
        resp.raise_for_status()

    def list_models(self) -> list[dict[str, Any]]:
        """List locally downloaded Ollama models (with retry)."""
        return cast(list[dict[str, Any]], self._retry_sync(self._list_models_once))

    def _list_models_once(self) -> list[dict[str, Any]]:
        client = self._get_sync_client()
        resp = client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return cast(list[dict[str, Any]], data.get("models", []))

    async def alist_models(self) -> list[dict[str, Any]]:
        """List locally downloaded Ollama models (async, with retry)."""
        return cast(list[dict[str, Any]], await self._retry_async(self._alist_models_once))

    async def _alist_models_once(self) -> list[dict[str, Any]]:
        client = self._get_async_client()
        resp = await client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return cast(list[dict[str, Any]], data.get("models", []))

    def pull_model(
        self,
        model: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Download a model from Ollama registry with streaming progress."""
        # Use a longer timeout for model downloads
        with httpx.Client(
            base_url=self._host,
            timeout=httpx.Timeout(600.0),
        ) as client:
            with client.stream("POST", "/api/pull", json={"model": model}) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    import json

                    data = json.loads(line)
                    if progress_callback:
                        progress_callback(data)

    def show_model(self, model: str) -> dict[str, Any]:
        """Get model details (with retry)."""

        def _once() -> dict[str, Any]:
            client = self._get_sync_client()
            resp = client.post("/api/show", json={"model": model})
            resp.raise_for_status()
            return cast(dict[str, Any], resp.json())

        return cast(dict[str, Any], self._retry_sync(_once))

    def delete_model(self, model: str) -> None:
        """Delete a locally downloaded model."""
        client = self._get_sync_client()
        resp = client.request("DELETE", "/api/delete", json={"model": model})
        resp.raise_for_status()

    def is_available(self) -> bool:
        """Check if Ollama is running and reachable."""
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                resp = client.get(f"{self._host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close HTTP clients."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client is not None:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_client.aclose())
            except RuntimeError:
                asyncio.run(self._async_client.aclose())
            self._async_client = None

    @staticmethod
    def _retry_sync(
        fn: Callable[..., Any],
        max_attempts: int = 3,
        backoff: float = 0.5,
    ) -> Any:
        """Retry a sync callable with exponential backoff for transient errors."""
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return fn()
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    time.sleep(backoff * (2**attempt))
                    logger.debug(
                        "Ollama retry %d/%d after %s",
                        attempt + 1,
                        max_attempts,
                        type(e).__name__,
                    )
        assert last_exc is not None
        raise last_exc

    @staticmethod
    async def _retry_async(
        fn: Callable[..., Any],
        max_attempts: int = 3,
        backoff: float = 0.5,
    ) -> Any:
        """Retry an async callable with exponential backoff for transient errors."""
        import asyncio

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await fn()
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    await asyncio.sleep(backoff * (2**attempt))
                    logger.debug(
                        "Ollama async retry %d/%d after %s",
                        attempt + 1,
                        max_attempts,
                        type(e).__name__,
                    )
        assert last_exc is not None
        raise last_exc

    def _parse_response(self, data: dict[str, Any], elapsed_ms: float) -> OllamaResponse:
        """Parse Ollama API response into OllamaResponse."""
        message = data.get("message", {})
        content = message.get("content", "")

        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0

        return OllamaResponse(
            model=data.get("model", ""),
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=elapsed_ms,
            raw=data,
        )
