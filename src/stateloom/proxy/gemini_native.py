"""Gemini-native /v1beta/models/{model}:generateContent endpoint for the StateLoom proxy.

Allows Gemini CLI and other Google AI SDK clients to use StateLoom as a
drop-in base URL replacement::

    export GOOGLE_GEMINI_BASE_URL=http://localhost:4782
    gemini "explain this code"

Uses HTTP reverse proxy (passthrough) instead of SDK instantiation, so
subscription users (Gemini Ultra) whose CLIs use OAuth/session tokens work
transparently — no API key required from StateLoom's side.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from stateloom.proxy.auth import (
    AuthResult,
    ProxyAuth,
    _StubKey,
    authenticate_request,
    enforce_vk_policies,
    format_policy_error,
    resolve_vk_rate_limit_id,
)
from stateloom.proxy.passthrough import PassthroughProxy, filter_headers
from stateloom.proxy.rate_limiter import ProxyRateLimiter
from stateloom.proxy.sticky_session import (
    StickySessionManager,
    derive_session_name,
    resolve_session_id,
)
from stateloom.proxy.stream_helpers import SSE_HEADERS, passthrough_stream_relay

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.proxy.gemini_native")


def create_gemini_router(
    gate: Gate,
    sticky_session: StickySessionManager | None = None,
    passthrough: PassthroughProxy | None = None,
) -> APIRouter:
    """Create the Gemini-native /v1beta router."""
    router = APIRouter()
    proxy_auth = ProxyAuth(gate)
    proxy_rate_limiter = ProxyRateLimiter(
        metrics=gate._metrics_collector,
        enabled=gate.config.rate_limiting_enabled,
    )

    def _authenticate(x_goog_api_key: str) -> AuthResult:
        """Validate auth and return AuthResult."""
        return authenticate_request(proxy_auth, x_goog_api_key, gate.config)

    async def _handle_generate(
        request: Request,
        model: str,
        x_goog_api_key: str,
        x_stateloom_session_id: str = "",
        stream: bool = False,
        x_stateloom_end_user: str = "",
    ) -> Any:
        """Shared handler for generateContent and streamGenerateContent."""
        # Auth
        auth = _authenticate(x_goog_api_key)
        vk, byok_key, raw_token = auth.vk, auth.byok_key, auth.raw_token
        if vk is None:
            msg = auth.error_hint or "Invalid or missing API key"
            return JSONResponse(
                status_code=401,
                content=_gemini_error(401, msg, "UNAUTHENTICATED"),
            )

        # Parse body
        try:
            raw_body = await request.body()
            body = json.loads(raw_body)
        except Exception:
            return JSONResponse(
                status_code=400,
                content=_gemini_error(400, "Invalid JSON in request body", "INVALID_ARGUMENT"),
            )

        contents = body.get("contents", [])
        system_instruction = body.get("systemInstruction")
        generation_config = body.get("generationConfig", {})

        if not contents:
            return JSONResponse(
                status_code=400,
                content=_gemini_error(
                    400, "'contents' is required and must be non-empty", "INVALID_ARGUMENT"
                ),
            )

        # VK policy enforcement (model access, budget, rate limit, scope)
        policy_error = await enforce_vk_policies(
            vk, model, "generate", proxy_auth, proxy_rate_limiter
        )
        if policy_error is not None:
            status, _error_code, msg = format_policy_error(policy_error, model, "generate")
            gemini_status = (
                "RESOURCE_EXHAUSTED" if status == 429 else "PERMISSION_DENIED"
            )
            return JSONResponse(
                status_code=status,
                content=_gemini_error(status, msg, gemini_status),
            )

        # End-user attribution
        from stateloom.proxy.auth import sanitize_end_user

        end_user = sanitize_end_user(x_stateloom_end_user) if x_stateloom_end_user else ""

        logger.info(
            "POST /v1beta/models/%s model=%s vk=%s",
            model, model, vk.id if hasattr(vk, "id") else "anonymous",
        )

        # Convert Gemini contents to OpenAI messages format for middleware
        openai_messages: list[dict[str, Any]] = []

        if system_instruction:
            sys_text = _extract_system_text(system_instruction)
            if sys_text:
                openai_messages.append({"role": "system", "content": sys_text})

        for item in contents:
            role = item.get("role", "user")
            openai_role = "assistant" if role == "model" else role
            parts = item.get("parts", [])
            text_parts: list[str] = []
            function_calls: list[dict[str, Any]] = []
            function_responses: list[dict[str, Any]] = []
            for part in parts:
                if isinstance(part, dict):
                    if "text" in part:
                        text_parts.append(part["text"])
                    elif "functionCall" in part:
                        function_calls.append(part["functionCall"])
                    elif "functionResponse" in part:
                        function_responses.append(part["functionResponse"])

            if function_responses:
                # Gemini tool results → OpenAI role="tool" messages
                # (enables _is_tool_continuation detection for dashboard collapsing)
                if text_parts:
                    openai_messages.append(
                        {"role": openai_role, "content": "\n".join(text_parts)}
                    )
                for fr in function_responses:
                    resp = fr.get("response", {})
                    openai_messages.append({
                        "role": "tool",
                        "content": json.dumps(resp) if isinstance(resp, dict) else str(resp),
                        "tool_call_id": fr.get("name", "unknown"),
                    })
            elif function_calls:
                # Gemini function calls → OpenAI assistant + tool_calls
                openai_messages.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": [
                        {
                            "id": fc.get("name", "unknown"),
                            "type": "function",
                            "function": {
                                "name": fc.get("name", "unknown"),
                                "arguments": json.dumps(fc.get("args", {})),
                            },
                        }
                        for fc in function_calls
                    ],
                })
            elif text_parts:
                openai_messages.append(
                    {
                        "role": openai_role,
                        "content": "\n".join(text_parts),
                    }
                )

        if not openai_messages:
            return JSONResponse(
                status_code=400,
                content=_gemini_error(
                    400, "No valid messages found in contents", "INVALID_ARGUMENT"
                ),
            )

        # Session ID and name
        session_id = resolve_session_id(x_stateloom_session_id, request, sticky_session)
        session_name = derive_session_name(request, model, "gemini")

        # Resolve provider keys
        provider_keys: dict[str, str] = {}
        if vk.org_id:
            provider_keys = proxy_auth.get_provider_keys(vk)
        if byok_key:
            provider_keys["google"] = byok_key

        # Rate limit slot tracking
        _vk_id = resolve_vk_rate_limit_id(vk)

        # Determine billing mode:
        #   1. VK explicit billing_mode (admin-set per key)
        #   2. Auto-detect from token format
        from stateloom.proxy.billing import detect_billing_mode

        billing_mode = vk.billing_mode or ""
        if not billing_mode and byok_key:
            billing_mode = detect_billing_mode(byok_key, "google")
        elif not billing_mode and raw_token and not byok_key:
            billing_mode = detect_billing_mode(raw_token, "google")
        if not billing_mode:
            billing_mode = "api"

        # Build upstream URL and headers — preserve query params (e.g. alt=sse)
        upstream_base = gate.config.proxy.upstream_gemini
        action = "streamGenerateContent" if stream else "generateContent"
        upstream_url = f"{upstream_base}/v1beta/models/{model}:{action}"
        if request.url.query:
            upstream_url += f"?{request.url.query}"

        auth_value = provider_keys.get("google", "") or raw_token
        upstream_headers = filter_headers(
            request.headers,
            auth_header_name="x-goog-api-key" if auth_value else "",
            auth_header_value=auth_value,
        )

        # Use passthrough proxy if available
        if passthrough is not None:
            return await _handle_passthrough(
                gate=gate,
                passthrough=passthrough,
                upstream_url=upstream_url,
                upstream_headers=upstream_headers,
                raw_body=raw_body,
                body=body,
                model=model,
                openai_messages=openai_messages,
                stream=stream,
                session_id=session_id,
                session_name=session_name,
                vk=vk,
                billing_mode=billing_mode,
                proxy_rate_limiter=proxy_rate_limiter,
                vk_id=_vk_id,
                end_user=end_user,
            )

        # Fallback: SDK-based flow via Client
        from stateloom.chat import Client

        extra_kwargs: dict[str, Any] = {}
        if "maxOutputTokens" in generation_config:
            extra_kwargs["max_tokens"] = generation_config["maxOutputTokens"]
        if "temperature" in generation_config:
            extra_kwargs["temperature"] = generation_config["temperature"]
        if "topP" in generation_config:
            extra_kwargs["top_p"] = generation_config["topP"]
        if "topK" in generation_config:
            extra_kwargs["top_k"] = generation_config["topK"]
        if "stopSequences" in generation_config:
            extra_kwargs["stop"] = generation_config["stopSequences"]

        try:
            client = Client(
                session_id=session_id,
                name=session_name,
                org_id=vk.org_id,
                team_id=vk.team_id,
                provider_keys=provider_keys or None,
                billing_mode=billing_mode,
            )

            if stream:
                return await _handle_streaming_legacy(
                    client,
                    model,
                    openai_messages,
                    extra_kwargs,
                    proxy_rate_limiter=proxy_rate_limiter,
                    vk_id=_vk_id,
                )
            else:
                try:
                    async with client:
                        response = await client.achat(
                            model=model, messages=openai_messages, **extra_kwargs
                        )
                        result = _response_to_gemini(response.raw, model)
                        return JSONResponse(content=result)
                finally:
                    if _vk_id:
                        proxy_rate_limiter.on_request_complete(_vk_id)

        except Exception as exc:
            status, content = _map_error(exc)
            return JSONResponse(status_code=status, content=content)

    @router.post("/models/{model}:generateContent")
    async def generate_content(
        model: str,
        request: Request,
        x_goog_api_key: str = Header(default="", alias="x-goog-api-key"),
        x_stateloom_session_id: str = Header(default="", alias="X-StateLoom-Session-Id"),
        x_stateloom_end_user: str = Header(default="", alias="X-StateLoom-End-User"),
    ) -> Any:
        """Gemini-native generateContent endpoint."""
        return await _handle_generate(
            request,
            model,
            x_goog_api_key,
            x_stateloom_session_id,
            stream=False,
            x_stateloom_end_user=x_stateloom_end_user,
        )

    @router.post("/models/{model}:streamGenerateContent")
    async def stream_generate_content(
        model: str,
        request: Request,
        x_goog_api_key: str = Header(default="", alias="x-goog-api-key"),
        x_stateloom_session_id: str = Header(default="", alias="X-StateLoom-Session-Id"),
        x_stateloom_end_user: str = Header(default="", alias="X-StateLoom-End-User"),
    ) -> Any:
        """Gemini-native streamGenerateContent endpoint."""
        return await _handle_generate(
            request,
            model,
            x_goog_api_key,
            x_stateloom_session_id,
            stream=True,
            x_stateloom_end_user=x_stateloom_end_user,
        )

    return router


async def _handle_passthrough(
    *,
    gate: Gate,
    passthrough: PassthroughProxy,
    upstream_url: str,
    upstream_headers: dict[str, str],
    raw_body: bytes,
    body: dict[str, Any],
    model: str,
    openai_messages: list[dict[str, Any]],
    stream: bool,
    session_id: str,
    session_name: str,
    vk: Any,
    billing_mode: str,
    proxy_rate_limiter: ProxyRateLimiter,
    vk_id: str | None,
    end_user: str = "",
) -> Response:
    """Handle request via HTTP passthrough with middleware pipeline."""
    from stateloom.middleware.base import MiddlewareContext

    try:
        async with gate.async_session(
            session_id=session_id,
            name=session_name,
            org_id=vk.org_id,
            team_id=vk.team_id,
        ) as session:
            session.billing_mode = billing_mode
            session.metadata["billing_mode"] = billing_mode
            if end_user:
                session.end_user = end_user
            session.next_step()

            ctx = MiddlewareContext(
                session=session,
                config=gate.config,
                provider="gemini",
                model=model,
                method="generateContent",
                request_kwargs={"messages": openai_messages, "model": model},
                request_hash="" if stream else gate.pipeline._hash_request(
                    {"messages": openai_messages, "model": model}
                ),
                provider_base_url=gate.config.proxy.upstream_gemini,
            )

            if stream:
                ctx.is_streaming = True
                await gate.pipeline.execute_streaming(ctx)

                if ctx.skip_call and ctx.cached_response is not None:
                    try:
                        result = _response_to_gemini(ctx.cached_response, model)
                        return _emit_cached_as_stream(result)
                    except Exception:
                        logger.warning(
                            "Cache-hit SSE conversion failed, falling through",
                            exc_info=True,
                        )
                        ctx.skip_call = False
                        ctx.cached_response = None

                return await _handle_streaming_passthrough(
                    passthrough,
                    upstream_url,
                    raw_body,
                    upstream_headers,
                    ctx=ctx,
                    proxy_rate_limiter=proxy_rate_limiter,
                    vk_id=vk_id,
                )

            else:

                async def llm_call() -> dict[str, Any]:
                    resp = await passthrough.forward(upstream_url, raw_body, upstream_headers)
                    if resp.status_code >= 400:
                        try:
                            error_data = resp.json()
                        except Exception:
                            error_data = _gemini_error(resp.status_code, resp.text, "INTERNAL")
                        return {
                            "_upstream_error": True,
                            "_status_code": resp.status_code,
                            **error_data,
                        }
                    return resp.json()

                result = await gate.pipeline.execute(ctx, llm_call)

                if isinstance(result, dict) and result.get("_upstream_error"):
                    status_code = result.pop("_status_code", 500)
                    result.pop("_upstream_error", None)
                    return JSONResponse(status_code=status_code, content=result)

                if isinstance(result, dict):
                    return JSONResponse(content=result)

                converted = _response_to_gemini(result, model)
                return JSONResponse(content=converted)

    except Exception as exc:
        status, content = _map_error(exc)
        return JSONResponse(status_code=status, content=content)
    finally:
        if vk_id:
            proxy_rate_limiter.on_request_complete(vk_id)


async def _handle_streaming_passthrough(
    passthrough: PassthroughProxy,
    upstream_url: str,
    body: bytes,
    headers: dict[str, str],
    ctx: Any = None,
    proxy_rate_limiter: ProxyRateLimiter | None = None,
    vk_id: str | None = None,
) -> StreamingResponse:
    """Forward streaming response from upstream Gemini API."""

    def _format_error(exc: Exception) -> bytes:
        _status, content = _map_error(exc)
        return f"data: {json.dumps(content)}\n\n".encode("utf-8")

    return await passthrough_stream_relay(
        passthrough,
        upstream_url,
        body,
        headers,
        ctx=ctx,
        track_usage=_track_stream_usage,
        format_error=_format_error,
        proxy_rate_limiter=proxy_rate_limiter,
        vk_id=vk_id,
    )


def _track_stream_usage(chunk_str: str, ctx: Any) -> None:
    """Parse Gemini SSE events to extract token usage for cost tracking."""
    if ctx is None:
        return
    try:
        for line in chunk_str.split("\n"):
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            meta = data.get("usageMetadata", {})
            if meta:
                ctx.prompt_tokens = meta.get("promptTokenCount", 0)
                ctx.completion_tokens = meta.get("candidatesTokenCount", 0)
    except Exception:
        logger.debug("Gemini stream usage extraction failed", exc_info=True)


def _emit_cached_as_stream(result: dict[str, Any]) -> StreamingResponse:
    """Emit a cached response as Gemini SSE."""

    async def generate() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _handle_streaming_legacy(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    extra_kwargs: dict[str, Any],
    proxy_rate_limiter: ProxyRateLimiter | None = None,
    vk_id: str | None = None,
) -> StreamingResponse:
    """Legacy buffer-then-emit as Gemini SSE events (used when no passthrough)."""

    async def generate() -> AsyncGenerator[str, None]:
        try:
            async with client:
                response = await client.achat(model=model, messages=messages, **extra_kwargs)
                result = _response_to_gemini(response.raw, model)
                yield f"data: {json.dumps(result)}\n\n"
        except Exception as exc:
            _status, content = _map_error(exc)
            yield f"data: {json.dumps(content)}\n\n"
        finally:
            if proxy_rate_limiter and vk_id:
                proxy_rate_limiter.on_request_complete(vk_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _response_to_gemini(raw_response: Any, model: str) -> dict[str, Any]:
    """Convert a pipeline response to Gemini REST API format."""
    # Dict response
    if isinstance(raw_response, dict):
        if "candidates" in raw_response and isinstance(raw_response.get("candidates"), list):
            return raw_response
        return _dict_to_gemini_response(raw_response, model)

    # Gemini SDK response
    if hasattr(raw_response, "candidates") and not isinstance(raw_response, dict):
        return _gemini_sdk_to_dict(raw_response, model)

    # Extract text from any other provider response
    text = _extract_text(raw_response)
    return _make_gemini_response(text, model)


def _gemini_sdk_to_dict(raw_response: Any, model: str) -> dict[str, Any]:
    """Convert a Gemini SDK response to REST API dict."""
    try:
        candidates = raw_response.candidates
        if not candidates:
            return _make_gemini_response("", model)

        cand = candidates[0]
        parts = []
        if hasattr(cand, "content") and hasattr(cand.content, "parts"):
            for part in cand.content.parts:
                if hasattr(part, "text"):
                    parts.append({"text": part.text})

        finish_reason = "STOP"
        if hasattr(cand, "finish_reason"):
            fr = cand.finish_reason
            if hasattr(fr, "name"):
                finish_reason = fr.name
            elif isinstance(fr, str):
                finish_reason = fr

        result: dict[str, Any] = {
            "candidates": [
                {
                    "content": {"parts": parts, "role": "model"},
                    "finishReason": finish_reason,
                }
            ],
            "modelVersion": model,
        }

        if hasattr(raw_response, "usage_metadata"):
            um = raw_response.usage_metadata
            result["usageMetadata"] = {
                "promptTokenCount": getattr(um, "prompt_token_count", 0),
                "candidatesTokenCount": getattr(um, "candidates_token_count", 0),
                "totalTokenCount": getattr(um, "total_token_count", 0),
            }

        return result
    except Exception:
        return _make_gemini_response("", model)


def _dict_to_gemini_response(data: dict[str, Any], model: str) -> dict[str, Any]:
    """Wrap a dict response in Gemini REST API format."""
    if "candidates" in data and isinstance(data.get("candidates"), list):
        return data

    content = ""
    choices = data.get("choices", [])
    if choices and isinstance(choices, list):
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
    elif "content" in data:
        content = data["content"]
    elif "message" in data:
        content = data["message"]

    usage = data.get("usage", {})
    return _make_gemini_response(
        content,
        model,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )


def _extract_text(raw_response: Any) -> str:
    """Extract plain text from any provider response object."""
    if raw_response is None:
        return ""
    try:
        if hasattr(raw_response, "choices") and raw_response.choices:
            return raw_response.choices[0].message.content or ""
        if hasattr(raw_response, "content") and hasattr(raw_response, "stop_reason"):
            blocks = raw_response.content
            if blocks and hasattr(blocks[0], "text"):
                return blocks[0].text
    except Exception:
        pass
    return str(raw_response)


def _make_gemini_response(
    text: str,
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Build a synthetic Gemini REST API response dict."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": prompt_tokens + completion_tokens,
        },
        "modelVersion": model,
    }


def _extract_system_text(system_instruction: Any) -> str:
    """Extract text from Gemini systemInstruction field."""
    if isinstance(system_instruction, str):
        return system_instruction
    if isinstance(system_instruction, dict):
        parts = system_instruction.get("parts", [])
        texts = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
        return "\n\n".join(texts)
    return ""


def _gemini_error(code: int, message: str, status: str) -> dict[str, Any]:
    """Build a Gemini-format error response."""
    return {
        "error": {
            "code": code,
            "message": message,
            "status": status,
        },
    }


def _map_error(exc: Exception) -> tuple[int, dict[str, Any]]:
    """Map an exception to (status_code, gemini_error_dict)."""
    from stateloom.core.errors import (
        StateLoomBlastRadiusError,
        StateLoomBudgetError,
        StateLoomCancellationError,
        StateLoomError,
        StateLoomKillSwitchError,
        StateLoomPIIBlockedError,
        StateLoomRateLimitError,
        StateLoomTimeoutError,
    )

    if isinstance(exc, StateLoomRateLimitError):
        return 429, _gemini_error(429, str(exc), "RESOURCE_EXHAUSTED")
    if isinstance(exc, StateLoomBudgetError):
        return 400, _gemini_error(400, str(exc), "INVALID_ARGUMENT")
    if isinstance(exc, StateLoomPIIBlockedError):
        return 400, _gemini_error(400, str(exc), "INVALID_ARGUMENT")
    if isinstance(exc, StateLoomKillSwitchError):
        return 503, _gemini_error(503, str(exc), "UNAVAILABLE")
    if isinstance(exc, StateLoomBlastRadiusError):
        return 503, _gemini_error(503, str(exc), "UNAVAILABLE")
    if isinstance(exc, StateLoomTimeoutError):
        return 504, _gemini_error(504, str(exc), "DEADLINE_EXCEEDED")
    if isinstance(exc, StateLoomCancellationError):
        return 499, _gemini_error(499, str(exc), "CANCELLED")
    if isinstance(exc, StateLoomError):
        return 500, _gemini_error(500, str(exc), "INTERNAL")

    logger.exception("Gemini proxy error")
    return 500, _gemini_error(500, f"Internal server error ({type(exc).__name__})", "INTERNAL")


# _StubKey is now imported from proxy.auth — kept as a comment for git history.
# class _StubKey: ...  (removed: see proxy/auth.py)
