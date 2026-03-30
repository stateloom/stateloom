"""Anthropic-native /v1/messages endpoint for the StateLoom proxy.

Allows Claude CLI and other Anthropic SDK clients to use StateLoom as a
drop-in base URL replacement::

    export ANTHROPIC_BASE_URL=http://localhost:4782
    claude "explain this code"

Uses HTTP reverse proxy (passthrough) instead of SDK instantiation, so
subscription users (Claude Max) whose CLIs use OAuth/session tokens work
transparently — no API key required from StateLoom's side.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
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
    strip_bearer,
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

logger = logging.getLogger("stateloom.proxy.anthropic_native")

# Claude CLI sends metadata.user_id as:
#   user_{hash}_account_{uuid}_session_{uuid}
_CLAUDE_SESSION_RE = re.compile(r"_session_([0-9a-f-]{36})$")


def _extract_claude_session_id(body: dict[str, Any]) -> str:
    """Extract the session UUID from Claude CLI's metadata.user_id field.

    Returns the session UUID string, or empty string if not found.
    """
    meta = body.get("metadata")
    if not isinstance(meta, dict):
        return ""
    user_id = meta.get("user_id", "")
    if not isinstance(user_id, str):
        return ""
    m = _CLAUDE_SESSION_RE.search(user_id)
    return m.group(1) if m else ""


def _is_cli_preflight(body: dict[str, Any], request: Any) -> bool:
    """Detect Claude CLI internal preflight calls (quota/count checks).

    These are short single-message requests using a cheap model (haiku) that
    Claude CLI makes at startup.  They should bypass the middleware pipeline
    to avoid cluttering the session waterfall.
    """
    model = body.get("model", "")
    if "haiku" not in model:
        return False
    messages = body.get("messages", [])
    if len(messages) != 1:
        return False
    content = messages[0].get("content", "")
    if isinstance(content, str) and len(content.strip()) <= 30:
        return True
    # Claude CLI also sends content as list of blocks: [{"type": "text", "text": "..."}]
    if isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        if len(text.strip()) <= 30:
            return True
    return False


def create_anthropic_router(
    gate: Gate,
    sticky_session: StickySessionManager | None = None,
    passthrough: PassthroughProxy | None = None,
) -> APIRouter:
    """Create the Anthropic-native /v1/messages router."""
    router = APIRouter()
    proxy_auth = ProxyAuth(gate)
    proxy_rate_limiter = ProxyRateLimiter(
        metrics=gate._metrics_collector,
        enabled=gate.config.rate_limiting_enabled,
    )

    def _authenticate(x_api_key: str, authorization: str) -> AuthResult:
        """Validate auth and return AuthResult."""
        token = ""
        if x_api_key:
            token = x_api_key
        else:
            token = strip_bearer(authorization)
        return authenticate_request(proxy_auth, token, gate.config)

    @router.post("/messages")
    async def messages(
        request: Request,
        x_api_key: str = Header(default="", alias="x-api-key"),
        authorization: str = Header(default=""),
        x_stateloom_session_id: str = Header(default="", alias="X-StateLoom-Session-Id"),
        x_stateloom_end_user: str = Header(default="", alias="X-StateLoom-End-User"),
    ) -> Any:
        """Anthropic-native messages endpoint via HTTP passthrough."""
        # Auth
        auth = _authenticate(x_api_key, authorization)
        vk, byok_key, raw_token = auth.vk, auth.byok_key, auth.raw_token
        if vk is None:
            msg = auth.error_hint or "Invalid or missing API key"
            return JSONResponse(
                status_code=401,
                content=_anthropic_error("authentication_error", msg),
            )

        # Parse body
        try:
            raw_body = await request.body()
            body = json.loads(raw_body)
        except Exception:
            return JSONResponse(
                status_code=400,
                content=_anthropic_error("invalid_request_error", "Invalid JSON in request body"),
            )

        model = body.get("model", "")
        messages_body = body.get("messages", [])
        system = body.get("system", "")
        max_tokens = body.get("max_tokens")
        stream = body.get("stream", False)

        # Claude CLI preflight calls (quota/count checks at startup) —
        # forward directly to upstream, bypass middleware pipeline entirely
        # so they don't clutter the session waterfall.
        if passthrough is not None and _is_cli_preflight(body, request):
            _pf_keys = proxy_auth.get_provider_keys(vk) if vk.org_id else {}
            auth_value = _pf_keys.get("anthropic", "") or byok_key or raw_token
            upstream_url = f"{gate.config.proxy.upstream_anthropic}/v1/messages"
            headers = filter_headers(
                request.headers,
                auth_header_name="x-api-key" if auth_value else "",
                auth_header_value=auth_value,
            )
            if stream:
                return StreamingResponse(
                    passthrough.forward_stream(upstream_url, raw_body, headers),
                    media_type="text/event-stream",
                    headers=SSE_HEADERS,
                )
            resp = await passthrough.forward(upstream_url, raw_body, headers)
            # Strip response hop-by-hop headers — httpx auto-decompresses
            # the body, so forwarding content-encoding causes Claude CLI
            # to attempt double-decompression (ZlibError).
            from stateloom.proxy.passthrough import RESPONSE_HOP_BY_HOP_HEADERS

            resp_headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in RESPONSE_HOP_BY_HOP_HEADERS
            }
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )

        if not model:
            return JSONResponse(
                status_code=400,
                content=_anthropic_error("invalid_request_error", "'model' is required"),
            )

        if max_tokens is None:
            return JSONResponse(
                status_code=400,
                content=_anthropic_error("invalid_request_error", "'max_tokens' is required"),
            )

        if not messages_body:
            return JSONResponse(
                status_code=400,
                content=_anthropic_error(
                    "invalid_request_error", "'messages' is required and must be non-empty"
                ),
            )

        # VK policy enforcement (model access, budget, rate limit, scope)
        policy_error = await enforce_vk_policies(
            vk, model, "messages", proxy_auth, proxy_rate_limiter
        )
        if policy_error is not None:
            status, _error_code, msg = format_policy_error(policy_error, model, "messages")
            error_type = "rate_limit_error" if status == 429 else "permission_error"
            return JSONResponse(
                status_code=status,
                content=_anthropic_error(error_type, msg),
            )

        # End-user attribution
        from stateloom.proxy.auth import sanitize_end_user

        end_user = sanitize_end_user(x_stateloom_end_user) if x_stateloom_end_user else ""

        # Session ID: prefer explicit header > Claude CLI metadata > sticky > random
        claude_session = _extract_claude_session_id(body)
        session_id = resolve_session_id(
            x_stateloom_session_id or claude_session, request, sticky_session,
        )
        session_name = derive_session_name(request, model, "anthropic")

        logger.info(
            "POST /v1/messages model=%s vk=%s session=%s",
            model, vk.id if hasattr(vk, "id") else "anonymous", session_id or "-",
        )

        # Resolve provider keys for VK mode
        provider_keys: dict[str, str] = {}
        if vk.org_id:
            provider_keys = proxy_auth.get_provider_keys(vk)
        if byok_key:
            provider_keys["anthropic"] = byok_key

        # Rate limit slot tracking
        _vk_id = resolve_vk_rate_limit_id(vk)

        # Determine billing mode:
        #   1. VK explicit billing_mode (admin-set per key)
        #   2. Auto-detect from token format
        from stateloom.proxy.billing import detect_billing_mode

        billing_mode = vk.billing_mode or ""
        if not billing_mode and byok_key:
            billing_mode = detect_billing_mode(byok_key, "anthropic")
        elif not billing_mode and raw_token and not byok_key:
            billing_mode = detect_billing_mode(raw_token, "anthropic")
        if not billing_mode:
            billing_mode = "api"

        # Build upstream URL and headers
        upstream_base = gate.config.proxy.upstream_anthropic
        upstream_url = f"{upstream_base}/v1/messages"

        auth_value = provider_keys.get("anthropic", "") or raw_token
        upstream_headers = filter_headers(
            request.headers,
            auth_header_name="x-api-key" if auth_value else "",
            auth_header_value=auth_value,
        )

        # Convert Anthropic messages to OpenAI format for the middleware pipeline
        openai_messages: list[dict[str, Any]] = []
        if system:
            if isinstance(system, str):
                openai_messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                text_parts = []
                for block in system:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    openai_messages.append({"role": "system", "content": "\n\n".join(text_parts)})

        for msg in messages_body:
            openai_messages.append(msg)

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

        # Fallback: SDK-based flow via Client (for backward compatibility)
        from stateloom.chat import Client

        extra_kwargs: dict[str, Any] = {"max_tokens": max_tokens}
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "tools",
            "tool_choice",
            "metadata",
        ):
            if key in body:
                extra_kwargs[key] = body[key]
        extra_kwargs.setdefault("timeout", 600.0)

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
                        result = _response_to_anthropic(response.raw, model)
                        return JSONResponse(content=result)
                finally:
                    if _vk_id:
                        proxy_rate_limiter.on_request_complete(_vk_id)

        except Exception as exc:
            logger.exception("Anthropic proxy error in /v1/messages")
            status, content = _map_error(exc)
            return JSONResponse(status_code=status, content=content)

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

            # Detect CLI internal overhead calls (e.g. haiku quota/token
            # counts with longer prompts that weren't caught by
            # _is_cli_preflight).  All calls in this handler are proxy-
            # originated, so no session-name guard is needed.
            request_kwargs: dict[str, Any] = {
                "messages": openai_messages,
                "model": model,
            }
            if "haiku" in model:
                non_system = [
                    m for m in openai_messages if m.get("role") != "system"
                ]
                if len(non_system) == 1 and non_system[0].get("role") == "user":
                    request_kwargs["_cli_internal"] = True

            ctx = MiddlewareContext(
                session=session,
                config=gate.config,
                provider="anthropic",
                model=model,
                method="messages.create",
                request_kwargs=request_kwargs,
                request_hash="" if stream else gate.pipeline._hash_request(
                    {"messages": openai_messages, "model": model}
                ),
                provider_base_url=gate.config.proxy.upstream_anthropic,
            )

            if stream:
                # For streaming: run pre-call middleware, then forward stream
                ctx.is_streaming = True
                await gate.pipeline.execute_streaming(ctx)

                if ctx.skip_call and ctx.cached_response is not None:
                    # Cache hit — emit as Anthropic SSE
                    try:
                        result = _response_to_anthropic(ctx.cached_response, model)
                        return _emit_cached_as_stream(result, ctx)
                    except Exception:
                        logger.warning(
                            "Cache-hit SSE conversion failed, falling through to upstream",
                            exc_info=True,
                        )
                        ctx.skip_call = False
                        ctx.cached_response = None

                # Check if PII Phase 1 stripped the active turn (user's
                # current message contained previously-blocked PII).
                # Return a visible content-policy response as SSE so the
                # CLI renders it instead of re-answering old context.
                if session.metadata.get("_pii_active_turn_stripped"):
                    msg = _make_anthropic_message(
                        "[Content Policy] Your message was removed because it "
                        "contained previously blocked sensitive information. "
                        "Please rephrase without including PII.",
                        model,
                    )
                    return _emit_cached_as_stream(msg, None)

                # Check if middleware stripped all messages
                remaining = [
                    m for m in ctx.request_kwargs.get("messages", []) if m.get("role") != "system"
                ]
                if not remaining:
                    msg = _make_anthropic_message(
                        "[Content Policy] All messages were removed by "
                        "content policy. Please start a new conversation.",
                        model,
                    )
                    return _emit_cached_as_stream(msg, None)

                # Forward upstream with real streaming
                forwarded_body = _apply_message_modifications(body, ctx)
                forwarded_bytes = json.dumps(forwarded_body).encode("utf-8")

                return await _handle_streaming_passthrough(
                    passthrough,
                    upstream_url,
                    forwarded_bytes,
                    upstream_headers,
                    ctx=ctx,
                    proxy_rate_limiter=proxy_rate_limiter,
                    vk_id=vk_id,
                )

            else:
                # Non-streaming: check for active turn stripped by PII Phase 1
                if session.metadata.get("_pii_active_turn_stripped"):
                    return JSONResponse(
                        status_code=400,
                        content=_anthropic_error(
                            "invalid_request_error",
                            "[Content Policy] Your message was removed because "
                            "it contained previously blocked sensitive "
                            "information. Please rephrase without including PII.",
                        ),
                    )

                remaining_ns = [
                    m for m in ctx.request_kwargs.get("messages", []) if m.get("role") != "system"
                ]
                if not remaining_ns:
                    return JSONResponse(
                        status_code=400,
                        content=_anthropic_error(
                            "invalid_request_error",
                            "All messages were removed by content policy. "
                            "Please start a new conversation.",
                        ),
                    )

                # Use pipeline.execute() with passthrough llm_call
                async def llm_call() -> dict[str, Any]:
                    forwarded_body = _apply_message_modifications(body, ctx)
                    forwarded_bytes = json.dumps(forwarded_body).encode("utf-8")
                    resp = await passthrough.forward(
                        upstream_url, forwarded_bytes, upstream_headers
                    )
                    if resp.status_code >= 400:
                        try:
                            error_data = resp.json()
                        except Exception:
                            error_data = _anthropic_error("api_error", resp.text)
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

                converted = _response_to_anthropic(result, model)
                return JSONResponse(content=converted)

    except Exception as exc:
        from stateloom.core.errors import StateLoomPIIBlockedError

        if isinstance(exc, StateLoomPIIBlockedError) and stream:
            # Return PII block as a synthetic Anthropic streaming response
            # so Claude CLI renders the error as visible text to the user.
            # A plain JSON 400 error on a streaming request is often silently
            # swallowed by CLIs — they expect SSE events.
            error_text = (
                f"[Content Policy] Your message was blocked: "
                f"detected {exc.pii_type}. "
                f"Please remove sensitive information and try again."
            )
            msg = _make_anthropic_message(error_text, model, output_tokens=0)
            return _emit_cached_as_stream(msg, None)

        logger.exception("Anthropic proxy error in /v1/messages")
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
    """Forward streaming response from upstream Anthropic API."""

    def _format_error(exc: Exception) -> bytes:
        _status, content = _map_error(exc)
        return _anthropic_sse("error", content).encode("utf-8")

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
    """Parse Anthropic SSE events to extract token usage for cost tracking."""
    if ctx is None:
        return
    try:
        for line in chunk_str.split("\n"):
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            event_type = data.get("type", "")
            if event_type == "message_start":
                usage = data.get("message", {}).get("usage", {})
                ctx.prompt_tokens = usage.get("input_tokens", 0)
            elif event_type == "message_delta":
                usage = data.get("usage", {})
                ctx.completion_tokens = usage.get("output_tokens", 0)
    except Exception:
        logger.debug("Anthropic stream usage extraction failed", exc_info=True)


def _emit_cached_as_stream(
    result: dict[str, Any],
    ctx: Any,
) -> StreamingResponse:
    """Emit a cached non-streaming result as Anthropic SSE events."""

    async def generate() -> AsyncGenerator[str, None]:
        msg_id = result.get("id", f"msg_{uuid.uuid4().hex[:24]}")
        resp_model = result.get("model", "")
        usage = result.get("usage", {})
        content_blocks = result.get("content", [])
        stop_reason = result.get("stop_reason", "end_turn")

        yield _anthropic_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": resp_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
                },
            },
        )

        for idx, block in enumerate(content_blocks):
            yield _anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            yield _anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": block.get("text", "")},
                },
            )
            yield _anthropic_sse(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": idx,
                },
            )

        yield _anthropic_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
        yield _anthropic_sse("message_stop", {"type": "message_stop"})

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
    """Legacy buffer-then-emit as Anthropic SSE events (used when no passthrough)."""

    async def generate() -> AsyncGenerator[str, None]:
        try:
            async with client:
                response = await client.achat(model=model, messages=messages, **extra_kwargs)
                result = _response_to_anthropic(response.raw, model)

                msg_id = result.get("id", f"msg_{uuid.uuid4().hex[:24]}")
                resp_model = result.get("model", model)
                usage = result.get("usage", {})
                content_blocks = result.get("content", [])
                stop_reason = result.get("stop_reason", "end_turn")

                yield _anthropic_sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": resp_model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": usage.get("input_tokens", 0),
                                "output_tokens": 0,
                            },
                        },
                    },
                )

                for idx, block in enumerate(content_blocks):
                    block_type = block.get("type", "text")
                    if block_type == "text":
                        yield _anthropic_sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                        yield _anthropic_sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "text_delta", "text": block.get("text", "")},
                            },
                        )
                    elif block_type == "tool_use":
                        yield _anthropic_sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": block.get("id", ""),
                                    "name": block.get("name", ""),
                                    "input": {},
                                },
                            },
                        )
                        yield _anthropic_sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(block.get("input", {})),
                                },
                            },
                        )
                    yield _anthropic_sse(
                        "content_block_stop",
                        {
                            "type": "content_block_stop",
                            "index": idx,
                        },
                    )

                if not content_blocks:
                    yield _anthropic_sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    yield _anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": ""},
                        },
                    )
                    yield _anthropic_sse(
                        "content_block_stop",
                        {
                            "type": "content_block_stop",
                            "index": 0,
                        },
                    )

                yield _anthropic_sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": usage.get("output_tokens", 0)},
                    },
                )
                yield _anthropic_sse("message_stop", {"type": "message_stop"})

        except Exception as exc:
            logger.exception("Anthropic proxy streaming error")
            _status, content = _map_error(exc)
            yield _anthropic_sse("error", content)
        finally:
            if proxy_rate_limiter and vk_id:
                proxy_rate_limiter.on_request_complete(vk_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _apply_message_modifications(
    original_body: dict[str, Any],
    ctx: Any,
) -> dict[str, Any]:
    """Apply middleware modifications (e.g. PII redaction) back to the body."""
    modified_messages = ctx.request_kwargs.get("messages", [])
    # Reconstruct body with any modifications
    new_body = dict(original_body)
    new_messages = []
    new_system = original_body.get("system", "")

    for msg in modified_messages:
        if msg.get("role") == "system":
            new_system = msg.get("content", "")
        else:
            new_messages.append(msg)

    # Always apply the modified messages — even if empty (Phase 1 may have
    # stripped all messages containing previously-blocked PII).
    new_body["messages"] = new_messages
    if new_system != original_body.get("system", ""):
        new_body["system"] = new_system

    return new_body


def _response_to_anthropic(raw_response: Any, model: str) -> dict[str, Any]:
    """Convert a pipeline response to Anthropic Message format."""
    # Dict response — already Anthropic or needs conversion
    if isinstance(raw_response, dict):
        if raw_response.get("type") == "message" and "content" in raw_response:
            return raw_response
        return _dict_to_anthropic_message(raw_response, model)

    # Native Anthropic response — pass through
    if hasattr(raw_response, "model_dump") and hasattr(raw_response, "stop_reason"):
        try:
            return raw_response.model_dump()
        except Exception:
            pass

    # Extract text content from any provider response
    text = _extract_text(raw_response)
    return _make_anthropic_message(text, model)


def _dict_to_anthropic_message(data: dict[str, Any], model: str) -> dict[str, Any]:
    """Wrap a dict response in Anthropic Message format."""
    if data.get("type") == "message" and "content" in data:
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
    return _make_anthropic_message(
        content,
        model,
        input_tokens=usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        output_tokens=usage.get("completion_tokens", usage.get("output_tokens", 0)),
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
        if hasattr(raw_response, "candidates"):
            cand = raw_response.candidates[0]
            return cand.content.parts[0].text
    except Exception:
        pass
    return str(raw_response)


def _make_anthropic_message(
    text: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict[str, Any]:
    """Build a synthetic Anthropic Message dict."""
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def _anthropic_sse(event_type: str, data: dict[str, Any]) -> str:
    """Format as Anthropic SSE (explicit event: line)."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _emit_error_as_stream(error_content: dict[str, Any]) -> StreamingResponse:
    """Emit an error as an Anthropic SSE error event for streaming requests."""

    async def generate() -> AsyncGenerator[str, None]:
        yield _anthropic_sse("error", error_content)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _anthropic_error(error_type: str, message: str) -> dict[str, Any]:
    """Build an Anthropic-format error response."""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def _map_error(exc: Exception) -> tuple[int, dict[str, Any]]:
    """Map an exception to (status_code, anthropic_error_dict)."""
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
        return 429, _anthropic_error("rate_limit_error", str(exc))
    if isinstance(exc, StateLoomBudgetError):
        return 400, _anthropic_error("invalid_request_error", str(exc))
    if isinstance(exc, StateLoomPIIBlockedError):
        return 400, _anthropic_error("invalid_request_error", str(exc))
    if isinstance(exc, StateLoomKillSwitchError):
        return 503, _anthropic_error("api_error", str(exc))
    if isinstance(exc, StateLoomBlastRadiusError):
        return 503, _anthropic_error("api_error", str(exc))
    if isinstance(exc, StateLoomTimeoutError):
        return 504, _anthropic_error("api_error", str(exc))
    if isinstance(exc, StateLoomCancellationError):
        return 499, _anthropic_error("api_error", str(exc))
    if isinstance(exc, StateLoomError):
        return 500, _anthropic_error("api_error", str(exc))

    logger.exception("Anthropic proxy error")
    return 500, _anthropic_error("api_error", f"Internal server error ({type(exc).__name__})")


# _StubKey is now imported from proxy.auth — kept as a comment for git history.
# class _StubKey: ...  (removed: see proxy/auth.py)
