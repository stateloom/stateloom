"""OpenAI Responses API proxy adapter for Codex CLI support.

Codex CLI uses the newer Responses API (``POST /v1/responses``) instead of
Chat Completions.  This module transparently proxies those requests so they
get full middleware benefits (PII scanning, cost tracking, budget enforcement,
caching, dashboard visibility).

Supports both **HTTP** (``POST /v1/responses``) and **WebSocket**
(``ws://host/v1/responses``) transport.  Codex CLI prefers WebSocket
for lower latency on multi-turn tool-heavy workflows.

User setup::

    export OPENAI_BASE_URL=http://localhost:4782/v1
    codex "explain what a linked list is"
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse

from stateloom.proxy.auth import (
    AuthResult,
    ProxyAuth,
    _StubKey,
    authenticate_request,
    format_policy_error,
    resolve_vk_rate_limit_id,
    sanitize_end_user,
    strip_bearer,
)
from stateloom.proxy.errors import error_status_code, to_openai_error_dict
from stateloom.proxy.passthrough import PassthroughProxy, filter_headers
from stateloom.proxy.rate_limiter import ProxyRateLimiter
from stateloom.proxy.sticky_session import (
    StickySessionManager,
    derive_session_name,
    resolve_session_id,
)
from stateloom.proxy.stream_helpers import SSE_HEADERS

if TYPE_CHECKING:
    from starlette.datastructures import Headers

    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.proxy.responses")

# ChatGPT OAuth users (Codex web login) use a different upstream than API key users.
# Codex CLI sets base_url to chatgpt.com/backend-api/codex for ChatGPT auth mode,
# but when OPENAI_BASE_URL is overridden, the client loses that routing info.
# We detect ChatGPT mode from the `chatgpt-account-id` header and route accordingly.
_CHATGPT_UPSTREAM = "https://chatgpt.com/backend-api/codex"


@dataclass
class _WSRelayState:
    """Mutable state shared between the two async relay tasks.

    Both client_to_upstream and upstream_to_client run as asyncio tasks
    in the same event loop.  Asyncio is single-threaded (cooperative),
    so there are no true concurrent mutations — only interleaving at
    await points.  Worst case: slightly stale latency or model name in
    event recording.  No locks needed.
    """

    current_model: str = ""
    call_start: float = 0.0
    prompt_preview: str = ""
    synthetic_ids: set[str] = field(default_factory=set)


def _resolve_upstream(
    headers: Headers | dict[str, str],
    config_upstream_openai: str,
    config_upstream_chatgpt: str = "",
) -> tuple[str, str]:
    """Return (http_url, ws_url) for the Responses API endpoint.

    ChatGPT-authenticated requests (identified by ``chatgpt-account-id`` header)
    must go to ``chatgpt.com/backend-api/codex/responses``.  API-key requests go
    to ``api.openai.com/v1/responses`` (or the configured upstream).

    The ChatGPT upstream is detected heuristically from the presence of
    ``chatgpt-account-id`` in the request headers.  If Codex CLI changes
    this header name, or if ChatGPT changes the upstream endpoint, set
    ``proxy_upstream_chatgpt`` in config to override.
    """
    get = headers.get if hasattr(headers, "get") else lambda k, d="": headers.get(k, d)
    if get("chatgpt-account-id", ""):
        chatgpt_base = config_upstream_chatgpt or _CHATGPT_UPSTREAM
        http_url = f"{chatgpt_base}/responses"
        ws_url = http_url.replace("https://", "wss://")
        return http_url, ws_url
    http_url = f"{config_upstream_openai}/v1/responses"
    ws_url = http_url.replace("https://", "wss://").replace("http://", "ws://")
    return http_url, ws_url


def _strip_synthetic_ids(
    msg: dict[str, Any],
    resp_obj: dict[str, Any],
    synthetic_ids: set[str],
) -> bool:
    """Strip ``previous_response_id`` if it references a synthetic StateLoom ID.

    Checks both ``msg``-level and ``resp_obj``-level (Codex may place
    the field in either location).  Mutates the dicts in place.

    Returns True if any ID was stripped.
    """
    stripped = False
    for obj in (resp_obj, msg):
        prev_id = obj.get("previous_response_id", "")
        if prev_id and (
            # Dual-check: set membership (fast, this session's IDs)
            # + prefix (catches IDs from prior sessions if Codex
            # persists them across reconnections).
            prev_id in synthetic_ids or prev_id.startswith(_SYNTHETIC_RESP_PREFIX)
        ):
            obj.pop("previous_response_id", None)
            stripped = True
            logger.debug(
                "Stripped synthetic previous_response_id %s",
                prev_id,
            )
    return stripped


async def _run_ws_middleware(
    ws: WebSocket,
    gate: Any,
    session: Any,
    ws_state: _WSRelayState,
    openai_msgs: list[dict[str, Any]],
    billing_mode: str,
) -> tuple[bool, Any]:
    """Execute middleware pipeline for a WS ``response.create``.

    Handles all terminal outcomes internally:

    - Middleware block → sends synthetic ``response.completed`` to *ws*,
      returns ``(True, None)``.
    - Cache hit → sends cached response to *ws*, returns ``(True, None)``.
    - Pass-through → returns ``(False, ctx)`` for caller to forward.

    The caller's loop becomes: if ``should_skip`` → ``continue``, else
    forward (optionally rebuilding the body from ``ctx``).
    """
    from stateloom.core.errors import StateLoomError
    from stateloom.middleware.base import MiddlewareContext

    ctx = MiddlewareContext(
        session=session,
        config=gate.config,
        provider="openai",
        model=ws_state.current_model,
        method="responses.create",
        request_kwargs={
            "messages": openai_msgs,
            "model": ws_state.current_model,
        },
        request_hash="",  # WS is always streaming; cache skips streaming
        provider_base_url=gate.config.proxy.upstream_openai,
    )
    ctx.is_streaming = True

    try:
        await gate.pipeline.execute_streaming(ctx)
    except StateLoomError as mw_err:
        # Middleware blocked — send a response.completed with
        # status=incomplete and reason=content_filter.  We use
        # response.completed (NOT response.failed) because Codex
        # CLI has a 5-retry reconnection loop for response.failed
        # and WS close events.  response.completed with
        # status=incomplete matches OpenAI's content-filter pattern
        # and is treated as a terminal (non-retryable) turn.
        logger.warning("WebSocket middleware blocked: %s", mw_err)
        syn_id = await _send_blocked_response(ws, str(mw_err))
        ws_state.synthetic_ids.add(syn_id)
        return True, None

    if ctx.skip_call and ctx.cached_response is not None:
        # Cache hit — return cached response
        try:
            cached_event = {
                "type": "response.completed",
                "response": ctx.cached_response if isinstance(ctx.cached_response, dict) else {},
            }
            await ws.send_text(json.dumps(cached_event))
            return True, None
        except Exception:
            logger.warning(
                "WS cache-hit conversion failed, falling through",
                exc_info=True,
            )
            ctx.skip_call = False
            ctx.cached_response = None

    return False, ctx


def create_responses_router(
    gate: Gate,
    sticky_session: StickySessionManager | None = None,
    passthrough: PassthroughProxy | None = None,
) -> APIRouter:
    """Create the ``/v1/responses`` router supporting dual transport.

    Args:
        gate: The Gate singleton.
        sticky_session: Optional sticky session manager.
        passthrough: HTTP reverse proxy (required for HTTP POST transport;
            WebSocket transport uses the ``websockets`` library directly).

    Returns:
        A FastAPI ``APIRouter`` with ``POST /responses`` (HTTP) and
        ``ws /responses`` (WebSocket) endpoints.
    """
    router = APIRouter()
    proxy_auth = ProxyAuth(gate)
    proxy_rate_limiter = ProxyRateLimiter(
        metrics=gate._metrics_collector,
        enabled=gate.config.rate_limiting_enabled,
    )

    def _authenticate(authorization: str) -> AuthResult:
        """Validate auth and return AuthResult."""
        token = strip_bearer(authorization)
        return authenticate_request(proxy_auth, token, gate.config)

    # ── WebSocket transport ─────────────────────────────────────────────
    @router.websocket("/responses")
    async def responses_ws(ws: WebSocket) -> None:
        """WebSocket proxy for the Responses API.

        Codex CLI connects here via ``ws://host/v1/responses``.  We open a
        mirror WebSocket to upstream OpenAI and relay messages bidirectionally.
        Each ``response.completed`` event creates an ``LLMCallEvent`` inside
        an StateLoom session so that calls appear in the dashboard.
        """
        # Extract auth from WebSocket headers (or query params)
        auth_header = ws.headers.get("authorization", "")
        if not auth_header:
            token_param = ws.query_params.get("token", "")
            if token_param:
                auth_header = f"Bearer {token_param}"

        auth = _authenticate(auth_header)
        vk, byok_key, raw_token = auth.vk, auth.byok_key, auth.raw_token
        if vk is None:
            reason = auth.error_hint or "Invalid or missing API key"
            await ws.close(code=4001, reason=reason[:123])  # WS close reason max 123 bytes
            return

        # VK scope check
        if vk.scopes:
            if not proxy_auth.check_scope(vk, "responses"):
                await ws.close(
                    code=4003,
                    reason="Virtual key does not have the 'responses' scope.",
                )
                return

        # End-user attribution
        ws_end_user_raw = ws.headers.get("x-stateloom-end-user", "")
        ws_end_user = sanitize_end_user(ws_end_user_raw) if ws_end_user_raw else ""

        await ws.accept()

        # Resolve the auth token to forward upstream
        upstream_token = byok_key or raw_token
        if not upstream_token:
            if vk.org_id:
                provider_keys = proxy_auth.get_provider_keys(vk)
                upstream_token = provider_keys.get("openai", "")
            if not upstream_token:
                bare = strip_bearer(auth_header)
                if bare and not bare.startswith("ag-"):
                    upstream_token = bare

        # Build upstream WebSocket URL — ChatGPT OAuth vs API key routing
        _http_url, upstream_ws_url = _resolve_upstream(
            ws.headers,
            gate.config.proxy.upstream_openai,
            gate.config.proxy.upstream_chatgpt,
        )

        # Forward all client headers to upstream (strip hop-by-hop + stateloom internal)
        _ws_skip = frozenset(
            {
                "host",
                "connection",
                "upgrade",
                "sec-websocket-key",
                "sec-websocket-version",
                "sec-websocket-extensions",
                "sec-websocket-protocol",
                "sec-websocket-accept",
                "transfer-encoding",
                "keep-alive",
                "te",
                "trailer",
                "proxy-connection",
                "content-length",
                "x-stateloom-session-id",
                "x-stateloom-openai-key",
                "x-stateloom-anthropic-key",
                "x-stateloom-google-key",
                "x-stateloom-end-user",
            }
        )
        upstream_headers: dict[str, str] = {}
        for key, value in ws.headers.items():
            if key.lower() in _ws_skip:
                continue
            if key.lower() == "authorization":
                continue
            upstream_headers[key] = value
        if upstream_token:
            upstream_headers["Authorization"] = f"Bearer {upstream_token}"

        # Billing mode
        from stateloom.proxy.billing import detect_billing_mode

        billing_mode = vk.billing_mode or ""
        if not billing_mode and byok_key:
            billing_mode = detect_billing_mode(byok_key, "openai")
        elif not billing_mode and raw_token and not byok_key:
            billing_mode = detect_billing_mode(raw_token, "openai")
        if not billing_mode:
            billing_mode = "api"

        # Session ID — stable per client fingerprint (same logic as StickySessionManager)
        import hashlib

        explicit_sid = ws.headers.get("x-stateloom-session-id", "")
        if explicit_sid:
            session_id = explicit_sid
        else:
            forwarded = ws.headers.get("x-forwarded-for", "")
            client_ip = (
                forwarded.split(",")[0].strip()
                if forwarded
                else (ws.client.host if ws.client else "unknown")
            )
            ua = ws.headers.get("user-agent", "")
            fp = hashlib.sha256(f"{client_ip}|{ua}".encode()).hexdigest()[:12]
            session_id = f"sticky-{fp}"

        logger.info(
            "WebSocket proxy connecting to %s (token present: %s, headers: %s)",
            upstream_ws_url,
            bool(upstream_token),
            list(upstream_headers.keys()),
        )

        try:
            import websockets

            async with gate.async_session(
                session_id=session_id,
                name="Codex CLI",
                org_id=vk.org_id,
                team_id=vk.team_id,
            ) as session:
                session.billing_mode = billing_mode
                session.metadata["billing_mode"] = billing_mode
                session.transport = "websocket"
                session.metadata["transport"] = "websocket"
                if ws_end_user:
                    session.end_user = ws_end_user

                # Shared mutable state between the client→upstream and
                # upstream→client relay tasks.
                ws_state = _WSRelayState()

                async with websockets.connect(
                    upstream_ws_url,
                    additional_headers=upstream_headers,
                    max_size=16 * 1024 * 1024,
                    close_timeout=5,
                ) as upstream:
                    logger.info("WebSocket proxy connected to %s", upstream_ws_url)

                    async def client_to_upstream() -> None:
                        """Relay messages from Codex CLI to upstream OpenAI."""
                        from stateloom.core.errors import StateLoomError

                        try:
                            while True:
                                data = await ws.receive_text()
                                forward_data = data  # may be modified by middleware
                                try:
                                    msg = json.loads(data)
                                    if msg.get("type") == "response.create":
                                        resp_obj = msg.get("response", {})

                                        # 1. Strip synthetic IDs from prior blocked turns
                                        if _strip_synthetic_ids(
                                            msg, resp_obj, ws_state.synthetic_ids
                                        ):
                                            forward_data = json.dumps(msg)

                                        # 2. Update relay state
                                        m = resp_obj.get("model", "") or msg.get("model", "")
                                        if m:
                                            ws_state.current_model = m
                                        ws_state.call_start = time.monotonic()

                                        input_field = resp_obj.get("input", msg.get("input", ""))
                                        instructions = resp_obj.get(
                                            "instructions", msg.get("instructions", "")
                                        )
                                        ws_state.prompt_preview = _extract_prompt_preview(
                                            input_field
                                        )
                                        session.next_step()

                                        # 3. Run middleware pipeline
                                        openai_msgs = _input_to_openai_messages(
                                            input_field, instructions
                                        )
                                        should_skip, ctx = await _run_ws_middleware(
                                            ws,
                                            gate,
                                            session,
                                            ws_state,
                                            openai_msgs,
                                            billing_mode,
                                        )
                                        if should_skip:
                                            continue

                                        # 4. Rebuild if middleware modified messages (PII redaction)
                                        modified_msgs = ctx.request_kwargs.get("messages", [])
                                        if modified_msgs != openai_msgs:
                                            rebuilt_body = json.loads(
                                                _rebuild_responses_body(resp_obj, modified_msgs)
                                            )
                                            new_msg = dict(msg)
                                            new_msg["response"] = rebuilt_body
                                            forward_data = json.dumps(new_msg)

                                        logger.info(
                                            "WebSocket response.create: model=%s billing=%s",
                                            ws_state.current_model,
                                            billing_mode,
                                        )
                                except (json.JSONDecodeError, Exception) as parse_err:
                                    if isinstance(parse_err, StateLoomError):
                                        raise  # already handled above
                                    logger.warning(
                                        "WS response.create parse/middleware failed, "
                                        "forwarding unprocessed: %s",
                                        parse_err,
                                    )
                                await upstream.send(forward_data)
                        except WebSocketDisconnect:
                            logger.debug("Client WebSocket disconnected")
                        except Exception as exc:
                            logger.debug("Client→upstream relay ended: %s", exc)

                    async def upstream_to_client() -> None:
                        """Relay messages from upstream OpenAI to Codex CLI."""
                        try:
                            async for message in upstream:
                                if isinstance(message, bytes):
                                    await ws.send_bytes(message)
                                else:
                                    _record_ws_event(
                                        message,
                                        session,
                                        gate,
                                        billing_mode,
                                        ws_state,
                                    )
                                    await ws.send_text(message)
                        except Exception as exc:
                            logger.info("Upstream→client relay ended: %s", exc)

                    tasks = [
                        asyncio.create_task(client_to_upstream()),
                        asyncio.create_task(upstream_to_client()),
                    ]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass

        except ImportError:
            logger.error("websockets library required for WebSocket proxy")
            await ws.close(code=4000, reason="WebSocket proxy unavailable")
        except Exception as exc:
            detail = ""
            try:
                from websockets.exceptions import InvalidStatus

                if isinstance(exc, InvalidStatus):
                    resp = exc.response
                    body_bytes = resp.body if resp.body else b""
                    detail = (
                        f" | upstream status={resp.status_code}"
                        f" reason={resp.reason_phrase}"
                        f" body={body_bytes.decode('utf-8', errors='replace')[:1000]}"
                    )
            except Exception as detail_err:
                detail = f" | (detail extraction failed: {detail_err})"
            logger.error("WebSocket proxy error: %s%s", exc, detail)
            try:
                await ws.close(code=1011, reason="Internal server error")
            except Exception:
                pass

    # ── HTTP transport ──────────────────────────────────────────────────
    @router.post("/responses")
    async def responses(
        request: Request,
        authorization: str = Header(default=""),
        x_stateloom_session_id: str = Header(default="", alias="X-StateLoom-Session-Id"),
        x_stateloom_openai_key: str = Header(default="", alias="X-StateLoom-OpenAI-Key"),
        x_stateloom_end_user: str = Header(default="", alias="X-StateLoom-End-User"),
    ) -> Any:
        """OpenAI Responses API endpoint via HTTP passthrough."""
        # Auth
        auth = _authenticate(authorization)
        vk, byok_key, raw_token = auth.vk, auth.byok_key, auth.raw_token
        if vk is None:
            msg = auth.error_hint or "Invalid or missing API key"
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": msg,
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
            )

        # VK scope check
        if vk.scopes:
            if not proxy_auth.check_scope(vk, "responses"):
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": {
                            "message": (
                                "Virtual key does not have the 'responses' scope. "
                                "Add 'responses' to the key's scopes list."
                            ),
                            "type": "permission_error",
                            "code": "scope_denied",
                        }
                    },
                )

        # Sanitize end-user header
        end_user = sanitize_end_user(x_stateloom_end_user) if x_stateloom_end_user else ""

        # Parse body (may be zstd-compressed — Codex CLI HTTPS fallback)
        try:
            raw_body = await request.body()
            # Detect zstd: magic bytes 0x28 0xB5 0x2F 0xFD
            if raw_body[:4] == b"\x28\xb5\x2f\xfd":
                import zstandard

                dctx = zstandard.ZstdDecompressor()
                try:
                    raw_body = dctx.decompress(raw_body)
                except zstandard.ZstdError:
                    # Frame may lack content size — use explicit max
                    raw_body = dctx.decompress(raw_body, max_output_size=10 * 1024 * 1024)
            body = json.loads(raw_body)
        except Exception as parse_exc:
            logger.warning(
                "Responses API invalid body: content_type=%s len=%d err=%s",
                request.headers.get("content-type", ""),
                len(raw_body) if raw_body else 0,
                parse_exc,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid JSON in request body",
                        "type": "invalid_request_error",
                        "code": "invalid_json",
                    }
                },
            )

        model = body.get("model", "")
        input_field = body.get("input", "")
        instructions = body.get("instructions", "")
        stream = body.get("stream", False)

        if not model:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "'model' is required",
                        "type": "invalid_request_error",
                        "code": "missing_model",
                    }
                },
            )

        if not input_field:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "'input' is required",
                        "type": "invalid_request_error",
                        "code": "missing_input",
                    }
                },
            )

        # Convert input/instructions to OpenAI messages for middleware
        openai_messages = _input_to_openai_messages(input_field, instructions)

        # VK policy enforcement (model access, budget, rate limit)
        # Note: scope "responses" was already checked above.
        from stateloom.core.errors import StateLoomRateLimitError

        _resp_policy_err: str | None = None
        if vk.allowed_models and not proxy_auth.check_model_access(vk, model):
            _resp_policy_err = f"model_not_allowed:{model}"
        elif vk.budget_limit is not None and not proxy_auth.check_budget(vk):
            _resp_policy_err = "key_budget_exceeded"
        else:
            if vk.rate_limit_tps is not None:
                try:
                    await proxy_rate_limiter.check(vk)
                except StateLoomRateLimitError:
                    _resp_policy_err = "key_rate_limit_exceeded"
        if _resp_policy_err is not None:
            status, error_code, msg = format_policy_error(_resp_policy_err, model, "responses")
            error_type = "rate_limit_error" if status == 429 else "permission_error"
            return JSONResponse(
                status_code=status,
                content={
                    "error": {
                        "message": msg,
                        "type": error_type,
                        "code": error_code,
                    }
                },
            )

        # Session ID and name
        session_id = resolve_session_id(x_stateloom_session_id, request, sticky_session)
        session_name = derive_session_name(request, model, "openai")

        logger.info(
            "POST /v1/responses model=%s vk=%s session=%s",
            model, vk.id if hasattr(vk, "id") else "anonymous", session_id or "-",
        )

        # Resolve provider keys: BYOK headers > org secrets > direct BYOK
        provider_keys: dict[str, str] = {}
        if vk.org_id:
            provider_keys = proxy_auth.get_provider_keys(vk)
        if x_stateloom_openai_key:
            provider_keys["openai"] = x_stateloom_openai_key
        elif byok_key:
            provider_keys["openai"] = byok_key

        # Rate limit slot tracking
        _vk_id = resolve_vk_rate_limit_id(vk)

        # Determine billing mode
        from stateloom.proxy.billing import detect_billing_mode

        billing_mode = vk.billing_mode or ""
        if not billing_mode and byok_key:
            billing_mode = detect_billing_mode(byok_key, "openai")
        elif not billing_mode and raw_token and not byok_key:
            billing_mode = detect_billing_mode(raw_token, "openai")
        if not billing_mode:
            billing_mode = "api"

        # Passthrough proxy required -- no Client fallback for Responses API
        if passthrough is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Responses API proxy requires HTTP passthrough",
                        "type": "server_error",
                        "code": "service_unavailable",
                    }
                },
            )

        return await _handle_passthrough(
            gate=gate,
            passthrough=passthrough,
            request=request,
            raw_body=raw_body,
            body=body,
            model=model,
            openai_messages=openai_messages,
            stream=stream,
            session_id=session_id,
            session_name=session_name,
            vk=vk,
            provider_keys=provider_keys,
            billing_mode=billing_mode,
            proxy_rate_limiter=proxy_rate_limiter,
            vk_id=_vk_id,
            raw_token=raw_token,
            end_user=end_user,
        )

    return router


# Synthetic response ID prefix — used to identify StateLoom-generated
# responses so we can strip ``previous_response_id`` when Codex references
# one in a subsequent turn (upstream doesn't know about our synthetic IDs).
_SYNTHETIC_RESP_PREFIX = "resp_ag_"


# ── WebSocket middleware-blocked response ──────────────────────────────


async def _send_blocked_response(
    ws: WebSocket,
    error_message: str,
) -> str:
    """Send a ``response.completed`` with ``status: incomplete`` on the WebSocket.

    This mirrors how OpenAI itself signals content-filter blocks: the turn
    is "complete" (Codex won't reconnect) but marked as incomplete with
    ``reason: content_filter``.  The assistant output contains the error
    message so the user sees why the request was blocked.

    Returns the synthetic response ID (callers may track it to strip
    ``previous_response_id`` in subsequent turns).
    """
    import uuid

    resp_id = f"{_SYNTHETIC_RESP_PREFIX}{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())

    # 1. response.created — marks the turn as started
    created_event = {
        "type": "response.created",
        "response": {
            "id": resp_id,
            "object": "response",
            "status": "in_progress",
            "created_at": created_at,
            "error": None,
            "output": [],
        },
    }
    await ws.send_text(json.dumps(created_event))

    # 2. response.completed with status=incomplete — terminates the turn
    completed_event = {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "status": "incomplete",
            "created_at": created_at,
            "incomplete_details": {"reason": "content_filter"},
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"[StateLoom] {error_message}",
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        },
    }
    await ws.send_text(json.dumps(completed_event))
    return resp_id


# ── WebSocket event recording ──────────────────────────────────────────


def _extract_prompt_preview(input_field: Any) -> str:
    """Extract the last user message text from a Responses API ``input`` field.

    Args:
        input_field: The ``input`` value from a Responses API request body.
            May be a string or a list of message items.

    Returns:
        A truncated preview (max 200 chars) suitable for dashboard display.
    """
    text = ""
    if isinstance(input_field, str):
        text = input_field
    elif isinstance(input_field, list):
        # Walk backwards to find the last user message
        for item in reversed(input_field):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message" or item.get("role") != "user":
                continue
            content = item.get("content", "")
            if isinstance(content, str):
                text = content
                break
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in (
                        "input_text",
                        "text",
                    ):
                        parts.append(part.get("text", ""))
                if parts:
                    text = " ".join(parts)
                    break
    return text[:200]


def _record_ws_event(
    message: str,
    session: Any,
    gate: Any,
    billing_mode: str,
    ws_state: _WSRelayState,
) -> None:
    """Parse upstream WS messages; on ``response.completed`` create an LLMCallEvent.

    Args:
        message: Raw JSON string from upstream WebSocket.
        session: Active StateLoom session.
        gate: The Gate singleton (for pricing and store access).
        billing_mode: ``"api"`` or ``"subscription"``.
        ws_state: Shared WebSocket relay state.

    This is fail-open — errors are logged but never break the relay.
    """
    try:
        data = json.loads(message)
        msg_type = data.get("type", "")
        if msg_type != "response.completed":
            return

        from stateloom.core.event import LLMCallEvent

        response = data.get("response", {})
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        model = response.get("model", "") or ws_state.current_model

        # Cost calculation
        api_cost = gate.pricing.calculate_cost(model, input_tokens, output_tokens)
        actual_cost = 0.0 if billing_mode == "subscription" else api_cost

        # Latency
        latency_ms = 0.0
        if ws_state.call_start > 0:
            latency_ms = (time.monotonic() - ws_state.call_start) * 1000
            ws_state.call_start = 0.0

        preview = ws_state.prompt_preview

        event = LLMCallEvent(
            session_id=session.id,
            step=session.step_counter,
            provider="openai",
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=actual_cost,
            estimated_api_cost=api_cost,
            latency_ms=latency_ms,
            is_streaming=True,
            prompt_preview=preview,
        )

        # Update session accumulators
        session.add_cost(
            cost=actual_cost,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            estimated_api_cost=api_cost,
        )

        # Persist event + session
        try:
            if hasattr(gate.store, "save_session_with_events"):
                gate.store.save_session_with_events(session, [event])
            else:
                gate.store.save_event(event)
                gate.store.save_session(session)
        except Exception as exc:
            logger.warning("Failed to persist WS event: %s", exc)

        logger.info(
            "WebSocket response.completed: model=%s input=%d output=%d "
            "cost=%.6f billing=%s session=%s",
            model,
            input_tokens,
            output_tokens,
            actual_cost,
            billing_mode,
            session.id,
        )
    except Exception:
        logger.warning("Failed to record WS event for cost tracking", exc_info=True)


# ── HTTP passthrough ────────────────────────────────────────────────────


async def _handle_passthrough(
    *,
    gate: Gate,
    passthrough: PassthroughProxy,
    request: Request,
    raw_body: bytes,
    body: dict[str, Any],
    model: str,
    openai_messages: list[dict[str, Any]],
    stream: bool,
    session_id: str,
    session_name: str,
    vk: Any,
    provider_keys: dict[str, str],
    billing_mode: str,
    proxy_rate_limiter: ProxyRateLimiter,
    vk_id: str | None,
    raw_token: str,
    end_user: str = "",
) -> Response:
    """Handle Responses API request via HTTP passthrough with middleware pipeline.

    Args:
        gate: The Gate singleton.
        passthrough: HTTP reverse proxy.
        request: Incoming FastAPI request.
        raw_body: Raw request body bytes (may have been decompressed from zstd).
        body: Parsed JSON body.
        model: Requested model.
        openai_messages: Converted OpenAI-format messages for middleware.
        stream: Whether streaming was requested.
        session_id: Resolved session ID.
        session_name: Human-readable session label.
        vk: Virtual key or stub.
        provider_keys: Resolved BYOK/org provider keys.
        billing_mode: ``"api"`` or ``"subscription"``.
        proxy_rate_limiter: Rate limiter for slot release.
        vk_id: Virtual key ID (or None).
        raw_token: CLI's original auth token for passthrough.
        end_user: Sanitized end-user identifier.

    Returns:
        A ``JSONResponse`` or ``StreamingResponse``.
    """
    from stateloom.middleware.base import MiddlewareContext

    upstream_url, _ = _resolve_upstream(
        request.headers,
        gate.config.proxy.upstream_openai,
        gate.config.proxy.upstream_chatgpt,
    )

    # Build upstream headers -- use BYOK key or resolved provider key
    auth_value = provider_keys.get("openai", "") or raw_token
    if not auth_value:
        raw_bearer = strip_bearer(request.headers.get("authorization", ""))
        if raw_bearer and not raw_bearer.startswith("ag-"):
            auth_value = raw_bearer

    upstream_headers = filter_headers(
        request.headers,
        auth_header_name="authorization" if auth_value else "",
        auth_header_value=f"Bearer {auth_value}" if auth_value else "",
    )

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
                provider="openai",
                model=model,
                method="responses.create",
                request_kwargs={"messages": openai_messages, "model": model},
                request_hash="" if stream else gate.pipeline._hash_request(
                    {"messages": openai_messages, "model": model}
                ),
                provider_base_url=gate.config.proxy.upstream_openai,
            )

            # Snapshot original messages for change detection
            original_messages = list(openai_messages)

            if stream:
                ctx.is_streaming = True
                await gate.pipeline.execute_streaming(ctx)

                if ctx.skip_call and ctx.cached_response is not None:
                    try:
                        return _emit_cached_as_stream(ctx.cached_response)
                    except Exception:
                        logger.warning(
                            "Cache-hit SSE conversion failed, falling through",
                            exc_info=True,
                        )
                        ctx.skip_call = False
                        ctx.cached_response = None

                # Rebuild body if middleware modified messages
                forwarded_body = raw_body
                current_messages = ctx.request_kwargs.get("messages", [])
                if current_messages != original_messages:
                    forwarded_body = _rebuild_responses_body(body, current_messages)
                else:
                    forwarded_body = raw_body

                return await _handle_streaming_passthrough(
                    passthrough,
                    upstream_url,
                    forwarded_body,
                    upstream_headers,
                    ctx=ctx,
                    proxy_rate_limiter=proxy_rate_limiter,
                    vk_id=vk_id,
                )

            else:

                async def llm_call() -> dict[str, Any]:
                    # Rebuild body if middleware modified messages
                    forwarded_body = raw_body
                    current_messages = ctx.request_kwargs.get("messages", [])
                    if current_messages != original_messages:
                        forwarded_body = _rebuild_responses_body(body, current_messages)

                    resp = await passthrough.forward(upstream_url, forwarded_body, upstream_headers)
                    if resp.status_code >= 400:
                        try:
                            error_data = resp.json()
                        except Exception:
                            error_data = {
                                "error": {
                                    "message": resp.text,
                                    "type": "api_error",
                                    "code": "upstream_error",
                                }
                            }
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

                return JSONResponse(content={"error": "unexpected response"}, status_code=500)

    except Exception as exc:
        from stateloom.core.errors import StateLoomError

        if isinstance(exc, StateLoomError):
            status = error_status_code(exc)
            content = to_openai_error_dict(exc)
            return JSONResponse(status_code=status, content=content)

        logger.exception("Responses API proxy error")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Internal server error ({type(exc).__name__})",
                    "type": "server_error",
                    "code": "internal_error",
                }
            },
        )
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
) -> Response:
    """Forward streaming response from upstream OpenAI Responses API.

    Eagerly peeks the first chunk to detect upstream errors (4xx/5xx)
    before committing to a 200 SSE stream.

    Args:
        passthrough: HTTP reverse proxy.
        upstream_url: Full upstream URL.
        body: Request body bytes (possibly rebuilt after PII redaction).
        headers: Filtered upstream headers.
        ctx: Pipeline context for stream-complete callbacks and usage tracking.
        proxy_rate_limiter: Rate limiter for slot release.
        vk_id: Virtual key ID (or None).

    Returns:
        A ``Response`` — either a proper error or an SSE ``StreamingResponse``.
    """
    from stateloom.proxy.passthrough import UpstreamStreamError

    gen = passthrough.forward_stream(upstream_url, body, headers)

    try:
        first_chunk = await gen.__anext__()
    except UpstreamStreamError as e:
        logger.debug("Upstream error %d for %s", e.status_code, upstream_url[:120])
        if ctx is not None:
            for cb in ctx._on_stream_complete:
                try:
                    cb()
                except Exception:
                    logger.warning("Stream completion callback failed", exc_info=True)
        if proxy_rate_limiter and vk_id:
            proxy_rate_limiter.on_request_complete(vk_id)
        return Response(
            content=e.content,
            status_code=e.status_code,
            media_type=e.content_type or "application/json",
        )
    except StopAsyncIteration:
        first_chunk = b""

    async def generate() -> AsyncGenerator[bytes, None]:
        try:
            if first_chunk:
                chunk_str = (
                    first_chunk.decode("utf-8") if isinstance(first_chunk, bytes) else first_chunk
                )
                _track_stream_usage(chunk_str, ctx)
                yield first_chunk
            async for chunk in gen:
                chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                _track_stream_usage(chunk_str, ctx)
                yield chunk
        except Exception as exc:
            logger.exception("Responses API proxy streaming error")
            from stateloom.core.errors import StateLoomError

            if isinstance(exc, StateLoomError):
                error = to_openai_error_dict(exc)
            else:
                error = {
                    "error": {
                        "message": f"Internal server error ({type(exc).__name__})",
                        "type": "server_error",
                        "code": "internal_error",
                    }
                }
            yield f"data: {json.dumps(error)}\n\n".encode("utf-8")
        finally:
            if ctx is not None:
                for cb in ctx._on_stream_complete:
                    try:
                        cb()
                    except Exception:
                        logger.warning("Stream completion callback failed", exc_info=True)
            if proxy_rate_limiter and vk_id:
                proxy_rate_limiter.on_request_complete(vk_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _track_stream_usage(chunk_str: str, ctx: Any) -> None:
    """Parse Responses API SSE events to extract token usage for cost tracking.

    The ``response.completed`` event contains the full response including
    ``usage`` with ``input_tokens`` and ``output_tokens``.
    """
    if ctx is None:
        return
    try:
        for line in chunk_str.split("\n"):
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            data = json.loads(data_str)
            usage = data.get("usage")
            if usage and "input_tokens" in usage:
                ctx.prompt_tokens = usage.get("input_tokens", 0)
                ctx.completion_tokens = usage.get("output_tokens", 0)
    except Exception:
        logger.warning("Responses API stream token extraction failed", exc_info=True)


def _emit_cached_as_stream(result: Any) -> StreamingResponse:
    """Emit a cached response as a ``response.completed`` SSE event."""

    async def generate() -> AsyncGenerator[str, None]:
        if isinstance(result, dict):
            yield f"event: response.completed\ndata: {json.dumps(result)}\n\n"
        else:
            yield f"event: response.completed\ndata: {json.dumps({})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _input_to_openai_messages(
    input_field: Any,
    instructions: str = "",
) -> list[dict[str, Any]]:
    """Convert Responses API ``input``/``instructions`` to OpenAI messages format.

    - ``instructions`` -> ``{"role": "system", "content": instructions}``
    - ``input`` as string -> ``[{"role": "user", "content": input}]``
    - ``input`` as array -> filter ``type == "message"`` items, extract role + text
    """
    messages: list[dict[str, Any]] = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_field, str):
        messages.append({"role": "user", "content": input_field})
    elif isinstance(input_field, list):
        for item in input_field:
            if not isinstance(item, dict):
                logger.debug("_input_to_openai_messages: skipping non-dict item: %s", type(item))
                continue
            # Only process message-type items for middleware
            if item.get("type") != "message":
                logger.debug(
                    "_input_to_openai_messages: skipping non-message item type=%s",
                    item.get("type"),
                )
                continue
            role = item.get("role", "user")
            content = item.get("content", "")
            # Content can be a string or array of parts
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        else:
                            logger.debug(
                                "_input_to_openai_messages: skipping unrecognized "
                                "content part type=%s",
                                part.get("type"),
                            )
                if text_parts:
                    messages.append({"role": role, "content": "\n".join(text_parts)})

    return messages


def _rebuild_responses_body(
    original_body: dict[str, Any],
    messages: list[dict[str, Any]],
) -> bytes:
    """Rebuild Responses API request body from (potentially modified) messages.

    After middleware modifies the OpenAI-format messages (e.g. PII redaction),
    this converts them back to Responses API format.
    """
    body = copy.deepcopy(original_body)

    # Extract system messages -> instructions
    system_parts: list[str] = []
    input_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg.get("content", ""))
        else:
            input_messages.append(msg)

    # Update instructions if system messages changed
    if system_parts:
        body["instructions"] = "\n\n".join(system_parts)
    elif "instructions" in body and not system_parts:
        # System messages were stripped -- keep original instructions
        pass

    # Rebuild input
    original_input = original_body.get("input", "")
    if isinstance(original_input, str):
        # String input -- replace with first user message content
        for msg in input_messages:
            if msg.get("role") == "user":
                body["input"] = msg.get("content", "")
                break
    elif isinstance(original_input, list):
        # Array input -- rebuild message items
        new_input: list[dict[str, Any]] = []
        msg_idx = 0
        for item in original_input:
            if not isinstance(item, dict):
                new_input.append(item)
                continue
            if item.get("type") != "message":
                # Non-message items (item_reference, function_call_output) pass through
                new_input.append(item)
                continue
            # Replace with corresponding modified message
            if msg_idx < len(input_messages):
                new_item = dict(item)
                new_item["content"] = input_messages[msg_idx].get("content", "")
                new_input.append(new_item)
                msg_idx += 1
            else:
                logger.warning(
                    "_rebuild_responses_body: message index %d exceeds modified "
                    "messages count %d — preserving original item",
                    msg_idx,
                    len(input_messages),
                )
                new_input.append(item)
        body["input"] = new_input

    return json.dumps(body).encode("utf-8")


# _StubKey is now imported from proxy.auth — kept as a comment for git history.
# class _StubKey: ...  (removed: see proxy/auth.py)
