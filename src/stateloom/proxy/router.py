"""FastAPI router for the OpenAI-compatible proxy endpoints.

Supports HTTP passthrough for direct-to-provider forwarding (when a
PassthroughProxy is provided) and SDK-based flow via Client as fallback.
The agent endpoint always uses Client (agents need model/prompt overrides).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from stateloom.chat import Client, _resolve_provider
from stateloom.core.errors import StateLoomRateLimitError
from stateloom.proxy.auth import (
    AuthResult,
    ProxyAuth,
    authenticate_request,
    enforce_vk_policies,
    format_policy_error,
    resolve_vk_rate_limit_id,
    strip_bearer,
)
from stateloom.proxy.errors import error_status_code, to_openai_error_dict
from stateloom.proxy.passthrough import PassthroughProxy, filter_headers
from stateloom.proxy.rate_limiter import ProxyRateLimiter
from stateloom.proxy.response_format import (
    to_openai_completion_dict,
    to_openai_done_event,
    to_openai_sse_event,
)
from stateloom.proxy.sticky_session import (
    StickySessionManager,
    derive_session_name,
    resolve_session_id,
)
from stateloom.proxy.stream_helpers import SSE_HEADERS, passthrough_stream_relay

if TYPE_CHECKING:
    from stateloom.gate import Gate
    from stateloom.proxy.virtual_key import VirtualKey

logger = logging.getLogger("stateloom.proxy.router")


def create_proxy_router(
    gate: Gate,
    sticky_session: StickySessionManager | None = None,
    passthrough: PassthroughProxy | None = None,
) -> APIRouter:
    """Create the ``/v1`` proxy router with chat completions and agent endpoints.

    Args:
        gate: The Gate singleton.
        sticky_session: Optional sticky session manager for fingerprint-based
            session grouping.
        passthrough: Optional HTTP reverse proxy.  When provided, OpenAI
            model requests use HTTP passthrough (direct forwarding).  When
            None, all requests fall back to the SDK-based ``Client`` flow.

    Returns:
        A FastAPI ``APIRouter`` with ``/chat/completions``,
        ``/agents/{ref}/chat/completions``, ``/models``, and ``/health``.
    """
    router = APIRouter()
    proxy_auth = ProxyAuth(gate)
    proxy_rate_limiter = ProxyRateLimiter(
        metrics=gate._metrics_collector,  # type: ignore[arg-type]
        enabled=gate.config.rate_limiting_enabled,
    )

    def _authenticate(authorization: str) -> AuthResult:
        """Validate auth and return AuthResult."""
        token = strip_bearer(authorization)
        if not token and authorization:
            token = authorization
        return authenticate_request(proxy_auth, token, gate.config)

    @router.post("/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: str = Header(default=""),
        x_stateloom_session_id: str = Header(default="", alias="X-StateLoom-Session-Id"),
        x_stateloom_openai_key: str = Header(default="", alias="X-StateLoom-OpenAI-Key"),
        x_stateloom_anthropic_key: str = Header(default="", alias="X-StateLoom-Anthropic-Key"),
        x_stateloom_google_key: str = Header(default="", alias="X-StateLoom-Google-Key"),
        x_stateloom_end_user: str = Header(default="", alias="X-StateLoom-End-User"),
    ) -> Any:
        """OpenAI-compatible chat completions endpoint.

        Auth flow: validate virtual key → check model access, budget, and
        rate limit → resolve session → detect billing mode → route via
        passthrough (OpenAI models) or Client (cross-provider fallback).
        """
        # Auth
        auth = _authenticate(authorization)
        vk = auth.vk
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

        # Parse body
        try:
            raw_body = await request.body()
            body = json.loads(raw_body)
        except Exception:
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
        messages = body.get("messages", [])
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

        if not messages:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "'messages' is required and must be non-empty",
                        "type": "invalid_request_error",
                        "code": "missing_messages",
                    }
                },
            )

        # Virtual key policy enforcement (model access, budget, rate limit, scope)
        policy_error = await enforce_vk_policies(vk, model, "chat", proxy_auth, proxy_rate_limiter)
        if policy_error is not None:
            status, error_code, msg = format_policy_error(policy_error, model, "chat")
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

        # End-user attribution
        from stateloom.proxy.auth import sanitize_end_user

        end_user = sanitize_end_user(x_stateloom_end_user) if x_stateloom_end_user else ""

        # Session ID and name
        session_id = resolve_session_id(x_stateloom_session_id, request, sticky_session)

        logger.info(
            "POST /v1/chat/completions model=%s vk=%s session=%s",
            model,
            vk.id or "anonymous",
            session_id or "-",
        )

        # Request ID for the response
        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        session_name = derive_session_name(request, model, _resolve_provider(model))

        # Resolve provider keys: BYOK headers > org secrets > global config
        provider_keys: dict[str, str] = {}
        if vk.org_id:
            provider_keys = proxy_auth.get_provider_keys(cast("VirtualKey", vk))
        if x_stateloom_openai_key:
            provider_keys["openai"] = x_stateloom_openai_key
        if x_stateloom_anthropic_key:
            provider_keys["anthropic"] = x_stateloom_anthropic_key
        if x_stateloom_google_key:
            provider_keys["google"] = x_stateloom_google_key

        # Determine if we need to release a rate limit slot on completion
        _vk_id = resolve_vk_rate_limit_id(vk)

        # Determine billing mode:
        #   1. VK explicit billing_mode (admin-set per key)
        #   2. Auto-detect from token format
        from stateloom.proxy.billing import detect_billing_mode

        billing_mode = vk.billing_mode or ""
        if not billing_mode:
            provider = _resolve_provider(model)
            byok_key = provider_keys.get(provider, "") or provider_keys.get(
                {"openai": "openai", "anthropic": "anthropic", "gemini": "google"}.get(
                    provider, ""
                ),
                "",
            )
            if byok_key:
                billing_mode = detect_billing_mode(byok_key, provider)
            elif not gate.config.proxy.require_virtual_key and authorization:
                raw_bearer = strip_bearer(authorization)
                if raw_bearer and not raw_bearer.startswith("ag-"):
                    billing_mode = detect_billing_mode(raw_bearer, provider)
        if not billing_mode:
            billing_mode = "api"

        # Use passthrough for OpenAI models when passthrough is available
        provider = _resolve_provider(model)
        if passthrough is not None and provider == "openai":
            logger.debug("Routing via HTTP passthrough: model=%s provider=openai", model)
            return await _handle_openai_passthrough(
                gate=gate,
                passthrough=passthrough,
                request=request,
                raw_body=raw_body,
                body=body,
                model=model,
                messages=messages,
                stream=stream,
                session_id=session_id,
                session_name=session_name,
                request_id=request_id,
                vk=vk,
                provider_keys=provider_keys,
                billing_mode=billing_mode,
                proxy_rate_limiter=proxy_rate_limiter,
                vk_id=_vk_id,
                end_user=end_user,
            )

        # Fallback: SDK-based flow via Client (cross-provider, or no passthrough)
        logger.debug("Routing via Client SDK fallback: model=%s provider=%s", model, provider)
        extra_kwargs: dict[str, Any] = {}
        for key in (
            "temperature",
            "max_tokens",
            "top_p",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "tools",
            "tool_choice",
            "response_format",
            "seed",
        ):
            if key in body:
                extra_kwargs[key] = body[key]

        try:
            client = Client(
                session_id=session_id,
                name=session_name,
                org_id=vk.org_id,
                team_id=vk.team_id,
                provider_keys=provider_keys or None,
                billing_mode=billing_mode,
            )

            if end_user and client._session:
                client._session.end_user = end_user

            if stream:
                return await _handle_streaming(
                    client,
                    model,
                    messages,
                    extra_kwargs,
                    request_id,
                    proxy_rate_limiter=proxy_rate_limiter,
                    vk_id=_vk_id,
                )
            else:
                try:
                    async with client:
                        if end_user and client._session:
                            client._session.end_user = end_user
                        response = await client.achat(
                            model=model, messages=messages, **extra_kwargs
                        )

                        provider = _resolve_provider(model)
                        result = to_openai_completion_dict(
                            response.raw, provider, model, request_id
                        )
                        return JSONResponse(content=result)
                finally:
                    if _vk_id:
                        proxy_rate_limiter.on_request_complete(_vk_id)

        except Exception as exc:
            from stateloom.core.errors import StateLoomError

            if isinstance(exc, StateLoomError):
                status = error_status_code(exc)
                content = to_openai_error_dict(exc)
                return JSONResponse(status_code=status, content=content)

            logger.exception("Proxy error in /v1/chat/completions")
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

    @router.post("/agents/{agent_ref}/chat/completions")
    async def agent_chat_completions(
        agent_ref: str,
        request: Request,
        authorization: str = Header(default=""),
        x_stateloom_session_id: str = Header(default="", alias="X-StateLoom-Session-Id"),
        x_stateloom_openai_key: str = Header(default="", alias="X-StateLoom-OpenAI-Key"),
        x_stateloom_anthropic_key: str = Header(default="", alias="X-StateLoom-Anthropic-Key"),
        x_stateloom_google_key: str = Header(default="", alias="X-StateLoom-Google-Key"),
        x_stateloom_end_user: str = Header(default="", alias="X-StateLoom-End-User"),
    ) -> Any:
        """Agent-scoped chat completions endpoint.

        Always uses the SDK-based ``Client`` flow (never HTTP passthrough)
        because agents need model and system-prompt overrides applied via
        ``apply_agent_overrides()`` before the LLM call.
        """
        from stateloom.agent.resolver import (
            AgentResolutionError,
            apply_agent_overrides,
            resolve_agent,
        )

        # Auth
        auth = _authenticate(authorization)
        vk = auth.vk
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

        # Parse body
        try:
            body = await request.json()
        except Exception:
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

        messages = body.get("messages", [])
        stream = body.get("stream", False)

        if not messages:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "'messages' is required and must be non-empty",
                        "type": "invalid_request_error",
                        "code": "missing_messages",
                    }
                },
            )

        # Virtual key scope enforcement: scope check (before agent resolution)
        if vk.scopes:
            if not proxy_auth.check_scope(vk, "agents"):
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": {
                            "message": (
                                "Virtual key does not have the 'agents' scope. "
                                "Add 'agents' to the key's scopes list."
                            ),
                            "type": "permission_error",
                            "code": "scope_denied",
                        }
                    },
                )

        # Resolve agent
        vk_team_id = vk.team_id
        vk_agent_ids = vk.agent_ids or []
        try:
            agent, version = resolve_agent(gate.store, agent_ref, vk_team_id, vk_agent_ids or None)
        except AgentResolutionError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "message": exc.message,
                        "type": "invalid_request_error",
                        "code": "agent_error",
                    }
                },
            )

        # Apply agent overrides (model, system prompt, request overrides)
        model, messages, extra_kwargs = apply_agent_overrides(version, messages, body)

        # Virtual key policy enforcement (model access, budget, rate limit)
        # Note: scope "agents" was already checked above (before agent resolution).
        # We still need model/budget/rate checks since the model comes from agent overrides.
        _agent_policy_err: str | None = None
        if vk.allowed_models and not proxy_auth.check_model_access(vk, model):
            _agent_policy_err = f"model_not_allowed:{model}"
        elif vk.budget_limit is not None and not proxy_auth.check_budget(vk):
            _agent_policy_err = "key_budget_exceeded"
        else:
            if vk.rate_limit_tps is not None:
                try:
                    await proxy_rate_limiter.check(cast("VirtualKey", vk))
                except StateLoomRateLimitError:
                    _agent_policy_err = "key_rate_limit_exceeded"
        if _agent_policy_err is not None:
            status, error_code, msg = format_policy_error(_agent_policy_err, model, "agents")
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

        # End-user attribution
        from stateloom.proxy.auth import sanitize_end_user

        end_user = sanitize_end_user(x_stateloom_end_user) if x_stateloom_end_user else ""

        # Session ID
        session_id = x_stateloom_session_id or f"agent-{agent.slug}"

        logger.info(
            "POST /v1/agents/%s/chat/completions model=%s vk=%s session=%s",
            agent_ref,
            model,
            vk.id or "anonymous",
            session_id or "-",
        )

        # Request ID for the response
        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]

        # Resolve provider keys: BYOK headers > org secrets > global config
        provider_keys: dict[str, str] = {}
        if vk.org_id:
            provider_keys = proxy_auth.get_provider_keys(cast("VirtualKey", vk))
        if x_stateloom_openai_key:
            provider_keys["openai"] = x_stateloom_openai_key
        if x_stateloom_anthropic_key:
            provider_keys["anthropic"] = x_stateloom_anthropic_key
        if x_stateloom_google_key:
            provider_keys["google"] = x_stateloom_google_key

        # Determine if we need to release a rate limit slot on completion
        _vk_id = resolve_vk_rate_limit_id(vk)

        # Determine billing mode for agent route
        from stateloom.proxy.billing import detect_billing_mode as _detect_bm

        agent_billing_mode = vk.billing_mode or ""
        if not agent_billing_mode:
            agent_provider = _resolve_provider(model)
            agent_byok_key = provider_keys.get(agent_provider, "") or provider_keys.get(
                {"openai": "openai", "anthropic": "anthropic", "gemini": "google"}.get(
                    agent_provider, ""
                ),
                "",
            )
            if agent_byok_key:
                agent_billing_mode = _detect_bm(agent_byok_key, agent_provider)
        if not agent_billing_mode:
            agent_billing_mode = "api"

        # Route through Client (agents always use SDK-based flow)
        try:
            agent_metadata = {
                "agent_id": agent.id,
                "agent_slug": agent.slug,
                "agent_version_id": version.id,
                "agent_version_number": version.version_number,
            }

            def _apply_agent_fields(s: Any) -> None:
                """Set typed agent fields + metadata dict on session."""
                s.agent_id = agent.id
                s.agent_slug = agent.slug
                s.agent_version_id = version.id
                s.agent_version_number = version.version_number
                s.agent_name = agent.slug
                s.metadata.update(agent_metadata)
                s.metadata["agent_name"] = agent.slug

            agent_session_name = f"Agent: {agent.slug} / {model}"

            client = Client(
                session_id=session_id,
                name=agent_session_name,
                org_id=vk.org_id,
                team_id=vk.team_id,
                provider_keys=provider_keys or None,
                budget=version.budget_per_session,
                billing_mode=agent_billing_mode,
            )

            if stream:
                return await _handle_streaming_agent(
                    client,
                    model,
                    messages,
                    extra_kwargs,
                    request_id,
                    agent_metadata=agent_metadata,
                    proxy_rate_limiter=proxy_rate_limiter,
                    vk_id=_vk_id,
                    end_user=end_user,
                )
            else:
                try:
                    async with client:
                        if client._session is not None:
                            _apply_agent_fields(client._session)
                            if end_user:
                                client._session.end_user = end_user
                        response = await client.achat(
                            model=model, messages=messages, **extra_kwargs
                        )

                        provider = _resolve_provider(model)
                        result = to_openai_completion_dict(
                            response.raw, provider, model, request_id
                        )
                        return JSONResponse(content=result)
                finally:
                    if _vk_id:
                        proxy_rate_limiter.on_request_complete(_vk_id)

        except Exception as exc:
            from stateloom.core.errors import StateLoomError

            if isinstance(exc, StateLoomError):
                status = error_status_code(exc)
                content = to_openai_error_dict(exc)
                return JSONResponse(status_code=status, content=content)

            logger.exception("Agent proxy error in /v1/agents/%s", agent_ref)
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

    @router.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint for load balancers and monitoring."""
        return {"status": "ok"}

    @router.get("/models")
    async def list_models(
        authorization: str = Header(default=""),
    ) -> Any:
        """OpenAI-compatible /v1/models endpoint."""
        auth = _authenticate(authorization)
        if auth.vk is None:
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

        models_list: list[dict[str, Any]] = []
        for model_id in sorted(gate.pricing._prices):
            models_list.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": _model_owner(model_id),
                }
            )

        return {
            "object": "list",
            "data": models_list,
        }

    return router


async def _handle_openai_passthrough(
    *,
    gate: Gate,
    passthrough: PassthroughProxy,
    request: Request,
    raw_body: bytes,
    body: dict[str, Any],
    model: str,
    messages: list[dict[str, Any]],
    stream: bool,
    session_id: str,
    session_name: str,
    request_id: str,
    vk: Any,
    provider_keys: dict[str, str],
    billing_mode: str,
    proxy_rate_limiter: ProxyRateLimiter,
    vk_id: str | None,
    end_user: str = "",
) -> Response:
    """Handle OpenAI request via HTTP passthrough with middleware pipeline.

    Args:
        gate: The Gate singleton.
        passthrough: HTTP reverse proxy instance.
        request: The incoming FastAPI request.
        raw_body: Raw request body bytes.
        body: Parsed JSON body.
        model: Requested model.
        messages: Chat messages.
        stream: Whether streaming was requested.
        session_id: Resolved session ID.
        session_name: Human-readable session label.
        request_id: Generated ``chatcmpl-*`` request ID.
        vk: Virtual key or stub.
        provider_keys: Resolved BYOK/org provider keys.
        billing_mode: ``"api"`` or ``"subscription"``.
        proxy_rate_limiter: Rate limiter for slot release.
        vk_id: Virtual key ID for rate limit tracking (or None).

    Returns:
        A ``JSONResponse`` or ``StreamingResponse``.
    """
    from stateloom.middleware.base import MiddlewareContext

    upstream_base = gate.config.proxy.upstream_openai
    upstream_url = f"{upstream_base}/v1/chat/completions"

    # Build upstream headers — use BYOK key or resolved provider key
    auth_value = provider_keys.get("openai", "")
    # Also check Authorization header for direct BYOK passthrough
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
                method="chat.completions.create",
                request_kwargs={"messages": messages, "model": model, "_proxy": True},
                request_hash=""
                if stream
                else gate.pipeline._hash_request({"messages": messages, "model": model}),
                provider_base_url=gate.config.proxy.upstream_openai,
            )

            if stream:
                ctx.is_streaming = True
                await gate.pipeline.execute_streaming(ctx)

                if ctx.skip_call and ctx.cached_response is not None:
                    try:
                        result = to_openai_completion_dict(
                            ctx.cached_response, "openai", model, request_id
                        )
                        return _emit_openai_cached_as_stream(result, request_id)
                    except Exception:
                        logger.warning(
                            "Cache-hit SSE conversion failed, falling through",
                            exc_info=True,
                        )
                        ctx.skip_call = False
                        ctx.cached_response = None

                return await _handle_streaming_openai_passthrough(
                    passthrough,
                    upstream_url,
                    raw_body,
                    upstream_headers,
                    request_id=request_id,
                    model=model,
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
                    return cast(dict[str, Any], resp.json())

                result = await gate.pipeline.execute(ctx, llm_call)

                # Detect the _upstream_error sentinel set by llm_call() when
                # the upstream returns 4xx/5xx — pass the error through to
                # the client with the original status code.
                if isinstance(result, dict) and result.get("_upstream_error"):
                    status_code = result.pop("_status_code", 500)
                    result.pop("_upstream_error", None)
                    return JSONResponse(status_code=status_code, content=result)

                if isinstance(result, dict):
                    return JSONResponse(content=result)

                # Non-dict response from middleware (shouldn't happen, but handle)
                converted = to_openai_completion_dict(result, "openai", model, request_id)
                return JSONResponse(content=converted)

    except Exception as exc:
        from stateloom.core.errors import StateLoomError

        if isinstance(exc, StateLoomError):
            status = error_status_code(exc)
            content = to_openai_error_dict(exc)
            return JSONResponse(status_code=status, content=content)

        logger.exception("Proxy error in OpenAI passthrough")
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


async def _handle_streaming_openai_passthrough(
    passthrough: PassthroughProxy,
    upstream_url: str,
    body: bytes,
    headers: dict[str, str],
    *,
    request_id: str,
    model: str,
    ctx: Any = None,
    proxy_rate_limiter: ProxyRateLimiter | None = None,
    vk_id: str | None = None,
) -> StreamingResponse:
    """Forward streaming response from upstream OpenAI API."""

    def _format_error(exc: Exception) -> bytes:
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
        return f"data: {json.dumps(error)}\n\ndata: [DONE]\n\n".encode()

    return await passthrough_stream_relay(  # type: ignore[return-value]
        passthrough,
        upstream_url,
        body,
        headers,
        ctx=ctx,
        track_usage=_track_openai_stream_usage,
        format_error=_format_error,
        proxy_rate_limiter=proxy_rate_limiter,
        vk_id=vk_id,
    )


def _track_openai_stream_usage(chunk_str: str, ctx: Any) -> None:
    """Parse OpenAI SSE events to extract token usage for cost tracking."""
    if ctx is None:
        return
    try:
        for line in chunk_str.split("\n"):
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                continue
            data = json.loads(data_str)
            usage = data.get("usage")
            if usage:
                ctx.prompt_tokens = usage.get("prompt_tokens", 0)
                ctx.completion_tokens = usage.get("completion_tokens", 0)
    except Exception:
        logger.debug("OpenAI stream usage extraction failed", exc_info=True)


def _emit_openai_cached_as_stream(
    result: dict[str, Any],
    request_id: str,
) -> StreamingResponse:
    """Emit a cached response as OpenAI SSE events."""

    async def generate() -> AsyncGenerator[str, None]:
        choices = result.get("choices", [])
        if choices:
            role_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": result.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield to_openai_sse_event(role_chunk)

            content_chunk = _completion_to_chunk(result, request_id)
            for c in content_chunk.get("choices", []):
                c.get("delta", {}).pop("role", None)
            yield to_openai_sse_event(content_chunk)

        yield to_openai_done_event()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _handle_streaming(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    extra_kwargs: dict[str, Any],
    request_id: str,
    proxy_rate_limiter: ProxyRateLimiter | None = None,
    vk_id: str | None = None,
) -> StreamingResponse:
    """SDK-based streaming handler (buffer-then-emit pattern).

    The full response is obtained via ``client.achat()`` first, then
    converted to SSE chunk events and emitted.  This is not true token-by-
    token streaming; it simulates it so the client receives a well-formed
    SSE stream.

    Args:
        client: ``Client`` instance (session will be opened in the generator).
        model: Requested model.
        messages: Chat messages.
        extra_kwargs: Additional provider-specific params.
        request_id: Generated ``chatcmpl-*`` ID.
        proxy_rate_limiter: Rate limiter for slot release.
        vk_id: Virtual key ID for rate limit tracking (or None).

    Returns:
        A ``StreamingResponse`` with ``text/event-stream`` media type.
    """

    async def generate() -> AsyncGenerator[str, None]:
        try:
            async with client:
                response = await client.achat(model=model, messages=messages, **extra_kwargs)

                provider = _resolve_provider(model)
                full = to_openai_completion_dict(response.raw, provider, model, request_id)

                choices = full.get("choices", [])
                if choices:
                    role_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": full.get("model", ""),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield to_openai_sse_event(role_chunk)

                    content_chunk = _completion_to_chunk(full, request_id)
                    for c in content_chunk.get("choices", []):
                        c.get("delta", {}).pop("role", None)
                    yield to_openai_sse_event(content_chunk)
                else:
                    chunk = _completion_to_chunk(full, request_id)
                    yield to_openai_sse_event(chunk)

                yield to_openai_done_event()

        except Exception as exc:
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
            yield f"data: {json.dumps(error)}\n\n"
            yield to_openai_done_event()
        finally:
            if proxy_rate_limiter and vk_id:
                proxy_rate_limiter.on_request_complete(vk_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _handle_streaming_agent(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    extra_kwargs: dict[str, Any],
    request_id: str,
    agent_metadata: dict[str, Any] | None = None,
    proxy_rate_limiter: ProxyRateLimiter | None = None,
    vk_id: str | None = None,
    end_user: str = "",
) -> StreamingResponse:
    """Agent-scoped streaming handler — injects agent metadata on session.

    Args:
        client: ``Client`` instance.
        model: Resolved model (from agent version override).
        messages: Messages with agent system prompt prepended.
        extra_kwargs: Merged request overrides from agent version.
        request_id: Generated ``chatcmpl-*`` ID.
        agent_metadata: Dict with ``agent_id``, ``agent_slug``, etc.
        proxy_rate_limiter: Rate limiter for slot release.
        vk_id: Virtual key ID (or None).

    Returns:
        A ``StreamingResponse`` with ``text/event-stream`` media type.
    """

    async def generate() -> AsyncGenerator[str, None]:
        try:
            async with client:
                if client._session is not None:
                    if agent_metadata:
                        client._session.metadata.update(agent_metadata)
                        client._session.agent_id = agent_metadata.get("agent_id", "")
                        client._session.agent_slug = agent_metadata.get("agent_slug", "")
                        client._session.agent_version_id = agent_metadata.get(
                            "agent_version_id", ""
                        )
                        client._session.agent_version_number = agent_metadata.get(
                            "agent_version_number", 0
                        )
                        slug = agent_metadata.get("agent_slug", "")
                        if slug:
                            client._session.agent_name = slug
                            client._session.metadata["agent_name"] = slug
                    if end_user:
                        client._session.end_user = end_user
                response = await client.achat(model=model, messages=messages, **extra_kwargs)

                provider = _resolve_provider(model)
                full = to_openai_completion_dict(response.raw, provider, model, request_id)

                choices = full.get("choices", [])
                if choices:
                    role_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": full.get("model", ""),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield to_openai_sse_event(role_chunk)

                    content_chunk = _completion_to_chunk(full, request_id)
                    for c in content_chunk.get("choices", []):
                        c.get("delta", {}).pop("role", None)
                    yield to_openai_sse_event(content_chunk)
                else:
                    chunk = _completion_to_chunk(full, request_id)
                    yield to_openai_sse_event(chunk)

                yield to_openai_done_event()

        except Exception as exc:
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
            yield f"data: {json.dumps(error)}\n\n"
            yield to_openai_done_event()
        finally:
            if proxy_rate_limiter and vk_id:
                proxy_rate_limiter.on_request_complete(vk_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _completion_to_chunk(completion: dict[str, Any], request_id: str) -> dict[str, Any]:
    """Convert a full ChatCompletion dict into a streaming chunk dict.

    Args:
        completion: Complete ``chat.completion`` dict.
        request_id: The ``chatcmpl-*`` ID to use for the chunk.

    Returns:
        A ``chat.completion.chunk`` dict with ``delta`` instead of ``message``.
    """
    choices = completion.get("choices", [])
    chunk_choices = []
    for choice in choices:
        msg = choice.get("message", {})
        delta: dict[str, Any] = {}
        if msg.get("content"):
            delta["content"] = msg["content"]
        if msg.get("role"):
            delta["role"] = msg["role"]
        if msg.get("tool_calls"):
            delta["tool_calls"] = msg["tool_calls"]
        chunk_choices.append(
            {
                "index": choice.get("index", 0),
                "delta": delta,
                "finish_reason": choice.get("finish_reason"),
            }
        )

    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": completion.get("model", ""),
        "choices": chunk_choices,
    }


def _model_owner(model_id: str) -> str:
    """Infer model owner from model ID prefix.

    Args:
        model_id: Model identifier (e.g. ``"gpt-4o"``).

    Returns:
        Owner string: ``"openai"``, ``"anthropic"``, ``"google"``, or
        ``"stateloom"`` for unrecognized models.
    """
    if model_id.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    if model_id.startswith("claude"):
        return "anthropic"
    if model_id.startswith("gemini"):
        return "google"
    return "stateloom"


# _StubKey is now imported from proxy.auth — kept as a comment for git history.
# class _StubKey: ...  (removed: see proxy/auth.py)
