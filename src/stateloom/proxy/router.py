"""FastAPI router for the OpenAI-compatible proxy endpoints.

Supports HTTP passthrough for direct-to-provider forwarding (when a
PassthroughProxy is provided) and dedicated per-provider SDK handlers as
fallback.  Both paths build ``MiddlewareContext`` with ``_proxy: True``
so the PII scanner takes the stateful proxy path (Phase 1 dedup).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from stateloom.chat import _resolve_provider
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
        passthrough (OpenAI models) or provider SDK handler (cross-provider).
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

        # Fallback: dedicated SDK handler (cross-provider, or no passthrough).
        # Builds MiddlewareContext directly with _proxy: True so the PII
        # scanner takes the stateful proxy path (Phase 1 dedup).
        logger.debug("Routing via provider SDK handler: model=%s provider=%s", model, provider)
        extra_kwargs: dict[str, Any] = {}
        for key in (
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
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

        return await _handle_provider_sdk(
            gate=gate,
            model=model,
            provider=provider,
            messages=messages,
            extra_kwargs=extra_kwargs,
            session_id=session_id,
            session_name=session_name,
            request_id=request_id,
            vk=vk,
            provider_keys=provider_keys,
            billing_mode=billing_mode,
            stream=stream,
            end_user=end_user,
            proxy_rate_limiter=proxy_rate_limiter,
            vk_id=_vk_id,
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

        Always uses the provider SDK handler (never HTTP passthrough)
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

        # Route through dedicated SDK handler with agent metadata.
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

        return await _handle_provider_sdk(
            gate=gate,
            model=model,
            provider=_resolve_provider(model),
            messages=messages,
            extra_kwargs=extra_kwargs,
            session_id=session_id,
            session_name=agent_session_name,
            request_id=request_id,
            vk=vk,
            provider_keys=provider_keys,
            billing_mode=agent_billing_mode,
            stream=stream,
            end_user=end_user,
            proxy_rate_limiter=proxy_rate_limiter,
            vk_id=_vk_id,
            budget=version.budget_per_session,
            agent_fields_fn=_apply_agent_fields,
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

    # Build upstream headers — resolve OpenAI API key for passthrough.
    # Priority: BYOK header > org secrets > env var > raw bearer passthrough
    import os

    auth_value = provider_keys.get("openai", "")
    if not auth_value:
        auth_value = os.environ.get("OPENAI_API_KEY", "")
    # Last resort: raw bearer passthrough (for clients sending their own key)
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

                # Inject stream_options to get usage in the final chunk
                stream_body = body.copy()
                stream_body.setdefault("stream_options", {})["include_usage"] = True
                raw_stream_body = json.dumps(stream_body).encode()

                return await _handle_streaming_openai_passthrough(
                    passthrough,
                    upstream_url,
                    raw_stream_body,
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


def _build_gemini_llm_call(
    request_kwargs: dict[str, Any],
    provider_keys: dict[str, str],
    model: str,
) -> Callable[[], Any]:
    """Build a Gemini SDK call closure for the proxy SDK fallback path.

    Mirrors ``GeminiGenaiAdapter.prepare_chat()`` / ``GeminiAdapter.prepare_chat()``.
    Re-reads ``messages`` from *request_kwargs* at call time so PII redaction
    propagates.  Uses ``get_original()`` to avoid double interception.
    """
    # Capture closure references — gen_config is pre-computed, but messages
    # are re-read at call time from the same dict middleware modifies.
    gen_config: dict[str, Any] = {}
    for k in ("temperature", "top_p", "top_k"):
        if k in request_kwargs:
            gen_config[k] = request_kwargs[k]
    if "max_tokens" in request_kwargs:
        gen_config["max_output_tokens"] = request_kwargs["max_tokens"]
    if "max_output_tokens" in request_kwargs:
        gen_config["max_output_tokens"] = request_kwargs["max_output_tokens"]

    _rk = request_kwargs
    keys = provider_keys

    def llm_call() -> Any:
        from stateloom.intercept.unpatch import get_original

        # Try the new google-genai SDK first, fall back to legacy google-generativeai
        try:
            from google.genai import Client as GenaiClient
            from google.genai import models as genai_models

            use_new_sdk = True
        except ImportError:
            use_new_sdk = False

        if use_new_sdk:
            # Import the full message converter from the genai adapter
            from stateloom.intercept.adapters.gemini_genai_adapter import (
                _convert_messages,
                _convert_openai_tools,
                _convert_tool_choice,
            )

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
                call_kwargs["config"] = {"system_instruction": live_system}
            if gen_config:
                config = call_kwargs.setdefault("config", {})
                config.update(gen_config)

            # Tools
            openai_tools = _rk.get("tools")
            if openai_tools:
                declarations = _convert_openai_tools(openai_tools)
                if declarations:
                    config = call_kwargs.setdefault("config", {})
                    config["tools"] = [{"function_declarations": declarations}]
            tool_choice = _rk.get("tool_choice")
            if tool_choice is not None:
                config = call_kwargs.setdefault("config", {})
                config["tool_config"] = _convert_tool_choice(tool_choice)

            original = get_original(genai_models.Models, "generate_content")
            if original:
                return original(client.models, **call_kwargs)
            return client.models.generate_content(**call_kwargs)

        # Legacy google-generativeai fallback
        try:
            from google.generativeai import GenerativeModel  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError(
                "google-genai or google-generativeai package is required for Gemini models. "
                "Install with: pip install google-genai"
            )

        if keys.get("google"):
            import google.generativeai as genai

            genai.configure(api_key=keys["google"])  # type: ignore[attr-defined]

        live_messages = _rk.get("messages", [])
        live_contents_legacy: list[dict[str, Any]] = []
        live_system: str | None = None  # type: ignore[no-redef]
        for msg in live_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                live_system = content if isinstance(content, str) else str(content)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            live_contents_legacy.append({"role": gemini_role, "parts": [{"text": str(content)}]})

        gen_model = GenerativeModel(model, system_instruction=live_system)
        original = get_original(GenerativeModel, "generate_content")
        if original:
            return original(gen_model, live_contents_legacy, generation_config=gen_config or None)
        return gen_model.generate_content(
            live_contents_legacy,  # type: ignore[arg-type]
            generation_config=gen_config or None,  # type: ignore[arg-type]
        )

    return llm_call


def _build_anthropic_llm_call(
    request_kwargs: dict[str, Any],
    provider_keys: dict[str, str],
    model: str,
) -> Callable[[], Any]:
    """Build an Anthropic SDK call closure for the proxy SDK fallback path.

    Mirrors ``AnthropicAdapter.prepare_chat()``.  Reads specific fields from
    *request_kwargs* at call time (never spreads ``**request_kwargs``).
    Uses ``get_original()`` to avoid double interception.
    """
    _rk = request_kwargs
    keys = provider_keys

    def llm_call() -> Any:
        import anthropic

        from stateloom.intercept.adapters.anthropic_adapter import (
            _convert_messages,
            _convert_openai_tools,
            _convert_tool_choice,
        )
        from stateloom.intercept.unpatch import get_original

        ctor_kwargs: dict[str, Any] = {
            "base_url": "https://api.anthropic.com",
        }
        if keys.get("anthropic"):
            ctor_kwargs["api_key"] = keys["anthropic"]
        client = anthropic.Anthropic(**ctor_kwargs)

        # Convert messages from OpenAI format to Anthropic format at call time
        # so middleware modifications (e.g. PII redaction) are reflected.
        live_messages = _rk.get("messages", [])
        converted_msgs, converted_system = _convert_messages(live_messages)

        sdk_kwargs: dict[str, Any] = {
            "model": _rk.get("model", model),
            "messages": converted_msgs,
            "max_tokens": _rk.get("max_tokens", 4096),
        }

        # System prompt: prefer already-extracted _rk["system"], fall back to converted
        system = _rk.get("system")
        if system:
            sdk_kwargs["system"] = system
        elif converted_system:
            sdk_kwargs["system"] = converted_system

        for k in ("temperature", "top_p", "stop"):
            if k in _rk:
                sdk_kwargs[k] = _rk[k]

        # Convert tools from OpenAI to Anthropic format
        openai_tools = _rk.get("tools")
        if openai_tools:
            sdk_kwargs["tools"] = _convert_openai_tools(openai_tools)

        # Convert tool_choice
        openai_tool_choice = _rk.get("tool_choice")
        if openai_tool_choice is not None:
            converted_tc = _convert_tool_choice(openai_tool_choice)
            if converted_tc is None:
                # "none" → omit tools entirely
                sdk_kwargs.pop("tools", None)
            else:
                sdk_kwargs["tool_choice"] = converted_tc

        original = get_original(type(client.messages), "create")
        if original:
            return original(client.messages, **sdk_kwargs)
        return client.messages.create(**sdk_kwargs)

    return llm_call


def _build_ollama_llm_call(
    request_kwargs: dict[str, Any],
    model: str,
    ollama_host: str,
) -> Callable[[], Any]:
    """Build an Ollama SDK call closure via Ollama's OpenAI-compat endpoint.

    Points an unpatched ``openai.OpenAI`` client at Ollama's ``/v1`` endpoint
    with a dummy API key.  Reads specific fields from *request_kwargs* at call
    time so middleware modifications (e.g. PII redaction) propagate.

    ``tools`` and ``tool_choice`` are always stripped — most local models
    do not support tool calling.
    """
    _rk = request_kwargs

    def llm_call() -> Any:
        import openai

        from stateloom.intercept.unpatch import get_original

        client = openai.OpenAI(base_url=f"{ollama_host}/v1", api_key="ollama")

        sdk_kwargs: dict[str, Any] = {
            "model": _rk.get("model", model),
            "messages": _rk.get("messages", []),
        }
        for k in (
            "temperature",
            "max_tokens",
            "top_p",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        ):
            if k in _rk:
                sdk_kwargs[k] = _rk[k]

        original = get_original(type(client.chat.completions), "create")
        if original:
            return original(client.chat.completions, **sdk_kwargs)
        return client.chat.completions.create(**sdk_kwargs)

    return llm_call


def _build_openai_compat_llm_call(
    request_kwargs: dict[str, Any],
    provider_keys: dict[str, str],
    model: str,
    provider: str,
) -> Callable[[], Any]:
    """Build an OpenAI-compatible SDK call closure for fallback providers.

    Handles Cohere, Mistral, and unknown providers that expose an
    OpenAI-compatible API.  Reads specific fields from *request_kwargs*.
    Uses ``get_original()`` to avoid double interception.
    """
    _rk = request_kwargs
    keys = provider_keys

    # Resolve API key and base URL for the provider
    _base_urls: dict[str, str] = {
        "mistral": "https://api.mistral.ai/v1",
        "cohere": "https://api.cohere.ai/compatibility/v1",
    }

    def llm_call() -> Any:
        import openai

        from stateloom.intercept.unpatch import get_original

        ctor_kwargs: dict[str, Any] = {}
        api_key = keys.get(provider) or keys.get("openai", "")
        if api_key:
            ctor_kwargs["api_key"] = api_key
        base_url = _base_urls.get(provider)
        if base_url:
            ctor_kwargs["base_url"] = base_url

        client = openai.OpenAI(**ctor_kwargs)

        # Build SDK kwargs from specific fields
        sdk_kwargs: dict[str, Any] = {
            "model": _rk.get("model", model),
            "messages": _rk.get("messages", []),
        }
        for k in (
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
            if k in _rk:
                sdk_kwargs[k] = _rk[k]

        original = get_original(type(client.chat.completions), "create")
        if original:
            return original(client.chat.completions, **sdk_kwargs)
        return client.chat.completions.create(**sdk_kwargs)

    return llm_call


async def _handle_provider_sdk(
    *,
    gate: Gate,
    model: str,
    provider: str,
    messages: list[dict[str, Any]],
    extra_kwargs: dict[str, Any],
    session_id: str,
    session_name: str,
    request_id: str,
    vk: Any,
    provider_keys: dict[str, str],
    billing_mode: str,
    stream: bool,
    end_user: str,
    proxy_rate_limiter: ProxyRateLimiter,
    vk_id: str | None,
    budget: float | None = None,
    agent_fields_fn: Callable[[Any], None] | None = None,
) -> Response:
    """Unified provider SDK handler for the proxy fallback path.

    Replaces the old ``Client.achat()``-based fallback.  Builds
    ``MiddlewareContext`` directly with ``_proxy: True`` so the PII
    scanner takes the stateful proxy path (Phase 1 dedup).

    Args:
        gate: The Gate singleton.
        model: Requested model.
        provider: Resolved provider name.
        messages: Chat messages (OpenAI format).
        extra_kwargs: Additional provider-specific params from request body.
        session_id: Resolved session ID.
        session_name: Human-readable session label.
        request_id: Generated ``chatcmpl-*`` ID.
        vk: Virtual key or stub.
        provider_keys: Resolved BYOK/org provider keys.
        billing_mode: ``"api"`` or ``"subscription"``.
        stream: Whether streaming was requested.
        end_user: Sanitized end-user attribution string.
        proxy_rate_limiter: Rate limiter for slot release.
        vk_id: Virtual key ID for rate limit tracking (or None).
        budget: Optional per-session budget (for agents).
        agent_fields_fn: Optional callback to set agent metadata on session.

    Returns:
        A ``JSONResponse`` or ``StreamingResponse``.
    """
    from stateloom.middleware.base import MiddlewareContext

    try:
        session_kwargs: dict[str, Any] = {
            "session_id": session_id,
            "name": session_name,
            "org_id": vk.org_id,
            "team_id": vk.team_id,
        }
        if budget is not None:
            session_kwargs["budget"] = budget

        async with gate.async_session(**session_kwargs) as session:
            session.billing_mode = billing_mode
            session.metadata["billing_mode"] = billing_mode
            if end_user:
                session.end_user = end_user
            if agent_fields_fn:
                agent_fields_fn(session)
            session.next_step()

            # Build request_kwargs with _proxy flag — this is the key fix.
            # The PII scanner checks this flag to dispatch to the stateful
            # proxy path (Phase 1 dedup) instead of the stateless SDK path.
            request_kwargs: dict[str, Any] = {
                "messages": messages,
                "model": model,
                "_proxy": True,
            }

            # Anthropic: extract system prompt from messages
            if provider == "anthropic":
                system_parts: list[str] = []
                non_system: list[dict[str, Any]] = []
                for msg in messages:
                    if msg.get("role") == "system":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            system_parts.append(content)
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    system_parts.append(block.get("text", ""))
                    else:
                        non_system.append(msg)
                request_kwargs["messages"] = non_system
                request_kwargs["max_tokens"] = extra_kwargs.pop("max_tokens", 4096)
                if system_parts:
                    request_kwargs["system"] = "\n\n".join(system_parts)

            # Copy known extra kwargs into request_kwargs
            for k in (
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
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
                if k in extra_kwargs:
                    request_kwargs[k] = extra_kwargs[k]

            # Strip ollama: prefix so downstream middleware sees the real model name
            if provider == "local" and model.startswith("ollama:"):
                actual_model = model.removeprefix("ollama:")
                request_kwargs["model"] = actual_model
                model = actual_model

            # Build the provider-specific llm_call closure
            if provider == "local":
                ollama_host = gate.config.local_model_host or "http://localhost:11434"
                llm_call = _build_ollama_llm_call(request_kwargs, model, ollama_host)
            elif provider == "gemini":
                llm_call = _build_gemini_llm_call(request_kwargs, provider_keys, model)
            elif provider == "anthropic":
                llm_call = _build_anthropic_llm_call(request_kwargs, provider_keys, model)
            else:
                llm_call = _build_openai_compat_llm_call(
                    request_kwargs, provider_keys, model, provider
                )

            ctx = MiddlewareContext(
                session=session,
                config=gate.config,
                provider=provider,
                model=model,
                method="chat.completions.create",
                request_kwargs=request_kwargs,
                request_hash=""
                if stream
                else gate.pipeline._hash_request(
                    {"messages": request_kwargs["messages"], "model": model}
                ),
            )

            if stream:
                # Buffer-then-emit: pipeline runs the full LLM call and gets
                # the complete response.  We do NOT set ctx.is_streaming here
                # because from the middleware perspective the response is fully
                # buffered (not chunked).  Setting is_streaming would cause
                # EventRecorder to defer persistence to _on_stream_complete
                # callbacks that are never fired in this path.
                result = await gate.pipeline.execute(ctx, llm_call)

                provider_for_convert = provider
                full = to_openai_completion_dict(result, provider_for_convert, model, request_id)

                async def generate() -> AsyncGenerator[str, None]:
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

                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers=SSE_HEADERS,
                )

            else:
                result = await gate.pipeline.execute(ctx, llm_call)
                converted = to_openai_completion_dict(result, provider, model, request_id)
                return JSONResponse(content=converted)

    except Exception as exc:
        from stateloom.core.errors import StateLoomError

        if isinstance(exc, StateLoomError):
            status = error_status_code(exc)
            content = to_openai_error_dict(exc)
            return JSONResponse(status_code=status, content=content)

        logger.exception("Proxy error in provider SDK handler")
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
