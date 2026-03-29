"""Code Assist proxy adapter for Gemini CLI OAuth/subscription users.

Gemini CLI users authenticated via Google OAuth (e.g. Gemini Code Assist
subscribers) talk to ``cloudcode-pa.googleapis.com/v1internal`` — a different
API surface than the standard Gemini REST endpoint.  This module transparently
proxies those requests so they get full middleware benefits (cost tracking,
PII scanning, budget enforcement, caching, dashboard visibility) while the
upstream responses are forwarded as-is.

User setup::

    export CODE_ASSIST_ENDPOINT=http://localhost:4782/code-assist
    gemini "hello"

Two-tier routing:
  - **LLM calls** (``generateContent``, ``streamGenerateContent``)
    go through the middleware pipeline + HTTP passthrough.
  - **Everything else** (``loadCodeAssist``, ``onboardUser``, etc.)
    is pure HTTP passthrough with no middleware.
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
    format_policy_error,
    resolve_vk_rate_limit_id,
    strip_bearer,
)
from stateloom.proxy.passthrough import (
    RESPONSE_HOP_BY_HOP_HEADERS,
    PassthroughProxy,
    filter_headers,
)
from stateloom.proxy.rate_limiter import ProxyRateLimiter
from stateloom.proxy.sticky_session import (
    StickySessionManager,
    derive_session_name,
    resolve_session_id,
)
from stateloom.proxy.stream_helpers import SSE_HEADERS

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.proxy.code_assist")

# Response header filtering uses the shared set from passthrough.py
_RESPONSE_HOP_BY_HOP = RESPONSE_HOP_BY_HOP_HEADERS

# Gemini CLI user_prompt_id values that indicate CLI-internal requests
# (not user-initiated prompts). These calls should be marked as cli_internal.
_CLI_INTERNAL_PROMPT_IDS = frozenset({
    "session-summary-generation",
})


def create_code_assist_router(
    gate: Gate,
    sticky_session: StickySessionManager | None = None,
    passthrough: PassthroughProxy | None = None,
) -> APIRouter:
    """Create the Code Assist router (mounted at ``/code-assist``)."""
    router = APIRouter()
    proxy_auth = ProxyAuth(gate)
    proxy_rate_limiter = ProxyRateLimiter(
        metrics=gate._metrics_collector,
        enabled=gate.config.rate_limiting_enabled,
    )

    def _authenticate(authorization: str) -> AuthResult:
        """Validate auth and return AuthResult.

        Code Assist uses ``Authorization: Bearer <token>``.
        - Token starting with ``ag-`` → virtual key mode.
        - Otherwise → forward OAuth token as-is to upstream.

        No BYOK concept — Code Assist users are always subscription-based.
        """
        token = strip_bearer(authorization)
        return authenticate_request(proxy_auth, token, gate.config, has_byok=False)

    async def _handle_generate(
        request: Request,
        version: str,
        authorization: str,
        x_stateloom_session_id: str = "",
        stream: bool = False,
    ) -> Any:
        """Shared handler for generateContent and streamGenerateContent."""
        # Auth
        auth = _authenticate(authorization)
        vk, raw_token = auth.vk, auth.raw_token
        if vk is None:
            msg = auth.error_hint or "Invalid or missing authorization"
            return JSONResponse(
                status_code=401,
                content=_code_assist_error(401, msg, "UNAUTHENTICATED"),
            )

        # Parse body
        try:
            raw_body = await request.body()
            body = json.loads(raw_body)
        except Exception:
            return JSONResponse(
                status_code=400,
                content=_code_assist_error(400, "Invalid JSON in request body", "INVALID_ARGUMENT"),
            )

        # Code Assist wraps the request: model at top, contents inside "request"
        model = body.get("model", "")
        inner_request = body.get("request", {})
        if not isinstance(inner_request, dict):
            inner_request = {}

        contents = inner_request.get("contents", [])
        system_instruction = inner_request.get("systemInstruction")

        # Extract Gemini CLI metadata for session tracking and CLI-internal detection
        user_prompt_id = body.get("user_prompt_id", "")
        gemini_session_id = inner_request.get("session_id", "")

        if not model:
            return JSONResponse(
                status_code=400,
                content=_code_assist_error(400, "'model' is required", "INVALID_ARGUMENT"),
            )

        if not contents:
            return JSONResponse(
                status_code=400,
                content=_code_assist_error(
                    400, "'request.contents' is required and must be non-empty", "INVALID_ARGUMENT"
                ),
            )

        # VK policy enforcement
        from stateloom.core.errors import StateLoomRateLimitError

        _ca_policy_err: str | None = None
        if vk.allowed_models and not proxy_auth.check_model_access(vk, model):
            _ca_policy_err = f"model_not_allowed:{model}"
        elif vk.budget_limit is not None and not proxy_auth.check_budget(vk):
            _ca_policy_err = "key_budget_exceeded"
        else:
            if vk.rate_limit_tps is not None:
                try:
                    await proxy_rate_limiter.check(vk)
                except StateLoomRateLimitError:
                    _ca_policy_err = "key_rate_limit_exceeded"
        if _ca_policy_err is not None:
            status, _error_code, msg = format_policy_error(_ca_policy_err, model, "generate")
            ca_status = "RESOURCE_EXHAUSTED" if status == 429 else "PERMISSION_DENIED"
            return JSONResponse(
                status_code=status,
                content=_code_assist_error(status, msg, ca_status),
            )

        # Convert Code Assist contents to OpenAI messages for middleware
        openai_messages = _contents_to_openai_messages(contents, system_instruction)

        if not openai_messages:
            return JSONResponse(
                status_code=400,
                content=_code_assist_error(
                    400, "No valid messages found in contents", "INVALID_ARGUMENT"
                ),
            )

        # Session ID: prefer explicit header, then Gemini CLI's own session_id,
        # then fall back to sticky session hashing.
        if x_stateloom_session_id:
            session_id = x_stateloom_session_id
        elif gemini_session_id:
            session_id = gemini_session_id
        else:
            session_id = resolve_session_id("", request, sticky_session)
        session_name = derive_session_name(request, model, "code-assist")

        # Detect CLI-internal calls from user_prompt_id sentinel values
        _cli_internal = user_prompt_id in _CLI_INTERNAL_PROMPT_IDS

        logger.info(
            "POST /code-assist/%s model=%s vk=%s session=%s prompt_id=%s%s",
            version, model, vk.id if hasattr(vk, "id") else "anonymous",
            session_id or "-", user_prompt_id or "-",
            " [cli-internal]" if _cli_internal else "",
        )

        # Resolve provider keys (for VK mode only)
        provider_keys: dict[str, str] = {}
        if vk.org_id:
            provider_keys = proxy_auth.get_provider_keys(vk)

        # Rate limit slot tracking
        _vk_id = resolve_vk_rate_limit_id(vk)

        # Billing mode: always subscription for Code Assist
        billing_mode = vk.billing_mode or "subscription"

        # Build upstream URL
        upstream_base = gate.config.proxy.upstream_code_assist
        action = "streamGenerateContent" if stream else "generateContent"
        upstream_url = f"{upstream_base}/{version}:{action}"
        if stream:
            # Code Assist streaming uses ?alt=sse
            upstream_url += "?alt=sse"

        # Auth header: VK mode → resolved google key; otherwise → forward OAuth
        auth_value = provider_keys.get("google", "") or raw_token
        upstream_headers = filter_headers(
            request.headers,
            auth_header_name="authorization" if auth_value else "",
            auth_header_value=f"Bearer {auth_value}" if auth_value else "",
        )

        # Passthrough proxy required — no Client fallback for Code Assist
        if passthrough is None:
            return JSONResponse(
                status_code=503,
                content=_code_assist_error(
                    503,
                    "Code Assist proxy requires HTTP passthrough (no SDK fallback)",
                    "UNAVAILABLE",
                ),
            )

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
            cli_internal=_cli_internal,
            user_prompt_id=user_prompt_id,
        )

    @router.post("/{version}:generateContent")
    async def generate_content(
        version: str,
        request: Request,
        authorization: str = Header(default="", alias="authorization"),
        x_stateloom_session_id: str = Header(default="", alias="X-StateLoom-Session-Id"),
    ) -> Any:
        """Code Assist generateContent endpoint."""
        return await _handle_generate(
            request,
            version,
            authorization,
            x_stateloom_session_id,
            stream=False,
        )

    @router.post("/{version}:streamGenerateContent")
    async def stream_generate_content(
        version: str,
        request: Request,
        authorization: str = Header(default="", alias="authorization"),
        x_stateloom_session_id: str = Header(default="", alias="X-StateLoom-Session-Id"),
    ) -> Any:
        """Code Assist streamGenerateContent endpoint."""
        return await _handle_generate(
            request,
            version,
            authorization,
            x_stateloom_session_id,
            stream=True,
        )

    @router.api_route("/{version}:{method}", methods=["GET", "POST"])
    async def catch_all_method(
        version: str,
        method: str,
        request: Request,
    ) -> Any:
        """Catch-all for non-LLM Code Assist methods (loadCodeAssist, etc.).

        Pure HTTP passthrough — no middleware pipeline.
        """
        if method in ("generateContent", "streamGenerateContent"):
            # Should not reach here — specific routes take priority.
            # But guard just in case.
            authorization = request.headers.get("authorization", "")
            session_id = request.headers.get("x-stateloom-session-id", "")
            stream = method == "streamGenerateContent"
            return await _handle_generate(
                request,
                version,
                authorization,
                session_id,
                stream=stream,
            )

        return await _forward_utility(request, version, method=method)

    @router.api_route("/{version}/{path:path}", methods=["GET", "POST"])
    async def catch_all_path(
        version: str,
        path: str,
        request: Request,
    ) -> Any:
        """Catch-all for operations polling and other path-based endpoints."""
        return await _forward_utility(request, version, path=path)

    async def _forward_utility(
        request: Request,
        version: str,
        *,
        method: str = "",
        path: str = "",
    ) -> Response:
        """Forward a non-LLM request to upstream as-is."""
        if passthrough is None:
            return JSONResponse(
                status_code=503,
                content=_code_assist_error(
                    503,
                    "Code Assist proxy requires HTTP passthrough",
                    "UNAVAILABLE",
                ),
            )

        upstream_base = gate.config.proxy.upstream_code_assist
        if method:
            upstream_url = f"{upstream_base}/{version}:{method}"
        else:
            upstream_url = f"{upstream_base}/{version}/{path}"

        # Preserve query string
        query = str(request.url.query)
        if query:
            upstream_url += f"?{query}"

        raw_body = await request.body()
        upstream_headers = filter_headers(request.headers)

        try:
            resp = await passthrough.forward(upstream_url, raw_body, upstream_headers)
            # Filter hop-by-hop headers from upstream response — Starlette
            # sets its own content-length/transfer-encoding and having both
            # causes "Content-Length can't be present with Transfer-Encoding".
            resp_headers = {
                k: v for k, v in resp.headers.items() if k.lower() not in _RESPONSE_HOP_BY_HOP
            }
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
            )
        except Exception:
            logger.exception("Code Assist utility passthrough error")
            return JSONResponse(
                status_code=502,
                content=_code_assist_error(502, "Upstream request failed", "INTERNAL"),
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
    cli_internal: bool = False,
    user_prompt_id: str = "",
) -> Response:
    """Handle an LLM request via HTTP passthrough with middleware pipeline."""
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
            session.next_step()

            request_kwargs: dict[str, Any] = {
                "messages": openai_messages,
                "model": model,
            }
            if cli_internal:
                request_kwargs["_cli_internal"] = True
            if user_prompt_id:
                request_kwargs["_user_prompt_id"] = user_prompt_id

            ctx = MiddlewareContext(
                session=session,
                config=gate.config,
                provider="gemini",
                model=model,
                method="generateContent",
                request_kwargs=request_kwargs,
                request_hash="" if stream else gate.pipeline._hash_request(
                    {"messages": openai_messages, "model": model}
                ),
                provider_base_url=gate.config.proxy.upstream_code_assist,
            )

            # Snapshot original messages so we can detect middleware changes
            original_messages = list(openai_messages)

            if stream:
                ctx.is_streaming = True
                await gate.pipeline.execute_streaming(ctx)

                if ctx.skip_call and ctx.cached_response is not None:
                    try:
                        result = ctx.cached_response
                        if isinstance(result, dict):
                            return _emit_cached_as_stream(result)
                    except Exception:
                        logger.warning(
                            "Cache-hit SSE conversion failed, falling through",
                            exc_info=True,
                        )
                    ctx.skip_call = False
                    ctx.cached_response = None

                # Rebuild body if middleware modified messages (e.g. PII stripping)
                forwarded_body = raw_body
                current_messages = ctx.request_kwargs.get("messages", [])
                if current_messages != original_messages:
                    forwarded_body = _patch_code_assist_body(body, original_messages, current_messages)

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
                    # Rebuild body if middleware modified messages (e.g. PII stripping)
                    forwarded_body = raw_body
                    current_messages = ctx.request_kwargs.get("messages", [])
                    if current_messages != original_messages:
                        forwarded_body = _patch_code_assist_body(body, original_messages, current_messages)

                    resp = await passthrough.forward(upstream_url, forwarded_body, upstream_headers)
                    if resp.status_code >= 400:
                        try:
                            error_data = resp.json()
                        except Exception:
                            error_data = _code_assist_error(resp.status_code, resp.text, "INTERNAL")
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

                # Shouldn't happen — passthrough always returns dict
                return JSONResponse(content={"response": str(result)})

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
) -> Response:
    """Forward streaming response from upstream Code Assist API.

    Eagerly peeks the first chunk to detect upstream errors (4xx/5xx)
    before committing to a 200 SSE stream.  Upstream errors are returned
    with the original status code so CLIs can display them.
    """
    from stateloom.proxy.passthrough import UpstreamStreamError

    gen = passthrough.forward_stream(upstream_url, body, headers)

    try:
        first_chunk = await gen.__anext__()
    except UpstreamStreamError as e:
        logger.debug("Upstream error %d for %s", e.status_code, upstream_url[:120])
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
            logger.exception("Code Assist proxy streaming error")
            _status, content = _map_error(exc)
            yield f"data: {json.dumps(content)}\n\n".encode("utf-8")
        finally:
            if ctx is not None:
                for cb in ctx._on_stream_complete:
                    try:
                        cb()
                    except Exception:
                        pass
            if proxy_rate_limiter and vk_id:
                proxy_rate_limiter.on_request_complete(vk_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _track_stream_usage(chunk_str: str, ctx: Any) -> None:
    """Parse Code Assist SSE events to extract token usage for cost tracking.

    Code Assist nests usage inside ``response.usageMetadata``.
    """
    if ctx is None:
        return
    try:
        for line in chunk_str.split("\n"):
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            # Code Assist: {"response": {"usageMetadata": {...}}}
            inner = data.get("response", {})
            if isinstance(inner, dict):
                meta = inner.get("usageMetadata", {})
                if meta:
                    ctx.prompt_tokens = meta.get("promptTokenCount", 0)
                    ctx.completion_tokens = meta.get("candidatesTokenCount", 0)
    except Exception:
        logger.debug("Code Assist stream usage extraction failed", exc_info=True)


def _emit_cached_as_stream(result: dict[str, Any]) -> StreamingResponse:
    """Emit a cached response as SSE."""

    async def generate() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _patch_code_assist_body(
    original_body: dict[str, Any],
    original_messages: list[dict[str, Any]],
    current_messages: list[dict[str, Any]],
) -> bytes:
    """Patch the original Code Assist body to reflect middleware changes.

    Instead of rebuilding the Gemini body from OpenAI messages (which loses
    provider-specific fields like ``thought_signature``), this uses the
    ``_content_idx`` mapping on each message to apply only the diff:

    - **Removed messages**: the corresponding Gemini contents entries are
      removed (all messages referencing a content index must be gone for
      the content to be removed).
    - **Modified text**: updated in the original Gemini content's text parts.
    - **System prompt changes**: patched in systemInstruction.
    """
    import copy

    body = copy.deepcopy(original_body)
    request = body.get("request", {})
    contents = request.get("contents", [])

    # Build set of content indices still referenced by current messages
    kept_content_indices: set[int] = set()
    # Build mapping of content_idx → set of text values for redaction detection
    current_texts: dict[int, list[str]] = {}
    has_system = False

    for msg in current_messages:
        ci = msg.get("_content_idx", -2)
        if ci == -1:
            has_system = True
            continue
        if ci >= 0:
            kept_content_indices.add(ci)
        # Track text for redaction detection
        text = msg.get("content", "")
        if isinstance(text, str) and text and ci >= 0:
            current_texts.setdefault(ci, []).append(text)

    # Build set of content indices from original messages
    original_content_indices: set[int] = set()
    original_texts: dict[int, list[str]] = {}
    for msg in original_messages:
        ci = msg.get("_content_idx", -2)
        if ci >= 0:
            original_content_indices.add(ci)
        text = msg.get("content", "")
        if isinstance(text, str) and text and ci >= 0:
            original_texts.setdefault(ci, []).append(text)

    # Determine which content indices were removed
    removed_indices = original_content_indices - kept_content_indices

    # Apply text redactions to kept contents
    for ci in kept_content_indices:
        if ci >= len(contents):
            continue
        orig_texts = original_texts.get(ci, [])
        curr_texts = current_texts.get(ci, [])
        if orig_texts != curr_texts and curr_texts:
            # Text was modified (redaction) — update text parts
            entry = contents[ci]
            text_part_idx = 0
            for part in entry.get("parts", []):
                if "text" in part and text_part_idx < len(curr_texts):
                    part["text"] = curr_texts[text_part_idx]
                    text_part_idx += 1

    # Remove stripped content entries (reverse order to preserve indices)
    for ci in sorted(removed_indices, reverse=True):
        if ci < len(contents):
            contents.pop(ci)

    request["contents"] = contents

    # Handle system prompt changes
    had_system = any(m.get("_content_idx") == -1 for m in original_messages)
    if had_system and not has_system:
        request.pop("systemInstruction", None)
    elif has_system:
        sys_msg = next((m for m in current_messages if m.get("role") == "system"), None)
        if sys_msg:
            request["systemInstruction"] = {"parts": [{"text": sys_msg.get("content", "")}]}

    body["request"] = request
    return json.dumps(body).encode()


def _contents_to_openai_messages(
    contents: list[dict[str, Any]],
    system_instruction: Any = None,
) -> list[dict[str, Any]]:
    """Convert Gemini-style contents to OpenAI messages format for middleware.

    Translates Gemini ``functionCall`` parts to OpenAI assistant ``tool_calls``
    and ``functionResponse`` parts to ``role="tool"`` messages so that
    ``_is_tool_continuation()`` can detect tool-use loops for dashboard
    step collapsing.

    Each message gets a ``_content_idx`` field tracking which Gemini contents
    entry it originated from (-1 for system instruction).  This mapping is
    used by ``_patch_code_assist_body`` to apply middleware changes (PII
    stripping/redaction) back to the original Gemini body without losing
    provider-specific fields like ``thought_signature``.
    """
    messages: list[dict[str, Any]] = []

    if system_instruction:
        sys_text = _extract_system_text(system_instruction)
        if sys_text:
            messages.append({"role": "system", "content": sys_text, "_content_idx": -1})

    for ci, item in enumerate(contents):
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
            if text_parts:
                messages.append({
                    "role": openai_role, "content": "\n".join(text_parts),
                    "_content_idx": ci,
                })
            for fr in function_responses:
                resp = fr.get("response", {})
                messages.append({
                    "role": "tool",
                    "content": json.dumps(resp) if isinstance(resp, dict) else str(resp),
                    "tool_call_id": fr.get("name", "unknown"),
                    "_content_idx": ci,
                })
        elif function_calls:
            # Gemini function calls → OpenAI assistant + tool_calls
            messages.append({
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
                "_content_idx": ci,
            })
        elif text_parts:
            messages.append({
                "role": openai_role, "content": "\n".join(text_parts),
                "_content_idx": ci,
            })

    return messages


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


def _code_assist_error(code: int, message: str, status: str) -> dict[str, Any]:
    """Build a Google-style error response."""
    return {
        "error": {
            "code": code,
            "message": message,
            "status": status,
        },
    }


def _map_error(exc: Exception) -> tuple[int, dict[str, Any]]:
    """Map an exception to (status_code, error_dict)."""
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
        return 429, _code_assist_error(429, str(exc), "RESOURCE_EXHAUSTED")
    if isinstance(exc, StateLoomBudgetError):
        return 400, _code_assist_error(400, str(exc), "INVALID_ARGUMENT")
    if isinstance(exc, StateLoomPIIBlockedError):
        return 400, _code_assist_error(400, str(exc), "INVALID_ARGUMENT")
    if isinstance(exc, StateLoomKillSwitchError):
        return 503, _code_assist_error(503, str(exc), "UNAVAILABLE")
    if isinstance(exc, StateLoomBlastRadiusError):
        return 503, _code_assist_error(503, str(exc), "UNAVAILABLE")
    if isinstance(exc, StateLoomTimeoutError):
        return 504, _code_assist_error(504, str(exc), "DEADLINE_EXCEEDED")
    if isinstance(exc, StateLoomCancellationError):
        return 499, _code_assist_error(499, str(exc), "CANCELLED")
    if isinstance(exc, StateLoomError):
        return 500, _code_assist_error(500, str(exc), "INTERNAL")

    logger.exception("Code Assist proxy error")
    return 500, _code_assist_error(500, f"Internal server error ({type(exc).__name__})", "INTERNAL")


# _StubKey is now imported from proxy.auth — kept as a comment for git history.
# class _StubKey: ...  (removed: see proxy/auth.py)
