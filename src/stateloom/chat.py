"""Unified stateloom.Client and stateloom.chat() API.

Provides a provider-agnostic chat interface that routes through the full
middleware pipeline with auto-routing eligibility. SDK direct calls (monkey-
patched) flow through the pipeline for observability but never auto-route.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from stateloom.core.session import Session
from stateloom.intercept.unpatch import get_original
from stateloom.middleware.base import MiddlewareContext

if TYPE_CHECKING:
    from stateloom.agent.models import Agent, AgentVersion
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.chat")

# Provider resolution patterns
_OPENAI_PATTERN = re.compile(r"^(gpt-|o1|o3|o4|chatgpt-)")
_ANTHROPIC_PATTERN = re.compile(r"^claude-")
_GEMINI_PATTERN = re.compile(r"^gemini-")


@dataclass
class ChatResponse:
    """Response from stateloom.chat() with metadata about what actually served the request."""

    content: str
    model: str
    provider: str
    raw: Any = None
    _stateloom: dict[str, Any] = field(default_factory=dict)


def _resolve_provider(model: str) -> str:
    """Resolve provider from model name prefix.

    Args:
        model: Model identifier (e.g. ``"gpt-4o"``, ``"claude-3-opus"``).

    Returns:
        Provider name string (e.g. ``"openai"``, ``"anthropic"``).
        Defaults to ``"openai"`` for unknown model prefixes (most common
        usage pattern).

    Tries the adapter registry first (covers all registered providers
    including Mistral, Cohere, etc.), then falls back to legacy regex
    patterns as a safety net for when the registry is empty (before init()).
    """
    # Explicit local model prefix — skip adapter registry entirely
    if model.startswith("ollama:"):
        return "local"

    from stateloom.intercept.provider_registry import resolve_provider

    result = resolve_provider(model)
    if result is not None:
        return result
    # Legacy fallback (registry empty before init(), or unknown model)
    if _OPENAI_PATTERN.match(model):
        return "openai"
    if _ANTHROPIC_PATTERN.match(model):
        return "anthropic"
    if _GEMINI_PATTERN.match(model):
        return "gemini"
    return "openai"


def _extract_content(response: Any, provider: str) -> str:
    """Extract text content from a provider-specific response object.

    Args:
        response: Raw provider SDK response (or dict for kill-switch/cache).
        provider: Provider name used for adapter lookup priority.

    Returns:
        The extracted text content, or ``""`` on failure.
    """
    if response is None:
        return ""
    try:
        # Dict fallback (kill switch response, etc.)
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                return cast(str, msg.get("content", ""))
            return cast(str, response.get("content", ""))

        # Use adapter for provider-specific extraction
        from stateloom.intercept.provider_registry import get_adapter

        adapter = get_adapter(provider)
        if adapter is not None:
            text = adapter.extract_content(response)
            if text:
                return text

        # Fallback: try all registered adapters
        from stateloom.intercept.provider_registry import get_all_adapters

        all_adapters = get_all_adapters()
        for name, fallback in all_adapters.items():
            if name == provider:
                continue
            text = fallback.extract_content(response)
            if text:
                return text

        # Structural fallback when registry is empty (tests, standalone use)
        if not all_adapters:
            from stateloom.intercept.adapters.anthropic_adapter import AnthropicAdapter
            from stateloom.intercept.adapters.gemini_adapter import GeminiAdapter
            from stateloom.intercept.adapters.openai_adapter import OpenAIAdapter

            for cls in (OpenAIAdapter, AnthropicAdapter, GeminiAdapter):
                text = cls().extract_content(response)
                if text:
                    return text
    except Exception:
        logger.debug("Content extraction failed", exc_info=True)
    return ""


def _get_gate() -> Gate:
    """Get the global Gate instance."""
    from stateloom import get_gate

    return get_gate()


def _resolve_agent_for_chat(gate: Gate, agent_ref: str) -> tuple[Agent, AgentVersion]:
    """Resolve an agent by slug or ID for SDK-level chat calls.

    Unlike proxy-level ``resolve_agent()``, this skips VK scoping (SDK calls
    are in-process and trusted).  First slug match wins.

    Args:
        gate: The Gate singleton.
        agent_ref: Agent slug (e.g. ``"support-bot"``) or ID (``"agt-..."``).

    Returns:
        ``(Agent, AgentVersion)`` tuple.

    Raises:
        StateLoomError: If the agent is not found, paused, archived, or has
            no active version.
    """
    from stateloom.agent.models import Agent as AgentModel
    from stateloom.core.errors import StateLoomError
    from stateloom.core.types import AgentStatus

    agent: AgentModel | None = None

    if agent_ref.startswith("agt-"):
        agent = gate.store.get_agent(agent_ref)
    else:
        # Slug lookup — scan all agents for first match (no team scoping)
        all_agents = gate.store.list_agents()
        for a in all_agents:
            if a.slug == agent_ref:
                agent = a
                break

    if agent is None:
        raise StateLoomError(
            f"Agent not found: {agent_ref}",
            details="Pass an agent slug or ID (agt-...) to the agent parameter.",
        )

    if agent.status == AgentStatus.PAUSED:
        raise StateLoomError(f"Agent '{agent_ref}' is paused")
    if agent.status == AgentStatus.ARCHIVED:
        raise StateLoomError(f"Agent '{agent_ref}' has been archived")

    if not agent.active_version_id:
        raise StateLoomError(f"Agent '{agent_ref}' has no active version")

    version = gate.store.get_agent_version(agent.active_version_id)
    if version is None:
        raise StateLoomError(f"Agent '{agent_ref}' has no active version")

    return agent, version


class Client:
    """Unified chat client with session ownership.

    Usage::

        # Context manager — session scoped to block
        with stateloom.Client(session_id="task-1", budget=5.0) as client:
            r1 = client.chat(model="gpt-4", messages=[...])
            r2 = client.chat(model="claude-3-opus", messages=[...])
            print(client.session.total_cost)

        # Standalone — session owned for lifetime
        client = stateloom.Client(session_id="agent-run")
        r = client.chat(model="gpt-4", messages=[...])
        client.close()

        # One-liner — uses default client
        r = stateloom.chat(model="gpt-4", messages=[...])
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        budget: float | None = None,
        org_id: str = "",
        team_id: str = "",
        provider_keys: dict[str, str] | None = None,
        billing_mode: str = "api",
        **session_kwargs: Any,
    ) -> None:
        """Initialize a unified chat client.

        Args:
            session_id: Explicit session ID, or None to auto-generate (or
                reuse the active ContextVar session).
            budget: Per-session budget in USD.
            org_id: Organization scope for multi-tenant isolation.
            team_id: Team scope for multi-tenant isolation.
            provider_keys: BYOK provider API keys (e.g.
                ``{"openai": "sk-..."}``).  When set, these override org-level
                secrets and global config.
            billing_mode: ``"api"`` for per-token billing or
                ``"subscription"`` for flat-rate.  Affects cost tracking.
            **session_kwargs: Additional kwargs forwarded to
                ``Gate.session()`` (e.g. ``name``, ``durable``, ``timeout``).
        """
        self._session_kwargs = {
            "session_id": session_id,
            "budget": budget,
            "org_id": org_id,
            "team_id": team_id,
            **session_kwargs,
        }
        self._provider_keys: dict[str, str] = provider_keys or {}
        self._billing_mode = billing_mode
        self._gate: Gate | None = None
        self._session: Session | None = None
        self._session_cm: Any = None

    def _ensure_session(self) -> Session:
        """Get or create the session for this Client.

        If there is already an active session in the ContextVar (e.g. the caller
        is inside ``with stateloom.session(...)``), reuse it instead of creating
        a nested session.  This ensures ``stateloom.chat()`` records events and
        cost in the surrounding session.
        """
        from stateloom.core.context import get_current_session

        gate = _get_gate()
        self._gate = gate

        # When no explicit session_id was requested, always prefer the
        # active session from the ContextVar.  This handles the case where
        # a global _default_client is reused across multiple session blocks.
        if self._session_kwargs.get("session_id") is None:
            existing = get_current_session()
            if existing is not None:
                self._session = existing
                return self._session

        if self._session is not None:
            return self._session

        self._session_cm = gate.session(**self._session_kwargs)
        self._session = self._session_cm.__enter__()
        if self._billing_mode != "api":
            self._session.billing_mode = self._billing_mode
            self._session.metadata["billing_mode"] = self._billing_mode
        return self._session

    def __enter__(self) -> Client:
        self._ensure_session()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def __aenter__(self) -> Client:
        from stateloom.core.context import get_current_session

        gate = _get_gate()
        self._gate = gate

        # Reuse active session from ContextVar when no explicit session_id
        if self._session_kwargs.get("session_id") is None:
            existing = get_current_session()
            if existing is not None:
                self._session = existing
                return self
        self._session_cm = gate.async_session(**self._session_kwargs)
        self._session = await self._session_cm.__aenter__()
        if self._billing_mode != "api":
            self._session.billing_mode = self._billing_mode
            self._session.metadata["billing_mode"] = self._billing_mode
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(*exc)
            self._session = None
            self._session_cm = None

    def close(self) -> None:
        """End the session and clean up (sync path)."""
        if self._session_cm is not None:
            self._session_cm.__exit__(None, None, None)
            self._session = None
            self._session_cm = None

    async def aclose(self) -> None:
        """End the session and clean up (async path)."""
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session = None
            self._session_cm = None

    @property
    def session(self) -> Session | None:
        """Access the underlying session for cost/token inspection."""
        return self._session

    def _resolve_model(self, gate: Gate, model: str | None) -> tuple[str, bool]:
        """Resolve the effective model and whether auto-routing is eligible.

        Returns (resolved_model, auto_route_eligible).
        - Explicit model → use it, no auto-routing.
        - No model → use config.default_model, auto-routing eligible.
        - No model and no default → raise.
        """
        if model is not None:
            return model, False

        default = gate.config.default_model
        if not default:
            from stateloom.core.errors import StateLoomError

            raise StateLoomError(
                "No model specified and no default_model configured.",
                details=(
                    "Either pass model= to chat(), or set default_model "
                    "in stateloom.init(default_model='gpt-4o')."
                ),
            )
        return default, True

    def chat(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        agent: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Sync chat through the full middleware pipeline.

        Args:
            model: Model to use. When provided, the request always goes to that
                model (no auto-routing). When omitted, uses config.default_model
                and the request is eligible for auto-routing to a local model.
            messages: Chat messages in OpenAI format.
            agent: Optional agent slug or ID (``"agt-..."``). When provided,
                the agent's model, system prompt, and request overrides are
                applied automatically.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A ``ChatResponse`` containing the content, model, provider, raw
            response, and ``_stateloom`` metadata dict.

        Raises:
            StateLoomError: On middleware policy violations (budget, PII,
                kill switch, etc.).
        """
        session = self._ensure_session()
        gate = self._gate
        assert gate is not None

        if agent is not None:
            messages, kwargs, model = self._apply_agent(gate, agent, session, messages, kwargs)

        resolved_model, auto_route_eligible = self._resolve_model(gate, model)
        provider = _resolve_provider(resolved_model)

        session.next_step()

        request_kwargs, llm_call = self._prepare_call(
            gate, provider, resolved_model, messages, **kwargs
        )
        ctx = self._build_ctx(
            gate, provider, resolved_model, request_kwargs, session, auto_route_eligible
        )

        logger.debug(
            "Client.chat: provider=%s model=%s session=%s auto_route=%s",
            provider,
            resolved_model,
            session.id,
            auto_route_eligible,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Nested event loop detected (Jupyter, async frameworks).
            # asyncio.run() would fail, so dispatch to a fresh thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, gate.pipeline.execute(ctx, llm_call))
                result = future.result()
        else:
            result = asyncio.run(gate.pipeline.execute(ctx, llm_call))

        return self._build_response(result, resolved_model, provider, ctx)

    async def achat(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        agent: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat through the full middleware pipeline.

        Args:
            model: Model to use. When provided, the request always goes to that
                model (no auto-routing). When omitted, uses config.default_model
                and the request is eligible for auto-routing to a local model.
            messages: Chat messages in OpenAI format.
            agent: Optional agent slug or ID (``"agt-..."``). When provided,
                the agent's model, system prompt, and request overrides are
                applied automatically.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A ``ChatResponse`` with content, model, provider, raw response,
            and ``_stateloom`` metadata.

        Raises:
            StateLoomError: On middleware policy violations.
        """
        from stateloom.core.context import get_current_session

        gate = _get_gate()
        self._gate = gate

        # Always prefer the active session from the ContextVar.  This handles
        # the case where _default_client is reused across multiple session
        # blocks (same pattern as sync _ensure_session).
        if self._session_kwargs.get("session_id") is None:
            existing = get_current_session()
            if existing is not None:
                self._session = existing

        if self._session is None:
            self._session_cm = gate.async_session(**self._session_kwargs)
            self._session = await self._session_cm.__aenter__()
            if self._billing_mode != "api":
                self._session.billing_mode = self._billing_mode
                self._session.metadata["billing_mode"] = self._billing_mode

        session = self._session
        gate = self._gate
        assert gate is not None
        assert session is not None

        if agent is not None:
            messages, kwargs, model = self._apply_agent(gate, agent, session, messages, kwargs)

        resolved_model, auto_route_eligible = self._resolve_model(gate, model)
        provider = _resolve_provider(resolved_model)

        session.next_step()

        request_kwargs, llm_call = self._prepare_call(
            gate, provider, resolved_model, messages, **kwargs
        )
        ctx = self._build_ctx(
            gate, provider, resolved_model, request_kwargs, session, auto_route_eligible
        )
        logger.debug(
            "Client.achat: provider=%s model=%s session=%s auto_route=%s",
            provider,
            resolved_model,
            session.id,
            auto_route_eligible,
        )
        result = await gate.pipeline.execute(ctx, llm_call)
        return self._build_response(result, resolved_model, provider, ctx)

    @staticmethod
    def _apply_agent(
        gate: Gate,
        agent_ref: str,
        session: Session,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        """Resolve an agent and apply its overrides.

        Returns:
            ``(messages, kwargs, model)`` with agent overrides applied.
        """
        from stateloom.agent.resolver import apply_agent_overrides

        agent_obj, version = _resolve_agent_for_chat(gate, agent_ref)
        model, messages, extra_kwargs = apply_agent_overrides(version, messages, body={})
        kwargs.update(extra_kwargs)

        # Set session typed fields (same pattern as proxy)
        session.agent_id = agent_obj.id
        session.agent_slug = agent_obj.slug
        session.agent_version_id = version.id
        session.agent_version_number = version.version_number
        session.agent_name = agent_obj.slug
        session.metadata.update(
            {
                "agent_id": agent_obj.id,
                "agent_slug": agent_obj.slug,
                "agent_version_id": version.id,
                "agent_version_number": version.version_number,
                "agent_name": agent_obj.slug,
            }
        )

        return messages, kwargs, model

    # Default provider base URLs for chat() calls
    _PROVIDER_BASE_URLS: dict[str, str] = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "gemini": "https://generativelanguage.googleapis.com",
    }

    def _build_ctx(
        self,
        gate: Gate,
        provider: str,
        model: str,
        request_kwargs: dict[str, Any],
        session: Session,
        auto_route_eligible: bool = False,
    ) -> MiddlewareContext:
        """Build a ``MiddlewareContext`` for a chat call.

        Args:
            gate: The Gate singleton.
            provider: Resolved provider name.
            model: Resolved model identifier.
            request_kwargs: Provider-specific request kwargs.
            session: Active session.
            auto_route_eligible: Whether auto-routing is allowed.

        Returns:
            A fully initialized ``MiddlewareContext``.
        """
        from stateloom.intercept.provider_registry import get_adapter

        # Resolve base URL: adapter first, then legacy dict
        base_url = ""
        adapter = get_adapter(provider)
        if adapter is not None:
            try:
                base_url = adapter.default_base_url
            except (AttributeError, NotImplementedError):
                pass
        if not base_url:
            base_url = self._PROVIDER_BASE_URLS.get(provider, "")

        return MiddlewareContext(
            session=session,
            config=gate.config,
            provider=provider,
            method="chat",
            model=model,
            request_kwargs=request_kwargs,
            request_hash=gate.pipeline._hash_request(request_kwargs),
            auto_route_eligible=auto_route_eligible,
            provider_base_url=base_url,
        )

    def _prepare_call(
        self,
        gate: Gate,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any]:
        """Build provider-specific kwargs and llm_call callable.

        Args:
            gate: The Gate singleton.
            provider: Resolved provider name.
            model: Model identifier.
            messages: Chat messages in OpenAI format.
            **kwargs: Extra provider-specific parameters.

        Returns:
            A 2-tuple of ``(request_kwargs, llm_call)`` where
            ``request_kwargs`` is the dict passed through the middleware
            pipeline and ``llm_call`` is a zero-arg callable that makes
            the actual SDK API call.

        Tries the adapter's ``prepare_chat()`` first for dynamic provider
        support, then falls back to legacy ``_prepare_*`` methods.
        """
        from stateloom.intercept.provider_registry import get_adapter

        adapter = get_adapter(provider)
        if adapter is not None:
            try:
                return adapter.prepare_chat(
                    model=model,
                    messages=messages,
                    provider_keys=self._provider_keys or None,
                    **kwargs,
                )
            except NotImplementedError:
                pass
        # Legacy fallback
        if provider == "anthropic":
            return self._prepare_anthropic(gate, model, messages, **kwargs)
        elif provider == "gemini":
            return self._prepare_gemini(gate, model, messages, **kwargs)
        return self._prepare_openai(gate, model, messages, **kwargs)

    def _prepare_openai(
        self,
        gate: Gate,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any]:
        """Prepare OpenAI request.

        Args:
            gate: The Gate singleton.
            model: Model identifier (e.g. ``"gpt-4o"``).
            messages: OpenAI-format messages.
            **kwargs: Extra params forwarded to the SDK.

        Returns:
            ``(request_kwargs, llm_call)`` tuple.
        """
        request_kwargs = {"model": model, "messages": messages, **kwargs}

        def llm_call() -> Any:
            import openai

            ctor_kwargs: dict[str, Any] = {
                "base_url": "https://api.openai.com/v1",
            }
            if self._provider_keys.get("openai"):
                ctor_kwargs["api_key"] = self._provider_keys["openai"]
            client = openai.OpenAI(**ctor_kwargs)
            original = get_original(type(client.chat.completions), "create")
            method = original or client.chat.completions.create
            if original:
                return method(client.chat.completions, **request_kwargs)  # type: ignore[call-overload]
            return method(**request_kwargs)

        return request_kwargs, llm_call

    def _prepare_anthropic(
        self,
        gate: Gate,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any]:
        """Prepare Anthropic request.

        System messages are extracted from ``messages`` and joined into a
        single ``system`` kwarg (Anthropic SDK uses a top-level ``system``
        param, not a system message in the messages array).  ``max_tokens``
        defaults to 4096 if not provided.

        Args:
            gate: The Gate singleton.
            model: Model identifier (e.g. ``"claude-3-opus"``).
            messages: OpenAI-format messages (system messages extracted).
            **kwargs: Extra params (``max_tokens`` popped with default 4096).

        Returns:
            ``(request_kwargs, llm_call)`` tuple.
        """
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

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": non_system,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system_parts:
            request_kwargs["system"] = "\n\n".join(system_parts)

        def llm_call() -> Any:
            import anthropic

            ctor_kwargs: dict[str, Any] = {
                "base_url": "https://api.anthropic.com",
            }
            if self._provider_keys.get("anthropic"):
                ctor_kwargs["api_key"] = self._provider_keys["anthropic"]
            client = anthropic.Anthropic(**ctor_kwargs)
            original = get_original(type(client.messages), "create")
            method = original or client.messages.create
            if original:
                return method(client.messages, **request_kwargs)  # type: ignore[call-overload]
            return method(**request_kwargs)

        return request_kwargs, llm_call

    def _prepare_gemini(
        self,
        gate: Gate,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any]:
        """Prepare Gemini request — convert to contents format.

        Stores ``messages`` (OpenAI format) in ``request_kwargs`` so middleware
        (PII scanner, cache, loop detector) can inspect and modify them.  The
        ``llm_call`` closure rebuilds ``contents`` from ``messages`` at call
        time so that any middleware modifications (e.g. PII redaction) are
        reflected in the actual Gemini API call.
        """
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

        # Extract generation config params before building request_kwargs
        gen_config: dict[str, Any] = {}
        for k in ("temperature", "max_tokens", "top_p", "top_k"):
            if k in kwargs:
                gen_config[k] = kwargs.pop(k)

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
        # Closure captures a reference to the SAME dict object that middleware
        # modifies (e.g. PII redaction on "messages").  At call time, llm_call()
        # re-reads "messages" from this dict and rebuilds "contents", so
        # middleware modifications propagate to the actual Gemini API call.
        _rk = request_kwargs

        def llm_call() -> Any:
            try:
                from google.generativeai import GenerativeModel  # type: ignore[attr-defined]
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required for Gemini models. "
                    "Install with: pip install google-generativeai"
                )

            if self._provider_keys.get("google"):
                import google.generativeai as genai

                genai.configure(api_key=self._provider_keys["google"])  # type: ignore[attr-defined]

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

            # Use original (pre-patch) method to avoid double interception
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

    @staticmethod
    def _build_response(
        raw_response: Any,
        requested_model: str,
        requested_provider: str,
        ctx: MiddlewareContext,
    ) -> ChatResponse:
        """Build a ``ChatResponse`` with ``_stateloom`` metadata from the pipeline.

        Args:
            raw_response: The raw provider SDK response object.
            requested_model: The model the caller originally asked for.
            requested_provider: The provider the caller originally targeted.
            ctx: The ``MiddlewareContext`` after pipeline execution.

        Returns:
            A ``ChatResponse`` with extracted content and metadata dict
            including actual model/provider, routing info, cost, and latency.
        """
        content = _extract_content(raw_response, requested_provider)

        # Calculate cost from events
        cost = 0.0
        event_types: list[str] = []
        for event in ctx.events:
            if hasattr(event, "cost"):
                cost += event.cost
            event_types.append(
                event.event_type.value
                if hasattr(event.event_type, "value")
                else str(event.event_type)
            )

        return ChatResponse(
            content=content,
            model=requested_model,
            provider=requested_provider,
            raw=raw_response,
            _stateloom={
                "actual_model": ctx.model,
                "actual_provider": ctx.provider,
                "routed_local": ctx.model != requested_model,
                "cached": ctx.skip_call and ctx.cached_response is not None,
                "cost": cost,
                "latency_ms": ctx.latency_ms,
                "prompt_tokens": ctx.prompt_tokens,
                "completion_tokens": ctx.completion_tokens,
                "session_id": ctx.session.id,
                "events": event_types,
            },
        )


# --- Module-level convenience functions ---

_default_client: Client | None = None


def chat(
    *,
    model: str | None = None,
    messages: list[dict[str, Any]],
    agent: str | None = None,
    **kwargs: Any,
) -> ChatResponse:
    """One-liner sync chat through the full middleware pipeline.

    When called outside an explicit ``stateloom.session()`` block, creates a
    throwaway session for the single call and closes it immediately after.
    Inside a session block, reuses the active session.

    Args:
        model: Model to use. When provided, no auto-routing. When omitted,
            uses config.default_model and is eligible for auto-routing.
        messages: Chat messages in OpenAI format.
        agent: Optional agent slug or ID. When provided, the agent's model,
            system prompt, and request overrides are applied.
    """
    global _default_client
    if _default_client is None:
        _default_client = Client()
    result = _default_client.chat(model=model, messages=messages, agent=agent, **kwargs)
    # Close the auto-created session so it doesn't leak into later calls.
    # When inside an explicit session block, _session_cm is None (session
    # came from ContextVar), so close() is a no-op.
    _default_client.close()
    return result


async def achat(
    *,
    model: str | None = None,
    messages: list[dict[str, Any]],
    agent: str | None = None,
    **kwargs: Any,
) -> ChatResponse:
    """One-liner async chat through the full middleware pipeline.

    When called outside an explicit ``stateloom.async_session()`` block,
    creates a throwaway session for the single call and closes it immediately
    after.  Inside a session block, reuses the active session.

    Args:
        model: Model to use. When provided, no auto-routing. When omitted,
            uses config.default_model and is eligible for auto-routing.
        messages: Chat messages in OpenAI format.
        agent: Optional agent slug or ID. When provided, the agent's model,
            system prompt, and request overrides are applied.
    """
    global _default_client
    if _default_client is None:
        _default_client = Client()
    result = await _default_client.achat(model=model, messages=messages, agent=agent, **kwargs)
    _default_client.close()
    return result
