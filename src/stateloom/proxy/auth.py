"""Proxy authentication — virtual key validation and provider key resolution.

Provides shared auth primitives used by all proxy endpoints:

* :class:`_StubKey` — duck-type stand-in for :class:`VirtualKey` when no
  virtual key is present.  Has every field ``VirtualKey`` has, with
  permissive defaults (empty lists, ``None`` budgets, etc.) so downstream
  code never needs ``hasattr`` guards.
* :class:`AuthResult` — unified return value from :func:`authenticate_request`.
* :func:`authenticate_request` — single auth function replacing the five
  per-endpoint ``_authenticate()`` implementations.
* :func:`enforce_vk_policies` — consolidated model-access / budget /
  rate-limit / scope enforcement (was duplicated ~36× across endpoints).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any

from stateloom.proxy.virtual_key import VirtualKey, hash_key

# End-user header sanitization
_NON_PRINTABLE = re.compile(r"[^\x20-\x7E]")
_MAX_END_USER_LEN = 256


def strip_bearer(authorization: str) -> str:
    """Extract the token from a ``Bearer <token>`` header value.

    Case-insensitive, handles extra whitespace.
    Returns empty string if header doesn't match.
    """
    if not authorization:
        return ""
    if authorization.lower().startswith("bearer "):
        return authorization[7:].lstrip()
    return ""


def sanitize_end_user(value: str) -> str:
    """Sanitize X-StateLoom-End-User header value.

    Strips non-printable characters and truncates to 256 chars
    to prevent log injection and UI breakage.
    """
    return _NON_PRINTABLE.sub("", value)[:_MAX_END_USER_LEN]


# Well-known proxy scopes
PROXY_SCOPES = {"chat", "agents", "models", "responses", "messages", "generate"}

if TYPE_CHECKING:
    from stateloom.core.config import StateLoomConfig
    from stateloom.gate import Gate
    from stateloom.proxy.rate_limiter import ProxyRateLimiter

logger = logging.getLogger("stateloom.proxy.auth")


# ── Shared auth types ────────────────────────────────────────────────


@dataclass
class _StubKey:
    """Stub for unauthenticated proxy requests.

    Duck-type compatible with :class:`VirtualKey` — every field that proxy
    code accesses is present with a permissive default so that no
    ``hasattr`` / ``getattr`` guards are needed downstream.
    """

    id: str = ""
    org_id: str = ""
    team_id: str = ""
    name: str = "anonymous"
    scopes: list[str] = field(default_factory=list)
    allowed_models: list[str] = field(default_factory=list)
    budget_limit: float | None = None
    budget_spent: float = 0.0
    rate_limit_tps: float | None = None
    billing_mode: str = ""
    agent_ids: list[str] = field(default_factory=list)
    revoked: bool = False


@dataclass
class AuthResult:
    """Unified auth result for all proxy endpoints."""

    vk: VirtualKey | _StubKey | None = None
    byok_key: str = ""
    raw_token: str = ""
    error_hint: str = ""


def authenticate_request(
    proxy_auth: ProxyAuth,
    token: str,
    config: StateLoomConfig,
    *,
    has_byok: bool = True,
) -> AuthResult:
    """Shared authentication for all proxy endpoints.

    Args:
        proxy_auth: The ``ProxyAuth`` instance for VK lookups.
        token: The bearer/API-key token already extracted from the header
            by the caller (no ``Bearer `` prefix).
        config: Gate config (to check ``proxy_require_virtual_key``).
        has_byok: When ``True`` (default), non-VK tokens in managed mode
            are treated as BYOK provider keys.  Set to ``False`` for
            endpoints where BYOK is not applicable (e.g. Code Assist).

    Returns:
        An :class:`AuthResult`.  ``result.vk`` is ``None`` when auth is
        required but the token is invalid — the caller should return 401.
    """
    if token:
        if token.startswith("ag-"):
            vk = proxy_auth.authenticate(f"Bearer {token}")
            if vk is not None:
                logger.debug(
                    "VK auth success: id=%s team=%s org=%s",
                    getattr(vk, "id", ""),
                    getattr(vk, "team_id", ""),
                    getattr(vk, "org_id", ""),
                )
                return AuthResult(vk=vk)
            # Invalid VK token
            logger.warning("VK auth failed: token not found or revoked (ag-***%s)", token[-4:])
            return AuthResult(
                vk=None,
                error_hint=(
                    "Virtual key not found or revoked. Check that your ag-* token is valid."
                ),
            )
        if config.proxy.require_virtual_key and has_byok:
            # Managed mode: treat as BYOK provider key
            logger.debug("BYOK auth: managed mode, token treated as provider key")
            return AuthResult(vk=_StubKey(), byok_key=token, raw_token=token)
        if not config.proxy.require_virtual_key or not has_byok:
            # No-auth mode OR non-BYOK endpoint (e.g. Code Assist):
            # forward CLI's auth token as-is
            logger.debug("Passthrough auth: forwarding CLI token as-is")
            return AuthResult(vk=_StubKey(), raw_token=token)
        # proxy_require_virtual_key=True but no BYOK and not ag-*
        logger.warning("Auth rejected: VK required but non-ag-* token provided")
        return AuthResult(
            vk=None,
            error_hint=(
                "A virtual key (ag-*) is required. The provided token is not a valid virtual key."
            ),
        )

    # No token at all
    if not config.proxy.require_virtual_key:
        logger.debug("No-auth mode: no token provided, proceeding with stub key")
        return AuthResult(vk=_StubKey())
    logger.warning("Auth rejected: no token provided and VK is required")
    return AuthResult(
        vk=None,
        error_hint=(
            "No authorization token provided. Send a Bearer token in the "
            "Authorization header (virtual key ag-*, or provider API key)."
        ),
    )


async def enforce_vk_policies(
    vk: VirtualKey | _StubKey,
    model: str,
    required_scope: str,
    proxy_auth: ProxyAuth,
    proxy_rate_limiter: ProxyRateLimiter,
) -> str | None:
    """Check VK policies (model access, budget, rate limit, scope).

    Returns ``None`` if all checks pass, or an error message string if
    blocked.  The caller wraps this into the provider-specific error
    format.

    The return convention encodes the check that failed:

    * ``"model_not_allowed:{model}"``
    * ``"key_budget_exceeded"``
    * ``"key_rate_limit_exceeded"``
    * ``"scope_denied:{scope}"``
    """
    from stateloom.core.errors import StateLoomRateLimitError

    # Model access
    if vk.allowed_models:
        if not proxy_auth.check_model_access(vk, model):
            logger.warning("VK policy: model '%s' not allowed for key %s", model, vk.id)
            return f"model_not_allowed:{model}"

    # Budget
    if vk.budget_limit is not None:
        if not proxy_auth.check_budget(vk):
            logger.warning(
                "VK policy: budget exceeded for key %s (spent=%.4f, limit=%.4f)",
                vk.id,
                vk.budget_spent,
                vk.budget_limit,
            )
            return "key_budget_exceeded"

    # Rate limit
    if vk.rate_limit_tps is not None:
        try:
            await proxy_rate_limiter.check(vk)  # type: ignore[arg-type]
        except StateLoomRateLimitError:
            logger.info(
                "VK policy: rate limit exceeded for key %s (tps=%.1f)",
                vk.id,
                vk.rate_limit_tps,
            )
            return "key_rate_limit_exceeded"

    # Scope
    if vk.scopes:
        if not proxy_auth.check_scope(vk, required_scope):
            logger.warning(
                "VK policy: scope '%s' denied for key %s (scopes=%s)",
                required_scope,
                vk.id,
                vk.scopes,
            )
            return f"scope_denied:{required_scope}"

    return None


def format_policy_error(policy_error: str, model: str, default_scope: str) -> tuple[int, str, str]:
    """Parse policy error string into (status_code, error_code, message).

    Returns a tuple usable by all proxy endpoint error formatters.
    """
    code = policy_error.split(":")[0]
    if code == "model_not_allowed":
        return (
            403,
            "model_not_allowed",
            f"Model '{model}' is not allowed for this virtual key. "
            f"Check allowed_models in your VK configuration.",
        )
    if code == "key_budget_exceeded":
        return (
            403,
            "key_budget_exceeded",
            "Virtual key budget exceeded. Contact your admin to increase "
            "the budget or create a new key.",
        )
    if code == "key_rate_limit_exceeded":
        return (
            429,
            "key_rate_limit_exceeded",
            "Rate limit exceeded for this virtual key. Wait and retry, "
            "or contact your admin to increase rate_limit_tps.",
        )
    if code == "scope_denied":
        scope = policy_error.split(":", 1)[1] if ":" in policy_error else default_scope
        return (
            403,
            "scope_denied",
            f"Virtual key does not have the '{scope}' scope. "
            f"Add '{scope}' to the key's scopes list.",
        )
    return 403, "policy_error", policy_error


def resolve_vk_rate_limit_id(vk: VirtualKey | _StubKey) -> str | None:
    """Return the VK id for rate-limit slot tracking, or None."""
    if vk.rate_limit_tps is not None and vk.id:
        return vk.id
    return None


_CACHE_MAX_SIZE = 256


class ProxyAuth:
    """Authenticates proxy requests via virtual API keys."""

    def __init__(self, gate: Gate) -> None:
        """Initialize proxy authenticator.

        Args:
            gate: The Gate singleton (used for store lookups and config).
                An in-memory LRU cache (max 256 entries) avoids repeated
                store reads for the same key hash.
        """
        self._gate = gate
        self._cache: dict[str, VirtualKey] = {}
        self._lock = threading.Lock()

    def authenticate(self, authorization: str) -> VirtualKey | None:
        """Validate a Bearer token and return the VirtualKey, or None.

        Args:
            authorization: The full Authorization header value (e.g. "Bearer ag-xxx").
        """
        token = strip_bearer(authorization)
        if not token:
            return None

        key_hash = hash_key(token)

        # Check in-memory cache first
        with self._lock:
            cached = self._cache.get(key_hash)
            if cached is not None:
                if cached.revoked:
                    logger.debug("VK cache hit but key is revoked: %s", cached.id)
                    return None
                return cached

        # Look up in store
        vk = self._gate.store.get_virtual_key_by_hash(key_hash)
        if vk is None:
            logger.debug("VK not found in store for hash %s...", key_hash[:8])
            return None
        if vk.revoked:
            logger.debug("VK found but revoked: %s", vk.id)
            return None

        # Cache the result.  When the cache exceeds 256 entries, evict the
        # oldest half (insertion-order, since dict preserves order in 3.7+).
        with self._lock:
            if len(self._cache) >= _CACHE_MAX_SIZE:
                keys = list(self._cache.keys())
                for k in keys[: len(keys) // 2]:
                    del self._cache[k]
            self._cache[key_hash] = vk

        return vk

    def invalidate_cache(self, key_hash: str) -> None:
        """Remove a key from the auth cache (e.g. on revocation).

        Args:
            key_hash: SHA-256 hash of the virtual key token.
        """
        with self._lock:
            self._cache.pop(key_hash, None)

    @staticmethod
    def check_scope(vk: VirtualKey | _StubKey | Any, required_scope: str) -> bool:
        """Check if a virtual key has the required scope.

        Returns True if access is allowed. Empty scopes means all allowed
        (backward compatible).
        """
        if not vk.scopes:
            return True
        return required_scope in vk.scopes

    @staticmethod
    def check_model_access(vk: VirtualKey | _StubKey | Any, model: str) -> bool:
        """Check if a virtual key is allowed to access a model.

        Returns True if access is allowed. Empty allowed_models means all allowed.
        Supports glob patterns (e.g. "gpt-*", "claude-*").
        """
        if not vk.allowed_models:
            return True
        return any(fnmatch(model, pattern) for pattern in vk.allowed_models)

    @staticmethod
    def check_budget(vk: VirtualKey | _StubKey | Any, cost: float = 0.0) -> bool:
        """Check if a virtual key has budget remaining.

        Returns True if the key has budget remaining or no budget limit.
        """
        if vk.budget_limit is None:
            return True
        return (vk.budget_spent + cost) <= vk.budget_limit

    def get_provider_keys(self, vk: VirtualKey) -> dict[str, str]:
        """Resolve provider API keys for a virtual key's org.

        Resolution order:
        1. Org-level secrets (stored in the secrets table as "org:{org_id}:provider_key_{provider}")
        2. Global config fallback (provider_api_key_openai, etc.)
        3. Secret vault (air-gapped key resolution)
        4. Environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)
        """
        keys: dict[str, str] = {}
        store = self._gate.store

        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }

        for provider in ("openai", "anthropic", "google"):
            # Try org-level secret
            if vk.org_id:
                org_key = store.get_secret(f"org:{vk.org_id}:provider_key_{provider}")
                if org_key:
                    keys[provider] = org_key
                    logger.debug(
                        "Provider key resolved for %s: org secret (org=%s)",
                        provider,
                        vk.org_id,
                    )
                    continue

            # Fall back to global config
            config_key = getattr(self._gate.config, f"provider_api_key_{provider}", "")
            if config_key:
                keys[provider] = config_key
                logger.debug("Provider key resolved for %s: global config", provider)
                continue

            # Check secret vault (air-gapped key resolution)
            vault = getattr(self._gate, "_secret_vault", None)
            if vault is not None:
                from stateloom.security.vault import SecretVault

                if isinstance(vault, SecretVault):
                    env_key = env_map.get(provider)
                    if env_key:
                        vaulted = vault.retrieve(env_key)
                        if vaulted:
                            keys[provider] = vaulted
                            logger.debug("Provider key resolved for %s: secret vault", provider)
                            continue

            # Fall back to environment variable
            env_key = env_map.get(provider)
            if env_key:
                env_val = os.environ.get(env_key, "")
                if env_val:
                    keys[provider] = env_val
                    logger.debug("Provider key resolved for %s: environment variable", provider)

        return keys
