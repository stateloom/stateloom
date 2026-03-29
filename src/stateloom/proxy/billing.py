"""Billing mode detection for BYOK proxy requests.

Infers whether a user-supplied API key is a standard per-token API key
or a subscription/session token (e.g. Claude Max, ChatGPT Plus).
"""

from __future__ import annotations

# Per-token API key prefixes by provider.
_API_KEY_PREFIXES: dict[str, tuple[str, ...]] = {
    "anthropic": ("sk-ant-api",),
    "openai": ("sk-",),
    "google": ("AIzaSy",),
}

# Subscription/OAuth token prefixes by provider.
# These are tokens from CLI OAuth flows (e.g. Claude Max "sk-ant-oat*").
_SUBSCRIPTION_PREFIXES: dict[str, tuple[str, ...]] = {
    "anthropic": ("sk-ant-oat",),
}


def detect_billing_mode(token: str, provider: str) -> str:
    """Infer billing mode from a BYOK key format.

    Returns ``"api"`` or ``"subscription"``.

    Detection order:
    1. Empty token → ``"api"`` (server key scenario).
    2. Matches subscription prefix (e.g. ``sk-ant-oat*``) → ``"subscription"``.
    3. Matches API key prefix (e.g. ``sk-ant-api*``) → ``"api"``.
    4. Non-matching token for a known provider → ``"subscription"``.
    5. Unknown provider → ``"api"``.
    """
    if not token:
        return "api"

    # Check subscription prefixes first (more specific match)
    for prefix in _SUBSCRIPTION_PREFIXES.get(provider, ()):
        if token.startswith(prefix):
            return "subscription"

    # Check API key prefixes
    for prefix in _API_KEY_PREFIXES.get(provider, ()):
        if token.startswith(prefix):
            return "api"

    # Non-matching token for a known provider → subscription
    if provider in _API_KEY_PREFIXES:
        return "subscription"

    return "api"
