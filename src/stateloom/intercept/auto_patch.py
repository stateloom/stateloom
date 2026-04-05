"""Auto-detect and patch installed LLM clients."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from stateloom.core.types import Provider

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.intercept")


def auto_patch(gate: Gate) -> list[dict[str, Any]]:
    """Detect installed LLM clients and patch them.

    Returns a list of dicts describing what was patched.
    """
    from stateloom.intercept.generic_interceptor import patch_provider
    from stateloom.intercept.provider_registry import get_all_adapters, register_builtin_adapters

    register_builtin_adapters()

    patched: list[dict[str, Any]] = []

    for name, adapter in get_all_adapters().items():
        results = patch_provider(gate, adapter)
        for desc in results:
            patched.append({"provider": name, "description": desc})

    # google-genai uses per-client classes — patch separately to avoid
    # overwriting the legacy google-generativeai GeminiAdapter.
    try:
        from stateloom.intercept.adapters.genai_adapter import patch_genai

        results = patch_genai(gate)
        for desc in results:
            patched.append({"provider": Provider.GEMINI, "description": desc})
    except Exception:
        pass  # google-genai not installed or patch failed

    # Promote GeminiGenaiAdapter in the registry for Client/proxy paths.
    # SDK interceptors already captured their adapter in closures above,
    # so this only affects get_adapter("gemini") callers (Client._prepare_call,
    # response_format, etc.) — giving them full tool/function-calling support
    # via the new google-genai SDK.
    from stateloom.intercept.adapters.gemini_genai_adapter import GeminiGenaiAdapter
    from stateloom.intercept.provider_registry import register_adapter

    register_adapter(GeminiGenaiAdapter())

    # LiteLLM uses module-level functions — patch separately
    try:
        from stateloom.intercept.adapters.litellm_adapter import patch_litellm

        results = patch_litellm(gate)
        for desc in results:
            patched.append({"provider": Provider.LITELLM, "description": desc})
    except Exception:
        pass  # litellm not installed or patch failed

    if patched:
        providers = set(str(p["provider"]) for p in patched)
        logger.info(
            f"[StateLoom] Auto-patched {len(patched)} methods across {', '.join(sorted(providers))}"
        )
    else:
        logger.warning(
            "[StateLoom] No LLM clients found to patch. "
            "Install openai, anthropic, google-generativeai, google-genai, mistralai, "
            "cohere, or litellm, "
            "or use gate.wrap() or stateloom.register_provider() for custom providers."
        )

    return patched


def wrap_client(gate: Gate, client: Any) -> None:
    """Wrap a specific client instance (explicit, no monkey-patching).

    This patches the client's methods directly on the instance,
    rather than on the class. Iterates registered adapters to find a match.
    """
    from stateloom.intercept.generic_interceptor import wrap_instance
    from stateloom.intercept.provider_registry import get_all_adapters, register_builtin_adapters

    register_builtin_adapters()

    client_module = type(client).__module__

    for _name, adapter in get_all_adapters().items():
        try:
            targets = adapter.get_instance_targets(client)
        except (AttributeError, TypeError):
            continue
        if targets:
            wrap_instance(gate, adapter, client)
            return

    # Fall back to module name heuristic for built-in providers
    from stateloom.intercept.provider_registry import get_adapter

    for keyword, adapter_name in [
        ("openai", Provider.OPENAI),
        ("anthropic", Provider.ANTHROPIC),
        ("google", Provider.GEMINI),
        ("genai", Provider.GEMINI),
        ("mistralai", Provider.MISTRAL),
        ("cohere", Provider.COHERE),
    ]:
        if keyword in client_module:
            found_adapter = get_adapter(adapter_name)
            if found_adapter:
                wrap_instance(gate, found_adapter, client)
                return

    logger.warning(
        f"[StateLoom] Unknown client type: {type(client)}. "
        "Use stateloom.register_provider() to add support for this provider, "
        "or use one of the built-in providers: openai, anthropic, gemini."
    )
