"""
StateLoom Unified Client — Provider-Agnostic Chat

Demonstrates:
  - stateloom.Client: one interface for every provider
  - stateloom.chat(): module-level one-liner convenience
  - Model name → provider auto-detection (gpt-* → OpenAI, claude-* → Anthropic, gemini-* → Gemini)
  - No provider SDK imports needed

Run with any provider key:

    export OPENAI_API_KEY=sk-...
    python examples/unified_client.py

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/unified_client.py

    export GOOGLE_API_KEY=AIza...
    python examples/unified_client.py
"""

import os

import stateloom

stateloom.init(
    pii=True,
    cache_backend="sqlite",  # cache across sessions, by default it uses in memory cache
)

# ── Model → Provider mapping ─────────────────────────────────────────
# StateLoom resolves the provider from the model name prefix:
#   gpt-*, o1-*, o3-*          → OpenAI
#   claude-*                   → Anthropic
#   gemini-*                   → Google Gemini
#   command-*                  → Cohere
#   mistral-*, open-mistral-*  → Mistral

MODELS = {
    "openai": ("gpt-4o-mini", "OPENAI_API_KEY"),
    "anthropic": ("claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
    "gemini": ("gemini-2.5-flash", "GOOGLE_API_KEY"),
}

# Filter to providers with keys set
available = {name: model for name, (model, env_var) in MODELS.items() if os.environ.get(env_var)}

if not available:
    print("No provider API keys found. Set at least one of:")
    for name, (_, env_var) in MODELS.items():
        print(f"  export {env_var}=...")
    raise SystemExit(1)

print(f"Providers available: {', '.join(available)}\n")

PROMPT = [{"role": "user", "content": "What is the CAP theorem? One sentence."}]

# ── Option 1: stateloom.Client (explicit session) ────────────────────
# Client dispatches to the correct provider based on the model name.
# No need to import openai, anthropic, or google.generativeai.

print("=" * 60)
print("stateloom.Client — explicit session")
print("=" * 60)

with stateloom.session("unified-client-demo", budget=1.0) as s:
    for provider_name, model in available.items():
        print(f"\n── {provider_name} ({model}) ──")
        response = stateloom.Client().chat(
            model=model,
            messages=PROMPT,
        )
        print(f"   {response.content[:120]}")
        print(f"   Provider: {response.provider} | Cost: ${response._stateloom.get('cost', 0):.4f}")

    print(f"\n  Total: ${s.total_cost:.4f} | {s.total_tokens} tokens | {s.call_count} calls")

# ── Option 2: stateloom.chat() (one-liner, auto-session) ─────────────
# Module-level function — creates a throwaway session per call.
# Ideal for scripts or one-off queries.

print("\n" + "=" * 60)
print("stateloom.chat() — one-liner convenience")
print("=" * 60)

for provider_name, model in available.items():
    print(f"\n── {provider_name} ({model}) ──")
    response = stateloom.chat(
        model=model,
        messages=[{"role": "user", "content": "What is eventual consistency? One sentence."}],
    )
    print(f"   {response.content[:120]}")

# ── Option 3: async client ───────────────────────────────────────────

import asyncio


async def main():
    async with stateloom.async_session("async-demo-1", budget=1.0) as s:
        for provider_name, model in available.items():
            response = await stateloom.achat(
                model=model,
                messages=[{"role": "user", "content": "What is CRDT? One sentence."}],
            )
            print(f"{provider_name}: {response.content[:100]}")
        print(f"Total: ${s.total_cost:.4f}")


asyncio.run(main())

print("\nDashboard: http://localhost:4782")
