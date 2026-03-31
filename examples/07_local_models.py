"""
StateLoom Local Models — Ollama Integration

Demonstrates:
  - Hardware detection and model recommendations
  - Direct Ollama chat via OllamaClient
  - Auto-routing (simple requests routed to local, complex to cloud)
  - Cost savings from local routing

Prerequisites:

    # Install Ollama: https://ollama.com
    ollama pull llama3.2:3b       # or any model you prefer

Run:

    export OPENAI_API_KEY=sk-...   # cloud provider for routing comparison
    python examples/07_local_models.py

    # or with Anthropic / Gemini as cloud provider
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/07_local_models.py
"""

import os
import stateloom
from stateloom.local import detect_hardware, recommend_models, OllamaClient

# ── Detect provider for cloud calls ──────────────────────────────────

if os.environ.get("OPENAI_API_KEY"):
    CLOUD_MODEL = "gpt-4o-mini"
elif os.environ.get("ANTHROPIC_API_KEY"):
    CLOUD_MODEL = "claude-haiku-4-5-20251001"
elif os.environ.get("GOOGLE_API_KEY"):
    CLOUD_MODEL = "gemini-2.5-flash"
else:
    print("Set at least one cloud API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
    raise SystemExit(1)

print(f"Cloud model: {CLOUD_MODEL}\n")


# ── 1. Hardware detection ────────────────────────────────────────────
# StateLoom detects your hardware and recommends models that fit.

print("=" * 60)
print("1. Hardware detection & model recommendations")
print("=" * 60)

hw = detect_hardware()
print(f"  OS:        {hw.os_name} ({hw.arch})")
print(f"  RAM:       {hw.ram_gb:.1f} GB")
print(f"  CPU cores: {hw.cpu_count}")
print(f"  Disk free: {hw.disk_free_gb:.1f} GB")
if hw.gpu_name:
    print(f"  GPU:       {hw.gpu_name} ({hw.gpu_vram_gb:.1f} GB VRAM)")
print()

recs = recommend_models()
if recs:
    print("  Recommended models for your hardware:")
    for r in recs[:5]:
        print(f"    {r['model']:20s}  {r['size_gb']:.1f} GB  {r['description']}")
else:
    print("  No model recommendations (not enough resources detected)")
print()


# ── 2. Direct Ollama chat ────────────────────────────────────────────
# Use OllamaClient directly for local-only inference — no cloud, no cost.

print("=" * 60)
print("2. Direct Ollama chat (zero cost)")
print("=" * 60)

LOCAL_MODEL = "llama3.2:3b"

client = OllamaClient()
try:
    models = client.list_models()
    available_names = [m.get("name", "") for m in models]
    print(f"  Ollama models installed: {', '.join(available_names[:5])}")

    if any(LOCAL_MODEL in name for name in available_names):
        response = client.chat(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": "What is a hash map? One sentence."}],
        )
        print(f"  Response: {response.content[:120]}")
        print(f"  Tokens: {response.total_tokens} | Latency: {response.latency_ms:.0f}ms | Cost: $0.00")
    else:
        print(f"  Model '{LOCAL_MODEL}' not installed. Run: ollama pull {LOCAL_MODEL}")
except Exception as e:
    print(f"  Ollama not running or not installed: {e}")
    print("  Install from https://ollama.com and run: ollama serve")
finally:
    client.close()

print()


# ── 3. Auto-routing ─────────────────────────────────────────────────
# Simple requests are routed to the local model (free), complex ones
# go to cloud. StateLoom scores complexity automatically.

print("=" * 60)
print("3. Auto-routing (simple→local, complex→cloud)")
print("=" * 60)

try:
    stateloom.init(
        local_model=LOCAL_MODEL,
        auto_route=True,
        budget=2.0,
    )

    SIMPLE_PROMPTS = [
        "What is 2 + 2?",
        "Say hello in French.",
        "What color is the sky?",
    ]

    COMPLEX_PROMPTS = [
        "Design a distributed consensus algorithm that handles Byzantine faults with proof of correctness.",
        "Compare the trade-offs between event sourcing and CQRS in a microservices architecture with eventual consistency requirements.",
    ]

    print("  Simple prompts (may route to local):")
    with stateloom.session("routing-simple", budget=2.0) as s:
        for prompt in SIMPLE_PROMPTS:
            response = stateloom.chat(
                model=CLOUD_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            print(f"    Q: {prompt[:50]}")
            print(f"    A: {response.content[:80]}")
        print(f"  Cost: ${s.total_cost:.6f} | Calls: {s.call_count}")

    print()
    print("  Complex prompts (should route to cloud):")
    with stateloom.session("routing-complex", budget=2.0) as s:
        for prompt in COMPLEX_PROMPTS:
            response = stateloom.chat(
                model=CLOUD_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            print(f"    Q: {prompt[:50]}...")
            print(f"    A: {response.content[:80]}...")
        print(f"  Cost: ${s.total_cost:.6f} | Calls: {s.call_count}")

except Exception as e:
    print(f"  Auto-routing requires Ollama running with {LOCAL_MODEL}: {e}")

print(f"\nDashboard: http://localhost:4782")
