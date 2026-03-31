"""
StateLoom Sessions & Budget Enforcement

Demonstrates:
  - Session context manager with named sessions
  - Per-session budget limits (hard stop vs warning)
  - Budget remaining tracking mid-session
  - Parent-child session hierarchy
  - StateLoomBudgetError when budget is exceeded

Run:

    export OPENAI_API_KEY=sk-...
    python examples/03_sessions_and_budget.py

    # or with Anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/03_sessions_and_budget.py
"""

import os
import stateloom

# ── Init ──────────────────────────────────────────────────────────────

stateloom.init(
    budget=5.0,  # default per-session budget when none specified
)

# Pick whichever provider is available
if os.environ.get("OPENAI_API_KEY"):
    MODEL = "gpt-4o-mini"
elif os.environ.get("ANTHROPIC_API_KEY"):
    MODEL = "claude-haiku-4-5-20251001"
elif os.environ.get("GOOGLE_API_KEY"):
    MODEL = "gemini-2.5-flash"
else:
    print("Set at least one API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
    raise SystemExit(1)

print(f"Using model: {MODEL}\n")


# ── 1. Basic session with budget ─────────────────────────────────────
# Everything inside the block is tracked: cost, tokens, cache hits.

print("=" * 60)
print("1. Basic session with $1.00 budget")
print("=" * 60)

with stateloom.session("basic-session", budget=1.0) as s:
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is a load balancer? One sentence."}],
    )
    print(f"  Response: {response.content[:100]}")
    print(f"  Cost so far: ${s.total_cost:.4f}")
    print(f"  Budget remaining: ${s.budget - s.total_cost:.4f}")

    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is a reverse proxy? One sentence."}],
    )
    print(f"  Response: {response.content[:100]}")
    print(f"  Cost so far: ${s.total_cost:.4f}")
    print(f"  Calls: {s.call_count} | Tokens: {s.total_tokens}")

print(f"  Session ended. Final cost: ${s.total_cost:.4f}\n")


# ── 2. Default budget from init() ────────────────────────────────────
# Sessions without explicit budget inherit the $5.00 default from init().

print("=" * 60)
print("2. Session with default budget from init(budget=5.0)")
print("=" * 60)

with stateloom.session("default-budget-session") as s:
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is DNS? One sentence."}],
    )
    print(f"  Response: {response.content[:100]}")
    print(f"  Budget: ${s.budget:.2f} (inherited from init)")
    print(f"  Cost: ${s.total_cost:.4f}\n")


# ── 3. Parent-child sessions ─────────────────────────────────────────
# Child sessions inherit org/team from parent. Budgets are independent.

print("=" * 60)
print("3. Parent-child session hierarchy")
print("=" * 60)

with stateloom.session("parent-session", budget=2.0) as parent:
    response = stateloom.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is microservices architecture? One sentence."}],
    )
    print(f"  Parent: {response.content[:80]}")

    # Child session — auto-links to parent
    with stateloom.session("child-task-1", budget=0.50) as child:
        response = stateloom.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "What is service mesh? One sentence."}],
        )
        print(f"  Child:  {response.content[:80]}")
        print(f"  Child cost: ${child.total_cost:.4f} (own budget: $0.50)")

    print(f"  Parent cost: ${parent.total_cost:.4f} (does NOT include child)")
    print(f"  Parent calls: {parent.call_count} | Child calls: {child.call_count}\n")


# ── 4. Budget enforcement (hard stop) ────────────────────────────────
# When the budget is exceeded, StateLoomBudgetError is raised.

print("=" * 60)
print("4. Budget enforcement — $0.00001 budget (will trigger)")
print("=" * 60)

try:
    with stateloom.session("tight-budget", budget=0.00001) as s:
        # First call may succeed if it costs less than $0.00001
        for i in range(5):
            response = stateloom.chat(
                model=MODEL,
                messages=[{"role": "user", "content": f"Count again to {i + 1} again and stop."}],
            )
            print(f"  Call {i + 1}: ${s.total_cost:.4f} spent")
except stateloom.StateLoomBudgetError as e:
    print(f"  Budget exceeded! {e}")
    print(f"  Session was stopped at ${s.total_cost:.4f} / ${s.budget:.4f}")


# ── 5. Cost tracking by model ────────────────────────────────────────

print("\n" + "=" * 60)
print("5. Per-model cost breakdown")
print("=" * 60)

with stateloom.session("multi-call-session", budget=1.0) as s:
    prompts = [
        "What is TCP? One sentence.",
        "What is UDP? One sentence.",
        "What is HTTP/2? One sentence.",
    ]
    for prompt in prompts:
        stateloom.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])

    print(f"  Total: ${s.total_cost:.4f} | {s.total_tokens} tokens | {s.call_count} calls")
    if s.cost_by_model:
        for model_name, cost in s.cost_by_model.items():
            tokens = s.tokens_by_model.get(model_name, {})
            print(f"  {model_name}: ${cost:.4f} ({tokens.get('total', 0)} tokens)")

print(f"\nDashboard: http://localhost:4782")
