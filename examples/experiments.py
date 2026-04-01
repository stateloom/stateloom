"""
StateLoom A/B Experiments — Model & Prompt Testing in Production

Demonstrates a realistic scenario: an e-commerce company is testing
different LLM configurations for their customer support chatbot.

  Part A — Setting Up Experiments
    - Create an experiment with prompt variants (same model, different personas)
    - Use hash-based assignment for deterministic user bucketing
    - Run sessions through the experiment pipeline
    - Mix stateloom.chat() and native SDK calls in the same session

  Part B — Collecting Feedback & Metrics
    - Record success/failure/partial feedback per session
    - View per-variant metrics (cost, latency, success rate)
    - Compare variants on the leaderboard
    - Conclude the experiment and pick a winner

  Part C — Advanced: Temperature Tuning Experiment
    - Same model, different temperature settings
    - Weighted random assignment (80/20 split)
    - Evaluate output quality via scoring

Requires at least ONE provider API key:

    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...
    export GOOGLE_API_KEY=AIza...

    python examples/experiments.py
"""

import json
import os
import random
import time

import stateloom

# Unique run prefix so session IDs don't collide across reruns
# (the experiment assigner returns cached assignments for existing session IDs)
_run = f"{int(time.time()) % 100000}"

# ── Init ──────────────────────────────────────────────────────────────

stateloom.init(budget=10.0, console_output=True)

# ── Detect available providers and pick models ────────────────────────

models = []

if os.environ.get("ANTHROPIC_API_KEY"):
    models.append("claude-haiku-4-5-20251001")

if os.environ.get("OPENAI_API_KEY"):
    models.append("gpt-4o-mini")

if os.environ.get("GOOGLE_API_KEY"):
    models.append("gemini-2.5-flash")

if len(models) < 1:
    print("Need at least 1 provider API key:")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    print("  export OPENAI_API_KEY=sk-...")
    print("  export GOOGLE_API_KEY=AIza...")
    raise SystemExit(1)

print(f"Available models: {', '.join(models)}\n")

# ── Helper: simulate a customer support interaction ───────────────────

CUSTOMER_QUERIES = [
    "I ordered a laptop 3 days ago and tracking still shows 'processing'. Can you help?",
    "I want to return a pair of shoes I bought last week. They don't fit.",
    "My discount code SAVE20 isn't working at checkout. It says 'invalid'.",
    "Can you tell me when the new iPhone cases will be back in stock?",
    "I was charged twice for my last order #ORD-9182. Please fix this.",
    "How do I change the shipping address on an order that hasn't shipped yet?",
]


def run_support_session(session_id: str, query: str, experiment_id: str) -> dict:
    """Simulate a customer support interaction and return quality metrics."""
    with stateloom.session(session_id, budget=2.0, experiment=experiment_id) as s:
        # Step 1: Classify the customer intent using stateloom.chat()
        # We pass model= as a baseline — the experiment middleware will
        # override it with the variant's model during pipeline execution.
        classify = stateloom.chat(
            model=models[0],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the customer query into one category: "
                        "order_status, return, billing, product_info, shipping. "
                        "Respond with just the category name."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        intent = classify.content.strip().lower()

        # Step 2: Generate the support response (this call gets experiment
        # variant overrides applied by the middleware pipeline)
        response = stateloom.chat(
            model=models[0],
            messages=[
                {"role": "user", "content": query},
            ],
        )

        # Step 3: Quality evaluation using stateloom.chat() — the auto-patch
        # ensures cost tracking and experiment overrides are applied.
        eval_prompt = (
            f"Customer query: {query}\n\n"
            f"Agent response: {response.content[:500]}\n\n"
            "Rate this support response 1-5. Respond with JSON only: "
            '{"score": N, "reason": "..."}'
        )

        eval_response = stateloom.chat(
            model=models[0],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer support quality reviewer. "
                        "Rate responses on helpfulness, empathy, and actionability."
                    ),
                },
                {"role": "user", "content": eval_prompt},
            ],
            max_tokens=100,
        )
        quality_text = eval_response.content

        # Parse the quality score
        try:
            score_text = quality_text.strip()
            # Handle markdown code blocks
            if "```" in score_text:
                score_text = score_text.split("```")[1]
                if score_text.startswith("json"):
                    score_text = score_text[4:]
            parsed = json.loads(score_text)
            score = float(parsed.get("score", 3))
        except (json.JSONDecodeError, ValueError, IndexError):
            score = 3.0

    return {
        "session_id": session_id,
        "intent": intent,
        "response_preview": response.content[:100],
        "score": score,
        "cost": s.total_cost,
        "tokens": s.total_tokens,
        "calls": s.call_count,
    }


# =====================================================================
# PART A — Setting Up a Model Comparison Experiment
# =====================================================================

print("=" * 60)
print("A1. Create a model comparison experiment")
print("=" * 60)

# We're testing which system prompt produces better support quality.
# Both variants use the same model — the experiment isolates the prompt
# as the single variable. (Cross-provider model comparison is what
# consensus is for — see example 11.)

variants = [
    {
        "name": "concise",
        "weight": 1.0,
        "model": models[0],
        "request_overrides": {
            "system_prompt": (
                "You are a helpful customer support agent for ShopMax, "
                "an online retailer. Be concise, empathetic, and action-oriented. "
                "Always offer a concrete next step. Keep responses under 2 sentences."
            ),
            "temperature": 0.3,
            "max_tokens": 256,
        },
    },
    {
        "name": "empathetic",
        "weight": 1.0,
        "model": models[0],
        "request_overrides": {
            "system_prompt": (
                "You are a senior customer support specialist at ShopMax. "
                "You're known for turning frustrated customers into loyal fans. "
                "Always acknowledge the customer's feelings first, then offer "
                "a solution with a personal touch. Use the customer's situation "
                "to show you understand."
            ),
            "temperature": 0.5,
            "max_tokens": 300,
        },
    },
]

experiment = stateloom.create_experiment(
    name="Support Bot Model Comparison",
    variants=variants,
    strategy="hash",  # Deterministic assignment based on session ID
    description="Testing model and prompt variants for customer support quality",
    metadata={"team": "customer-experience", "sprint": "2026-Q1"},
)

print(f"  Experiment:  {experiment.name}")
print(f"  ID:          {experiment.id}")
print(f"  Strategy:    {experiment.strategy.value}")
print(f"  Variants:    {[v['name'] for v in variants]}  (same model, different prompts)")
print(f"  Status:      {experiment.status.value}")
print()


# ── Start the experiment ──────────────────────────────────────────────

print("=" * 60)
print("A2. Start experiment and run sessions")
print("=" * 60)

experiment = stateloom.start_experiment(experiment.id)
print(f"  Status: {experiment.status.value}")
print()

# Run sessions through the experiment. Each session is assigned a variant
# by the ExperimentMiddleware based on the hash strategy.

results = []
for i, query in enumerate(CUSTOMER_QUERIES):
    session_id = f"support-{_run}-{i + 1:03d}"
    print(f"  Ticket {i + 1}: {query[:60]}...")

    result = run_support_session(session_id, query, experiment.id)
    results.append(result)

    # Record feedback based on the quality score
    rating = (
        "success" if result["score"] >= 4 else ("partial" if result["score"] >= 3 else "failure")
    )
    stateloom.feedback(
        session_id=session_id,
        rating=rating,
        score=result["score"] / 5.0,  # Normalize to 0-1
        comment=f"Auto-evaluated: intent={result['intent']}, score={result['score']}/5",
    )

    print(f"    Intent: {result['intent']}")
    print(f"    Score: {result['score']}/5 → {rating}")
    print(f"    Cost: ${result['cost']:.6f} | {result['tokens']} tokens")
    print()


# =====================================================================
# PART B — Analyzing Results
# =====================================================================

print("=" * 60)
print("B1. Per-variant metrics")
print("=" * 60)

metrics = stateloom.experiment_metrics(experiment.id)

print(f"  Experiment: {metrics.get('experiment_name', experiment.name)}")
print(f"  Total sessions: {metrics.get('total_sessions', len(results))}")
print()

for variant_name, stats in metrics.get("variants", {}).items():
    print(f"  Variant: {variant_name}")
    print(f"    Sessions:     {stats.get('count', 0)}")
    print(f"    Avg cost:     ${stats.get('avg_cost', 0):.6f}")
    print(f"    Avg tokens:   {stats.get('avg_tokens', 0):.0f}")
    print(f"    Success rate: {stats.get('success_rate', 0):.1%}")
    print(f"    Avg score:    {stats.get('avg_score', 0):.2f}")
    print()


# ── Leaderboard ───────────────────────────────────────────────────────

print("=" * 60)
print("B2. Cross-experiment leaderboard")
print("=" * 60)

board = stateloom.leaderboard()
for i, entry in enumerate(board, 1):
    print(
        f"  #{i}  {entry.get('variant_name', '?'):12s}  "
        f"success={entry.get('success_rate', 0):.1%}  "
        f"avg_cost=${entry.get('avg_cost', 0):.6f}  "
        f"({entry.get('experiment_name', '')})"
    )

print()


# ── Conclude ──────────────────────────────────────────────────────────

print("=" * 60)
print("B3. Conclude experiment")
print("=" * 60)

final = stateloom.conclude_experiment(experiment.id)
print(f"  Status: {experiment.status.value}")
print(f"  Final metrics returned: {list(final.keys())}")
print()


# =====================================================================
# PART C — Temperature Tuning Experiment (Weighted Random)
# =====================================================================

print("=" * 60)
print("C1. Temperature tuning experiment (80/20 split)")
print("=" * 60)

# Same model, different temperatures — which produces more consistent output?

temp_experiment = stateloom.create_experiment(
    name="Temperature Tuning - Support Tone",
    variants=[
        {
            "name": "conservative",
            "weight": 4.0,  # 80% traffic
            "model": models[0],
            "request_overrides": {
                "system_prompt": ("You are a customer support agent. Be helpful and professional."),
                "temperature": 0.2,
                "max_tokens": 200,
            },
        },
        {
            "name": "creative",
            "weight": 1.0,  # 20% traffic
            "model": models[0],
            "request_overrides": {
                "system_prompt": ("You are a customer support agent. Be helpful and professional."),
                "temperature": 0.9,
                "max_tokens": 200,
            },
        },
    ],
    strategy="random",
    description="Testing temperature impact on response consistency",
)

temp_experiment = stateloom.start_experiment(temp_experiment.id)
print(f"  Experiment: {temp_experiment.name}")
print(f"  ID: {temp_experiment.id}")
print("  Strategy: random (80/20 weighted)")
print()

# Run a batch of sessions
print("=" * 60)
print("C2. Run sessions with temperature variants")
print("=" * 60)

temp_results = {"conservative": [], "creative": []}

for i in range(6):
    query = random.choice(CUSTOMER_QUERIES)
    sid = f"temp-{_run}-{i + 1:03d}"

    with stateloom.session(sid, budget=1.0, experiment=temp_experiment.id) as s:
        r = stateloom.chat(
            model=models[0],
            messages=[{"role": "user", "content": query}],
        )

        # Check which variant was assigned
        variant_name = s.metadata.get("variant", "unknown")
        temp_results.setdefault(variant_name, [])

        # Score: shorter, more focused responses score higher for support
        word_count = len(r.content.split())
        # Support sweet spot: 30-80 words
        if 30 <= word_count <= 80:
            score = 5.0
        elif 20 <= word_count <= 100:
            score = 4.0
        elif word_count <= 120:
            score = 3.0
        else:
            score = 2.0

        temp_results[variant_name].append(score)

    rating = "success" if score >= 4 else ("partial" if score >= 3 else "failure")
    stateloom.feedback(sid, rating, score=score / 5.0)

    print(f"  Session {sid}: variant={variant_name}, words={word_count}, score={score}/5")

print()


# ── Temperature experiment results ────────────────────────────────────

print("=" * 60)
print("C3. Temperature experiment results")
print("=" * 60)

for variant_name, scores in temp_results.items():
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  {variant_name}: {len(scores)} sessions, avg score={avg:.1f}/5")

temp_metrics = stateloom.experiment_metrics(temp_experiment.id)
for vname, stats in temp_metrics.get("variants", {}).items():
    rate = stats.get("success_rate", 0)
    avg = stats.get("avg_cost", 0)
    print(f"  {vname}: success_rate={rate:.1%}, avg_cost=${avg:.6f}")

stateloom.conclude_experiment(temp_experiment.id)
print(f"\n  Experiment concluded: {temp_experiment.status.value}")
print()


# ── Final leaderboard (both experiments) ──────────────────────────────

print("=" * 60)
print("C4. Final leaderboard (all experiments)")
print("=" * 60)

board = stateloom.leaderboard()
for i, entry in enumerate(board, 1):
    print(
        f"  #{i}  {entry.get('variant_name', '?'):15s}  "
        f"success={entry.get('success_rate', 0):.1%}  "
        f"avg_cost=${entry.get('avg_cost', 0):.6f}  "
        f"({entry.get('experiment_name', '')})"
    )

print()
print("Dashboard: http://localhost:4782")
