"""
StateLoom Agents & Multi-Agent Consensus

Demonstrates:
  Part A — Managed Agents (Prompts-as-an-API)
    - Create agents with model + system prompt + request overrides
    - Version an agent (immutable snapshots, auto-increment)
    - Rollback to a previous version
    - Use agents via stateloom.chat(agent=...) and native SDK
    - Per-agent session tracking and cost attribution

  Part B — Multi-Agent Consensus (Cross-Provider Debate)
    - Vote strategy: fast parallel poll across models
    - Debate strategy: multi-round deliberation with judge synthesis
    - Self-consistency: multiple samples from one model
    - Greedy downgrade: auto-switch to cheaper models after easy consensus
    - Agent-guided consensus: debate using an agent's system prompt
    - Persona-driven consensus: named debaters with custom prompts and visibility
    - Mixed stateloom.chat() + consensus in one session

Requires API keys for at least TWO providers:

    export ANTHROPIC_API_KEY=sk-ant-...
    export GOOGLE_API_KEY=AIza...
    export OPENAI_API_KEY=sk-...       # optional third model

    python examples/agents_and_consensus.py
"""

import os

import stateloom

# ── Init ──────────────────────────────────────────────────────────────

stateloom.init(budget=10.0, console_output=True)

# ── Detect available providers and pick models ────────────────────────

models = []

if os.environ.get("ANTHROPIC_API_KEY"):
    models.append("claude-haiku-4-5-20251001")

if os.environ.get("GOOGLE_API_KEY"):
    models.append("gemini-2.5-flash")

if os.environ.get("OPENAI_API_KEY"):
    models.append("gpt-4o-mini")

if len(models) < 2:
    print("Need at least 2 provider API keys:")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    print("  export GOOGLE_API_KEY=AIza...")
    print("  export OPENAI_API_KEY=sk-...")
    raise SystemExit(1)

print(f"Models: {', '.join(models)}\n")


# =====================================================================
# PART A — Managed Agents (Prompts-as-an-API)
# =====================================================================


# ── 1. Create an agent ────────────────────────────────────────────────
# An agent bundles model + system prompt + overrides into a reusable
# definition. No code needed to deploy — just create and call by slug.

print("=" * 60)
print("A1. Create a managed agent")
print("=" * 60)

agent = stateloom.create_agent(
    slug="tech-advisor",
    name="Tech Advisor",
    description="Senior staff engineer that gives opinionated architecture advice",
    model=models[0],
    system_prompt=(
        "You are a senior staff engineer with 20 years of experience. "
        "Give direct, opinionated technical advice. Always state trade-offs. "
        "Keep answers under 3 sentences."
    ),
    request_overrides={"temperature": 0.3, "max_tokens": 256},
    budget_per_session=1.0,
)

v1_id = agent.active_version_id

print(f"  Agent ID:    {agent.id}")
print(f"  Slug:        {agent.slug}")
print(f"  Version:     {v1_id}")
print(f"  Model:       {models[0]}")
print()


# ── 2. Use the agent via stateloom.chat() ─────────────────────────────
# Just pass agent="slug" — the agent's model, system prompt, and
# overrides are applied automatically.

print("=" * 60)
print("A2. Chat with the agent via stateloom.chat(agent=...)")
print("=" * 60)

with stateloom.session("agent-chat-demo", budget=2.0) as s:
    r1 = stateloom.chat(
        agent="tech-advisor",
        messages=[
            {"role": "user", "content": "Should I use Redis or Memcached for session storage?"}
        ],
    )
    print("  Q: Should I use Redis or Memcached for session storage?")
    print(f"  A: {r1.content[:150]}")
    print()

    r2 = stateloom.chat(
        agent="tech-advisor",
        messages=[
            {
                "role": "user",
                "content": "Kafka vs RabbitMQ for an event-driven microservices system?",
            }
        ],
    )
    print("  Q: Kafka vs RabbitMQ for event-driven microservices?")
    print(f"  A: {r2.content[:150]}")

print(f"\n  Session: ${s.total_cost:.6f} | {s.call_count} calls | {s.total_tokens} tokens")
print()


# ── 3. Version the agent — new prompt, different model ────────────────
# Versions are immutable snapshots. "Updating" an agent creates a new
# version. The old version remains available for rollback.

print("=" * 60)
print("A3. Create agent version 2 (new model + refined prompt)")
print("=" * 60)

v2 = stateloom.create_agent_version(
    agent.id,
    model=models[1],
    system_prompt=(
        "You are a principal architect specializing in distributed systems. "
        "Give precise, actionable recommendations backed by real-world "
        "experience. Include a concrete next step. Under 3 sentences."
    ),
    request_overrides={"temperature": 0.2, "max_tokens": 300},
    created_by="examples/11",
)

# Activate v2 — all subsequent calls use the new version
stateloom.activate_agent_version(agent.id, v2.id)

print(f"  Version 2 ID:    {v2.id}")
print(f"  Version number:  v{v2.version_number}")
print(f"  New model:       {models[1]}")
print()

# Use the updated agent — same slug, new behavior
r3 = stateloom.chat(
    agent="tech-advisor",
    messages=[{"role": "user", "content": "Monorepo or polyrepo for a 10-person startup?"}],
)
print("  Q: Monorepo or polyrepo for a 10-person startup?")
print(f"  A (v2): {r3.content[:150]}")
print()


# ── 4. Rollback to version 1 ─────────────────────────────────────────
# Instant rollback — just point the agent back to the old version.

print("=" * 60)
print("A4. Rollback agent to version 1")
print("=" * 60)

# List all versions and find v1
versions = stateloom.list_agent_versions(agent.id)
print(f"  All versions: {[f'v{v.version_number} ({v.model})' for v in versions]}")

v1 = next(v for v in versions if v.version_number == 1)
stateloom.activate_agent_version(agent.id, v1.id)
print(f"  Rolled back to: {v1.id} (v1, model={v1.model})")

r4 = stateloom.chat(
    agent="tech-advisor",
    messages=[{"role": "user", "content": "gRPC or REST for internal service communication?"}],
)
print("  Q: gRPC or REST for internal services?")
print(f"  A (v1): {r4.content[:150]}")

# Re-activate v2 for subsequent sections
stateloom.activate_agent_version(agent.id, v2.id)

print()


# ── 5. Create a second agent ─────────────────────────────────────────
# Multiple agents can coexist. Each has its own persona and model.

print("=" * 60)
print("A5. Create a second agent (code-reviewer)")
print("=" * 60)

reviewer = stateloom.create_agent(
    slug="code-reviewer",
    name="Code Reviewer",
    description="Senior code reviewer focused on correctness and maintainability",
    model=models[-1],
    system_prompt=(
        "You are a meticulous code reviewer. Focus on correctness, "
        "edge cases, and maintainability. Be constructive but direct. "
        "Always suggest a specific improvement."
    ),
    request_overrides={"temperature": 0.1},
)

print(f"  Agent: {reviewer.slug} ({reviewer.id})")
print(f"  Model: {models[-1]}")

r5 = stateloom.chat(
    agent="code-reviewer",
    messages=[
        {
            "role": "user",
            "content": "Review this: `def div(a, b): return a / b`",
        }
    ],
)
print(f"  Review: {r5.content[:150]}")
print()


# =====================================================================
# PART B — Multi-Agent Consensus (Cross-Provider Debate)
# =====================================================================


def print_result(result):
    """Pretty-print a ConsensusResult."""
    print(f"  Strategy:     {result.strategy}")
    print(f"  Models:       {', '.join(result.models)}")
    print(f"  Answer:       {result.answer[:140]}...")
    print(f"  Confidence:   {result.confidence:.2f}")
    print(f"  Winner:       {result.winner_model}")
    print(f"  Aggregation:  {result.aggregation_method}")
    print(f"  Rounds:       {result.total_rounds}  (early stop: {result.early_stopped})")
    print(f"  Cost:         ${result.cost:.6f}  ({result.duration_ms:.0f}ms)")

    for rnd in result.rounds:
        print(f"\n  Round {rnd.round_number}  (agreement: {rnd.agreement_score:.2f})")
        for resp in rnd.responses:
            print(f"    {resp.model}: [{resp.confidence:.2f}] {resp.content[:80]}...")


# ── 6. Vote strategy — fast parallel poll ─────────────────────────────

print("=" * 60)
print("B1. Vote Strategy — parallel one-round poll")
print("=" * 60)

result = stateloom.consensus_sync(
    prompt="Is Rust or Go a better choice for a high-throughput web API? "
    "Give a clear recommendation with one key reason.",
    models=models,
    strategy="vote",
    budget=1.0,
)

print_result(result)
print()


# ── 7. Debate strategy — multi-round deliberation ────────────────────

print("=" * 60)
print("B2. Debate Strategy — 2-round deliberation")
print("=" * 60)

result = stateloom.consensus_sync(
    prompt="A startup is choosing between PostgreSQL and DynamoDB for their "
    "e-commerce platform that handles flash sales (100x traffic spikes). "
    "Small team, strong SQL experience. What should they pick and why?",
    models=models,
    strategy="debate",
    rounds=2,
    budget=2.0,
    aggregation="confidence_weighted",
)

print_result(result)
print()


# ── 8. Self-consistency — multiple samples from one model ─────────────

print("=" * 60)
print("B3. Self-Consistency — 3 samples, majority vote")
print("=" * 60)

result = stateloom.consensus_sync(
    prompt="A farmer has 17 sheep. All but 9 run away. How many sheep "
    "does the farmer have left? Think step by step.",
    models=[models[0]],
    strategy="self_consistency",
    samples=3,
    temperature=0.8,
    budget=1.0,
)

print_result(result)
print()


# ── 9. Greedy debate — auto-downgrade after easy consensus ────────────

print("=" * 60)
print("B4. Greedy Debate — auto-downgrade on easy consensus")
print("=" * 60)

result = stateloom.consensus_sync(
    prompt="What is the time complexity of binary search? "
    "State the answer and briefly explain why.",
    models=models,
    strategy="debate",
    rounds=2,
    greedy=True,
    greedy_agreement_threshold=0.6,
    budget=1.0,
)

print_result(result)
print()


# ── 10. Agent-guided consensus ────────────────────────────────────────
# Pass agent="slug" to consensus — all debater models use the agent's
# system prompt, so the debate is framed by the agent's persona.

print("=" * 60)
print("B5. Agent-Guided Consensus — debate with agent persona")
print("=" * 60)

result = stateloom.consensus_sync(
    prompt="Our team is debating whether to adopt GraphQL or stick with REST "
    "for our public API. We have 50+ third-party integrations. "
    "What's your recommendation?",
    models=models,
    strategy="debate",
    rounds=2,
    budget=2.0,
    agent="tech-advisor",
)

print("  Agent persona: tech-advisor (v2)")
print_result(result)
print()


# ── 11. Persona-driven consensus ──────────────────────────────────
# Instead of flat model lists, define named debaters with their own
# system prompts and visibility rules. Each persona has a name, model,
# system_prompt, and optional 'sees' list (who they can read).

print("=" * 60)
print("B6. Persona-Driven Consensus — named debaters")
print("=" * 60)

result = stateloom.consensus_sync(
    prompt="A fintech startup needs to choose a primary programming language. "
    "They need high performance, strong type safety, and good hiring pool. "
    "What should they pick?",
    personas=[
        {
            "name": "CTO",
            "model": models[0],
            "system_prompt": (
                "You are a startup CTO focused on shipping speed and hiring. "
                "Prioritize developer productivity and ecosystem maturity."
            ),
        },
        {
            "name": "Security Lead",
            "model": models[1],
            "system_prompt": (
                "You are a security engineer focused on memory safety, "
                "type safety, and secure-by-default practices."
            ),
        },
        *(
            [
                {
                    "name": "Performance Engineer",
                    "model": models[2],
                    "system_prompt": (
                        "You are a performance engineer focused on latency, "
                        "throughput, and runtime efficiency."
                    ),
                }
            ]
            if len(models) >= 3
            else []
        ),
    ],
    strategy="debate",
    rounds=2,
    budget=2.0,
)

print_result(result)
if result.winner_persona:
    print(f"  Winner persona: {result.winner_persona}")
print()


# ── 12. Persona visibility rules ─────────────────────────────────
# The 'sees' field controls which other personas a debater can read
# in subsequent rounds. None = sees all (default).

print("=" * 60)
print("B7. Persona Visibility — selective information flow")
print("=" * 60)

result = stateloom.consensus_sync(
    prompt="Should we open-source our core product?",
    personas=[
        {
            "name": "CEO",
            "model": models[0],
            "system_prompt": "You are a CEO focused on business strategy and revenue.",
            "sees": ["CTO"],  # CEO only sees CTO's arguments
        },
        {
            "name": "CTO",
            "model": models[1],
            "system_prompt": "You are a CTO focused on engineering culture and talent.",
            "sees": ["CEO", *(["Community Lead"] if len(models) >= 3 else [])],
        },
        *(
            [
                {
                    "name": "Community Lead",
                    "model": models[2],
                    "system_prompt": (
                        "You are a developer relations lead focused on community "
                        "growth, adoption, and contributor experience."
                    ),
                    # sees=None → sees all others (default)
                }
            ]
            if len(models) >= 3
            else []
        ),
    ],
    strategy="debate",
    rounds=2,
    budget=2.0,
)

print_result(result)
print("  (Each persona only saw arguments from personas in their 'sees' list)")
print()


# ── 13. Mixed session — agents + consensus + stateloom.chat() ─────────
# Combine everything in one tracked session: agent calls for routine
# work, consensus for the critical decision, regular chat for polish.

print("=" * 60)
print("B8. Mixed Session — agents + consensus + stateloom.chat()")
print("=" * 60)

with stateloom.session("full-pipeline-demo", budget=5.0) as s:
    # Step 1: Agent gathers context
    print("  Step 1: Tech advisor gathers context...")
    context = stateloom.chat(
        agent="tech-advisor",
        messages=[
            {
                "role": "user",
                "content": "List 3 key factors when choosing between microservices "
                "and monolith. One line each, no explanation.",
            }
        ],
    )
    print(f"    {context.content[:120]}")

    # Step 2: Code reviewer weighs in
    print("\n  Step 2: Code reviewer adds perspective...")
    review = stateloom.chat(
        agent="code-reviewer",
        messages=[
            {
                "role": "user",
                "content": f"From a code quality perspective, which architecture "
                f"is easier to maintain for a 5-person team? "
                f"Context: {context.content[:200]}",
            }
        ],
    )
    print(f"    {review.content[:120]}")

    # Step 3: Multi-model consensus on the final decision
    print("\n  Step 3: Cross-provider consensus on the decision...")
    result = stateloom.consensus_sync(
        prompt=f"Given this analysis:\n"
        f"- Tech advisor: {context.content[:200]}\n"
        f"- Code reviewer: {review.content[:200]}\n\n"
        "A 5-person startup building B2B SaaS with 100 customers. "
        "Monolith or microservices? Give a clear, final recommendation.",
        models=models,
        strategy="debate",
        rounds=2,
        budget=2.0,
    )
    print(f"    Consensus: {result.answer[:140]}...")
    print(f"    Confidence: {result.confidence:.2f} | Winner: {result.winner_model}")

    # Step 4: Plain stateloom.chat() summarizes for stakeholders
    print("\n  Step 4: Summarize for non-technical stakeholders...")
    summary = stateloom.chat(
        model=models[-1],
        messages=[
            {
                "role": "system",
                "content": "Explain technical decisions to a CEO. Two sentences max.",
            },
            {"role": "user", "content": f"Summarize: {result.answer[:300]}"},
        ],
    )
    print(f"    {summary.content[:140]}")

print(f"\n  Session total: ${s.total_cost:.6f} | {s.call_count} calls | {s.total_tokens} tokens")

print()
print("Dashboard: http://localhost:4782")
