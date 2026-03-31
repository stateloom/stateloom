# StateLoom Examples

Runnable scripts demonstrating each StateLoom feature.

## Prerequisites

```bash
pip install stateloom
```

Set at least one provider API key:

```bash
export OPENAI_API_KEY=sk-...
# and/or
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AIza...
```

Some examples require extra dependencies:

```bash
pip install stateloom[semantic]   # 05 guardrails NLI, 06 semantic cache
pip install stateloom[prompts]    # 11 file-based agents (watchdog)
pip install langchain langchain-openai  # 14 LangChain/LangGraph
pip install litellm               # 15 LiteLLM
```

## Running

Start the dashboard first, then run any example in a separate terminal:

```bash
# Terminal 1 — start the dashboard + proxy
stateloom serve

# Terminal 2 — run an example
python examples/01_quick_start.py
```

Open the dashboard at [http://localhost:4782](http://localhost:4782) to see sessions, cost breakdown, cache hits, PII detections, guardrail events, and the waterfall trace timeline as each example runs.

## Examples

| # | Script | Feature | API Key |
|---|--------|---------|:-------:|
| 01 | `01_quick_start.py` | Auto-patching OpenAI/Anthropic/Gemini SDKs, session tracking, per-model cost breakdown | Yes |
| 02 | `02_unified_client.py` | Provider-agnostic `stateloom.chat()` and `Client` — no SDK imports needed | Yes |
| 03 | `03_sessions_and_budget.py` | Session lifecycle, per-session budget enforcement, parent-child hierarchy | Yes |
| 04 | `04_pii_detection.py` | PII scanning modes (audit, redact, block), custom rules, GDPR compliance profiles | Yes |
| 05 | `05_guardrails.py` | Prompt injection & jailbreak detection (32 heuristic patterns), audit vs enforce mode, NLI classifier | Yes |
| 06 | `06_caching.py` | Exact-match request caching, cache hit tracking, cost savings | Yes |
| 07 | `07_local_models.py` | Ollama integration, hardware detection, model recommendations, auto-routing | No* |
| 08 | `08_durable_resumption.py` | Crash-recovery checkpointing (Temporal-like), durable sessions, resume from cached steps | Yes |
| 09 | `09_semantic_retries.py` | Self-healing LLM output: `retry_loop()` for bad JSON, `durable_task()` decorator | Yes |
| 10 | `10_time_travel.py` | Record-replay debugging, session export/import, VCR-cassette mock | Yes |
| 11 | `11_agents_and_consensus.py` | Managed agents (Prompts-as-an-API), multi-agent consensus (vote, debate, self-consistency) | Yes |
| 12 | `12_experiments.py` | A/B experiments: variant assignment, model/prompt overrides, feedback, metrics, leaderboard | Yes |
| 13 | `13_kill_switch.py` | Global & granular emergency stop: model globs, provider blocks, error vs response mode | Yes |
| 14 | `14_langchain_langgraph.py` | LangChain chains + LangGraph ReAct agents with full middleware pipeline | Yes |
| 15 | `15_litellm.py` | LiteLLM multi-provider routing with cost tracking, budget, guardrails, streaming | Yes |
| 16 | `16_zero_trust_security.py` | Secret vault (environ scrub), CPython audit hooks (PEP 578), layered defense | Yes |
| 17 | `17_circuit_breaker.py` | Per-provider circuit breaker, tier-based fallback, manual reset, layered protection | Yes |
| 18 | `18_compliance.py` | GDPR/HIPAA/CCPA profiles, tamper-proof audit trail, Right to Be Forgotten, custom profiles | Yes |
| 19 | `19_model_testing.py` | Dark launch candidates alongside production, similarity scoring, sampling, smart skip logic | Yes* |
| 20 | `20_proxy_gateway.py` | OpenAI-compatible proxy with PII, guardrails, caching, kill switch, dashboard API queries | Yes |

\* Example 07 requires a running [Ollama](https://ollama.com) instance instead of a cloud API key. Example 19 requires two provider API keys for cross-provider comparison.

## Dashboard Controls

The dashboard at [http://localhost:4782](http://localhost:4782) is not just for monitoring — many features can be **configured at runtime** without code changes or restarts.

### Monitoring & Observability

| Feature | What you see |
|---------|-------------|
| **Live Overview** | Real-time cost counter, request rate, cache savings, live event stream |
| **Sessions** | List/filter/search sessions, waterfall trace timeline, per-step cost/latency bars |
| **Observability** | Request rate timeseries, latency percentiles (P50/P95/P99), cost over time, provider breakdown |
| **Async Jobs** | Job status, retry history, webhook delivery logs |

### Runtime Configuration (no code changes needed)

| Feature | What you can control |
|---------|---------------------|
| **Kill Switch** | Activate/deactivate global stop, add/remove model glob rules, set error vs response mode |
| **Guardrails** | Toggle heuristic scanning, enable/disable NLI classifier, set audit vs enforce mode, adjust NLI threshold |
| **PII Rules** | Add/remove PII detection rules, switch between audit/redact/block modes |
| **Budget** | Set per-session and global budget limits, choose warn vs hard-stop action |
| **Rate Limits** | Set per-team TPS, priority, max queue size, queue timeout |
| **Caching** | Enable/disable, set TTL, toggle semantic similarity matching |
| **Blast Radius** | Unpause auto-paused sessions and agents after failure containment |
| **Audit Hooks** | Configure CPython audit hook deny/allow lists, set audit vs enforce mode |
| **Secret Vault** | Store/retrieve API keys in-memory, scrub environment variables |
| **Compliance** | Assign GDPR/HIPAA/CCPA profiles per org or team, trigger data purge |

### Agent & Experiment Management

| Feature | What you can do |
|---------|----------------|
| **Agents** | Create/update agents, push new prompt versions, rollback to previous versions, pause/archive |
| **Experiments** | Create A/B experiments, start/pause/conclude, view per-variant metrics, leaderboard rankings |
| **Consensus** | Browse multi-agent consensus runs, view round-by-round debate responses, aggregation results |
| **Virtual Keys** | Create scoped API keys, set per-key rate limits, revoke keys |
| **Organizations** | Create orgs/teams, assign per-team budgets, compliance profiles, and rate limits |

## Example Details

### 01 — Quick Start
Auto-patches installed SDKs so native `openai.OpenAI()`, `anthropic.Anthropic()`, and `genai.GenerativeModel()` calls flow through the middleware pipeline. Shows per-model cost/token breakdown in a single session.

### 02 — Unified Client
Use `stateloom.chat(model="gpt-4o-mini", messages=[...])` without importing any provider SDK. The `Client` class auto-detects providers from model name prefixes (gpt-* → OpenAI, claude-* → Anthropic, gemini-* → Google).

### 03 — Sessions & Budget
Named sessions track cost, tokens, and call count. Budget enforcement raises `StateLoomBudgetError` when spend exceeds the limit. Parent-child sessions provide hierarchical isolation.

### 04 — PII Detection
Three modes: **audit** (detect and log), **redact** (replace PII with placeholders), **block** (raise `StateLoomPIIBlockedError`). Custom rules for domain-specific patterns. GDPR compliance profiles enforce data handling policies.

### 05 — Guardrails
32 regex patterns detect prompt injection, jailbreak, encoding attacks, and multi-turn manipulation. **Audit mode** logs violations as events. **Enforce mode** blocks high-severity attacks before they reach the LLM. Optional NLI classifier adds semantic scoring.

### 06 — Caching
Exact-match caching deduplicates identical requests within a session. Cache hits appear in the dashboard waterfall and save real API costs. Shows hit/miss tracking and cost savings.

### 07 — Local Models
Hardware detection recommends Ollama models based on available RAM/GPU. Auto-routing steers simple prompts to local models, saving cloud API costs. Requires Ollama running locally.

### 08 — Durable Resumption
Temporal-like checkpointing: LLM responses are persisted so a crashed process can resume from the last completed step without re-calling the API. Supports both sync and async sessions.

### 09 — Semantic Retries
`retry_loop(retries=3)` retries LLM calls when output validation fails (bad JSON, missing fields, hallucinated tool calls). `durable_task()` combines durable sessions with automatic retries for zero wasted API calls on resume.

### 10 — Time-Travel Debugging
Record a multi-step session, then replay it deterministically with network blocking. Export sessions as portable JSON bundles. `stateloom.mock()` provides VCR-cassette style recording for test suites.

### 11 — Agents & Consensus
**Agents**: deploy system prompts as versioned API endpoints (`/v1/agents/{slug}/chat/completions`). **Consensus**: run multiple models on the same prompt and aggregate answers via vote, multi-round debate, or self-consistency strategies.

### 12 — A/B Experiments
Create experiments with model/prompt variants, assign sessions via hash or random strategy, collect feedback, and compare variants on a leaderboard. Cross-provider model overrides are safely guarded.

### 13 — Kill Switch
Emergency traffic control. **Global**: `stateloom.kill_switch(active=True)` blocks all LLM calls instantly. **Granular**: `stateloom.add_kill_switch_rule(model="gpt-4*")` blocks by model glob, provider, or environment. Two modes: **error** (raise `StateLoomKillSwitchError`) or **response** (return a static dict for graceful degradation). Can be toggled at runtime or via the dashboard API.

### 14 — LangChain / LangGraph
**LangChain**: `StateLoomCallbackHandler` adds tool-call tracking and chain metadata alongside the full middleware pipeline. Auto-patching ensures underlying SDK calls flow through PII scanning, guardrails, budget enforcement, and cost tracking. **LangGraph**: `patch_langgraph_tools()` wraps `ToolNode` for automatic tool-call observability. Budget enforcement stops runaway agent loops. Each agent step appears in the dashboard waterfall.

### 15 — LiteLLM
StateLoom auto-patches `litellm.completion()` and `litellm.acompletion()` so every call to any of LiteLLM's 100+ supported providers flows through the middleware pipeline. Cost tracking, PII scanning, guardrails, budget enforcement, caching, and kill switch all work transparently — including on LiteLLM's retries, fallbacks, and streaming.

### 16 — Zero-Trust Security
**Secret Vault**: moves API keys from `os.environ` into protected in-memory storage, optionally scrubbing the environment so leaked env dumps expose nothing. Store/retrieve custom secrets programmatically. **CPython Audit Hooks** (PEP 578): intercept dangerous interpreter operations (subprocess, exec, socket, ctypes) at the C level — catches supply-chain attacks in third-party libraries. Audit mode logs violations; enforce mode blocks them. No new dependencies — uses only Python stdlib.

### 17 — Circuit Breaker
Per-provider health tracking with a three-state machine: **closed** (healthy) -> **open** (failing, traffic blocked) -> **half-open** (probing for recovery). When a provider's circuit opens, tier-based fallback maps suggest a same-quality model from a healthy provider. Custom fallback maps for explicit model-to-model overrides. Manual reset for ops team intervention. Layered protection with budget enforcement.

### 18 — Compliance
Declarative compliance profiles for regulated industries. **GDPR**: EU data residency, 30-day session TTL, PII block/redact rules. **HIPAA**: zero-retention logs, no caching, strict PII blocking. **CCPA**: 90-day TTL, consumer deletion rights. Tamper-proof audit trail with SHA-256 integrity hashes and legal rule citations. Right to Be Forgotten purge engine deletes all user data on request. Custom profiles for domain-specific standards (e.g., PCI-DSS).

### 19 — Model Testing / Dark Launch
Run candidate models alongside production calls without affecting users. Candidate calls are fire-and-forget — the primary response is never delayed. **Cloud-to-cloud** testing compares any two providers (e.g., gpt-4o-mini vs claude-haiku). **Similarity scoring** compares responses via difflib (0.0–1.0 score). **Sampling** controls what percentage of traffic gets tested (cost control). **Smart skip logic** auto-excludes streaming, PII, and cached responses. Also works with local models (Ollama). Requires two provider API keys.

### 20 — Proxy Gateway
Run StateLoom as an OpenAI-compatible proxy gateway. Point any SDK at the proxy and get the full middleware pipeline: PII scanning, guardrails, caching, budget enforcement, and kill switch — all without changing your application code. Supports OpenAI (`/v1/chat/completions`), Anthropic-native (`/v1/messages`), and Gemini-native (`/v1beta/models/{m}:generateContent`) endpoints. All requests flow through the proxy using the provider's native SDK or HTTP calls, and session stats (PII detections, guardrail events, cache hits) are verified via the dashboard REST API. All middleware settings — PII rules, guardrails mode, budget limits, kill switch, caching, rate limits — can be changed at runtime through the dashboard at `http://localhost:4782` with no code changes or restarts needed.
