"""
StateLoom + LangChain / LangGraph — Full Middleware for Agent Workflows

Demonstrates:
  Part A — LangChain Chains
    - StateLoomCallbackHandler for LLM + tool event tracking
    - Auto-patching: SDK calls flow through the middleware pipeline
    - PII scanning, budget enforcement, and cost tracking on chain calls
    - Cache savings visible in the dashboard

  Part B — LangGraph Tool Agent
    - ReAct-style agent with tool use (search + calculator)
    - patch_langgraph_tools() for automatic tool-call observability
    - Per-step cost tracking in the waterfall timeline
    - Budget enforcement stops runaway agent loops

Why StateLoom + LangChain?
  LangChain handles orchestration (chains, agents, tools).
  StateLoom handles production concerns (cost tracking, PII, guardrails,
  budget limits, caching, kill switch). They compose naturally:
  stateloom.init() auto-patches the underlying SDK, and the callback
  handler adds LangChain-specific metadata (tool names, chain tracking).

Requires:

    pip install stateloom langchain langchain-openai
    # or: pip install stateloom langchain langchain-anthropic
    export OPENAI_API_KEY=sk-...
    python examples/langchain_langgraph.py
"""

import os

import stateloom

# ── Init ──────────────────────────────────────────────────────────────
# auto_patch=True (default) ensures that LangChain's underlying SDK
# calls (openai, anthropic) flow through StateLoom's middleware pipeline.

stateloom.init(
    budget=5.0,
    pii=True,
    console_output=True,
)

# ── Detect available LangChain chat model ────────────────────────────

ChatModel = None
MODEL_NAME = ""

if os.environ.get("OPENAI_API_KEY"):
    try:
        from langchain_openai import ChatOpenAI

        ChatModel = ChatOpenAI
        MODEL_NAME = "gpt-4o-mini"
    except ImportError:
        print("  pip install langchain-openai")

if ChatModel is None and os.environ.get("ANTHROPIC_API_KEY"):
    try:
        from langchain_anthropic import ChatAnthropic

        ChatModel = ChatAnthropic
        MODEL_NAME = "claude-haiku-4-5-20251001"
    except ImportError:
        print("  pip install langchain-anthropic")

if ChatModel is None:
    print("Need langchain + provider package:")
    print("  pip install langchain langchain-openai && export OPENAI_API_KEY=sk-...")
    print("  pip install langchain langchain-anthropic && export ANTHROPIC_API_KEY=sk-ant-...")
    raise SystemExit(1)

print(f"Using: {MODEL_NAME}\n")


# =====================================================================
# PART A — LangChain Chains with StateLoom Observability
# =====================================================================

print("=" * 60)
print("A1. Simple chain with StateLoomCallbackHandler")
print("=" * 60)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from stateloom.ext.langchain import StateLoomCallbackHandler

# The callback handler adds LangChain metadata (tool names, run IDs)
# to the events that auto-patch already records through the pipeline.
handler = StateLoomCallbackHandler()

llm = ChatModel(model=MODEL_NAME)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Be concise."),
        ("human", "{question}"),
    ]
)

chain = prompt | llm | StrOutputParser()

with stateloom.session("langchain-chain-demo", budget=1.0) as s:
    # Simple invocation — goes through full middleware pipeline
    result = chain.invoke(
        {"question": "What is the CAP theorem in distributed systems? Two sentences max."},
        config={"callbacks": [handler]},
    )
    print(f"  Response: {result[:120]}")
    print(f"  Cost: ${s.total_cost:.4f} | Tokens: {s.total_tokens}")
    print()

    # Second call — demonstrates caching (same chain, same input)
    result2 = chain.invoke(
        {"question": "What is the CAP theorem in distributed systems? Two sentences max."},
        config={"callbacks": [handler]},
    )
    print(f"  Cached:   {result2[:120]}")
    print(f"  Cost: ${s.total_cost:.4f} (should be same if cached)")
    print(f"  Cache hits: {s.cache_hits}")

print()


# ── A2. Multi-step chain ─────────────────────────────────────────────

print("=" * 60)
print("A2. Multi-step chain — classify then respond")
print("=" * 60)

classify_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Classify the question into: tech, science, or general. Reply with one word."),
        ("human", "{question}"),
    ]
)
respond_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in {category}. Answer concisely in 1-2 sentences."),
        ("human", "{question}"),
    ]
)

classify_chain = classify_prompt | llm | StrOutputParser()
respond_chain = respond_prompt | llm | StrOutputParser()

questions = [
    "How does TCP handle packet loss?",
    "What causes the northern lights?",
    "What's a good recipe for sourdough bread?",
]

with stateloom.session("langchain-multistep-demo", budget=2.0) as s:
    for q in questions:
        category = (
            classify_chain.invoke(
                {"question": q},
                config={"callbacks": [handler]},
            )
            .strip()
            .lower()
        )
        answer = respond_chain.invoke(
            {"question": q, "category": category},
            config={"callbacks": [handler]},
        )
        print(f"  [{category}] {q}")
        print(f"    {answer[:100]}")

    print()
    print(f"  Total: ${s.total_cost:.4f} | {s.total_tokens} tokens | {s.call_count} calls")

print()


# =====================================================================
# PART B — LangGraph ReAct Agent with Tool Observability
# =====================================================================
# LangGraph agents make multiple LLM calls + tool executions in a loop.
# StateLoom tracks each step: LLM calls (via auto-patch), tool calls
# (via patch_langgraph_tools), all with cost and latency.

print("=" * 60)
print("B1. LangGraph ReAct agent with tools")
print("=" * 60)

try:
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False
    print("  LangGraph not installed. Install with: pip install langgraph")
    print("  Skipping Part B.\n")


if _HAS_LANGGRAPH:
    from stateloom.ext.langgraph import patch_langgraph_tools

    # Patch LangGraph's ToolNode so tool invocations are recorded as events
    patch_langgraph_tools()

    # Define simple tools for the agent
    @tool
    def search(query: str) -> str:
        """Search the web for information."""
        # Simulated search results
        results = {
            "population": "The world population is approximately 8.1 billion as of 2024.",
            "python": "Python 3.13 was released in October 2024.",
            "default": f"Search results for '{query}': No specific results found.",
        }
        for key, value in results.items():
            if key in query.lower():
                return value
        return results["default"]

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            # Safe eval for simple math
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expression):
                return str(eval(expression))
            return "Error: only basic arithmetic is supported"
        except Exception as e:
            return f"Error: {e}"

    # Create the ReAct agent
    agent = create_react_agent(
        ChatModel(model=MODEL_NAME),
        tools=[search, calculator],
    )

    with stateloom.session("langgraph-agent-demo", budget=2.0) as s:
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        "What is the current world population? "
                        "Multiply it by 2 and tell me the result.",
                    )
                ]
            },
            config={"callbacks": [handler]},
        )

        # Extract the final answer
        final_msg = result["messages"][-1]
        print(f"  Agent answer: {final_msg.content[:200]}")
        print(f"  Steps: {s.call_count} | Cost: ${s.total_cost:.4f} | Tokens: {s.total_tokens}")

    print()

    # ── B2. Budget enforcement stops runaway agents ──────────────────

    print("=" * 60)
    print("B2. Budget enforcement — stop runaway agent loops")
    print("=" * 60)

    # With a tiny budget, the agent gets cut off mid-loop
    try:
        with stateloom.session("langgraph-budget-demo", budget=0.001) as s:
            result = agent.invoke(
                {
                    "messages": [
                        (
                            "human",
                            "Research the top 5 programming languages, "
                            "their market share, and calculate the total.",
                        )
                    ]
                },
                config={"callbacks": [handler]},
            )
            print("  Completed (unlikely with $0.001 budget)")
    except stateloom.StateLoomBudgetError as e:
        print(f"  Budget enforced: {e}")
        print(f"  Agent stopped at ${s.total_cost:.4f} / ${s.budget:.4f}")
        print(f"  Steps completed: {s.call_count}")

    print()


# ── Summary ──────────────────────────────────────────────────────────

print("=" * 60)
print("Summary")
print("=" * 60)
print("""
  LangChain + StateLoom:
    - stateloom.init() auto-patches the underlying SDK
    - StateLoomCallbackHandler adds tool names and chain metadata
    - Full middleware pipeline: PII, guardrails, budget, caching, cost tracking
    - Works with any LangChain chat model (OpenAI, Anthropic, Google, etc.)

  LangGraph + StateLoom:
    - patch_langgraph_tools() wraps ToolNode for automatic tool observability
    - Each agent step (LLM call + tool call) tracked in the waterfall
    - Budget enforcement stops runaway agent loops
    - Cost per agent run visible in the dashboard
""")

print("Dashboard: http://localhost:4782")
print("Check the session waterfall to see LLM calls and tool executions as separate steps.")
