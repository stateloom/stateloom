"""Debate strategy — multi-round debate with optional greedy downgrade."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from stateloom.consensus.aggregation import (
    compute_agreement,
    confidence_weighted,
    judge_synthesis,
)
from stateloom.consensus.confidence import extract_confidence
from stateloom.consensus.models import ConsensusResult, DebateRound, DebaterResponse
from stateloom.consensus.prompts import (
    DEBATE_ROUND_TEMPLATE,
    DEFAULT_CONFIDENCE_INSTRUCTION,
    VOTE_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from stateloom.consensus.models import ConsensusConfig
    from stateloom.core.session import Session
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.consensus.strategies.debate")

_MAX_PREV_RESPONSE_LEN = 500  # Truncate previous responses after round 2


def _sanitize_session_id(name: str) -> str:
    """Convert a persona name into a safe session-id fragment."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:40]


def _downgrade_model(model: str) -> str:
    """Map a tier-1 model to its cheaper tier-2 equivalent.

    Uses the tier mapping from circuit_breaker.py.
    """
    from stateloom.middleware.circuit_breaker import _DEFAULT_MODEL_TIERS

    tier1 = _DEFAULT_MODEL_TIERS.get("tier-1-flagship", [])
    tier2 = _DEFAULT_MODEL_TIERS.get("tier-2-fast", [])

    if model not in tier1:
        return model

    # Find the tier-2 model from the same provider family
    model_lower = model.lower()
    for t2 in tier2:
        t2_lower = t2.lower()
        # Match by provider prefix
        if model_lower.startswith("gpt") and t2_lower.startswith("gpt"):
            return t2
        if model_lower.startswith("claude") and t2_lower.startswith("claude"):
            return t2
        if model_lower.startswith("gemini") and t2_lower.startswith("gemini"):
            return t2
    return model


async def _call_debater(
    model: str,
    messages: list[dict[str, Any]],
    parent_session: Session,
    budget: float | None,
    round_number: int,
    durable: bool = True,
    persona_name: str = "",
) -> DebaterResponse:
    """Make a single debater call and return a DebaterResponse."""
    from stateloom.chat import Client

    start = time.monotonic()
    if persona_name:
        slug = _sanitize_session_id(persona_name)
        session_id = f"{parent_session.id}-debate-{slug}-r{round_number}"
        client_name = f"debate-{slug}-r{round_number}"
    else:
        session_id = f"{parent_session.id}-debate-{model}-r{round_number}"
        client_name = f"debate-{model}-r{round_number}"
    async with Client(
        session_id=session_id,
        parent=parent_session.id,
        budget=budget,
        durable=durable,
        name=client_name,
    ) as client:
        result = await client.achat(model=model, messages=messages)
        elapsed_ms = (time.monotonic() - start) * 1000
        text = result.content
        conf = extract_confidence(text)
        session = client.session
        cost = session.total_cost if session else 0.0
        tokens = (session.total_prompt_tokens + session.total_completion_tokens) if session else 0
        return DebaterResponse(
            model=model,
            content=text,
            confidence=conf,
            cost=cost,
            latency_ms=elapsed_ms,
            tokens=tokens,
            session_id=session_id,
            round_number=round_number,
            persona_name=persona_name,
        )


def _build_debate_messages(
    base_messages: list[dict[str, Any]],
    previous_responses: list[DebaterResponse],
    current_identity: str,
    round_number: int,
    use_personas: bool = False,
    sees: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build messages for a debate round, including previous responses."""
    # Format previous responses from other models/personas
    parts: list[str] = []
    for resp in previous_responses:
        if use_personas:
            # Skip self by persona name
            if resp.persona_name == current_identity:
                continue
            # Filter by sees list when provided
            if sees is not None and resp.persona_name not in sees:
                continue
            label = resp.persona_name
        else:
            if resp.model == current_identity:
                continue
            label = resp.model
        content = resp.content
        # Truncate after round 2 to manage context growth
        if round_number > 2 and len(content) > _MAX_PREV_RESPONSE_LEN:
            content = content[:_MAX_PREV_RESPONSE_LEN] + "..."
        parts.append(f"**{label}** (Confidence: {resp.confidence:.2f}):\n{content}")

    prev_text = "\n\n---\n\n".join(parts) if parts else "(No other responses)"
    debate_prompt = DEBATE_ROUND_TEMPLATE.format(previous_responses=prev_text)

    return base_messages + [{"role": "system", "content": debate_prompt}]


class DebateStrategy:
    """Multi-round debate with early stopping and optional greedy downgrade."""

    async def execute(
        self,
        config: ConsensusConfig,
        gate: Gate,
        parent_session: Session,
    ) -> ConsensusResult:
        start = time.monotonic()
        models = list(config.models)
        total_rounds = config.rounds
        use_personas = bool(config.personas)
        personas = list(config.personas) if use_personas else []

        # Build base messages (non-persona path)
        base_messages = list(config.messages) if config.messages else []
        if config.prompt and not base_messages:
            base_messages = [{"role": "user", "content": config.prompt}]

        if not use_personas:
            # Add system prompt for round 1
            if config.agent_system_prompt:
                system_prompt = config.agent_system_prompt + DEFAULT_CONFIDENCE_INSTRUCTION
            else:
                system_prompt = VOTE_SYSTEM_PROMPT
            r1_messages = [{"role": "system", "content": system_prompt}] + base_messages

        # Per-model budget (spread across all rounds + judge)
        debater_count = len(personas) if use_personas else len(models)
        total_calls = debater_count * total_rounds + 1  # +1 for judge
        per_call_budget = config.budget / total_calls if config.budget else None

        all_rounds: list[DebateRound] = []
        all_responses: list[DebaterResponse] = []
        total_cost = 0.0
        early_stopped = False

        for round_num in range(1, total_rounds + 1):
            round_start = time.monotonic()

            if use_personas:
                tasks = []
                for p in personas:
                    if round_num == 1:
                        # Round 1: each persona gets its own system_prompt + prompt
                        sys_prompt = (
                            p.system_prompt + DEFAULT_CONFIDENCE_INSTRUCTION
                            if p.system_prompt
                            else VOTE_SYSTEM_PROMPT
                        )
                        msgs: list[dict[str, Any]] = [
                            {"role": "system", "content": sys_prompt},
                        ]
                        if p.prompt:
                            msgs.append({"role": "user", "content": p.prompt})
                        elif base_messages:
                            msgs.extend(base_messages)
                    else:
                        # Round 2+: keep persona system prompt, add filtered debate context
                        sys_prompt = (
                            p.system_prompt + DEFAULT_CONFIDENCE_INSTRUCTION
                            if p.system_prompt
                            else VOTE_SYSTEM_PROMPT
                        )
                        persona_base: list[dict[str, Any]] = [
                            {"role": "system", "content": sys_prompt},
                        ]
                        if p.prompt:
                            persona_base.append({"role": "user", "content": p.prompt})
                        elif base_messages:
                            persona_base.extend(base_messages)
                        prev_round_responses = all_rounds[-1].responses
                        msgs = _build_debate_messages(
                            persona_base,
                            prev_round_responses,
                            p.name,
                            round_num,
                            use_personas=True,
                            sees=p.sees,
                        )
                    tasks.append(
                        _call_debater(
                            p.model,
                            msgs,
                            parent_session,
                            per_call_budget,
                            round_num,
                            durable=config.ee_consensus,
                            persona_name=p.name,
                        )
                    )
            else:
                if round_num == 1:
                    # Round 1: independent parallel calls
                    messages_per_model = {m: r1_messages for m in models}
                else:
                    # Subsequent rounds: include previous round's responses
                    prev_round_responses = all_rounds[-1].responses
                    messages_per_model = {
                        m: _build_debate_messages(base_messages, prev_round_responses, m, round_num)
                        for m in models
                    }

                tasks = [
                    _call_debater(
                        m,
                        messages_per_model[m],
                        parent_session,
                        per_call_budget,
                        round_num,
                        durable=config.ee_consensus,
                    )
                    for m in models
                ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            valid: list[DebaterResponse] = []
            for r in responses:
                if isinstance(r, DebaterResponse):
                    valid.append(r)
                else:
                    logger.warning("Debater call failed in round %d: %s", round_num, r)

            round_cost = sum(r.cost for r in valid)
            total_cost += round_cost
            agreement = compute_agreement(valid)
            round_elapsed = (time.monotonic() - round_start) * 1000

            debate_round = DebateRound(
                round_number=round_num,
                responses=valid,
                consensus_reached=agreement > config.early_stop_threshold,
                agreement_score=agreement,
                cost=round_cost,
                duration_ms=round_elapsed,
            )
            all_rounds.append(debate_round)
            all_responses.extend(valid)

            # Early stop check
            if config.early_stop_enabled and round_num < total_rounds:
                all_high_confidence = all(
                    r.confidence >= config.early_stop_threshold for r in valid
                )
                if all_high_confidence and agreement > 0.8:
                    early_stopped = True
                    break

            # Greedy downgrade: if agreement > threshold after round 1, swap to cheaper models
            if config.greedy and round_num == 1 and agreement > config.greedy_agreement_threshold:
                if use_personas:
                    for p in personas:
                        p.model = _downgrade_model(p.model)
                    logger.info(
                        "Greedy downgrade: persona models changed to %s",
                        [p.model for p in personas],
                    )
                else:
                    models = [_downgrade_model(m) for m in models]
                    logger.info("Greedy downgrade: models changed to %s", models)

        # Judge synthesis (EE only) — falls back to confidence_weighted
        agg_method = "confidence_weighted"
        if config.ee_consensus:
            judge_model = config.judge_model or config.models[0]
            try:
                answer, conf = await judge_synthesis(
                    all_responses, judge_model, gate, parent_session
                )
                agg_method = "judge_synthesis"
            except Exception:
                logger.warning(
                    "Judge synthesis failed, falling back to confidence_weighted",
                    exc_info=True,
                )
                answer, conf = confidence_weighted(all_rounds[-1].responses if all_rounds else [])
        else:
            answer, conf = confidence_weighted(all_rounds[-1].responses if all_rounds else [])

        elapsed_ms = (time.monotonic() - start) * 1000
        winner = max(all_responses, key=lambda r: r.confidence) if all_responses else None

        # Build persona metadata for result
        result_personas: list[dict[str, str]] = []
        winner_persona = ""
        if use_personas:
            result_personas = [{"name": p.name, "model": p.model} for p in personas]
            if winner and winner.persona_name:
                winner_persona = winner.persona_name

        return ConsensusResult(
            answer=answer,
            confidence=conf,
            cost=total_cost,
            session_id=parent_session.id,
            strategy="debate",
            models=config.models,
            rounds=all_rounds,
            total_rounds=len(all_rounds),
            early_stopped=early_stopped,
            aggregation_method=agg_method,
            winner_model=winner.model if winner else "",
            duration_ms=elapsed_ms,
            personas=result_personas,
            winner_persona=winner_persona,
        )
