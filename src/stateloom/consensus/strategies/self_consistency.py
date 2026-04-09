"""Self-consistency strategy — multiple samples from one model, majority vote."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from stateloom.consensus.aggregation import compute_agreement, majority_vote
from stateloom.consensus.confidence import extract_confidence
from stateloom.consensus.models import ConsensusResult, DebateRound, DebaterResponse
from stateloom.consensus.prompts import DEFAULT_CONFIDENCE_INSTRUCTION, VOTE_SYSTEM_PROMPT

if TYPE_CHECKING:
    from stateloom.consensus.models import ConsensusConfig
    from stateloom.core.session import Session
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.consensus.strategies.self_consistency")


async def _sample_call(
    model: str,
    messages: list[dict[str, Any]],
    parent_session: Session,
    budget: float | None,
    sample_index: int,
    temperature: float,
    durable: bool = True,
    persona_name: str = "",
) -> DebaterResponse:
    """Make a single sample call."""
    from stateloom.chat import Client

    start = time.monotonic()
    session_id = f"{parent_session.id}-sc-{model}-s{sample_index}"
    async with Client(
        session_id=session_id,
        parent=parent_session.id,
        budget=budget,
        durable=durable,
        name=f"sc-{model}-s{sample_index}",
    ) as client:
        result = await client.achat(
            model=model,
            messages=messages,
            temperature=temperature,
        )
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
            round_number=1,
            persona_name=persona_name,
        )


class SelfConsistencyStrategy:
    """Multiple samples from one model, aggregated by majority vote."""

    async def execute(
        self,
        config: ConsensusConfig,
        gate: Gate,
        parent_session: Session,
    ) -> ConsensusResult:
        if config.personas and len(config.personas) > 1:
            raise ValueError("self_consistency requires exactly 1 persona")

        start = time.monotonic()
        persona = config.personas[0] if config.personas else None
        model = persona.model if persona else config.models[0]
        persona_name = persona.name if persona else ""
        samples = config.samples
        temperature = config.temperature

        # Build messages
        messages = list(config.messages) if config.messages else []
        if persona and persona.prompt:
            messages = [{"role": "user", "content": persona.prompt}]
        elif config.prompt and not messages:
            messages = [{"role": "user", "content": config.prompt}]
        if persona and persona.system_prompt:
            system_prompt = persona.system_prompt + DEFAULT_CONFIDENCE_INSTRUCTION
        elif config.agent_system_prompt:
            system_prompt = config.agent_system_prompt + DEFAULT_CONFIDENCE_INSTRUCTION
        else:
            system_prompt = VOTE_SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_prompt}] + messages

        per_sample_budget = config.budget / samples if config.budget else None

        tasks = [
            _sample_call(
                model,
                messages,
                parent_session,
                per_sample_budget,
                i,
                temperature,
                durable=config.ee_consensus,
                persona_name=persona_name,
            )
            for i in range(samples)
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid: list[DebaterResponse] = []
        for r in responses:
            if isinstance(r, DebaterResponse):
                valid.append(r)
            else:
                logger.warning("Sample call failed: %s", r)

        answer, conf = majority_vote(valid)
        total_cost = sum(r.cost for r in valid)
        elapsed_ms = (time.monotonic() - start) * 1000
        agreement = compute_agreement(valid)

        debate_round = DebateRound(
            round_number=1,
            responses=valid,
            consensus_reached=True,
            agreement_score=agreement,
            cost=total_cost,
            duration_ms=elapsed_ms,
        )

        # Build persona metadata
        result_personas: list[dict[str, str]] = []
        if persona:
            result_personas = [{"name": persona.name, "model": persona.model}]

        return ConsensusResult(
            answer=answer,
            confidence=conf,
            cost=total_cost,
            session_id=parent_session.id,
            strategy="self_consistency",
            models=[model],
            rounds=[debate_round],
            total_rounds=1,
            early_stopped=False,
            aggregation_method="majority_vote",
            winner_model=model,
            duration_ms=elapsed_ms,
            personas=result_personas,
            winner_persona=persona_name,
        )
