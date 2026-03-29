"""Vote strategy — all models answer independently in parallel."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from stateloom.consensus.aggregation import (
    compute_agreement,
    confidence_weighted,
    majority_vote,
)
from stateloom.consensus.confidence import extract_confidence
from stateloom.consensus.models import ConsensusResult, DebateRound, DebaterResponse
from stateloom.consensus.prompts import DEFAULT_CONFIDENCE_INSTRUCTION, VOTE_SYSTEM_PROMPT

if TYPE_CHECKING:
    from stateloom.consensus.models import ConsensusConfig
    from stateloom.core.session import Session
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.consensus.strategies.vote")


async def _call_debater(
    model: str,
    messages: list[dict[str, Any]],
    parent_session: Session,
    budget: float | None,
    round_number: int,
    durable: bool = True,
) -> DebaterResponse:
    """Make a single debater call and return a DebaterResponse."""
    from stateloom.chat import Client

    start = time.monotonic()
    session_id = f"{parent_session.id}-vote-{model}-r{round_number}"
    async with Client(
        session_id=session_id,
        parent=parent_session.id,
        budget=budget,
        durable=durable,
        name=f"vote-{model}",
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
        )


class VoteStrategy:
    """All models answer independently in parallel, then aggregate."""

    async def execute(
        self,
        config: ConsensusConfig,
        gate: Gate,
        parent_session: Session,
    ) -> ConsensusResult:
        start = time.monotonic()
        models = config.models

        # Build messages
        messages = list(config.messages) if config.messages else []
        if config.prompt and not messages:
            messages = [{"role": "user", "content": config.prompt}]

        # Prepend system prompt with confidence instruction
        if config.agent_system_prompt:
            system_prompt = config.agent_system_prompt + DEFAULT_CONFIDENCE_INSTRUCTION
        else:
            system_prompt = VOTE_SYSTEM_PROMPT
        vote_messages = [{"role": "system", "content": system_prompt}] + messages

        # Calculate per-model budget
        per_model_budget = config.budget / len(models) if config.budget else None

        # All models vote in parallel
        tasks = [
            _call_debater(
                m, vote_messages, parent_session, per_model_budget, 1,
                durable=config.ee_consensus,
            )
            for m in models
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_responses: list[DebaterResponse] = []
        for r in responses:
            if isinstance(r, DebaterResponse):
                valid_responses.append(r)
            else:
                logger.warning("Debater call failed: %s", r, exc_info=True)

        # Aggregate
        if config.aggregation == "majority_vote":
            answer, conf = majority_vote(valid_responses)
        else:
            answer, conf = confidence_weighted(valid_responses)

        round_cost = sum(r.cost for r in valid_responses)
        elapsed_ms = (time.monotonic() - start) * 1000

        debate_round = DebateRound(
            round_number=1,
            responses=valid_responses,
            consensus_reached=True,
            agreement_score=compute_agreement(valid_responses),
            cost=round_cost,
            duration_ms=elapsed_ms,
        )

        winner = max(valid_responses, key=lambda r: r.confidence) if valid_responses else None

        return ConsensusResult(
            answer=answer,
            confidence=conf,
            cost=round_cost,
            session_id=parent_session.id,
            strategy="vote",
            models=models,
            rounds=[debate_round],
            total_rounds=1,
            early_stopped=False,
            aggregation_method=config.aggregation,
            winner_model=winner.model if winner else "",
            duration_ms=elapsed_ms,
        )
