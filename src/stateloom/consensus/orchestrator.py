"""ConsensusOrchestrator — main entry point for multi-agent debate."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from stateloom.consensus.models import ConsensusConfig, ConsensusResult
from stateloom.consensus.strategies import get_strategy
from stateloom.core.event import ConsensusEvent, DebateRoundEvent

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom.consensus.orchestrator")


class ConsensusOrchestrator:
    """Orchestrates multi-agent consensus runs."""

    def __init__(self, gate: Gate) -> None:
        self._gate = gate

    async def run(self, config: ConsensusConfig) -> ConsensusResult:
        """Run a consensus session.

        Creates a durable parent session, dispatches to the chosen strategy,
        records events, and returns the result.
        """
        gate = self._gate
        start = time.monotonic()

        async with gate.async_session(
            session_id=config.session_id,
            name=f"consensus-{config.strategy}",
            budget=config.budget,
            durable=config.ee_consensus,
        ) as parent_session:
            # Set agent session fields for dashboard visibility
            if config.agent_id:
                parent_session.agent_id = config.agent_id
                parent_session.agent_slug = config.agent_slug
                parent_session.agent_version_id = config.agent_version_id
                parent_session.agent_version_number = config.agent_version_number
                parent_session.agent_name = config.agent_slug

            strategy = get_strategy(config.strategy)

            try:
                result = await strategy.execute(config, gate, parent_session)
            except Exception as exc:
                # On budget error or other failure, try to return partial results
                from stateloom.core.errors import StateLoomBudgetError

                if isinstance(exc, StateLoomBudgetError):
                    logger.warning("Budget exhausted during consensus, returning partial result")
                    result = ConsensusResult(
                        answer="(Budget exhausted — partial result)",
                        confidence=0.0,
                        cost=parent_session.total_cost,
                        session_id=parent_session.id,
                        strategy=config.strategy,
                        models=config.models,
                        total_rounds=0,
                        duration_ms=(time.monotonic() - start) * 1000,
                        personas=[{"name": p.name, "model": p.model} for p in config.personas],
                    )
                else:
                    raise

            # Ensure session_id is set
            result.session_id = parent_session.id

            # Record DebateRoundEvent for each round
            for rnd in result.rounds:
                try:
                    round_event = DebateRoundEvent(
                        session_id=parent_session.id,
                        round_number=rnd.round_number,
                        strategy=config.strategy,
                        models=[r.model for r in rnd.responses],
                        responses_summary=[
                            {
                                "model": r.model,
                                "confidence": r.confidence,
                                "cost": r.cost,
                                "content_preview": r.content[:200],
                                **({"persona_name": r.persona_name} if r.persona_name else {}),
                            }
                            for r in rnd.responses
                        ],
                        agreement_score=rnd.agreement_score,
                        consensus_reached=rnd.consensus_reached,
                        round_cost=rnd.cost,
                        round_duration_ms=rnd.duration_ms,
                        persona_names=[r.persona_name for r in rnd.responses if r.persona_name],
                    )
                    gate.store.save_event(round_event)
                except Exception:
                    logger.warning("Failed to save DebateRoundEvent", exc_info=True)

            # Record ConsensusEvent
            try:
                consensus_event = ConsensusEvent(
                    session_id=parent_session.id,
                    strategy=config.strategy,
                    models=config.models,
                    total_rounds=result.total_rounds,
                    final_answer_preview=result.answer[:200],
                    confidence=result.confidence,
                    total_cost=result.cost,
                    total_duration_ms=result.duration_ms,
                    early_stopped=result.early_stopped,
                    aggregation_method=result.aggregation_method,
                    winner_model=result.winner_model,
                    personas=result.personas,
                    winner_persona=result.winner_persona,
                )
                gate.store.save_event(consensus_event)
            except Exception:
                logger.warning("Failed to save ConsensusEvent", exc_info=True)

            return result
