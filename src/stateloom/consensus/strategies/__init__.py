"""Consensus strategy registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from stateloom.consensus.models import ConsensusResult
    from stateloom.core.session import Session
    from stateloom.gate import Gate


class ConsensusStrategyProtocol(Protocol):
    """Protocol for consensus strategies."""

    async def execute(
        self,
        config: Any,
        gate: Gate,
        parent_session: Session,
    ) -> ConsensusResult: ...


def get_strategy(name: str) -> ConsensusStrategyProtocol:
    """Look up a strategy by name."""
    from stateloom.consensus.strategies.debate import DebateStrategy
    from stateloom.consensus.strategies.self_consistency import SelfConsistencyStrategy
    from stateloom.consensus.strategies.vote import VoteStrategy

    strategies: dict[str, ConsensusStrategyProtocol] = {
        "vote": VoteStrategy(),
        "debate": DebateStrategy(),
        "self_consistency": SelfConsistencyStrategy(),
    }
    strategy = strategies.get(name)
    if strategy is None:
        raise ValueError(f"Unknown consensus strategy: {name!r}. Available: {list(strategies)}")
    return strategy
