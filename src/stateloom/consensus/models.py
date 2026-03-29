"""Pydantic v2 models for the consensus framework."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class ConsensusConfig:
    """Internal configuration passed from orchestrator to strategy."""

    prompt: str = ""
    messages: list[dict] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    rounds: int = 2
    strategy: str = "debate"
    budget: float | None = None
    session_id: str | None = None
    greedy: bool = False
    greedy_agreement_threshold: float = 0.7
    early_stop_enabled: bool = True
    early_stop_threshold: float = 0.9
    temperature: float = 0.7
    samples: int = 5
    judge_model: str | None = None
    aggregation: str = "confidence_weighted"
    ee_consensus: bool = True  # Internal — set by Gate based on EE availability
    agent: str | None = None  # slug or ID passed by user
    agent_system_prompt: str = ""  # resolved system prompt (set by Gate)
    agent_id: str = ""  # resolved agent ID (for session fields)
    agent_slug: str = ""  # resolved agent slug
    agent_version_id: str = ""  # resolved version ID
    agent_version_number: int = 0  # resolved version number


class DebaterResponse(BaseModel):
    """A single model's response in a debate round."""

    model_config = ConfigDict(extra="ignore")

    model: str = ""
    content: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    cost: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    tokens: int = Field(default=0, ge=0)
    session_id: str = ""
    round_number: int = Field(default=0, ge=0)


class DebateRound(BaseModel):
    """One round of debate/voting."""

    model_config = ConfigDict(extra="ignore")

    round_number: int = Field(default=0, ge=0)
    responses: list[DebaterResponse] = Field(default_factory=list)
    consensus_reached: bool = False
    agreement_score: float = 0.0
    cost: float = Field(default=0.0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0)


class ConsensusResult(BaseModel):
    """Final output of a consensus run."""

    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    cost: float = Field(default=0.0, ge=0)
    session_id: str = ""
    strategy: str = ""
    models: list[str] = Field(default_factory=list)
    rounds: list[DebateRound] = Field(default_factory=list)
    total_rounds: int = Field(default=0, ge=0)
    early_stopped: bool = False
    human_verdict: str | None = None
    aggregation_method: str = ""
    winner_model: str = ""
    duration_ms: float = Field(default=0.0, ge=0)
