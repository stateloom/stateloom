"""Event models for StateLoom. Every intercepted call produces an event."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field

from stateloom.core.types import EventType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class Event(BaseModel):
    """Base event. All events in a session inherit from this."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=_new_id)
    session_id: str = ""
    step: int = Field(default=0, ge=0)
    event_type: EventType = EventType.LLM_CALL
    timestamp: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMCallEvent(Event):
    """An intercepted LLM API call."""

    event_type: Literal[EventType.LLM_CALL] = EventType.LLM_CALL
    provider: str = ""
    model: str = ""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0)
    estimated_api_cost: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    is_streaming: bool = False
    request_hash: str = ""
    prompt_preview: str = ""
    is_cli_internal: bool = False
    is_tool_continuation: bool = False
    cached_response_json: str | None = None


class ToolCallEvent(Event):
    """An intercepted @gate.tool() call."""

    event_type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    tool_name: str = ""
    mutates_state: bool = False
    args_hash: str = ""
    result_hash: str = ""
    latency_ms: float = Field(default=0.0, ge=0)
    cached_result: Any = None
    cached_response_json: str | None = None


class CacheHitEvent(Event):
    """A cache hit that saved an LLM call."""

    event_type: Literal[EventType.CACHE_HIT] = EventType.CACHE_HIT
    original_model: str = ""
    saved_cost: float = Field(default=0.0, ge=0)
    request_hash: str = ""
    match_type: str = "exact"  # "exact" or "semantic"
    similarity_score: float | None = None  # cosine sim for semantic hits
    matched_hash: str = ""  # hash of matched entry (differs for semantic)
    prompt_preview: str = ""


class PIIDetectionEvent(Event):
    """A PII detection during scanning."""

    event_type: Literal[EventType.PII_DETECTION] = EventType.PII_DETECTION
    pii_type: str = ""
    mode: str = ""  # "audit", "redact", "block"
    pii_field: str = ""
    action_taken: str = ""


class BudgetEnforcementEvent(Event):
    """A budget enforcement action."""

    event_type: Literal[EventType.BUDGET_ENFORCEMENT] = EventType.BUDGET_ENFORCEMENT
    limit: float = 0.0
    spent: float = 0.0
    action: str = ""  # "warn", "hard_stop"
    budget_level: str = ""  # "session", "team", "org"


class LoopDetectionEvent(Event):
    """A loop detection event."""

    event_type: Literal[EventType.LOOP_DETECTION] = EventType.LOOP_DETECTION
    pattern_hash: str = ""
    repeat_count: int = Field(default=0, ge=0)
    action: str = ""  # "warn", "flag", "circuit_break"


class FeedbackEvent(Event):
    """A feedback event for session quality tracking."""

    event_type: Literal[EventType.FEEDBACK] = EventType.FEEDBACK
    rating: str = ""
    score: float | None = None
    comment: str = ""


class ShadowDraftEvent(Event):
    """A shadow draft comparison event (local model vs cloud)."""

    event_type: Literal[EventType.SHADOW_DRAFT] = EventType.SHADOW_DRAFT
    cloud_provider: str = ""
    cloud_model: str = ""
    cloud_latency_ms: float = Field(default=0.0, ge=0)
    cloud_tokens: int = Field(default=0, ge=0)
    cloud_cost: float = Field(default=0.0, ge=0)
    local_model: str = ""
    local_latency_ms: float = Field(default=0.0, ge=0)
    local_prompt_tokens: int = Field(default=0, ge=0)
    local_completion_tokens: int = Field(default=0, ge=0)
    local_tokens: int = Field(default=0, ge=0)
    latency_ratio: float = 0.0
    cost_saved: float = Field(default=0.0, ge=0)
    shadow_status: str = ""
    error_message: str = ""
    similarity_score: float | None = None
    similarity_method: str = ""
    cloud_preview: str = ""
    local_preview: str = ""
    length_ratio: float | None = None
    prompt_preview: str = ""


class LocalRoutingEvent(Event):
    """A local routing decision event (auto-router middleware)."""

    event_type: Literal[EventType.LOCAL_ROUTING] = EventType.LOCAL_ROUTING
    original_cloud_provider: str = ""
    original_cloud_model: str = ""
    local_model: str = ""
    complexity_score: float = 0.0
    budget_pressure: float = 0.0
    routing_reason: str = ""
    routing_success: bool = False
    probed: bool = False
    probe_confidence: float | None = None
    historical_success_rate: float | None = None
    local_latency_ms: float = Field(default=0.0, ge=0)
    estimated_cloud_cost: float = Field(default=0.0, ge=0)
    semantic_complexity: float | None = None
    custom_scorer_used: bool = False


class KillSwitchEvent(Event):
    """A kill switch activation event — all LLM traffic blocked."""

    event_type: Literal[EventType.KILL_SWITCH] = EventType.KILL_SWITCH
    reason: str = "kill_switch_active"
    message: str = ""
    matched_rule: dict[str, Any] = Field(default_factory=dict)
    blocked_model: str = ""
    blocked_provider: str = ""
    webhook_fired: bool = False
    webhook_url: str = ""


class BlastRadiusEvent(Event):
    """A blast radius containment event — session paused due to repeated failures."""

    event_type: Literal[EventType.BLAST_RADIUS] = EventType.BLAST_RADIUS
    trigger: str = ""  # "consecutive_failures" or "budget_violations"
    count: int = Field(default=0, ge=0)
    threshold: int = Field(default=0, ge=0)
    action: str = "paused"
    webhook_fired: bool = False
    webhook_url: str = ""
    agent_id: str = ""


class ComplianceAuditEvent(Event):
    """A compliance audit event — tamper-proof regulatory record."""

    event_type: Literal[EventType.COMPLIANCE_AUDIT] = EventType.COMPLIANCE_AUDIT
    compliance_standard: str = ""
    action: str = ""
    legal_rule: str = ""
    justification: str = ""
    target_type: str = ""
    target_id: str = ""
    org_id: str = ""
    team_id: str = ""
    integrity_hash: str = ""


class RateLimitEvent(Event):
    """Rate limiting event — request queued, released, rejected, or timed out."""

    event_type: Literal[EventType.RATE_LIMIT] = EventType.RATE_LIMIT
    team_id: str = ""
    queued: bool = False
    wait_ms: float = Field(default=0.0, ge=0)
    rejected: bool = False
    timed_out: bool = False
    virtual_key_id: str = ""


class SemanticRetryEvent(Event):
    """A semantic retry event — LLM output failed validation."""

    event_type: Literal[EventType.SEMANTIC_RETRY] = EventType.SEMANTIC_RETRY
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=0, ge=0)
    error_type: str = ""
    error_message: str = ""
    provider: str = ""
    model: str = ""
    resolved: bool = False


class CheckpointEvent(Event):
    """A named checkpoint within a session."""

    event_type: Literal[EventType.CHECKPOINT] = EventType.CHECKPOINT
    label: str = ""
    description: str = ""


class CircuitBreakerEvent(Event):
    """A circuit breaker state change or fallback event."""

    event_type: Literal[EventType.CIRCUIT_BREAKER] = EventType.CIRCUIT_BREAKER
    provider: str = ""
    state: str = ""  # "closed", "open", "half_open"
    previous_state: str = ""
    failure_count: int = Field(default=0, ge=0)
    failure_threshold: int = Field(default=0, ge=0)
    fallback_provider: str = ""
    fallback_model: str = ""
    original_model: str = ""
    probe_success: bool | None = None


class SuspensionEvent(Event):
    """A session suspension/resumption event (human-in-the-loop)."""

    event_type: Literal[EventType.SUSPENSION] = EventType.SUSPENSION
    action: str = ""  # "suspended" or "resumed"
    reason: str = ""  # Why suspended (agent-provided)
    suspend_data: dict[str, Any] = Field(default_factory=dict)
    signal_payload: Any = None  # What the human sent back (on "resumed")
    suspended_duration_ms: float = Field(default=0.0, ge=0)


class GuardrailEvent(Event):
    """A guardrail validation event — injection/jailbreak/leak detection."""

    event_type: Literal[EventType.GUARDRAIL] = EventType.GUARDRAIL
    rule_name: str = ""
    category: str = ""
    severity: str = "medium"
    score: float = 0.0
    action_taken: str = ""
    violation_text: str = ""
    scan_phase: str = "input"
    validator_type: str = ""


class SecurityAuditEvent(Event):
    """A security audit event — CPython hook or vault operation."""

    event_type: Literal[EventType.SECURITY_AUDIT] = EventType.SECURITY_AUDIT
    audit_event: str = ""
    action_taken: str = ""
    detail: str = ""
    source: str = ""
    severity: str = "medium"
    blocked: bool = False


class AsyncJobEvent(Event):
    """An async job lifecycle event."""

    event_type: Literal[EventType.ASYNC_JOB] = EventType.ASYNC_JOB
    job_id: str = ""
    job_status: str = ""
    provider: str = ""
    model: str = ""
    webhook_url: str = ""
    webhook_status: str = ""
    error: str = ""
    processing_time_ms: float = Field(default=0.0, ge=0)


class DebateRoundEvent(Event):
    """A debate round event — one round of multi-agent consensus."""

    event_type: Literal[EventType.DEBATE_ROUND] = EventType.DEBATE_ROUND
    round_number: int = Field(default=0, ge=0)
    strategy: str = ""
    models: list[str] = Field(default_factory=list)
    responses_summary: list[dict[str, Any]] = Field(default_factory=list)
    agreement_score: float = 0.0
    consensus_reached: bool = False
    round_cost: float = Field(default=0.0, ge=0)
    round_duration_ms: float = Field(default=0.0, ge=0)


class ConsensusEvent(Event):
    """A consensus completion event — final result of multi-agent debate."""

    event_type: Literal[EventType.CONSENSUS] = EventType.CONSENSUS
    strategy: str = ""
    models: list[str] = Field(default_factory=list)
    total_rounds: int = Field(default=0, ge=0)
    final_answer_preview: str = ""
    confidence: float = 0.0
    total_cost: float = Field(default=0.0, ge=0)
    total_duration_ms: float = Field(default=0.0, ge=0)
    early_stopped: bool = False
    aggregation_method: str = ""
    winner_model: str = ""


class SessionLifecycleEvent(Event):
    """A session lifecycle event (timeout, cancellation, etc.)."""

    event_type: Literal[EventType.SESSION_LIFECYCLE] = EventType.SESSION_LIFECYCLE
    action: str = ""  # "timed_out", "cancelled", "suspended"
    reason: str = ""  # e.g. "session_timeout", "idle_timeout"
    elapsed: float = 0.0
    limit: float = 0.0


# Discriminated union for automatic subclass resolution from dicts
AnyEvent = Annotated[
    LLMCallEvent
    | ToolCallEvent
    | CacheHitEvent
    | PIIDetectionEvent
    | BudgetEnforcementEvent
    | LoopDetectionEvent
    | FeedbackEvent
    | ShadowDraftEvent
    | LocalRoutingEvent
    | KillSwitchEvent
    | BlastRadiusEvent
    | ComplianceAuditEvent
    | RateLimitEvent
    | SemanticRetryEvent
    | CheckpointEvent
    | CircuitBreakerEvent
    | SuspensionEvent
    | AsyncJobEvent
    | GuardrailEvent
    | SecurityAuditEvent
    | DebateRoundEvent
    | ConsensusEvent
    | SessionLifecycleEvent,
    Discriminator("event_type"),
]
