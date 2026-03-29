"""Shared types and enums for StateLoom."""

from __future__ import annotations

from enum import Enum


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    BUDGET_EXCEEDED = "budget_exceeded"
    LOOP_KILLED = "loop_killed"
    ERROR = "error"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD = "dead"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    HTTP_CALL = "http_call"
    CACHE_HIT = "cache_hit"
    PII_DETECTION = "pii_detection"
    BUDGET_ENFORCEMENT = "budget_enforcement"
    LOOP_DETECTION = "loop_detection"
    FEEDBACK = "feedback"
    SHADOW_DRAFT = "shadow_draft"
    LOCAL_ROUTING = "local_routing"
    KILL_SWITCH = "kill_switch"
    BLAST_RADIUS = "blast_radius"
    COMPLIANCE_AUDIT = "compliance_audit"
    RATE_LIMIT = "rate_limit"
    ASYNC_JOB = "async_job"
    SEMANTIC_RETRY = "semantic_retry"
    CHECKPOINT = "checkpoint"
    CIRCUIT_BREAKER = "circuit_breaker"
    SUSPENSION = "suspension"
    GUARDRAIL = "guardrail"
    SECURITY_AUDIT = "security_audit"
    DEBATE_ROUND = "debate_round"
    CONSENSUS = "consensus"
    SESSION_LIFECYCLE = "session_lifecycle"


class PIIMode(str, Enum):
    AUDIT = "audit"
    REDACT = "redact"
    BLOCK = "block"


class BudgetAction(str, Enum):
    WARN = "warn"
    HARD_STOP = "hard_stop"


class LoopAction(str, Enum):
    WARN = "warn"
    FLAG = "flag"
    CIRCUIT_BREAK = "circuit_break"


class FailureAction(str, Enum):
    """What to do when a security middleware fails internally."""

    BLOCK = "block"
    PASS = "pass"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    COHERE = "cohere"
    LITELLM = "litellm"
    LOCAL = "local"
    UNKNOWN = "unknown"


class BillingMode(str, Enum):
    API = "api"
    SUBSCRIPTION = "subscription"


class AgentStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    CONCLUDED = "concluded"


class AssignmentStrategy(str, Enum):
    RANDOM = "random"
    HASH = "hash"
    MANUAL = "manual"


class OrgStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class TeamStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class ComplianceStandard(str, Enum):
    NONE = "none"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CCPA = "ccpa"


class DataRegion(str, Enum):
    GLOBAL = "global"
    EU = "eu"
    US_EAST = "us-east"
    US_WEST = "us-west"
    APAC = "apac"


class FeedbackRating(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class GuardrailMode(str, Enum):
    AUDIT = "audit"
    ENFORCE = "enforce"


class ConsensusStrategy(str, Enum):
    VOTE = "vote"
    DEBATE = "debate"
    MOA = "moa"
    SELF_CONSISTENCY = "self_consistency"


class Role(str, Enum):
    ORG_ADMIN = "org_admin"
    TEAM_ADMIN = "team_admin"
    TEAM_EDITOR = "team_editor"
    TEAM_VIEWER = "team_viewer"
    ORG_AUDITOR = "org_auditor"
