"""REST API endpoints for the StateLoom dashboard."""

from __future__ import annotations

import asyncio
import json as json_module
import logging
import os
import queue
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from stateloom.core.types import ActionTaken, BudgetAction, PIIMode

logger = logging.getLogger("stateloom.dashboard.api")

if TYPE_CHECKING:
    from stateloom.gate import Gate


class FeedbackRequest(BaseModel):
    rating: str = Field(..., max_length=32)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    comment: str = Field(default="", max_length=10000)


class ConfigUpdate(BaseModel):
    default_model: str | None = Field(default=None, max_length=256)
    shadow_enabled: bool | None = None
    shadow_model: str | None = Field(default=None, max_length=256)
    shadow_sample_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    shadow_max_context_tokens: int | None = Field(default=None, ge=256, le=1000000)
    shadow_models: list[str] | None = None
    shadow_similarity_method: str | None = Field(default=None, max_length=32)
    local_model_enabled: bool | None = None
    local_model_default: str | None = Field(default=None, max_length=256)
    pii_enabled: bool | None = None
    pii_default_mode: str | None = Field(default=None, max_length=32)
    budget_per_session: float | None = None
    budget_global: float | None = None
    budget_action: str | None = Field(default=None, max_length=32)
    cache_max_size: int | None = Field(default=None, ge=1, le=100000)
    cache_ttl_seconds: int | None = Field(default=None, ge=0, le=86400)
    cache_semantic_enabled: bool | None = None
    cache_similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    cache_scope: str | None = Field(default=None, max_length=32)
    loop_detection_enabled: bool | None = None
    loop_exact_threshold: int | None = Field(default=None, ge=0, le=1000)
    store_retention_days: int | None = Field(default=None, ge=1, le=3650)
    console_verbose: bool | None = None
    auto_route_enabled: bool | None = None
    auto_route_force_local: bool | None = None
    auto_route_model: str | None = Field(default=None, max_length=256)
    auto_route_complexity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    auto_route_probe_enabled: bool | None = None
    kill_switch_active: bool | None = None
    kill_switch_message: str | None = Field(default=None, max_length=1000)
    kill_switch_response_mode: str | None = Field(default=None, max_length=32)
    blast_radius_enabled: bool | None = None
    blast_radius_consecutive_failures: int | None = Field(default=None, ge=1, le=1000)
    blast_radius_budget_violations_per_hour: int | None = Field(default=None, ge=1, le=10000)
    provider_api_key_openai: str | None = Field(default=None, max_length=256)
    provider_api_key_anthropic: str | None = Field(default=None, max_length=256)
    provider_api_key_google: str | None = Field(default=None, max_length=256)


class PullRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)


class HotSwapRequest(BaseModel):
    new_model: str = Field(..., min_length=1, max_length=256)
    delete_old: bool = True


class CreateOrgRequest(BaseModel):
    name: str = Field(default="", max_length=256)
    budget: float | None = None
    pii_rules: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateOrgRequest(BaseModel):
    name: str | None = Field(default=None, max_length=256)
    budget: float | None = None
    pii_rules: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


class CreateTeamRequest(BaseModel):
    org_id: str = Field(..., min_length=1, max_length=256)
    name: str = Field(default="", max_length=256)
    budget: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateTeamRequest(BaseModel):
    name: str | None = Field(default=None, max_length=256)
    budget: float | None = None
    metadata: dict[str, Any] | None = None


class UpdateTeamRateLimitRequest(BaseModel):
    rate_limit_tps: float | None = Field(default=None, ge=0.1, le=10000)
    rate_limit_priority: int | None = Field(default=None, ge=0, le=100)
    rate_limit_max_queue: int | None = Field(default=None, ge=1, le=10000)
    rate_limit_queue_timeout: float | None = Field(default=None, ge=1.0, le=300.0)


class SetComplianceProfileRequest(BaseModel):
    standard: str = Field(..., min_length=1, max_length=32)
    region: str = Field(default="global", max_length=32)
    session_ttl_days: int = Field(default=0, ge=0)
    cache_ttl_seconds: int = Field(default=0, ge=0)
    zero_retention_logs: bool = False
    block_local_routing: bool = False
    block_shadow: bool = False
    allowed_endpoints: list[str] = Field(default_factory=list)
    audit_salt: str = Field(default="", max_length=256)
    default_consent: bool = False


class PurgeRequest(BaseModel):
    user_identifier: str = Field(..., min_length=1, max_length=1024)
    standard: str = Field(default="gdpr", max_length=32)


class SubmitJobRequest(BaseModel):
    provider: str = Field(default="", max_length=64)
    model: str = Field(default="", max_length=256)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    request_kwargs: dict[str, Any] = Field(default_factory=dict)
    webhook_url: str = Field(default="", max_length=2048)
    webhook_secret: str = Field(default="", max_length=256)
    session_id: str = Field(default="", max_length=256)
    org_id: str = Field(default="", max_length=256)
    team_id: str = Field(default="", max_length=256)
    max_retries: int = Field(default=3, ge=0, le=10)
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateVirtualKeyRequest(BaseModel):
    team_id: str = Field(..., min_length=1, max_length=256)
    name: str = Field(default="", max_length=256)
    scopes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    rate_limit_tps: float | None = Field(default=None, ge=0.1, le=10000)
    rate_limit_max_queue: int = Field(default=100, ge=1, le=10000)
    rate_limit_queue_timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    agent_ids: list[str] = Field(default_factory=list)


class CreateAgentRequest(BaseModel):
    slug: str = Field(
        ..., min_length=3, max_length=64, pattern=r"^[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?$"
    )
    team_id: str = Field(..., min_length=1)
    name: str = Field(default="", max_length=256)
    description: str = Field(default="", max_length=4096)
    model: str = Field(..., min_length=1)
    system_prompt: str = Field(default="", max_length=65536)
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    budget_per_session: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateAgentVersionRequest(BaseModel):
    model: str = Field(..., min_length=1)
    system_prompt: str = Field(default="", max_length=65536)
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    budget_per_session: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_by: str = Field(default="", max_length=256)


class LockSettingRequest(BaseModel):
    setting: str = Field(..., min_length=1, max_length=128)
    value: Any = None
    reason: str = Field(default="", max_length=500)


class ImportSessionRequest(BaseModel):
    bundle: dict[str, Any]
    session_id_override: str | None = None


class UpdateAgentRequest(BaseModel):
    name: str | None = Field(default=None, max_length=256)
    description: str | None = Field(default=None, max_length=4096)
    status: str | None = None
    metadata: dict[str, Any] | None = None


class ShadowConfigureRequest(BaseModel):
    enabled: bool | None = None
    model: str | None = Field(default=None, max_length=256)
    sample_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_context_tokens: int | None = Field(default=None, ge=256, le=1000000)
    models: list[str] | None = None


class VariantConfigRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    weight: float = Field(default=1.0, ge=0.0)
    model: str | None = Field(default=None, max_length=256)
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_version_id: str | None = Field(default=None, max_length=256)


class CreateExperimentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    variants: list[VariantConfigRequest] = Field(..., min_length=1)
    strategy: str = Field(default="random", max_length=32)
    description: str = Field(default="", max_length=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_id: str | None = Field(default=None, max_length=256)


class UpdateExperimentRequest(BaseModel):
    name: str | None = Field(default=None, max_length=256)
    description: str | None = Field(default=None, max_length=4096)
    variants: list[VariantConfigRequest] | None = None
    strategy: str | None = Field(default=None, max_length=32)
    metadata: dict[str, Any] | None = None
    agent_id: str | None = Field(default=None, max_length=256)


class KillSwitchActivateRequest(BaseModel):
    message: str | None = None
    response_mode: str | None = None


class KillSwitchRuleRequest(BaseModel):
    model: str | None = Field(default=None, max_length=256)
    provider: str | None = Field(default=None, max_length=64)
    environment: str | None = Field(default=None, max_length=128)
    agent_version: str | None = Field(default=None, max_length=64)
    message: str = Field(default="", max_length=1000)
    reason: str = Field(default="", max_length=1000)


class KillSwitchRulesReplaceRequest(BaseModel):
    rules: list[KillSwitchRuleRequest]


class PIIRuleRequest(BaseModel):
    pattern: str
    mode: str = "audit"
    on_middleware_failure: str | None = None


class PIIRulesReplaceRequest(BaseModel):
    rules: list[PIIRuleRequest]


class UpdateVKRateLimitRequest(BaseModel):
    rate_limit_tps: float | None = Field(default=None, ge=0.1, le=10000)
    rate_limit_max_queue: int | None = Field(default=None, ge=1, le=10000)
    rate_limit_queue_timeout: float | None = Field(default=None, ge=1.0, le=300.0)


@dataclass
class _HotSwapState:
    """In-memory hot-swap operation state (single operation at a time)."""

    active: bool = False
    new_model: str = ""
    old_model: str = ""
    phase: str = ""  # "pulling" | "switching" | "cleaning" | "complete" | "error"
    progress_pct: float = 0.0
    error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)


def create_api_router(gate: Gate) -> APIRouter:
    """Create the API router with all dashboard endpoints."""
    from fastapi import Depends

    from stateloom.auth.dependencies import require_feature

    router = APIRouter()

    # Feature gate dependencies
    _registry = getattr(gate, "_feature_registry", None)
    _mt_deps: list[Any] = []
    if _registry is not None:
        _mt_deps = [Depends(require_feature(_registry, "multi_tenant"))]

    _shadow_deps: list[Any] = []
    if _registry is not None:
        _shadow_deps = [Depends(require_feature(_registry, "model_testing"))]

    # In-memory pull progress tracking
    _pull_progress: dict[str, dict[str, Any]] = {}
    _pull_lock = threading.Lock()

    # In-memory hot-swap state
    _hot_swap = _HotSwapState()

    def _toggle_auto_route(enabled: bool) -> None:
        """Dynamically insert/remove AutoRouterMiddleware from the pipeline."""
        if enabled and not gate.config.auto_route_enabled:
            from stateloom.middleware.auto_router import AutoRouterMiddleware
            from stateloom.middleware.event_recorder import EventRecorder

            gate.config.auto_route_enabled = True
            if not gate.config.local_model_enabled:
                gate.config.local_model_enabled = True
            mw = AutoRouterMiddleware(gate.config, gate.store, pricing=gate.pricing)
            gate._auto_router = mw
            idx = next(
                (
                    i
                    for i, m in enumerate(gate.pipeline.middlewares)
                    if isinstance(m, EventRecorder)
                ),
                len(gate.pipeline.middlewares),
            )
            gate.pipeline.insert(idx, mw)
        elif not enabled and gate.config.auto_route_enabled:
            gate.config.auto_route_enabled = False
            gate.config.auto_route_force_local = False
            if hasattr(gate, "_auto_router") and gate._auto_router is not None:
                gate.pipeline.remove(gate._auto_router)
                gate._auto_router.shutdown()
                gate._auto_router = None  # type: ignore[assignment]
        else:
            gate.config.auto_route_enabled = enabled

    def _reload_pii_middleware() -> None:
        """Reload PII rules in the running middleware after config mutation."""
        for mw in gate.pipeline.middlewares:
            if hasattr(mw, "reload_rules"):
                mw.reload_rules()

    @router.get("/health")
    async def health() -> dict[str, Any]:
        ready = True
        checks: dict[str, str] = {}

        # Store check
        try:
            gate.store.get_global_stats()
            checks["store"] = "ok"
        except Exception:
            checks["store"] = "error"
            ready = False

        store_path: str | None = None
        if hasattr(gate.store, "_path"):
            store_path = os.path.abspath(gate.store._path)

        return {
            "status": "ok",
            "ready": ready,
            "version": "0.1.0",
            "api_version": "1",
            "checks": checks,
            "store_path": store_path,
        }

    @router.get("/features")
    async def feature_status() -> dict[str, Any]:
        """List all features with tier, availability, and upgrade hints."""
        if _registry is None:
            return {"features": [], "summary": {"community": 0, "enterprise": 0, "unlocked": 0}}

        status = _registry.status()
        features = []
        community_count = 0
        enterprise_count = 0
        unlocked_count = 0
        for name, info in status["features"].items():
            tier = info["tier"]
            enabled = info["enabled"]
            entry: dict[str, Any] = {
                "name": name,
                "tier": tier,
                "enabled": enabled,
                "description": info.get("description", ""),
            }
            if tier == "community":
                community_count += 1
            else:
                enterprise_count += 1
                if enabled:
                    unlocked_count += 1
                else:
                    entry["upgrade_hint"] = (
                        "Requires an enterprise license. "
                        "Set STATELOOM_LICENSE_KEY or use STATELOOM_ENV=development."
                    )
            features.append(entry)

        return {
            "features": features,
            "summary": {
                "community": community_count,
                "enterprise": enterprise_count,
                "unlocked": unlocked_count,
            },
        }

    @router.get("/sessions")
    async def list_sessions(
        limit: int = Query(default=50, le=500),
        offset: int = Query(default=0, ge=0),
        status: str | None = Query(default=None),
        org_id: str | None = Query(default=None),
        team_id: str | None = Query(default=None),
        end_user: str | None = Query(default=None),
    ) -> dict[str, Any]:
        sessions = gate.store.list_sessions(
            limit=limit,
            offset=offset,
            status=status,
            org_id=org_id,
            team_id=team_id,
            end_user=end_user,
        )
        # Filter out sessions that contain only CLI-internal calls
        # (e.g. Gemini CLI init requests).
        visible = [s for s in sessions if not s.metadata.get("_cli_internal_only")]
        total = gate.store.count_sessions(
            status=status,
            org_id=org_id,
            team_id=team_id,
            end_user=end_user,
        )
        return {
            "sessions": [_session_to_dict(gate, s) for s in visible],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        session = gate.store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        result = {
            "id": session.id,
            "name": session.name,
            "org_id": session.org_id,
            "team_id": session.team_id,
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "status": session.status.value,
            "total_cost": round(session.total_cost, 6),
            "estimated_api_cost": round(session.estimated_api_cost, 6),
            "total_tokens": session.total_tokens,
            "total_prompt_tokens": session.total_prompt_tokens,
            "total_completion_tokens": session.total_completion_tokens,
            "call_count": session.call_count,
            "cache_hits": session.cache_hits,
            "cache_savings": round(session.cache_savings, 6),
            "pii_detections": session.pii_detections,
            "guardrail_detections": session.guardrail_detections,
            "budget": session.budget,
            "step_counter": session.step_counter,
            "parent_session_id": session.parent_session_id,
            "timeout": session.timeout,
            "idle_timeout": session.idle_timeout,
            "end_user": session.end_user,
            "cost_by_model": {k: round(v, 6) for k, v in session.cost_by_model.items()},
            "tokens_by_model": session.tokens_by_model,
        }
        # Include experiment info from metadata
        if session.metadata:
            result["experiment_id"] = session.metadata.get("experiment_id")
            result["variant"] = session.metadata.get("variant")
            result["durable"] = session.metadata.get("durable", False)
            result["suspend_reason"] = session.metadata.get("_suspend_reason", "")
            result["suspend_data"] = session.metadata.get("_suspend_data")
            result["billing_mode"] = session.metadata.get("billing_mode", "api")

        # Aggregate child session stats for parent visibility
        children = gate.store.list_child_sessions(session_id, limit=500)
        if children:
            child_cost = sum(c.total_cost for c in children)
            child_tokens = sum(c.total_tokens for c in children)
            child_prompt = sum(c.total_prompt_tokens for c in children)
            child_completion = sum(c.total_completion_tokens for c in children)
            child_calls = sum(c.call_count for c in children)
            result["total_cost_with_children"] = round(session.total_cost + child_cost, 6)
            result["total_tokens_with_children"] = session.total_tokens + child_tokens
            result["total_prompt_tokens_with_children"] = session.total_prompt_tokens + child_prompt
            result["total_completion_tokens_with_children"] = (
                session.total_completion_tokens + child_completion
            )
            result["call_count_with_children"] = session.call_count + child_calls

            # Merge per-model breakdowns from children
            merged_cost_by_model: dict[str, float] = dict(session.cost_by_model)
            merged_tokens_by_model: dict[str, dict[str, int]] = {
                k: dict(v) for k, v in session.tokens_by_model.items()
            }
            for child in children:
                for model, cost in child.cost_by_model.items():
                    merged_cost_by_model[model] = merged_cost_by_model.get(model, 0.0) + cost
                for model, toks in child.tokens_by_model.items():
                    existing = merged_tokens_by_model.get(
                        model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    )
                    merged_tokens_by_model[model] = {
                        "prompt_tokens": existing["prompt_tokens"] + toks.get("prompt_tokens", 0),
                        "completion_tokens": existing["completion_tokens"]
                        + toks.get("completion_tokens", 0),
                        "total_tokens": existing["total_tokens"] + toks.get("total_tokens", 0),
                    }
            result["cost_by_model"] = {k: round(v, 6) for k, v in merged_cost_by_model.items()}
            result["tokens_by_model"] = merged_tokens_by_model

        return result

    @router.get("/sessions/{session_id}/events")
    async def get_session_events(
        session_id: str,
        event_type: str | None = Query(default=None),
        exclude_types: str | None = Query(
            default=None,
            description="Comma-separated event types to exclude (e.g. 'compliance_audit')",
        ),
        primary_only: bool = Query(
            default=False,
            description="When true, exclude tool-continuation and CLI-internal LLM calls, "
            "attaching aggregated _tool_summary to parent events instead.",
        ),
        limit: int = Query(default=1000, le=5000),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        # When primary_only, fetch a larger batch internally to compute summaries
        if primary_only:
            fetch_limit = limit * 5
        elif exclude_types:
            fetch_limit = limit * 3
        else:
            # Fetch one extra to detect has_more
            fetch_limit = limit + 1
        events = gate.store.get_session_events(
            session_id,
            event_type=event_type,
            limit=fetch_limit,
            offset=offset,
        )
        if exclude_types:
            excluded = {t.strip() for t in exclude_types.split(",")}
            events = [e for e in events if e.event_type.value not in excluded]

        if primary_only:
            primary_events, tool_summaries = _compute_tool_summaries(events)  # type: ignore[arg-type]
            has_more = len(primary_events) > limit
            primary_events = primary_events[:limit]
            result = []
            for e in primary_events:
                d = _event_to_dict(e)
                step = e.step  # type: ignore[attr-defined]
                if step in tool_summaries:
                    d["_tool_summary"] = tool_summaries[step]
                result.append(d)
            return {"events": result, "total": len(result), "has_more": has_more}

        has_more = len(events) > limit
        events = events[:limit]
        return {
            "events": [_event_to_dict(e) for e in events],
            "total": len(events),
            "has_more": has_more,
        }

    @router.get("/sessions/{session_id}/events/tool-steps")
    async def get_tool_steps(
        session_id: str,
        parent_step: int = Query(..., description="Step number of the parent user-prompt event"),
        limit: int = Query(default=200, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """Return tool-continuation and ToolCall events that follow a parent step.

        Collects both ``is_tool_continuation=True`` LLM calls (proxy tool-use)
        and ``ToolCallEvent``s (framework callback tool tracking) that
        immediately follow the parent LLM call step.
        """
        from stateloom.core.event import LLMCallEvent, ToolCallEvent

        events = gate.store.get_session_events(session_id, limit=10000)
        collecting = False
        all_tool_steps: list[object] = []
        for e in events:
            is_llm = isinstance(e, LLMCallEvent)
            if not collecting:
                # Find the parent LLM call (not compliance_audit or other events
                # that may share the same step number).
                if is_llm and e.step == parent_step:
                    collecting = True
                continue
            # Skip non-LLM, non-ToolCall events (compliance_audit, checkpoint,
            # etc.) that are interleaved between tool steps.
            is_tool = isinstance(e, ToolCallEvent)
            if not is_llm and not is_tool:
                continue
            is_tc = is_llm and getattr(e, "is_tool_continuation", False)
            is_cli = is_llm and getattr(e, "is_cli_internal", False)
            if is_tc or is_tool:
                all_tool_steps.append(e)
            elif is_cli:
                continue  # skip CLI-internal events interleaved between tool steps
            else:
                break  # reached next non-tool LLM event
        # Apply offset and limit, with has_more detection
        page = all_tool_steps[offset : offset + limit + 1]
        has_more = len(page) > limit
        page = page[:limit]
        return {"events": [_event_to_dict(e) for e in page], "has_more": has_more}

    @router.get("/events/{event_id}/messages")
    async def get_event_messages(event_id: str) -> dict[str, Any]:
        """Lazy-load the full request messages for a single event."""
        raw = gate.store.get_event_messages(event_id)
        if raw is None:
            return {"messages": None}
        try:
            return {"messages": json_module.loads(raw)}
        except (json_module.JSONDecodeError, TypeError):
            return {"messages": None}

    @router.get("/sessions/{session_id}/children")
    async def get_session_children(
        session_id: str,
        limit: int = Query(default=100, le=500),
    ) -> dict[str, Any]:
        children = gate.store.list_child_sessions(session_id, limit=limit)
        return {
            "children": [_session_to_dict(gate, s) for s in children],
            "total": len(children),
        }

    @router.post("/sessions/{session_id}/cancel")
    async def cancel_session(session_id: str) -> dict[str, Any]:
        result = gate.cancel_session(session_id)
        if not result:
            session = gate.store.get_session(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            raise HTTPException(
                status_code=400,
                detail=f"Session is not active (status: {session.status.value})",
            )
        return {"status": "cancelled", "session_id": session_id}

    @router.post("/sessions/{session_id}/end")
    async def end_session(session_id: str) -> dict[str, Any]:
        result = gate.end_session(session_id)
        if not result:
            session = gate.store.get_session(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            raise HTTPException(
                status_code=400,
                detail=f"Session is not active or suspended (status: {session.status.value})",
            )
        return {"status": "completed", "session_id": session_id}

    @router.post("/sessions/{session_id}/suspend")
    async def suspend_session(session_id: str, request: Request) -> dict[str, Any]:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        reason = body.get("reason", "")
        data = body.get("data")
        result = gate.suspend_session(session_id, reason=reason, data=data)
        if not result:
            session = gate.store.get_session(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            raise HTTPException(
                status_code=400,
                detail=f"Session is not active (status: {session.status.value})",
            )
        return {"status": "suspended", "session_id": session_id}

    @router.post("/sessions/{session_id}/signal")
    async def signal_session(session_id: str, request: Request) -> dict[str, Any]:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        payload = body.get("payload")
        result = gate.signal_session(session_id, payload)
        if not result:
            session = gate.session_manager.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            raise HTTPException(
                status_code=400,
                detail=f"Session is not suspended (status: {session.status.value})",
            )
        return {"status": "resumed", "session_id": session_id}

    @router.patch("/sessions/{session_id}")
    async def update_session(session_id: str, request: Request) -> dict[str, Any]:
        session = gate.store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        body = await request.json()
        if "budget" in body:
            val = body["budget"]
            if val is not None:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    raise HTTPException(status_code=422, detail="budget must be a number or null")
                if val < 0:
                    raise HTTPException(status_code=422, detail="budget must be non-negative")
            session.budget = val
            gate.store.save_session(session)
            # Also update in-memory session if it is live in the session manager
            live = gate.session_manager.get(session_id)
            if live is not None:
                live.budget = val
        return _session_to_dict(gate, session)

    @router.get("/stats")
    async def get_stats() -> dict[str, Any]:
        stats = gate.store.get_global_stats()
        call_counts = gate.store.get_call_counts()
        stats["cloud_calls"] = call_counts["cloud_calls"]
        stats["local_calls"] = call_counts["local_calls"]
        stats["total_estimated_api_cost"] = stats.get("total_estimated_api_cost", 0.0)
        stats["username"] = gate.config.username
        return stats

    @router.get("/stats/cost-by-model")
    async def get_cost_by_model() -> dict[str, Any]:
        return {"models": gate.store.get_cost_by_model()}

    @router.get("/models/cloud")
    async def get_cloud_models() -> dict[str, Any]:
        """Return supported cloud models from the pricing registry."""
        models: list[dict[str, Any]] = []
        for model_id in sorted(gate.pricing._prices):
            if (
                model_id.startswith("gpt-")
                or model_id.startswith("o1")
                or model_id.startswith("o3")
            ):
                provider = "openai"
            elif model_id.startswith("claude"):
                provider = "anthropic"
            elif model_id.startswith("gemini"):
                provider = "google"
            else:
                provider = "other"
            price = gate.pricing._prices[model_id]
            models.append(
                {
                    "model": model_id,
                    "provider": provider,
                    "input_per_1k": round(price.input_per_token * 1000, 4),
                    "output_per_1k": round(price.output_per_token * 1000, 4),
                }
            )
        return {"models": models}

    @router.get("/pii")
    async def get_pii_detections(
        limit: int = Query(default=100, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        # Paginated detections (newest first)
        events = gate.store.get_session_events(
            "",
            event_type="pii_detection",
            limit=limit,
            offset=offset,
            desc=True,
        )

        # True total count via efficient COUNT(*)
        total = gate.store.count_events(event_type="pii_detection")

        # Aggregate stats across ALL events (not just this page)
        stats = gate.store.get_pii_stats()

        return {
            "detections": [
                {
                    "id": e.id,
                    "session_id": e.session_id,
                    "timestamp": e.timestamp.isoformat(),
                    "pii_type": getattr(e, "pii_type", ""),
                    "mode": getattr(e, "mode", ""),
                    "field": getattr(e, "pii_field", ""),
                    "action": getattr(e, "action_taken", ""),
                    "redacted_preview": e.metadata.get("redacted_preview", ""),
                    "match_length": e.metadata.get("match_length", 0),
                }
                for e in events
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
            "by_type": stats["by_type"],
            "by_action": stats["by_action"],
            "sessions_affected": stats["sessions_affected"],
        }

    # --- Experiment endpoints ---

    @router.get("/experiments")
    async def list_experiments(
        status: str | None = Query(default=None),
    ) -> dict[str, Any]:
        experiments = gate.store.list_experiments(status=status)
        return {
            "experiments": [
                {
                    "id": e.id,
                    "name": e.name,
                    "description": e.description,
                    "status": e.status.value,
                    "strategy": e.strategy.value,
                    "variant_count": len(e.variants),
                    "assignment_counts": e.assignment_counts,
                    "created_at": e.created_at.isoformat(),
                    "updated_at": e.updated_at.isoformat(),
                }
                for e in experiments
            ],
            "total": len(experiments),
        }

    @router.get("/experiments/{experiment_id}")
    async def get_experiment(experiment_id: str) -> dict[str, Any]:
        experiment = gate.store.get_experiment(experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail="Experiment not found")

        metrics = gate.experiment_manager.get_metrics(experiment_id)
        return {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status.value,
            "strategy": experiment.strategy.value,
            "variants": [v.to_dict() for v in experiment.variants],
            "assignment_counts": experiment.assignment_counts,
            "metadata": experiment.metadata,
            "agent_id": experiment.agent_id,
            "created_at": experiment.created_at.isoformat(),
            "updated_at": experiment.updated_at.isoformat(),
            "metrics": metrics,
        }

    @router.get("/experiments/{experiment_id}/metrics")
    async def get_experiment_metrics(experiment_id: str) -> dict[str, Any]:
        return gate.experiment_manager.get_metrics(experiment_id)

    @router.get("/leaderboard")
    async def get_leaderboard() -> dict[str, Any]:
        entries = gate.experiment_manager.get_leaderboard()
        return {"entries": entries}

    @router.post("/experiments")
    async def create_experiment(body: CreateExperimentRequest) -> dict[str, Any]:
        try:
            experiment = gate.experiment_manager.create_experiment(
                name=body.name,
                variants=[v.model_dump() for v in body.variants],
                strategy=body.strategy,
                description=body.description,
                metadata=body.metadata,
                agent_id=body.agent_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status.value,
            "strategy": experiment.strategy.value,
            "variants": [v.to_dict() for v in experiment.variants],
            "agent_id": experiment.agent_id,
            "created_at": experiment.created_at.isoformat(),
        }

    @router.patch("/experiments/{experiment_id}")
    async def update_experiment(
        experiment_id: str,
        body: UpdateExperimentRequest,
    ) -> dict[str, Any]:
        try:
            experiment = gate.experiment_manager.update_experiment(
                experiment_id,
                name=body.name,
                description=body.description,
                variants=[v.model_dump() for v in body.variants] if body.variants else None,
                strategy=body.strategy,
                metadata=body.metadata,
                agent_id=body.agent_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {
            "id": experiment.id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status.value,
            "strategy": experiment.strategy.value,
            "variants": [v.to_dict() for v in experiment.variants],
            "agent_id": experiment.agent_id,
            "metadata": experiment.metadata,
            "updated_at": experiment.updated_at.isoformat(),
        }

    @router.post("/experiments/{experiment_id}/start")
    async def start_experiment(experiment_id: str) -> dict[str, Any]:
        try:
            experiment = gate.experiment_manager.start_experiment(experiment_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"id": experiment.id, "status": experiment.status.value}

    @router.post("/experiments/{experiment_id}/pause")
    async def pause_experiment(experiment_id: str) -> dict[str, Any]:
        try:
            experiment = gate.experiment_manager.pause_experiment(experiment_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"id": experiment.id, "status": experiment.status.value}

    @router.post("/experiments/{experiment_id}/conclude")
    async def conclude_experiment(experiment_id: str) -> dict[str, Any]:
        try:
            metrics = gate.experiment_manager.conclude_experiment(experiment_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"id": experiment_id, "status": "concluded", "metrics": metrics}

    @router.post("/sessions/{session_id}/feedback")
    async def post_feedback(session_id: str, body: FeedbackRequest) -> dict[str, str]:
        gate.feedback(
            session_id=session_id,
            rating=body.rating,
            score=body.score,
            comment=body.comment,
        )
        return {"status": "ok"}

    @router.get("/sessions/{session_id}/feedback")
    async def get_feedback(session_id: str) -> dict[str, Any]:
        fb = gate.store.get_feedback(session_id)
        if fb is None:
            return {"feedback": None}
        return {
            "feedback": {
                "session_id": fb.session_id,
                "rating": fb.rating,
                "score": fb.score,
                "comment": fb.comment,
                "created_at": fb.created_at.isoformat(),
            }
        }

    # --- Session export/import ---

    @router.get("/sessions/{session_id}/export")
    async def export_session_bundle(
        session_id: str,
        include_children: bool = Query(default=False),
        scrub_pii: bool = Query(default=False),
    ) -> Any:
        from fastapi.responses import Response

        session = gate.store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        bundle = gate.export_session(
            session_id,
            include_children=include_children,
            scrub_pii=scrub_pii,
        )
        content = json_module.dumps(bundle, indent=2)
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="session-{session_id}.json"',
            },
        )

    @router.post("/sessions/import")
    async def import_session_bundle(request: ImportSessionRequest) -> dict[str, Any]:
        from stateloom.core.errors import StateLoomError as _AGError

        try:
            session = gate.import_session(
                request.bundle,
                session_id_override=request.session_id_override,
            )
            return {
                "status": "imported",
                "session_id": session.id,
                "step_counter": session.step_counter,
            }
        except _AGError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # --- Config endpoints ---

    def _config_dict() -> dict[str, Any]:
        """Return the current config as a dict (shared by GET and PATCH)."""
        return {
            "default_model": gate.config.default_model,
            "local_model_enabled": gate.config.local_model_enabled,
            "local_model_host": gate.config.local_model_host,
            "local_model_default": gate.config.local_model_default,
            "shadow_enabled": gate.config.shadow.enabled,
            "shadow_model": gate.config.shadow.model,
            "shadow_sample_rate": gate.config.shadow.sample_rate,
            "shadow_max_context_tokens": gate.config.shadow.max_context_tokens,
            "shadow_models": gate.config.shadow.models,
            "shadow_similarity_method": gate.config.shadow_similarity_method,
            "pii_enabled": gate.config.pii.enabled,
            "pii_default_mode": gate.config.pii.default_mode.value,
            "pii_rules": [r.model_dump() for r in gate.config.pii.rules],
            "budget_per_session": gate.config.budget_per_session,
            "budget_global": gate.config.budget_global,
            "budget_action": gate.config.budget_action.value,
            "cache_enabled": gate.config.cache.enabled,
            "cache_max_size": gate.config.cache.max_size,
            "cache_ttl_seconds": gate.config.cache.ttl_seconds,
            "cache_backend": gate.config.cache.backend,
            "cache_scope": gate.config.cache.scope,
            "cache_semantic_enabled": gate.config.cache.semantic_enabled,
            "cache_similarity_threshold": gate.config.cache.similarity_threshold,
            "cache_embedding_model": gate.config.cache.embedding_model,
            "loop_detection_enabled": gate.config.loop_detection_enabled,
            "loop_exact_threshold": gate.config.loop_exact_threshold,
            "store_retention_days": gate.config.store_retention_days,
            "console_output": gate.config.console_output,
            "console_verbose": gate.config.console_verbose,
            "auto_route_enabled": gate.config.auto_route.enabled,
            "auto_route_force_local": gate.config.auto_route.force_local,
            "auto_route_model": gate.config.auto_route.model,
            "auto_route_complexity_threshold": gate.config.auto_route.complexity_threshold,
            "auto_route_probe_enabled": gate.config.auto_route.probe_enabled,
            "kill_switch_active": gate.config.kill_switch_active,
            "kill_switch_message": gate.config.kill_switch_message,
            "kill_switch_response_mode": gate.config.kill_switch_response_mode,
            "blast_radius_enabled": gate.config.blast_radius.enabled,
            "blast_radius_consecutive_failures": (gate.config.blast_radius.consecutive_failures),
            "blast_radius_budget_violations_per_hour": (
                gate.config.blast_radius.budget_violations_per_hour
            ),
            "_locked": [lk["setting"] for lk in gate.list_locked_settings()],
        }

    @router.get("/config")
    async def get_config() -> dict[str, Any]:
        return _config_dict()

    @router.patch("/config")
    async def patch_config(body: ConfigUpdate) -> dict[str, Any]:
        # Enforce admin locks — reject changes to locked settings
        locks = {lk["setting"]: lk for lk in gate.list_locked_settings()}
        for field_name, value in body.model_dump(exclude_unset=True).items():
            if field_name in locks:
                locked_value = json_module.loads(locks[field_name]["value"])
                if value != locked_value:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Setting '{field_name}' is admin-locked. Contact admin to unlock.",
                    )

        if body.default_model is not None:
            if _registry and not _registry.is_available("model_override"):
                raise HTTPException(
                    status_code=403,
                    detail="Emergency model override requires an enterprise license.",
                )
            gate.config.default_model = body.default_model
        if body.local_model_enabled is not None:
            gate.config.local_model_enabled = body.local_model_enabled
        if body.local_model_default is not None:
            gate.config.local_model_default = body.local_model_default

        # Gate shadow fields behind enterprise license
        _shadow_fields = {
            "shadow_enabled",
            "shadow_model",
            "shadow_sample_rate",
            "shadow_max_context_tokens",
            "shadow_models",
            "shadow_similarity_method",
        }
        submitted = body.model_dump(exclude_unset=True)
        if _registry is not None and not _registry.is_available("model_testing"):
            if _shadow_fields & submitted.keys():
                raise HTTPException(
                    status_code=403,
                    detail="Feature 'model_testing' requires an enterprise license.",
                )

        if body.shadow_model is not None:
            gate.config.shadow_model = body.shadow_model
        if body.shadow_sample_rate is not None:
            gate.config.shadow_sample_rate = body.shadow_sample_rate
        if body.shadow_max_context_tokens is not None:
            gate.config.shadow_max_context_tokens = body.shadow_max_context_tokens
        if body.shadow_models is not None:
            gate.config.shadow_models = body.shadow_models
        if body.shadow_similarity_method is not None:
            if body.shadow_similarity_method in ("auto", "semantic", "difflib"):
                gate.config.shadow_similarity_method = body.shadow_similarity_method

        if body.shadow_enabled is not None:
            if body.shadow_enabled and not gate.config.shadow_enabled:
                # Turn shadow ON: lazy-create middleware and insert at position 1
                from stateloom.middleware.shadow import ShadowMiddleware

                gate.config.shadow_enabled = True
                mw = ShadowMiddleware(gate.config, gate.store)
                gate._shadow_middleware = mw
                gate.pipeline.insert(1, mw)
            elif not body.shadow_enabled and gate.config.shadow_enabled:
                # Turn shadow OFF: remove middleware, shut it down
                gate.config.shadow_enabled = False
                if hasattr(gate, "_shadow_middleware"):
                    gate.pipeline.remove(gate._shadow_middleware)
                    gate._shadow_middleware.shutdown()
                    del gate._shadow_middleware

        # PII settings
        if body.pii_enabled is not None:
            if body.pii_enabled and not gate.config.pii_enabled:
                from stateloom.middleware.pii_scanner import PIIScannerMiddleware

                gate.config.pii_enabled = True
                mw = PIIScannerMiddleware(gate.config, gate.store)  # type: ignore[assignment]
                gate._pii_scanner = mw  # type: ignore[attr-defined]
                # Insert after kill switch and blast radius, before cache
                from stateloom.middleware.cache import CacheMiddleware

                idx = next(
                    (
                        i
                        for i, m in enumerate(gate.pipeline.middlewares)
                        if isinstance(m, CacheMiddleware)
                    ),
                    len(gate.pipeline.middlewares),
                )
                gate.pipeline.insert(idx, mw)
            elif not body.pii_enabled and gate.config.pii_enabled:
                gate.config.pii_enabled = False
                if hasattr(gate, "_pii_scanner") and gate._pii_scanner is not None:
                    gate.pipeline.remove(gate._pii_scanner)
                    gate._pii_scanner = None
        if body.pii_default_mode is not None:
            gate.config.pii_default_mode = PIIMode(body.pii_default_mode)
            _reload_pii_middleware()
        if body.pii_enabled is not None or body.pii_default_mode is not None:
            _persist_pii_state(gate)

        # Budget settings
        if body.budget_per_session is not None:
            val = None if body.budget_per_session == -1 else body.budget_per_session
            gate.config.budget_per_session = val
            gate.session_manager.set_default_budget(val)
        if body.budget_global is not None:
            gate.config.budget_global = None if body.budget_global == -1 else body.budget_global
        if body.budget_action is not None:
            gate.config.budget_action = BudgetAction(body.budget_action)
        if (
            body.budget_per_session is not None
            or body.budget_global is not None
            or body.budget_action is not None
        ):
            _persist_budget_state(gate)

        # Cache settings
        if body.cache_max_size is not None:
            gate.config.cache_max_size = body.cache_max_size
            for mw in gate.pipeline.middlewares:  # type: ignore[assignment]
                if hasattr(mw, "_max_size"):
                    mw._max_size = body.cache_max_size
        if body.cache_ttl_seconds is not None:
            gate.config.cache_ttl_seconds = body.cache_ttl_seconds
            for mw in gate.pipeline.middlewares:  # type: ignore[assignment]
                if hasattr(mw, "_ttl_seconds"):
                    mw._ttl_seconds = body.cache_ttl_seconds
        if body.cache_semantic_enabled is not None:
            gate.config.cache_semantic_enabled = body.cache_semantic_enabled
        if body.cache_similarity_threshold is not None:
            gate.config.cache_similarity_threshold = body.cache_similarity_threshold
            for mw in gate.pipeline.middlewares:  # type: ignore[assignment]
                if hasattr(mw, "_similarity_threshold"):
                    mw._similarity_threshold = body.cache_similarity_threshold
        if body.cache_scope is not None:
            gate.config.cache_scope = body.cache_scope  # type: ignore[assignment]
            for mw in gate.pipeline.middlewares:  # type: ignore[assignment]
                if hasattr(mw, "_scope"):
                    mw._scope = body.cache_scope

        # Loop detection settings
        if body.loop_detection_enabled is not None:
            gate.config.loop_detection_enabled = body.loop_detection_enabled
            if body.loop_detection_enabled:
                # Add LoopDetector if not already present
                has_loop = any(hasattr(mw, "_threshold") for mw in gate.pipeline.middlewares)
                if not has_loop:
                    try:
                        from stateloom.middleware.loop_detector import LoopDetector

                        gate.pipeline.add(LoopDetector(gate.config, store=gate.store))
                    except ImportError:
                        pass
            else:
                # Remove LoopDetector from pipeline
                gate.pipeline.middlewares = [
                    mw for mw in gate.pipeline.middlewares if not hasattr(mw, "_threshold")
                ]
        if body.loop_exact_threshold is not None:
            gate.config.loop_exact_threshold = body.loop_exact_threshold
            for mw in gate.pipeline.middlewares:  # type: ignore[assignment]
                if hasattr(mw, "_threshold"):
                    mw._threshold = body.loop_exact_threshold

        # Storage settings
        if body.store_retention_days is not None:
            gate.config.store_retention_days = body.store_retention_days

        # Console settings
        if body.console_verbose is not None:
            gate.config.console_verbose = body.console_verbose

        # Auto-route settings — dynamic pipeline management
        if body.auto_route_enabled is not None:
            _toggle_auto_route(body.auto_route_enabled)
        if body.auto_route_force_local is not None:
            gate.config.auto_route_force_local = body.auto_route_force_local
            if body.auto_route_force_local:
                # Auto-select first downloaded model if none is set
                if not gate.config.local_model_default and not gate.config.auto_route_model:
                    try:
                        from stateloom.local.client import OllamaClient

                        client = OllamaClient(host=gate.config.local_model_host)
                        try:
                            models = client.list_models()
                        finally:
                            client.close()
                        if models:
                            first_model = models[0].get("name", "")
                            if first_model:
                                gate.config.local_model_default = first_model
                    except Exception:
                        pass  # Ollama unavailable — user must select manually
                # Auto-enable auto_route when force_local is turned on
                if not gate.config.auto_route_enabled:
                    _toggle_auto_route(True)
        if body.auto_route_model is not None:
            gate.config.auto_route_model = body.auto_route_model
        if body.auto_route_complexity_threshold is not None:
            gate.config.auto_route_complexity_threshold = body.auto_route_complexity_threshold
        if body.auto_route_probe_enabled is not None:
            gate.config.auto_route_probe_enabled = body.auto_route_probe_enabled

        # Persist local routing config if any relevant field changed
        if any(
            getattr(body, f, None) is not None
            for f in (
                "auto_route_enabled",
                "auto_route_force_local",
                "local_model_default",
                "auto_route_model",
            )
        ):
            _persist_local_routing_state(gate)

        # Kill switch settings
        ks_changed = False
        if body.kill_switch_active is not None:
            gate.config.kill_switch_active = body.kill_switch_active
            ks_changed = True
        if body.kill_switch_message is not None:
            gate.config.kill_switch_message = body.kill_switch_message
            ks_changed = True
        if body.kill_switch_response_mode is not None:
            gate.config.kill_switch_response_mode = body.kill_switch_response_mode  # type: ignore[assignment]
            ks_changed = True
        if ks_changed:
            _persist_kill_switch_state(gate)

        # Blast radius settings
        if body.blast_radius_enabled is not None:
            gate.config.blast_radius_enabled = body.blast_radius_enabled
        if body.blast_radius_consecutive_failures is not None:
            gate.config.blast_radius_consecutive_failures = body.blast_radius_consecutive_failures
        if body.blast_radius_budget_violations_per_hour is not None:
            gate.config.blast_radius_budget_violations_per_hour = (
                body.blast_radius_budget_violations_per_hour
            )
        if (
            body.blast_radius_enabled is not None
            or body.blast_radius_consecutive_failures is not None
            or body.blast_radius_budget_violations_per_hour is not None
        ):
            _persist_blast_radius_state(gate)

        # Provider API keys — persist to encrypted secret store
        for provider in ("openai", "anthropic", "google"):
            field = f"provider_api_key_{provider}"
            val = getattr(body, field, None)
            if val is not None:
                setattr(gate.config, field, val)
                if val:
                    gate.store.save_secret(f"provider_key_{provider}", val)
                else:
                    gate.store.delete_secret(f"provider_key_{provider}")  # type: ignore[attr-defined]

        return _config_dict()

    # --- Admin lock endpoints ---

    @router.post("/admin/locks", status_code=201)
    async def create_admin_lock(body: LockSettingRequest) -> dict[str, Any]:
        valid_fields = set(ConfigUpdate.model_fields.keys())
        if body.setting not in valid_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown config setting: '{body.setting}'",
            )

        value = body.value
        if value is None:
            value = getattr(gate.config, body.setting, None)

        lock = gate.lock_setting(body.setting, value, reason=body.reason)
        return lock

    @router.get("/admin/locks")
    async def list_admin_locks() -> dict[str, Any]:
        return {"locks": gate.list_locked_settings()}

    @router.delete("/admin/locks/{setting}")
    async def delete_admin_lock(setting: str) -> dict[str, str]:
        removed = gate.unlock_setting(setting)
        if not removed:
            raise HTTPException(status_code=404, detail="Lock not found")
        return {"status": "unlocked", "setting": setting}

    # --- Provider API key endpoints ---

    @router.get("/provider-keys")
    async def get_provider_keys() -> dict[str, Any]:
        def status(key: str) -> dict[str, bool]:
            return {"set": bool(key)}

        return {
            "openai": status(gate.config.provider_api_key_openai),
            "anthropic": status(gate.config.provider_api_key_anthropic),
            "google": status(gate.config.provider_api_key_google),
        }

    # --- Local model endpoints ---

    @router.get("/local/status")
    async def local_status() -> dict[str, Any]:
        from stateloom.local.client import OllamaClient

        client = OllamaClient(host=gate.config.local_model_host)
        try:
            available = client.is_available()
            models = client.list_models() if available else []
        except Exception:
            available = False
            models = []
        finally:
            client.close()

        return {
            "available": available,
            "host": gate.config.local_model_host,
            "model_count": len(models),
            "local_model_enabled": gate.config.local_model_enabled,
            "local_model_default": gate.config.local_model_default,
            "shadow_enabled": gate.config.shadow.enabled,
            "shadow_model": gate.config.shadow.model,
        }

    @router.get("/local/models")
    async def list_local_models() -> dict[str, Any]:
        from stateloom.local.client import OllamaClient

        client = OllamaClient(host=gate.config.local_model_host)
        try:
            models = client.list_models()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")
        finally:
            client.close()
        return {"models": models}

    @router.get("/local/recommend")
    async def recommend_local_models() -> dict[str, Any]:
        from stateloom.local.hardware import detect_hardware
        from stateloom.local.hardware import recommend_models as _recommend

        hardware = detect_hardware()
        models = _recommend(hardware)
        return {
            "hardware": {
                "ram_gb": round(hardware.ram_gb, 1),
                "disk_free_gb": round(hardware.disk_free_gb, 1),
                "gpu_name": hardware.gpu_name,
                "gpu_vram_gb": round(hardware.gpu_vram_gb, 1),
                "os": hardware.os_name,
                "arch": hardware.arch,
                "cpu_count": hardware.cpu_count,
            },
            "recommended_models": models,
        }

    @router.post("/local/pull")
    async def pull_local_model(body: PullRequest) -> StreamingResponse:
        from stateloom.local.client import OllamaClient

        q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=200)
        cancelled = threading.Event()

        with _pull_lock:
            _pull_progress[body.model] = {"status": "pulling", "progress_pct": 0}

        def _pull_thread() -> None:
            client = OllamaClient(host=gate.config.local_model_host, timeout=600.0)
            max_pct = 0.0
            try:

                def cb(data: dict[str, Any]) -> None:
                    nonlocal max_pct
                    if cancelled.is_set():
                        return
                    total = data.get("total", 0)
                    completed = data.get("completed", 0)
                    status_text = data.get("status", "pulling")
                    if total > 0:
                        layer_pct = completed / total * 100
                    else:
                        layer_pct = max_pct
                    max_pct = max(layer_pct, max_pct)
                    progress = {
                        "status": "pulling",
                        "progress_pct": max_pct,
                        "detail": status_text,
                    }
                    with _pull_lock:
                        _pull_progress[body.model] = progress
                    try:
                        q.put_nowait(progress)
                    except queue.Full:
                        pass

                client.pull_model(body.model, progress_callback=cb)
                done = {"status": "complete", "progress_pct": 100}
                with _pull_lock:
                    _pull_progress[body.model] = done
                try:
                    q.put_nowait(done)
                except queue.Full:
                    pass
            except Exception as e:
                err = {"status": "error", "progress_pct": 0, "error": str(e)}
                with _pull_lock:
                    _pull_progress[body.model] = err
                try:
                    q.put_nowait(err)
                except queue.Full:
                    pass
            finally:
                client.close()
                q.put(None, timeout=5)  # Sentinel

        thread = threading.Thread(target=_pull_thread, daemon=True)
        thread.start()

        async def _stream() -> AsyncIterator[str]:
            try:
                loop = asyncio.get_running_loop()
                while True:
                    try:
                        item = await loop.run_in_executor(None, lambda: q.get(timeout=2.0))
                    except queue.Empty:
                        yield "\n"  # Keepalive
                        continue
                    if item is None:
                        break
                    yield json_module.dumps(item) + "\n"
                    if item.get("status") in ("complete", "error"):
                        break
            except (asyncio.CancelledError, GeneratorExit):
                cancelled.set()
            except Exception as exc:
                err = {"status": "error", "error": str(exc)}
                yield json_module.dumps(err) + "\n"

        return StreamingResponse(
            _stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/local/pull/{model:path}/progress")
    async def pull_progress(model: str) -> dict[str, Any]:
        with _pull_lock:
            state = _pull_progress.get(model)
        if state is None:
            return {"status": "unknown", "progress_pct": 0}
        return state

    # --- Hot-swap endpoint ---

    @router.post("/local/hot-swap")
    async def start_hot_swap(body: HotSwapRequest) -> dict[str, Any]:
        with _hot_swap.lock:
            if _hot_swap.active:
                raise HTTPException(status_code=409, detail="Hot-swap already in progress")
            old_model = gate.config.local_model_default
            if body.new_model == old_model:
                raise HTTPException(status_code=400, detail="New model is same as current model")
            _hot_swap.active = True
            _hot_swap.new_model = body.new_model
            _hot_swap.old_model = old_model
            _hot_swap.phase = "pulling"
            _hot_swap.progress_pct = 0.0
            _hot_swap.error = ""

        def _swap() -> None:
            from stateloom.dashboard.ws import broadcast_sync
            from stateloom.local.client import OllamaClient

            client = OllamaClient(host=gate.config.local_model_host, timeout=600.0)
            try:
                # Phase: pulling
                def cb(data: dict[str, Any]) -> None:
                    total = data.get("total", 0)
                    completed = data.get("completed", 0)
                    with _hot_swap.lock:
                        _hot_swap.progress_pct = (completed / total * 100) if total > 0 else 0
                    broadcast_sync(
                        {
                            "type": "hot_swap_progress",
                            "phase": "pulling",
                            "progress_pct": _hot_swap.progress_pct,
                            "new_model": body.new_model,
                        }
                    )

                client.pull_model(body.new_model, progress_callback=cb)

                # Phase: switching
                with _hot_swap.lock:
                    _hot_swap.phase = "switching"
                    _hot_swap.progress_pct = 100.0
                broadcast_sync(
                    {
                        "type": "hot_swap_progress",
                        "phase": "switching",
                        "new_model": body.new_model,
                    }
                )

                # Atomically update config
                gate.config.local_model_default = body.new_model
                if gate.config.auto_route_model == _hot_swap.old_model:
                    gate.config.auto_route_model = body.new_model
                if gate.config.shadow_model == _hot_swap.old_model:
                    gate.config.shadow_model = body.new_model

                # Phase: cleaning (fail-open)
                if body.delete_old and _hot_swap.old_model:
                    with _hot_swap.lock:
                        _hot_swap.phase = "cleaning"
                    broadcast_sync(
                        {
                            "type": "hot_swap_progress",
                            "phase": "cleaning",
                            "new_model": body.new_model,
                        }
                    )
                    try:
                        client.delete_model(_hot_swap.old_model)
                    except Exception:
                        pass  # fail-open

                # Phase: complete
                with _hot_swap.lock:
                    _hot_swap.phase = "complete"
                broadcast_sync(
                    {
                        "type": "hot_swap_progress",
                        "phase": "complete",
                        "new_model": body.new_model,
                    }
                )
            except Exception as e:
                with _hot_swap.lock:
                    _hot_swap.phase = "error"
                    _hot_swap.error = str(e)
                broadcast_sync(
                    {
                        "type": "hot_swap_progress",
                        "phase": "error",
                        "error": str(e),
                        "new_model": body.new_model,
                    }
                )
            finally:
                client.close()
                with _hot_swap.lock:
                    _hot_swap.active = False

        thread = threading.Thread(target=_swap, daemon=True)
        thread.start()
        return {
            "status": "started",
            "new_model": body.new_model,
            "old_model": _hot_swap.old_model,
        }

    @router.get("/local/hot-swap")
    async def get_hot_swap_status() -> dict[str, Any]:
        with _hot_swap.lock:
            return {
                "active": _hot_swap.active,
                "new_model": _hot_swap.new_model,
                "old_model": _hot_swap.old_model,
                "phase": _hot_swap.phase,
                "progress_pct": _hot_swap.progress_pct,
                "error": _hot_swap.error,
            }

    @router.delete("/local/models/{model}")
    async def delete_local_model(model: str) -> dict[str, str]:
        from stateloom.local.client import OllamaClient

        client = OllamaClient(host=gate.config.local_model_host)
        try:
            client.delete_model(model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            client.close()
        return {"status": "deleted", "model": model}

    # --- Shadow metrics endpoint ---

    @router.get("/shadow/metrics", dependencies=_shadow_deps)
    async def shadow_metrics() -> dict[str, Any]:
        from stateloom.core.event import LocalRoutingEvent, ShadowDraftEvent

        # Query all shadow events, excluding cancelled ones (routed locally / cached)
        all_events = gate.store.get_session_events("", event_type="shadow_draft", limit=10000)
        events = [
            e
            for e in all_events
            if isinstance(e, ShadowDraftEvent) and e.shadow_status != "cancelled"
        ]
        total = len(events)
        if total == 0:
            return {
                "total_calls": 0,
                "success_count": 0,
                "timeout_count": 0,
                "error_count": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "avg_latency_ratio": 0.0,
                "total_cost_saved": 0.0,
                "avg_similarity": None,
                "similarity_count": 0,
                "high_quality_count": 0,
                "high_quality_pct": 0.0,
                "by_model": {},
            }

        success = sum(
            1 for e in events if isinstance(e, ShadowDraftEvent) and e.shadow_status == "success"
        )
        timeouts = sum(
            1 for e in events if isinstance(e, ShadowDraftEvent) and e.shadow_status == "timeout"
        )
        errors = total - success - timeouts

        latencies = [
            e.local_latency_ms
            for e in events
            if isinstance(e, ShadowDraftEvent) and e.shadow_status == "success"
        ]
        ratios = [
            e.latency_ratio
            for e in events
            if isinstance(e, ShadowDraftEvent) and e.latency_ratio > 0
        ]
        total_saved = sum(e.cost_saved for e in events if isinstance(e, ShadowDraftEvent))

        # Include routing savings from auto-router
        routing_events = gate.store.get_session_events("", event_type="local_routing")
        routing_saved = sum(
            e.estimated_cloud_cost
            for e in routing_events
            if isinstance(e, LocalRoutingEvent) and e.routing_success
        )
        total_saved += routing_saved

        # Similarity aggregates
        similarity_scores = [
            e.similarity_score
            for e in events
            if isinstance(e, ShadowDraftEvent) and e.similarity_score is not None
        ]
        similarity_count = len(similarity_scores)
        avg_similarity = sum(similarity_scores) / similarity_count if similarity_count > 0 else None
        high_quality_count = sum(1 for s in similarity_scores if s >= 0.7)
        high_quality_pct = high_quality_count / similarity_count if similarity_count > 0 else 0.0

        # Per-model breakdown
        by_model: dict[str, dict[str, Any]] = {}
        for e in events:
            if not isinstance(e, ShadowDraftEvent):
                continue
            model = e.local_model
            if model not in by_model:
                by_model[model] = {
                    "calls": 0,
                    "success": 0,
                    "avg_latency_ms": 0.0,
                    "total_cost_saved": 0.0,
                    "latencies": [],
                    "similarity_scores": [],
                }
            by_model[model]["calls"] += 1
            if e.shadow_status == "success":
                by_model[model]["success"] += 1
                by_model[model]["latencies"].append(e.local_latency_ms)
            by_model[model]["total_cost_saved"] += e.cost_saved
            if e.similarity_score is not None:
                by_model[model]["similarity_scores"].append(e.similarity_score)

        for model, data in by_model.items():
            lats = data.pop("latencies")
            data["avg_latency_ms"] = sum(lats) / len(lats) if lats else 0.0
            sims = data.pop("similarity_scores")
            data["avg_similarity"] = sum(sims) / len(sims) if sims else None

        # Skip stats from eligibility filter
        skip_reasons: dict[str, int] = {}
        if hasattr(gate, "_shadow_middleware") and gate._shadow_middleware is not None:
            skip_reasons = gate._shadow_middleware.get_skip_stats()

        return {
            "total_calls": total,
            "success_count": success,
            "timeout_count": timeouts,
            "error_count": errors,
            "success_rate": success / total if total > 0 else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "avg_latency_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
            "total_cost_saved": total_saved,
            "avg_similarity": avg_similarity,
            "similarity_count": similarity_count,
            "high_quality_count": high_quality_count,
            "high_quality_pct": high_quality_pct,
            "by_model": by_model,
            "skip_reasons": skip_reasons,
        }

    @router.get("/shadow/readiness", dependencies=_shadow_deps)
    async def shadow_readiness() -> dict[str, Any]:
        """Model testing readiness scores per candidate model.

        Scoring formula per model (0–100):
        - Quality (0–40 pts): avg similarity
        - Reliability (0–30 pts): success rate
        - Speed (0–20 pts): latency ratio (local/cloud)
        - Confidence (0–10 pts): sample count
        """
        from stateloom.core.event import ShadowDraftEvent

        all_events = gate.store.get_session_events("", event_type="shadow_draft", limit=10000)
        events = [
            e
            for e in all_events
            if isinstance(e, ShadowDraftEvent) and e.shadow_status != "cancelled"
        ]

        # Group by model
        by_model: dict[str, list[ShadowDraftEvent]] = {}
        for e in events:
            if not isinstance(e, ShadowDraftEvent):
                continue
            model = e.local_model
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(e)

        results: dict[str, dict[str, Any]] = {}
        for model, model_events in by_model.items():
            total = len(model_events)
            successes = [e for e in model_events if e.shadow_status == "success"]
            success_count = len(successes)
            success_rate = success_count / total if total > 0 else 0.0

            sim_scores = [
                e.similarity_score for e in model_events if e.similarity_score is not None
            ]
            avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0

            ratios = [e.latency_ratio for e in model_events if e.latency_ratio > 0]
            avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0

            # Quality (0–40)
            if avg_sim >= 0.9:
                quality = 40
            elif avg_sim >= 0.8:
                quality = 32
            elif avg_sim >= 0.7:
                quality = 24
            elif avg_sim >= 0.5:
                quality = 12
            else:
                quality = 0

            # Reliability (0–30)
            if success_rate >= 0.99:
                reliability = 30
            elif success_rate >= 0.95:
                reliability = 24
            elif success_rate >= 0.90:
                reliability = 18
            elif success_rate >= 0.80:
                reliability = 10
            else:
                reliability = 0

            # Speed (0–20)
            if avg_ratio <= 0.5:
                speed = 20
            elif avg_ratio <= 1.0:
                speed = 15
            elif avg_ratio <= 2.0:
                speed = 8
            else:
                speed = 0

            # Confidence (0–10)
            if total >= 100:
                confidence = 10
            elif total >= 50:
                confidence = 7
            elif total >= 20:
                confidence = 4
            elif total >= 5:
                confidence = 2
            else:
                confidence = 0

            score = quality + reliability + speed + confidence

            # Recommendation
            if score >= 80:
                recommendation = "Safe to switch — high-quality replacement"
            elif score >= 60:
                recommendation = "Good for simple tasks — consider auto-routing"
            elif score >= 40:
                recommendation = "Partial coverage — suitable for specific use cases"
            else:
                recommendation = "Not ready — quality gap too large"

            # Similarity distribution
            excellent = sum(1 for s in sim_scores if s >= 0.9)
            good = sum(1 for s in sim_scores if 0.7 <= s < 0.9)
            fair = sum(1 for s in sim_scores if 0.4 <= s < 0.7)
            poor = sum(1 for s in sim_scores if s < 0.4)

            results[model] = {
                "score": score,
                "quality": quality,
                "reliability": reliability,
                "speed": speed,
                "confidence": confidence,
                "recommendation": recommendation,
                "total_tests": total,
                "success_rate": round(success_rate, 4),
                "avg_similarity": round(avg_sim, 4) if sim_scores else None,
                "avg_latency_ratio": round(avg_ratio, 4) if ratios else None,
                "similarity_distribution": {
                    "excellent": excellent,
                    "good": good,
                    "fair": fair,
                    "poor": poor,
                },
            }

        return {"models": results}

    @router.get("/shadow/config", dependencies=_shadow_deps)
    async def get_shadow_config() -> dict[str, Any]:
        """Get current shadow drafting configuration."""
        return gate.shadow_status()

    @router.post("/shadow/configure", dependencies=_shadow_deps)
    async def configure_shadow(body: ShadowConfigureRequest) -> dict[str, Any]:
        """Configure shadow drafting at runtime. Enterprise-gated."""
        return gate.configure_shadow(
            enabled=body.enabled,
            model=body.model,
            sample_rate=body.sample_rate,
            max_context_tokens=body.max_context_tokens,
            models=body.models,
        )

    # --- Kill switch endpoints ---

    @router.get("/kill-switch")
    async def get_kill_switch() -> dict[str, Any]:
        return {
            "active": gate.config.kill_switch_active,
            "message": gate.config.kill_switch_message,
            "response_mode": gate.config.kill_switch_response_mode,
            "rules": [r.model_dump() for r in gate.config.kill_switch_rules],
            "environment": gate.config.kill_switch_environment,
            "agent_version": gate.config.kill_switch_agent_version,
        }

    @router.post("/kill-switch/activate")
    async def activate_kill_switch(body: KillSwitchActivateRequest | None = None) -> dict[str, Any]:
        gate.config.kill_switch_active = True
        if body:
            if body.message is not None:
                gate.config.kill_switch_message = body.message
            if body.response_mode is not None:
                gate.config.kill_switch_response_mode = body.response_mode  # type: ignore[assignment]
        _persist_kill_switch_state(gate)
        return {"status": "activated", "active": True}

    @router.post("/kill-switch/deactivate")
    async def deactivate_kill_switch() -> dict[str, Any]:
        gate.config.kill_switch_active = False
        _persist_kill_switch_state(gate)
        return {"status": "deactivated", "active": False}

    @router.post("/kill-switch/rules")
    async def add_kill_switch_rule(body: KillSwitchRuleRequest) -> dict[str, Any]:
        from stateloom.core.config import KillSwitchRule

        rule = KillSwitchRule(**body.model_dump())
        gate.config.kill_switch_rules.append(rule)
        _persist_kill_switch_state(gate)
        return {"status": "added", "rule_count": len(gate.config.kill_switch_rules)}

    @router.put("/kill-switch/rules")
    async def replace_kill_switch_rules(body: KillSwitchRulesReplaceRequest) -> dict[str, Any]:
        from stateloom.core.config import KillSwitchRule

        gate.config.kill_switch_rules = [KillSwitchRule(**r.model_dump()) for r in body.rules]
        _persist_kill_switch_state(gate)
        return {"status": "replaced", "rule_count": len(gate.config.kill_switch_rules)}

    @router.delete("/kill-switch/rules")
    async def clear_kill_switch_rules() -> dict[str, Any]:
        gate.config.kill_switch_rules.clear()
        _persist_kill_switch_state(gate)
        return {"status": "cleared", "rule_count": 0}

    # --- PII rule endpoints ---

    @router.get("/pii/rules")
    async def get_pii_rules() -> dict[str, Any]:
        return {
            "rules": [r.model_dump() for r in gate.config.pii.rules],
            "pii_enabled": gate.config.pii.enabled,
            "pii_default_mode": gate.config.pii.default_mode.value,
        }

    def _pii_init_warning() -> dict[str, str] | None:
        """Return a warning if PII rules were set via init() (code takes precedence)."""
        for mw in gate.pipeline.middlewares:
            if hasattr(mw, "_rules_from_init") and mw._rules_from_init:
                return {
                    "warning": "PII rules were set via init(). API changes apply to "
                    "this process but init() rules will take precedence on restart."
                }
        return None

    @router.post("/pii/rules")
    async def add_pii_rule(body: PIIRuleRequest) -> dict[str, Any]:
        from stateloom.core.config import PIIRule

        rule_data: dict[str, Any] = {
            "pattern": body.pattern,
            "mode": PIIMode(body.mode),
        }
        if body.on_middleware_failure is not None:
            from stateloom.core.types import FailureAction

            rule_data["on_middleware_failure"] = FailureAction(body.on_middleware_failure)
        rule = PIIRule(**rule_data)
        gate.config.pii_rules.append(rule)
        _reload_pii_middleware()
        _persist_pii_state(gate)
        result: dict[str, Any] = {"status": "added", "rule_count": len(gate.config.pii_rules)}
        warning = _pii_init_warning()
        if warning:
            result.update(warning)
        return result

    @router.put("/pii/rules")
    async def replace_pii_rules(body: PIIRulesReplaceRequest) -> dict[str, Any]:
        from stateloom.core.config import PIIRule

        rules = []
        for r in body.rules:
            rule_data: dict[str, Any] = {
                "pattern": r.pattern,
                "mode": PIIMode(r.mode),
            }
            if r.on_middleware_failure is not None:
                from stateloom.core.types import FailureAction

                rule_data["on_middleware_failure"] = FailureAction(r.on_middleware_failure)
            rules.append(PIIRule(**rule_data))
        gate.config.pii_rules = rules
        _reload_pii_middleware()
        _persist_pii_state(gate)
        return {"status": "replaced", "rule_count": len(gate.config.pii_rules)}

    @router.delete("/pii/rules")
    async def clear_pii_rules() -> dict[str, Any]:
        gate.config.pii_rules.clear()
        _reload_pii_middleware()
        _persist_pii_state(gate)
        return {"status": "cleared", "rule_count": 0}

    @router.delete("/pii/rules/{pattern}")
    async def delete_pii_rule(pattern: str) -> dict[str, Any]:
        gate.config.pii_rules = [r for r in gate.config.pii_rules if r.pattern != pattern]
        _reload_pii_middleware()
        _persist_pii_state(gate)
        return {"status": "deleted", "rule_count": len(gate.config.pii_rules)}

    # --- Organization endpoints ---

    @router.get("/organizations")
    async def list_organizations() -> dict[str, Any]:
        orgs = gate.list_organizations()
        return {
            "organizations": [_org_to_dict(o) for o in orgs],
            "total": len(orgs),
        }

    @router.post("/organizations", dependencies=_mt_deps)
    async def create_organization(body: CreateOrgRequest) -> dict[str, Any]:
        from stateloom.core.config import PIIRule as PIIRuleModel

        pii_rules = [PIIRuleModel(**r) for r in body.pii_rules] if body.pii_rules else []
        org = gate.create_organization(
            name=body.name,
            budget=body.budget,
            pii_rules=pii_rules,
            metadata=body.metadata,
        )
        return _org_to_dict(org)

    @router.get("/organizations/{org_id}")
    async def get_organization(org_id: str) -> dict[str, Any]:
        org = gate.get_organization(org_id)
        if org is None:
            raise HTTPException(status_code=404, detail="Organization not found")
        result = _org_to_dict(org)
        result["stats"] = gate.store.get_org_stats(org_id)
        return result

    @router.patch("/organizations/{org_id}")
    async def update_organization(org_id: str, body: UpdateOrgRequest) -> dict[str, Any]:
        org = gate.get_organization(org_id)
        if org is None:
            raise HTTPException(status_code=404, detail="Organization not found")
        if body.name is not None:
            org.name = body.name
        if body.budget is not None:
            org.budget = body.budget
        if body.pii_rules is not None:
            from stateloom.core.config import PIIRule as PIIRuleModel

            org.pii_rules = [PIIRuleModel(**r) for r in body.pii_rules]
        if body.metadata is not None:
            org.metadata = body.metadata
        gate.store.save_organization(org)
        return _org_to_dict(org)

    @router.get("/organizations/{org_id}/teams")
    async def list_org_teams(org_id: str) -> dict[str, Any]:
        teams = gate.list_teams(org_id=org_id)
        return {
            "teams": [_team_to_dict(t) for t in teams],
            "total": len(teams),
        }

    @router.get("/organizations/{org_id}/sessions")
    async def list_org_sessions(
        org_id: str,
        limit: int = Query(default=50, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        sessions = gate.store.list_sessions(limit=limit, offset=offset, org_id=org_id)
        return {
            "sessions": [_session_to_dict(gate, s) for s in sessions],
            "total": len(sessions),
        }

    # --- Team endpoints ---

    @router.get("/teams")
    async def list_all_teams(
        org_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        teams = gate.list_teams(org_id=org_id)
        return {
            "teams": [_team_to_dict(t) for t in teams],
            "total": len(teams),
        }

    @router.post("/teams", dependencies=_mt_deps)
    async def create_team(body: CreateTeamRequest) -> dict[str, Any]:
        try:
            from stateloom.ee.setup import _check_dev_scale_cap

            _check_dev_scale_cap(gate, "teams")
        except Exception as exc:
            if "LICENSE_REQUIRED" in str(exc):
                raise HTTPException(status_code=403, detail=str(exc))
            raise
        team = gate.create_team(
            org_id=body.org_id,
            name=body.name,
            budget=body.budget,
            metadata=body.metadata,
        )
        return _team_to_dict(team)

    @router.get("/teams/{team_id}")
    async def get_team(team_id: str) -> dict[str, Any]:
        team = gate.get_team(team_id)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")
        result = _team_to_dict(team)
        result["stats"] = gate.store.get_team_stats(team_id)
        return result

    @router.patch("/teams/{team_id}")
    async def update_team(team_id: str, body: UpdateTeamRequest) -> dict[str, Any]:
        team = gate.get_team(team_id)
        if team is None:
            raise HTTPException(status_code=404, detail="Team not found")
        if body.name is not None:
            team.name = body.name
        if body.budget is not None:
            team.budget = body.budget
        if body.metadata is not None:
            team.metadata = body.metadata
        gate.store.save_team(team)
        return _team_to_dict(team)

    @router.get("/teams/{team_id}/sessions")
    async def list_team_sessions(
        team_id: str,
        limit: int = Query(default=50, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        sessions = gate.store.list_sessions(limit=limit, offset=offset, team_id=team_id)
        return {
            "sessions": [_session_to_dict(gate, s) for s in sessions],
            "total": len(sessions),
        }

    # --- Blast radius endpoints ---

    @router.get("/blast-radius")
    async def get_blast_radius() -> dict[str, Any]:
        if gate._blast_radius is None:
            return {
                "enabled": False,
                "paused_sessions": [],
                "paused_agents": [],
                "session_failure_counts": {},
                "agent_failure_counts": {},
                "session_budget_violations": {},
                "agent_budget_violations": {},
            }
        status: dict[str, Any] = gate._blast_radius.get_status()
        status["enabled"] = True
        return status

    @router.post("/blast-radius/unpause-session/{session_id}")
    async def unpause_session(session_id: str) -> dict[str, Any]:
        if gate._blast_radius is None:
            raise HTTPException(status_code=400, detail="Blast radius not enabled")
        result = gate._blast_radius.unpause_session(session_id)
        return {"unpaused": result, "session_id": session_id}

    @router.post("/blast-radius/unpause-agent/{agent_id:path}")
    async def unpause_agent(agent_id: str) -> dict[str, Any]:
        if gate._blast_radius is None:
            raise HTTPException(status_code=400, detail="Blast radius not enabled")
        result = gate._blast_radius.unpause_agent(agent_id)
        return {"unpaused": result, "agent_id": agent_id}

    @router.get("/blast-radius/events")
    async def get_blast_radius_events(
        limit: int = Query(default=100, le=500),
    ) -> dict[str, Any]:
        events = gate.store.get_session_events("", event_type="blast_radius", limit=limit)
        return {
            "events": [
                {
                    "id": e.id,
                    "session_id": e.session_id,
                    "step": e.step,
                    "timestamp": e.timestamp.isoformat(),
                    **_event_details(e),
                }
                for e in events
            ],
            "total": len(events),
        }

    # --- Circuit breaker endpoints ---

    @router.get("/circuit-breaker")
    async def get_circuit_breaker() -> dict[str, Any]:
        """Get circuit breaker status for all providers."""
        return gate.circuit_breaker_status()

    @router.post("/circuit-breaker/{provider}/reset")
    async def reset_circuit_breaker(provider: str) -> dict[str, str]:
        """Reset a provider's circuit breaker to closed."""
        result = gate.reset_circuit_breaker(provider)
        if not result:
            if gate._circuit_breaker is None:
                raise HTTPException(status_code=400, detail="Circuit breaker not enabled")
            raise HTTPException(
                status_code=404, detail=f"No circuit state for provider '{provider}'"
            )
        return {"status": "reset", "provider": provider}

    # --- Compliance endpoints ---

    @router.get("/compliance/profiles")
    async def list_compliance_profiles() -> dict[str, Any]:
        """List available compliance profile presets."""
        from stateloom.compliance.profiles import PROFILE_PRESETS

        profiles = {}
        for name, factory in PROFILE_PRESETS.items():
            p = factory()
            profiles[name] = {
                "standard": p.standard,
                "region": p.region,
                "session_ttl_days": p.session_ttl_days,
                "cache_ttl_seconds": p.cache_ttl_seconds,
                "zero_retention_logs": p.zero_retention_logs,
                "block_local_routing": p.block_local_routing,
                "block_shadow": p.block_shadow,
                "pii_rules_count": len(p.pii_rules),
                "default_consent": p.default_consent,
            }
        # Include current global profile and overlay its overrides
        cp = gate.config.compliance_profile
        active = None
        if cp and cp.standard != "none":
            active = cp.standard
            if active in profiles:
                profiles[active]["default_consent"] = cp.default_consent
        return {"profiles": profiles, "active_global": active}

    @router.get("/compliance/global")
    async def get_global_compliance() -> dict[str, Any]:
        """Get the global compliance profile."""
        cp = gate.config.compliance_profile
        if not cp or cp.standard == "none":
            return {"compliance_profile": None}
        return {"compliance_profile": cp.model_dump()}

    @router.put("/compliance/global")
    async def set_global_compliance(request: SetComplianceProfileRequest) -> dict[str, Any]:
        """Set the global compliance profile (applies to all sessions without org/team profile)."""
        from stateloom.compliance.profiles import resolve_profile

        profile = resolve_profile(request.standard)
        if request.region != "global":
            profile.region = request.region
        if request.session_ttl_days > 0:
            profile.session_ttl_days = request.session_ttl_days
        if request.cache_ttl_seconds > 0:
            profile.cache_ttl_seconds = request.cache_ttl_seconds
        if request.zero_retention_logs:
            profile.zero_retention_logs = True
        if request.block_local_routing:
            profile.block_local_routing = True
        if request.block_shadow:
            profile.block_shadow = True
        if request.allowed_endpoints:
            profile.allowed_endpoints = request.allowed_endpoints
        if request.audit_salt:
            profile.audit_salt = request.audit_salt
        if request.default_consent:
            profile.default_consent = True
        gate.config.compliance_profile = profile
        _apply_compliance_pii_rules(gate, profile)
        _persist_pii_state(gate)
        # Persist so it survives restart
        gate.store.save_secret("global_compliance_standard", request.standard)
        return {"compliance_profile": profile.model_dump()}

    @router.delete("/compliance/global")
    async def clear_global_compliance() -> dict[str, str]:
        """Clear the global compliance profile."""
        old_profile = gate.config.compliance_profile
        gate.config.compliance_profile = None
        # Remove compliance PII rules that were injected
        if old_profile and old_profile.pii_rules:
            compliance_patterns = {r.pattern for r in old_profile.pii_rules}
            gate.config.pii_rules = [
                r for r in gate.config.pii_rules if r.pattern not in compliance_patterns
            ]
            _reload_pii_scanner(gate)
            _persist_pii_state(gate)
        # Clear persisted setting
        gate.store.save_secret("global_compliance_standard", "")
        return {"status": "cleared"}

    @router.get("/compliance/audit")
    async def list_compliance_audit_events(
        org_id: str | None = Query(default=None),
        standard: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        """Query compliance audit events."""
        events = gate.store.get_session_events("", event_type="compliance_audit", limit=limit)
        results = []
        for e in events:
            event_dict = {
                "id": e.id,
                "session_id": e.session_id,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value,
            }
            if hasattr(e, "compliance_standard"):
                event_dict.update(
                    {
                        "compliance_standard": getattr(e, "compliance_standard", ""),
                        "action": getattr(e, "action", ""),
                        "legal_rule": getattr(e, "legal_rule", ""),
                        "justification": getattr(e, "justification", ""),
                        "target_type": getattr(e, "target_type", ""),
                        "target_id": getattr(e, "target_id", ""),
                        "org_id": getattr(e, "org_id", ""),
                        "team_id": getattr(e, "team_id", ""),
                        "integrity_hash": getattr(e, "integrity_hash", ""),
                    }
                )
            # Filter by org_id or standard if provided
            if org_id and event_dict.get("org_id") != org_id:
                continue
            if standard and event_dict.get("compliance_standard") != standard:
                continue
            results.append(event_dict)
        return {"events": results, "count": len(results)}

    @router.post("/compliance/purge")
    async def purge_user_data(request: PurgeRequest) -> dict[str, Any]:
        """Right to Be Forgotten — purge all data for a user."""
        from stateloom.compliance.purge import PurgeEngine

        engine = PurgeEngine(gate.store, cache_store=gate._cache_store)
        result = engine.purge(
            request.user_identifier,
            standard=request.standard,
        )
        return {
            "user_identifier": result.user_identifier,
            "sessions_deleted": result.sessions_deleted,
            "events_deleted": result.events_deleted,
            "cache_entries_deleted": result.cache_entries_deleted,
            "audit_event_id": result.audit_event_id,
        }

    @router.get("/organizations/{org_id}/compliance")
    async def get_org_compliance(org_id: str) -> dict[str, Any]:
        """Get an organization's compliance profile."""
        org = gate.get_organization(org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        profile = org.compliance_profile
        if not profile:
            return {"compliance_profile": None}
        return {"compliance_profile": profile.model_dump()}

    @router.put("/organizations/{org_id}/compliance")
    async def set_org_compliance(
        org_id: str,
        request: SetComplianceProfileRequest,
    ) -> dict[str, Any]:
        """Set an organization's compliance profile."""
        org = gate.get_organization(org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        from stateloom.compliance.profiles import resolve_profile

        profile = resolve_profile(request.standard)
        # Override defaults with any explicit values
        if request.region != "global":
            profile.region = request.region
        if request.session_ttl_days > 0:
            profile.session_ttl_days = request.session_ttl_days
        if request.cache_ttl_seconds > 0:
            profile.cache_ttl_seconds = request.cache_ttl_seconds
        if request.zero_retention_logs:
            profile.zero_retention_logs = True
        if request.block_local_routing:
            profile.block_local_routing = True
        if request.block_shadow:
            profile.block_shadow = True
        if request.allowed_endpoints:
            profile.allowed_endpoints = request.allowed_endpoints
        if request.audit_salt:
            profile.audit_salt = request.audit_salt
        if request.default_consent:
            profile.default_consent = True
        org.compliance_profile = profile
        gate.store.save_organization(org)
        return {"compliance_profile": profile.model_dump()}

    @router.get("/teams/{team_id}/compliance")
    async def get_team_compliance(team_id: str) -> dict[str, Any]:
        """Get a team's compliance profile."""
        team = gate.get_team(team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        profile = team.compliance_profile
        if not profile:
            return {"compliance_profile": None}
        return {"compliance_profile": profile.model_dump()}

    @router.put("/teams/{team_id}/compliance")
    async def set_team_compliance(
        team_id: str,
        request: SetComplianceProfileRequest,
    ) -> dict[str, Any]:
        """Set a team's compliance profile."""
        team = gate.get_team(team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        from stateloom.compliance.profiles import resolve_profile

        profile = resolve_profile(request.standard)
        if request.region != "global":
            profile.region = request.region
        if request.session_ttl_days > 0:
            profile.session_ttl_days = request.session_ttl_days
        if request.cache_ttl_seconds > 0:
            profile.cache_ttl_seconds = request.cache_ttl_seconds
        if request.zero_retention_logs:
            profile.zero_retention_logs = True
        if request.block_local_routing:
            profile.block_local_routing = True
        if request.block_shadow:
            profile.block_shadow = True
        if request.allowed_endpoints:
            profile.allowed_endpoints = request.allowed_endpoints
        if request.audit_salt:
            profile.audit_salt = request.audit_salt
        if request.default_consent:
            profile.default_consent = True
        team.compliance_profile = profile
        gate.store.save_team(team)
        return {"compliance_profile": profile.model_dump()}

    # --- Rate limiter endpoints ---

    @router.put("/teams/{team_id}/rate-limit")
    async def set_team_rate_limit(team_id: str, body: UpdateTeamRateLimitRequest) -> dict[str, Any]:
        """Set rate limit config for a team."""
        team = gate.get_team(team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        if body.rate_limit_tps is not None:
            team.rate_limit_tps = body.rate_limit_tps
        if body.rate_limit_priority is not None:
            team.rate_limit_priority = body.rate_limit_priority
        if body.rate_limit_max_queue is not None:
            team.rate_limit_max_queue = body.rate_limit_max_queue
        if body.rate_limit_queue_timeout is not None:
            team.rate_limit_queue_timeout = body.rate_limit_queue_timeout
        gate.store.save_team(team)
        return _team_to_dict(team)

    @router.get("/teams/{team_id}/rate-limit")
    async def get_team_rate_limit(team_id: str) -> dict[str, Any]:
        """Get rate limit config and live status for a team."""
        team = gate.get_team(team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        result = {
            "team_id": team_id,
            "rate_limit_tps": team.rate_limit_tps,
            "rate_limit_priority": team.rate_limit_priority,
            "rate_limit_max_queue": team.rate_limit_max_queue,
            "rate_limit_queue_timeout": team.rate_limit_queue_timeout,
        }
        # Include live status if rate limiter is active
        if hasattr(gate, "_rate_limiter") and gate._rate_limiter is not None:
            status = gate._rate_limiter.get_status()
            team_status = status.get("teams", {}).get(team_id, {})
            result["live_status"] = team_status
        return result

    @router.delete("/teams/{team_id}/rate-limit")
    async def delete_team_rate_limit(team_id: str) -> dict[str, Any]:
        """Remove rate limit for a team (unlimited)."""
        team = gate.get_team(team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        team.rate_limit_tps = None
        team.rate_limit_priority = 0
        team.rate_limit_max_queue = 100
        team.rate_limit_queue_timeout = 30.0
        gate.store.save_team(team)
        # Clean up bucket
        if hasattr(gate, "_rate_limiter") and gate._rate_limiter is not None:
            gate._rate_limiter.remove_bucket(team_id)
        return {"status": "removed", "team_id": team_id}

    @router.get("/rate-limiter")
    async def get_rate_limiter_status() -> dict[str, Any]:
        """Global rate limiter status across all teams."""
        if not hasattr(gate, "_rate_limiter") or gate._rate_limiter is None:
            return {"teams": {}}
        return cast(dict[str, Any], gate._rate_limiter.get_status())

    # --- Async Job endpoints ---

    @router.post("/jobs", status_code=202)
    async def submit_job(body: SubmitJobRequest) -> dict[str, Any]:
        """Submit an async job for background processing."""
        if not gate.config.async_jobs_enabled:
            raise HTTPException(status_code=400, detail="Async jobs are not enabled")
        job = gate.submit_job(
            provider=body.provider,
            model=body.model,
            messages=body.messages,
            request_kwargs=body.request_kwargs,
            webhook_url=body.webhook_url,
            webhook_secret=body.webhook_secret,
            session_id=body.session_id,
            org_id=body.org_id,
            team_id=body.team_id,
            max_retries=body.max_retries,
            ttl_seconds=body.ttl_seconds,
            metadata=body.metadata,
        )
        return _job_to_dict(job)

    @router.get("/jobs")
    async def list_jobs(
        status: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
        limit: int = Query(default=50, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """List jobs with optional filters."""
        jobs = gate.list_jobs(status=status, session_id=session_id, limit=limit, offset=offset)
        return {
            "jobs": [_job_to_dict(j) for j in jobs],
            "total": len(jobs),
        }

    @router.get("/jobs/stats")
    async def get_job_stats() -> dict[str, Any]:
        """Get aggregate job statistics."""
        return gate.job_stats()

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        """Get a job by ID."""
        job = gate.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_to_dict(job)

    @router.delete("/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict[str, str]:
        """Cancel a pending job."""
        result = gate.cancel_job(job_id)
        if not result:
            job = gate.get_job(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            raise HTTPException(
                status_code=400, detail=f"Job is not pending (status: {job.status.value})"
            )
        return {"status": "cancelled", "job_id": job_id}

    # --- Virtual key endpoints (proxy) ---

    @router.post("/virtual-keys")
    async def create_virtual_key(body: CreateVirtualKeyRequest) -> dict[str, Any]:
        """Create a virtual API key. Returns the full key once (never again)."""
        try:
            from stateloom.ee.setup import _check_dev_scale_cap

            _check_dev_scale_cap(gate, "virtual_keys")
        except Exception as exc:
            if "LICENSE_REQUIRED" in str(exc):
                raise HTTPException(status_code=403, detail=str(exc))
            raise
        from stateloom.proxy.virtual_key import (
            VirtualKey,
            generate_virtual_key,
            make_key_preview,
            make_virtual_key_id,
        )

        # Resolve org_id from team
        team = gate.get_team(body.team_id)
        org_id = team.org_id if team else ""

        full_key, key_hash = generate_virtual_key()
        vk = VirtualKey(
            id=make_virtual_key_id(),
            key_hash=key_hash,
            key_preview=make_key_preview(full_key),
            team_id=body.team_id,
            org_id=org_id,
            name=body.name,
            scopes=body.scopes,
            metadata=body.metadata,
            rate_limit_tps=body.rate_limit_tps,
            rate_limit_max_queue=body.rate_limit_max_queue,
            rate_limit_queue_timeout=body.rate_limit_queue_timeout,
            agent_ids=body.agent_ids,
        )
        gate.store.save_virtual_key(vk)
        return {
            "id": vk.id,
            "key": full_key,
            "key_preview": vk.key_preview,
            "team_id": vk.team_id,
            "org_id": vk.org_id,
            "name": vk.name,
            "created_at": vk.created_at.isoformat(),
        }

    @router.get("/virtual-keys")
    async def list_virtual_keys(
        team_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """List virtual keys (previews only, never the full key)."""
        keys = gate.store.list_virtual_keys(team_id=team_id)
        return {
            "virtual_keys": [
                {
                    "id": vk.id,
                    "key_preview": vk.key_preview,
                    "team_id": vk.team_id,
                    "org_id": vk.org_id,
                    "name": vk.name,
                    "created_at": vk.created_at.isoformat(),
                    "revoked": vk.revoked,
                    "rate_limit_tps": vk.rate_limit_tps,
                    "rate_limit_max_queue": vk.rate_limit_max_queue,
                    "rate_limit_queue_timeout": vk.rate_limit_queue_timeout,
                }
                for vk in keys
            ],
            "total": len(keys),
        }

    @router.delete("/virtual-keys/{key_id}")
    async def revoke_virtual_key(key_id: str) -> dict[str, str]:
        """Revoke a virtual key."""
        result = gate.store.revoke_virtual_key(key_id)
        if not result:
            raise HTTPException(status_code=404, detail="Virtual key not found or already revoked")
        return {"status": "revoked", "key_id": key_id}

    # --- Virtual key rate limit CRUD ---

    @router.put("/virtual-keys/{key_id}/rate-limit")
    async def set_vk_rate_limit(key_id: str, body: UpdateVKRateLimitRequest) -> dict[str, Any]:
        """Set rate limit config for a virtual key."""
        vk = gate.store.get_virtual_key(key_id)
        if vk is None:
            raise HTTPException(status_code=404, detail="Virtual key not found")
        if body.rate_limit_tps is not None:
            vk.rate_limit_tps = body.rate_limit_tps
        if body.rate_limit_max_queue is not None:
            vk.rate_limit_max_queue = body.rate_limit_max_queue
        if body.rate_limit_queue_timeout is not None:
            vk.rate_limit_queue_timeout = body.rate_limit_queue_timeout
        gate.store.save_virtual_key(vk)
        return {
            "key_id": key_id,
            "rate_limit_tps": vk.rate_limit_tps,
            "rate_limit_max_queue": vk.rate_limit_max_queue,
            "rate_limit_queue_timeout": vk.rate_limit_queue_timeout,
        }

    @router.get("/virtual-keys/{key_id}/rate-limit")
    async def get_vk_rate_limit(key_id: str) -> dict[str, Any]:
        """Get rate limit config for a virtual key."""
        vk = gate.store.get_virtual_key(key_id)
        if vk is None:
            raise HTTPException(status_code=404, detail="Virtual key not found")
        return {
            "key_id": key_id,
            "rate_limit_tps": vk.rate_limit_tps,
            "rate_limit_max_queue": vk.rate_limit_max_queue,
            "rate_limit_queue_timeout": vk.rate_limit_queue_timeout,
        }

    @router.delete("/virtual-keys/{key_id}/rate-limit")
    async def delete_vk_rate_limit(key_id: str) -> dict[str, str]:
        """Remove rate limit for a virtual key (unlimited)."""
        vk = gate.store.get_virtual_key(key_id)
        if vk is None:
            raise HTTPException(status_code=404, detail="Virtual key not found")
        vk.rate_limit_tps = None
        vk.rate_limit_max_queue = 100
        vk.rate_limit_queue_timeout = 30.0
        gate.store.save_virtual_key(vk)
        return {"status": "removed", "key_id": key_id}

    # --- Agent endpoints ---

    @router.post("/agents")
    async def create_agent(body: CreateAgentRequest) -> dict[str, Any]:
        """Create an agent with an initial version (v1)."""
        try:
            agent = gate.create_agent(
                slug=body.slug,
                team_id=body.team_id,
                name=body.name,
                description=body.description,
                model=body.model,
                system_prompt=body.system_prompt,
                request_overrides=body.request_overrides,
                budget_per_session=body.budget_per_session,
                metadata=body.metadata,
            )
        except Exception as exc:
            if "Invalid agent slug" in str(exc):
                raise HTTPException(status_code=400, detail=str(exc))
            raise HTTPException(status_code=500, detail=str(exc))

        # Fetch the initial version
        version = gate.store.get_agent_version(agent.active_version_id)
        return {
            "agent": _agent_to_dict(agent),
            "version": _agent_version_to_dict(version) if version else None,
        }

    @router.get("/agents")
    async def list_agents(
        team_id: str | None = Query(default=None),
        org_id: str | None = Query(default=None),
        status: str | None = Query(default=None),
        limit: int = Query(default=50, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """List agents with optional filters."""
        agents = gate.store.list_agents(
            team_id=team_id,
            org_id=org_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "agents": [_agent_to_dict(a) for a in agents],
            "total": len(agents),
        }

    @router.get("/agents/{agent_ref}")
    async def get_agent(
        agent_ref: str,
        team_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Get agent detail with active version."""
        agent = _resolve_agent_ref(gate, agent_ref, team_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        result = _agent_to_dict(agent)
        if agent.active_version_id:
            version = gate.store.get_agent_version(agent.active_version_id)
            result["active_version"] = _agent_version_to_dict(version) if version else None
        return result

    @router.patch("/agents/{agent_ref}")
    async def update_agent(
        agent_ref: str,
        body: UpdateAgentRequest,
        team_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Update agent fields."""
        agent = _resolve_agent_ref(gate, agent_ref, team_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        updated = gate.update_agent(
            agent.id,
            name=body.name,
            description=body.description,
            status=body.status,
            metadata=body.metadata,
        )
        return _agent_to_dict(updated)

    @router.delete("/agents/{agent_ref}")
    async def delete_agent(
        agent_ref: str,
        team_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Archive an agent (soft delete)."""
        agent = _resolve_agent_ref(gate, agent_ref, team_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        archived = gate.archive_agent(agent.id)
        return _agent_to_dict(archived)

    @router.post("/agents/{agent_ref}/versions")
    async def create_agent_version(
        agent_ref: str,
        body: CreateAgentVersionRequest,
        team_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Create a new version for an agent."""
        agent = _resolve_agent_ref(gate, agent_ref, team_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        version = gate.create_agent_version(
            agent.id,
            model=body.model,
            system_prompt=body.system_prompt,
            request_overrides=body.request_overrides,
            budget_per_session=body.budget_per_session,
            metadata=body.metadata,
            created_by=body.created_by,
        )
        return _agent_version_to_dict(version)

    @router.get("/agents/{agent_ref}/versions")
    async def list_agent_versions(
        agent_ref: str,
        team_id: str | None = Query(default=None),
        limit: int = Query(default=50, le=500),
    ) -> dict[str, Any]:
        """List versions for an agent (newest first)."""
        agent = _resolve_agent_ref(gate, agent_ref, team_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        versions = gate.store.list_agent_versions(agent.id, limit=limit)
        return {
            "versions": [_agent_version_to_dict(v) for v in versions],
            "total": len(versions),
        }

    @router.put("/agents/{agent_ref}/versions/{version_id}/activate")
    async def activate_agent_version(
        agent_ref: str,
        version_id: str,
        team_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Activate a specific version (rollback)."""
        agent = _resolve_agent_ref(gate, agent_ref, team_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        try:
            updated = gate.activate_agent_version(agent.id, version_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return _agent_to_dict(updated)

    @router.get("/agents/{agent_ref}/sessions")
    async def list_agent_sessions(
        agent_ref: str,
        team_id: str | None = Query(default=None),
        limit: int = Query(default=50, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """List sessions associated with this agent."""
        agent = _resolve_agent_ref(gate, agent_ref, team_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        # Query sessions that have agent_id in metadata
        all_sessions = gate.store.list_sessions(limit=limit, offset=offset)
        agent_sessions = [
            s
            for s in all_sessions
            if s.agent_id == agent.id or (s.metadata and s.metadata.get("agent_id") == agent.id)
        ]
        return {
            "sessions": [_session_to_dict(gate, s) for s in agent_sessions],
            "total": len(agent_sessions),
        }

    # ── Server restart ──────────────────────────────────────────────────

    @router.post("/restart")
    async def restart_server() -> dict[str, Any]:
        """Restart the StateLoom server process.

        Sends SIGUSR1 to self, which triggers a graceful re-exec in the CLI.
        Falls back to SIGTERM (clean shutdown) if SIGUSR1 is not handled.
        """
        import os
        import signal

        pid = os.getpid()
        # Schedule the signal slightly in the future so the response gets sent
        loop = asyncio.get_running_loop()
        loop.call_later(0.5, os.kill, pid, signal.SIGUSR1)
        return {"status": "restarting", "pid": pid}

    # --- Prompt Watcher ---

    @router.get("/prompts/status")
    async def get_prompts_status() -> dict[str, Any]:
        """Get prompt watcher status."""
        if gate._prompt_watcher is None:
            return {"enabled": False}
        return cast(dict[str, Any], gate._prompt_watcher.get_status())

    @router.post("/prompts/rescan")
    async def rescan_prompts() -> dict[str, Any]:
        """Force an immediate full scan of the prompts directory."""
        if gate._prompt_watcher is None:
            raise HTTPException(status_code=404, detail="Prompt watcher not enabled")
        gate._prompt_watcher.scan()
        return cast(dict[str, Any], gate._prompt_watcher.get_status())

    @router.get("/license")
    async def license_status() -> dict[str, Any]:
        """Get enterprise license status with feature availability."""
        from stateloom.ee import is_dev_mode, is_licensed, license_info

        info = license_info()
        features = {}
        if _registry is not None:
            features = _registry.status().get("features", {})
        return {
            "valid": is_licensed(),
            "dev_mode": is_dev_mode(),
            **(info if info else {}),
            "features": features,
        }

    # --- Debug mode endpoints ---

    @router.get("/debug")
    async def debug_status() -> dict[str, Any]:
        """Check if debug mode is enabled."""
        return {"debug": getattr(gate.config, "debug", False)}

    @router.get("/logs")
    async def get_server_logs(
        limit: int = Query(default=200, ge=1, le=2000),
        level: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Get recent server logs (debug mode only)."""
        if not getattr(gate.config, "debug", False):
            raise HTTPException(status_code=404, detail="Debug mode is not enabled")
        from stateloom.dashboard.log_buffer import get_log_buffer

        buf = get_log_buffer()
        if buf is None:
            return {"logs": [], "debug": False}
        return {"logs": buf.get_logs(limit=limit, level=level), "debug": True}

    @router.delete("/logs")
    async def clear_server_logs() -> dict[str, bool]:
        """Clear the server log buffer (debug mode only)."""
        if not getattr(gate.config, "debug", False):
            raise HTTPException(status_code=404, detail="Debug mode is not enabled")
        from stateloom.dashboard.log_buffer import get_log_buffer

        buf = get_log_buffer()
        if buf is not None:
            buf.clear()
        return {"cleared": True}

    # --- Security endpoints ---

    @router.get("/security")
    async def get_security_status() -> dict[str, Any]:
        """Get full security engine status (audit hooks + vault)."""
        return gate.security_status()

    @router.post("/security/audit-hooks/configure")
    async def configure_audit_hooks(request: Request) -> dict[str, Any]:
        """Update audit hook configuration at runtime."""
        body = await request.json()
        enabled = body.get("enabled")
        mode = body.get("mode")
        deny_events = body.get("deny_events")
        allow_paths = body.get("allow_paths")

        # Update config fields
        if enabled is not None:
            gate.config.security_audit_hooks_enabled = bool(enabled)
        if mode is not None:
            from stateloom.core.types import GuardrailMode

            gate.config.security_audit_hooks_mode = GuardrailMode(mode)
        if deny_events is not None:
            gate.config.security_audit_hooks_deny_events = list(deny_events)
        if allow_paths is not None:
            gate.config.security_audit_hooks_allow_paths = list(allow_paths)

        # If hooks not installed yet but enabling, set up security
        if enabled and gate._audit_hook_manager is None:
            gate._setup_security()
        elif gate._audit_hook_manager is not None:
            gate._audit_hook_manager.configure(
                enabled=gate.config.security_audit_hooks_enabled,
                mode=(
                    gate.config.security_audit_hooks_mode.value
                    if hasattr(gate.config.security_audit_hooks_mode, "value")
                    else gate.config.security_audit_hooks_mode
                ),
                deny_events=gate.config.security_audit_hooks_deny_events,
                allow_paths=gate.config.security_audit_hooks_allow_paths,
            )

        return gate.security_status()

    @router.get("/security/vault")
    async def get_vault_status() -> dict[str, Any]:
        """Get vault status (key names, NOT values)."""
        if gate._secret_vault is None:
            return {"enabled": False}
        return cast(dict[str, Any], gate._secret_vault.get_status())

    @router.post("/security/vault/store")
    async def store_vault_secret(request: Request) -> dict[str, Any]:
        """Store a new secret in the vault."""
        body = await request.json()
        name = body.get("name", "")
        value = body.get("value", "")
        if not name or not value:
            raise HTTPException(status_code=400, detail="name and value required")

        if gate._secret_vault is None:
            from stateloom.security.vault import SecretVault

            gate._secret_vault = SecretVault()
            gate._secret_vault.configure(enabled=True)

        gate._secret_vault.store(name, value)
        return {"stored": True, "key": name}

    @router.get("/security/events")
    async def get_security_events(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        """Get recent security audit events."""
        if gate._audit_hook_manager is None:
            return {"events": []}
        status = gate._audit_hook_manager.get_status()
        events = status.get("recent_events", [])
        return {"events": events[-limit:]}

    @router.get("/security/guardrails")
    async def get_guardrails_status() -> dict[str, Any]:
        """Get guardrails config status and aggregated detection stats."""
        from stateloom.core.event import GuardrailEvent

        gr_cfg = _read_guardrails_config(gate)

        config_status = {
            **gr_cfg,
            "nli_available": _check_nli_available(),
            "pattern_count": (len(gate._guardrails._heuristic_patterns) if gate._guardrails else 0),
        }

        # Aggregate detection stats from guardrail events
        events = gate.store.get_session_events("", event_type="guardrail", limit=50000)
        total = len(events)
        blocked = 0
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for e in events:
            if isinstance(e, GuardrailEvent):
                if e.action_taken == ActionTaken.BLOCKED:
                    blocked += 1
                cat = e.category or "unknown"
                by_category[cat] = by_category.get(cat, 0) + 1
                sev = e.severity or "medium"
                by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "config": config_status,
            "stats": {
                "total": total,
                "blocked": blocked,
                "by_category": by_category,
                "by_severity": by_severity,
            },
        }

    @router.post("/security/guardrails/configure")
    async def configure_guardrails(request: Request) -> dict[str, Any]:
        """Update guardrails configuration at runtime."""
        body = await request.json()

        enabled = body.get("enabled")
        nli_enabled = body.get("nli_enabled")
        nli_threshold = body.get("nli_threshold")
        heuristic_enabled = body.get("heuristic_enabled")
        mode = body.get("mode")
        local_model_enabled = body.get("local_model_enabled")
        output_scanning_enabled = body.get("output_scanning_enabled")

        if enabled is not None:
            gate.config.guardrails_enabled = bool(enabled)
        if nli_enabled is not None:
            gate.config.guardrails_nli_enabled = bool(nli_enabled)
        if nli_threshold is not None:
            gate.config.guardrails_nli_threshold = float(nli_threshold)
        if heuristic_enabled is not None:
            gate.config.guardrails_heuristic_enabled = bool(heuristic_enabled)
        if mode is not None:
            from stateloom.core.types import GuardrailMode

            gate.config.guardrails_mode = GuardrailMode(mode)
        if local_model_enabled is not None:
            gate.config.guardrails_local_model_enabled = bool(local_model_enabled)
        if output_scanning_enabled is not None:
            gate.config.guardrails_output_scanning_enabled = bool(output_scanning_enabled)

        _persist_guardrails_state(gate)

        # Return updated status
        return await get_guardrails_status()

    @router.get("/security/guardrails/events")
    async def get_guardrail_events(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        """Get recent guardrail events with all fields."""
        from stateloom.core.event import GuardrailEvent

        events = gate.store.get_session_events("", event_type="guardrail", limit=limit)
        result = []
        for e in events:
            if isinstance(e, GuardrailEvent):
                result.append(
                    {
                        "timestamp": e.timestamp.isoformat() if e.timestamp else "",
                        "session_id": e.session_id,
                        "rule_name": e.rule_name,
                        "category": e.category,
                        "severity": e.severity,
                        "score": round(e.score, 3),
                        "action_taken": e.action_taken,
                        "violation_text": (e.violation_text[:200] if e.violation_text else ""),
                        "scan_phase": e.scan_phase,
                        "validator_type": e.validator_type,
                    }
                )
        return {"events": result}

    # --- Consensus endpoints ---

    @router.get("/consensus-runs")
    async def list_consensus_runs(
        limit: int = Query(default=20, le=500),
        offset: int = Query(default=0, ge=0),
        strategy: str | None = Query(default=None),
    ) -> dict[str, Any]:
        if strategy:
            # Strategy filter is applied in-memory, so fetch all and paginate after
            all_events = gate.store.get_session_events(
                "", event_type="consensus", limit=10000, desc=True
            )
            filtered = [e for e in all_events if getattr(e, "strategy", "") == strategy]
            total = len(filtered)
            page = filtered[offset : offset + limit]
        else:
            total = gate.store.count_events(event_type="consensus")
            page = gate.store.get_session_events(
                "", event_type="consensus", limit=limit, offset=offset, desc=True
            )
        runs = []
        for e in page:
            runs.append(
                {
                    "session_id": e.session_id,
                    "strategy": getattr(e, "strategy", ""),
                    "models": getattr(e, "models", []),
                    "total_rounds": getattr(e, "total_rounds", 0),
                    "confidence": round(getattr(e, "confidence", 0.0), 4),
                    "total_cost": round(getattr(e, "total_cost", 0.0), 6),
                    "total_duration_ms": round(getattr(e, "total_duration_ms", 0.0), 1),
                    "early_stopped": getattr(e, "early_stopped", False),
                    "aggregation_method": getattr(e, "aggregation_method", ""),
                    "winner_model": getattr(e, "winner_model", ""),
                    "final_answer_preview": getattr(e, "final_answer_preview", ""),
                    "timestamp": e.timestamp.isoformat(),
                }
            )
        return {"runs": runs, "total": total}

    @router.get("/consensus-runs/{session_id}")
    async def get_consensus_run(session_id: str) -> dict[str, Any]:
        session = gate.store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get consensus event for this session
        consensus_events = gate.store.get_session_events(
            session_id, event_type="consensus", limit=1
        )
        if not consensus_events:
            raise HTTPException(status_code=404, detail="No consensus data for this session")
        ce = consensus_events[0]

        # Get debate round events
        round_events = gate.store.get_session_events(
            session_id, event_type="debate_round", limit=100
        )
        rounds = []
        for re in round_events:
            rounds.append(
                {
                    "round_number": getattr(re, "round_number", 0),
                    "strategy": getattr(re, "strategy", ""),
                    "models": getattr(re, "models", []),
                    "responses_summary": getattr(re, "responses_summary", []),
                    "agreement_score": round(getattr(re, "agreement_score", 0.0), 4),
                    "consensus_reached": getattr(re, "consensus_reached", False),
                    "round_cost": round(getattr(re, "round_cost", 0.0), 6),
                    "round_duration_ms": round(getattr(re, "round_duration_ms", 0.0), 1),
                    "timestamp": re.timestamp.isoformat(),
                }
            )

        # Get child sessions (debater sessions)
        children = gate.store.list_child_sessions(session_id, limit=100)

        return {
            "session_id": session_id,
            "session": _session_to_dict(gate, session),
            "consensus": {
                "strategy": getattr(ce, "strategy", ""),
                "models": getattr(ce, "models", []),
                "total_rounds": getattr(ce, "total_rounds", 0),
                "confidence": round(getattr(ce, "confidence", 0.0), 4),
                "total_cost": round(getattr(ce, "total_cost", 0.0), 6),
                "total_duration_ms": round(getattr(ce, "total_duration_ms", 0.0), 1),
                "early_stopped": getattr(ce, "early_stopped", False),
                "aggregation_method": getattr(ce, "aggregation_method", ""),
                "winner_model": getattr(ce, "winner_model", ""),
                "final_answer_preview": getattr(ce, "final_answer_preview", ""),
            },
            "rounds": rounds,
            "children": [_session_to_dict(gate, c) for c in children],
        }

    return router


def _persist_kill_switch_state(gate: Any) -> None:
    """Persist kill switch state to the store for cross-process propagation."""
    import json

    try:
        gate.store.save_secret("kill_switch_active", "1" if gate.config.kill_switch_active else "0")
        gate.store.save_secret("kill_switch_message", gate.config.kill_switch_message)
        gate.store.save_secret("kill_switch_response_mode", gate.config.kill_switch_response_mode)
        rules = [r.model_dump() for r in gate.config.kill_switch_rules]
        gate.store.save_secret("kill_switch_rules_json", json.dumps(rules))
    except Exception:
        logger.debug("Failed to persist kill switch state to store", exc_info=True)


def _persist_local_routing_state(gate: Any) -> None:
    """Persist local routing config to the store so it survives re-init."""
    import json

    try:
        blob = json.dumps(
            {
                "auto_route_force_local": gate.config.auto_route_force_local,
                "auto_route_enabled": gate.config.auto_route_enabled,
                "local_model_default": gate.config.local_model_default,
                "local_model_enabled": gate.config.local_model_enabled,
            }
        )
        gate.store.save_secret("local_routing_config_json", blob)
    except Exception:
        logger.debug("Failed to persist local routing state to store", exc_info=True)


def _persist_pii_state(gate: Any) -> None:
    """Persist PII config to the store for cross-process propagation."""
    import json

    try:
        blob = json.dumps(
            {
                "enabled": gate.config.pii_enabled,
                "default_mode": gate.config.pii_default_mode.value,
                "rules": [r.model_dump() for r in gate.config.pii_rules],
            }
        )
        gate.store.save_secret("pii_config_json", blob)
    except Exception:
        logger.debug("Failed to persist PII state to store", exc_info=True)


def _persist_budget_state(gate: Any) -> None:
    """Persist budget config to the store for cross-process propagation."""
    import json

    try:
        blob = json.dumps(
            {
                "budget_per_session": gate.config.budget_per_session,
                "budget_global": gate.config.budget_global,
                "budget_action": gate.config.budget_action.value,
            }
        )
        gate.store.save_secret("budget_config_json", blob)
    except Exception:
        logger.debug("Failed to persist budget state to store", exc_info=True)


def _persist_blast_radius_state(gate: Any) -> None:
    """Persist blast radius config to the store for cross-process propagation."""
    import json

    try:
        blob = json.dumps(
            {
                "enabled": gate.config.blast_radius_enabled,
                "consecutive_failures": gate.config.blast_radius_consecutive_failures,
                "budget_violations_per_hour": gate.config.blast_radius_budget_violations_per_hour,
            }
        )
        gate.store.save_secret("blast_radius_config_json", blob)
    except Exception:
        logger.debug("Failed to persist blast radius state to store", exc_info=True)


def _check_nli_available() -> bool:
    """Check if the NLI classifier backend (sentence-transformers) is installed."""
    try:
        from stateloom.guardrails.nli_classifier import _ST_AVAILABLE

        return _ST_AVAILABLE
    except Exception:
        return False


def _read_guardrails_config(gate: Any) -> dict[str, Any]:
    """Read guardrails config, merging store (source of truth) with gate defaults.

    The store holds values persisted by the dashboard POST endpoint.
    Gate config provides defaults for any fields not yet persisted.
    This is a pure read — gate.config is NOT mutated.
    """
    import json

    cfg = gate.config
    result: dict[str, Any] = {
        "enabled": cfg.guardrails_enabled,
        "mode": cfg.guardrails_mode.value if cfg.guardrails_enabled else "audit",
        "heuristic_enabled": cfg.guardrails_heuristic_enabled,
        "nli_enabled": cfg.guardrails_nli_enabled,
        "nli_threshold": cfg.guardrails_nli_threshold,
        "local_model_enabled": cfg.guardrails_local_model_enabled,
        "local_model": cfg.guardrails_local_model,
        "output_scanning_enabled": cfg.guardrails_output_scanning_enabled,
    }

    try:
        blob = gate.store.get_secret("guardrails_config_json")
        if blob:
            data = json.loads(blob)
            if "enabled" in data:
                result["enabled"] = bool(data["enabled"])
            if "mode" in data:
                result["mode"] = data["mode"]
            if "nli_enabled" in data:
                result["nli_enabled"] = bool(data["nli_enabled"])
            if "heuristic_enabled" in data:
                result["heuristic_enabled"] = bool(data["heuristic_enabled"])
            if "nli_threshold" in data:
                result["nli_threshold"] = float(data["nli_threshold"])
            if "local_model_enabled" in data:
                result["local_model_enabled"] = bool(data["local_model_enabled"])
            if "output_scanning_enabled" in data:
                result["output_scanning_enabled"] = bool(data["output_scanning_enabled"])
    except Exception:
        logger.warning("Failed to read guardrails config from store", exc_info=True)

    return result


def _persist_guardrails_state(gate: Any) -> None:
    """Persist guardrails config to the store for cross-process propagation."""
    import json

    try:
        blob = json.dumps(
            {
                "enabled": gate.config.guardrails_enabled,
                "mode": gate.config.guardrails_mode.value,
                "heuristic_enabled": gate.config.guardrails_heuristic_enabled,
                "nli_enabled": gate.config.guardrails_nli_enabled,
                "nli_threshold": gate.config.guardrails_nli_threshold,
                "local_model_enabled": gate.config.guardrails_local_model_enabled,
                "output_scanning_enabled": gate.config.guardrails_output_scanning_enabled,
                "disabled_rules": gate.config.guardrails_disabled_rules,
            }
        )
        gate.store.save_secret("guardrails_config_json", blob)
    except Exception:
        logger.warning("Failed to persist guardrails state to store", exc_info=True)


def _apply_compliance_pii_rules(gate: Any, profile: Any) -> None:
    """Enable PII scanning and inject compliance profile PII rules."""
    if not profile.pii_rules:
        return
    gate.config.pii_enabled = True
    # Merge rules — compliance rules override existing ones for same pattern
    existing_patterns = {r.pattern for r in gate.config.pii_rules}
    for rule in profile.pii_rules:
        if rule.pattern not in existing_patterns:
            gate.config.pii_rules.append(rule)
            existing_patterns.add(rule.pattern)
    # Add PII scanner to pipeline if not already present
    has_pii_scanner = any(hasattr(mw, "reload_rules") for mw in gate.pipeline.middlewares)
    if not has_pii_scanner:
        try:
            from stateloom.middleware.pii_scanner import PIIScannerMiddleware

            scanner = PIIScannerMiddleware(
                gate.config,
                store=gate.store,
                org_rules_fn=gate._get_org_pii_rules,
            )
            gate.pipeline.add(scanner)
        except ImportError:
            pass
    else:
        _reload_pii_scanner(gate)


def _reload_pii_scanner(gate: Any) -> None:
    """Reload PII scanner rules from config."""
    for mw in gate.pipeline.middlewares:
        if hasattr(mw, "reload_rules"):
            mw.reload_rules()
            break


def _org_to_dict(org: Any) -> dict[str, Any]:
    """Convert an organization to a dict."""
    result = {
        "id": org.id,
        "name": org.name,
        "status": org.status.value,
        "created_at": org.created_at.isoformat(),
        "budget": org.budget,
        "total_cost": round(org.total_cost, 6),
        "total_tokens": org.total_tokens,
        "pii_rules": [r.model_dump() for r in org.pii_rules],
        "metadata": org.metadata,
    }
    cp = getattr(org, "compliance_profile", None)
    result["compliance_profile"] = cp.model_dump() if cp else None
    return result


def _team_to_dict(team: Any) -> dict[str, Any]:
    """Convert a team to a dict."""
    result = {
        "id": team.id,
        "org_id": team.org_id,
        "name": team.name,
        "status": team.status.value,
        "created_at": team.created_at.isoformat(),
        "budget": team.budget,
        "total_cost": round(team.total_cost, 6),
        "total_tokens": team.total_tokens,
        "metadata": team.metadata,
        "rate_limit_tps": getattr(team, "rate_limit_tps", None),
        "rate_limit_priority": getattr(team, "rate_limit_priority", 0),
        "rate_limit_max_queue": getattr(team, "rate_limit_max_queue", 100),
        "rate_limit_queue_timeout": getattr(team, "rate_limit_queue_timeout", 30.0),
    }
    cp = getattr(team, "compliance_profile", None)
    result["compliance_profile"] = cp.model_dump() if cp else None
    return result


def _session_to_dict(gate: Any, s: Any) -> dict[str, Any]:
    """Convert a session to a dict, including experiment info from metadata."""
    result = {
        "id": s.id,
        "name": s.name,
        "org_id": s.org_id,
        "team_id": s.team_id,
        "started_at": s.started_at.isoformat(),
        "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        "status": s.status.value,
        "total_cost": round(s.total_cost, 6),
        "estimated_api_cost": round(getattr(s, "estimated_api_cost", 0.0), 6),
        "total_tokens": s.total_tokens,
        "call_count": s.call_count,
        "cache_hits": s.cache_hits,
        "cache_savings": round(s.cache_savings, 6),
        "pii_detections": s.pii_detections,
        "guardrail_detections": getattr(s, "guardrail_detections", 0),
        "parent_session_id": s.parent_session_id,
        "budget": s.budget,
        "end_user": s.end_user,
        "cost_by_model": {k: round(v, 6) for k, v in getattr(s, "cost_by_model", {}).items()},
        "tokens_by_model": getattr(s, "tokens_by_model", {}),
    }
    if s.metadata:
        result["experiment_id"] = s.metadata.get("experiment_id")
        result["experiment_name"] = s.metadata.get("experiment_name")
        result["variant"] = s.metadata.get("variant")
        result["billing_mode"] = s.metadata.get("billing_mode", "api")
        result["metadata"] = s.metadata
    return result


def _job_to_dict(job: Any) -> dict[str, Any]:
    """Convert a Job to a JSON-safe dict."""
    from stateloom.core.job import Job

    if isinstance(job, Job):
        d = job.model_dump(mode="json")
        # Exclude sensitive/internal fields; keep backwards compat
        d.pop("messages", None)
        d.pop("request_kwargs", None)
        d.pop("webhook_secret", None)
        # Only include result/error when present
        if d.get("result") is None:
            d.pop("result", None)
        if not d.get("error"):
            d.pop("error", None)
            d.pop("error_code", None)
        return d

    # Fallback for non-Pydantic objects
    result: dict[str, Any] = {
        "id": job.id,
        "session_id": job.session_id,
        "org_id": job.org_id,
        "team_id": job.team_id,
        "status": job.status.value,
        "provider": job.provider,
        "model": job.model,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "retry_count": job.retry_count,
        "max_retries": job.max_retries,
        "ttl_seconds": job.ttl_seconds,
        "webhook_url": job.webhook_url,
        "webhook_status": job.webhook_status,
        "metadata": job.metadata,
    }
    if job.result is not None:
        result["result"] = job.result
    if job.error:
        result["error"] = job.error
        result["error_code"] = job.error_code
    return result


def _resolve_agent_ref(gate: Any, agent_ref: str, team_id: str | None) -> Any:
    """Resolve an agent reference (ID or slug) in dashboard context."""
    if agent_ref.startswith("agt-"):
        return gate.get_agent(agent_ref)
    if team_id:
        return gate.get_agent_by_slug(agent_ref, team_id)
    return None


def _agent_to_dict(agent: Any) -> dict[str, Any]:
    """Convert an Agent to a dict."""
    from stateloom.agent.models import Agent

    if isinstance(agent, Agent):
        return agent.model_dump(mode="json")

    return {
        "id": agent.id,
        "slug": agent.slug,
        "team_id": agent.team_id,
        "org_id": agent.org_id,
        "name": agent.name,
        "description": agent.description,
        "status": agent.status.value,
        "active_version_id": agent.active_version_id,
        "metadata": agent.metadata,
        "created_at": agent.created_at.isoformat(),
        "updated_at": agent.updated_at.isoformat(),
    }


def _agent_version_to_dict(version: Any) -> dict[str, Any]:
    """Convert an AgentVersion to a dict."""
    from stateloom.agent.models import AgentVersion

    if isinstance(version, AgentVersion):
        return version.model_dump(mode="json")

    return {
        "id": version.id,
        "agent_id": version.agent_id,
        "version_number": version.version_number,
        "model": version.model,
        "system_prompt": version.system_prompt,
        "request_overrides": version.request_overrides,
        "compliance_profile_json": version.compliance_profile_json,
        "budget_per_session": version.budget_per_session,
        "metadata": version.metadata,
        "created_at": version.created_at.isoformat(),
        "created_by": version.created_by,
    }


_EVENT_BASE_FIELDS = {"id", "session_id", "step", "event_type", "timestamp", "metadata"}


def _event_details(event: object) -> dict[str, Any]:
    """Extract type-specific details from an event."""
    from stateloom.core.event import Event

    if isinstance(event, Event):
        d = event.model_dump(mode="json", exclude=_EVENT_BASE_FIELDS)
        details = {
            k: v
            for k, v in d.items()
            if v is not None
            and v != ""
            and v != 0
            and v != 0.0
            and v is not False
            and v != {}
            and v != []
        }
        # Mark durable checkpoint steps
        if hasattr(event, "cached_response_json") and event.cached_response_json:
            details.pop("cached_response_json", None)
            details["has_cached_response"] = True
        # Mark events with stored request messages (lazy-loaded via separate endpoint)
        if hasattr(event, "request_messages_json") and event.request_messages_json:
            details.pop("request_messages_json", None)
            details["has_request_messages"] = True
        return details

    # Fallback for non-Pydantic event objects (shouldn't happen in practice)
    fallback: dict[str, Any] = {}
    for attr in dir(event):
        if attr.startswith("_") or attr in _EVENT_BASE_FIELDS:
            continue
        val = getattr(event, attr, None)
        if val is not None and not callable(val):
            fallback[attr] = val
    return fallback


def _event_to_dict(e: object) -> dict[str, Any]:
    """Convert an event to a serializable dict (shared by multiple endpoints)."""
    return {
        "id": e.id,  # type: ignore[attr-defined]
        "step": e.step,  # type: ignore[attr-defined]
        "event_type": e.event_type.value,  # type: ignore[attr-defined]
        "timestamp": e.timestamp.isoformat(),  # type: ignore[attr-defined]
        **({"metadata": e.metadata} if e.metadata else {}),  # type: ignore[attr-defined]
        **_event_details(e),
    }


def _compute_tool_summaries(
    events: list[object],
) -> tuple[list[object], dict[int, dict[str, Any]]]:
    """Separate primary events from tool continuations, computing summaries.

    Returns ``(primary_events, tool_summaries)`` where *tool_summaries* maps
    parent step numbers to aggregate dicts.

    Absorbed events (collapsed into the parent's tool summary):
    - ``is_tool_continuation=True`` LLM calls (proxy tool-use sub-calls)
    - ``ToolCallEvent``s that immediately follow an LLM call (framework
      callback handler tool tracking, e.g. LangChain/LangGraph)
    - ``is_cli_internal=True`` LLM calls are dropped entirely
    """
    from stateloom.core.event import LLMCallEvent, ToolCallEvent

    primary: list[object] = []
    tool_summaries: dict[int, dict[str, Any]] = {}

    current_parent_step: int | None = None
    current_tools: list[object] = []

    def _flush_tools() -> None:
        nonlocal current_tools
        if current_tools and current_parent_step is not None:
            tool_summaries[current_parent_step] = {
                "count": len(current_tools),
                "total_tokens": sum(
                    (getattr(t, "prompt_tokens", 0) or 0)
                    + (getattr(t, "completion_tokens", 0) or 0)
                    for t in current_tools
                ),
                "total_cost": round(sum(getattr(t, "cost", 0) or 0 for t in current_tools), 6),
                "total_latency_ms": round(
                    sum(getattr(t, "latency_ms", 0) or 0 for t in current_tools), 1
                ),
                "first_step": current_tools[0].step,  # type: ignore[attr-defined]
                "last_step": current_tools[-1].step,  # type: ignore[attr-defined]
            }
            current_tools = []

    for e in events:
        is_llm = isinstance(e, LLMCallEvent)
        is_tool = isinstance(e, ToolCallEvent)
        is_tc = is_llm and getattr(e, "is_tool_continuation", False)
        is_cli = is_llm and getattr(e, "is_cli_internal", False)

        if is_tc or is_cli:
            if is_tc and current_parent_step is not None:
                current_tools.append(e)
            # CLI internal events are dropped entirely
            continue

        # ToolCallEvents following an LLM parent are absorbed into the group
        if is_tool and current_parent_step is not None:
            current_tools.append(e)
            continue

        # Non-tool-continuation, non-CLI-internal event — flush accumulated tools
        _flush_tools()

        primary.append(e)
        if is_llm:
            current_parent_step = e.step  # type: ignore[attr-defined]
        else:
            # Non-LLM primary event breaks the tool group chain
            current_parent_step = None

    # Flush any trailing tool continuations
    _flush_tools()

    return primary, tool_summaries
