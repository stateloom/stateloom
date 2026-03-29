"""Observability REST API endpoints for the StateLoom dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

if TYPE_CHECKING:
    from stateloom.gate import Gate


def create_observability_router(gate: Gate) -> APIRouter:
    """Create the observability API router."""
    router = APIRouter()

    @router.get("/observability/timeseries")
    async def get_timeseries(
        window: str = Query(default="1h", pattern="^(1h|6h|24h|7d)$"),
        org_id: str = Query(default=""),
        team_id: str = Query(default=""),
    ):
        return gate._observability_aggregator.get_timeseries(window, org_id=org_id, team_id=team_id)

    @router.get("/observability/latency")
    async def get_latency(
        window: str = Query(default="1h", pattern="^(1h|6h|24h|7d)$"),
        org_id: str = Query(default=""),
        team_id: str = Query(default=""),
    ):
        return gate._observability_aggregator.get_latency(window, org_id=org_id, team_id=team_id)

    @router.get("/observability/breakdown")
    async def get_breakdown(
        window: str = Query(default="1h", pattern="^(1h|6h|24h|7d)$"),
        org_id: str = Query(default=""),
        team_id: str = Query(default=""),
    ):
        return gate._observability_aggregator.get_breakdown(window, org_id=org_id, team_id=team_id)

    return router


def create_metrics_endpoint(gate: Gate) -> PlainTextResponse:
    """Generate a Prometheus /metrics response."""
    if gate._metrics_collector is not None:
        body = gate._metrics_collector.generate_metrics()
    else:
        body = "# StateLoom metrics not enabled (set metrics_enabled=True)\n"
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4; charset=utf-8")
