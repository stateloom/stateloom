"""Rich terminal output for StateLoom events."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.text import Text

from stateloom.core.config import StateLoomConfig
from stateloom.core.event import (
    BlastRadiusEvent,
    BudgetEnforcementEvent,
    CacheHitEvent,
    KillSwitchEvent,
    LLMCallEvent,
    LocalRoutingEvent,
    LoopDetectionEvent,
    PIIDetectionEvent,
)
from stateloom.middleware.base import MiddlewareContext

logger = logging.getLogger("stateloom.export.console")

_console = Console(stderr=True)


class ConsoleOutput:
    """Prints a one-liner for every LLM call to the terminal."""

    def __init__(self, config: StateLoomConfig) -> None:
        self._config = config

    async def process(
        self,
        ctx: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        result = await call_next(ctx)

        if ctx.is_streaming:
            ctx._on_stream_complete.append(lambda: self._print_events(ctx))
        else:
            self._print_events(ctx)

        return result

    def _print_events(self, ctx: MiddlewareContext) -> None:
        """Print all events in the context."""
        for event in ctx.events:
            if isinstance(event, LLMCallEvent):
                self._print_llm_call(event)
            elif isinstance(event, CacheHitEvent):
                self._print_cache_hit(event)
            elif isinstance(event, PIIDetectionEvent):
                self._print_pii(event)
            elif isinstance(event, LoopDetectionEvent):
                self._print_loop(event)
            elif isinstance(event, BudgetEnforcementEvent):
                self._print_budget(event)
            elif isinstance(event, LocalRoutingEvent):
                self._print_routing(event)
            elif isinstance(event, KillSwitchEvent):
                self._print_kill_switch(event)
            elif isinstance(event, BlastRadiusEvent):
                self._print_blast_radius(event)

    def _print_llm_call(self, event: LLMCallEvent) -> None:
        line = Text()
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  ", style="green")
        line.append(f"{event.model} ", style="bold")
        line.append(f"| {event.total_tokens} tok ", style="dim")
        line.append(f"| ${event.cost:.4f} ", style="yellow")
        line.append(f"| {event.latency_ms:.0f}ms ", style="dim")
        line.append(f"| session:{event.session_id[:20]}", style="blue")
        _console.print(line)

    def _print_cache_hit(self, event: CacheHitEvent) -> None:
        line = Text()
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  CACHE HIT ", style="bold yellow")
        if event.match_type == "semantic" and event.similarity_score is not None:
            line.append(
                f"| semantic match ({event.similarity_score:.3f}) | saved ${event.saved_cost:.4f}",
                style="dim",
            )
        else:
            line.append(f"| exact match | saved ${event.saved_cost:.4f}", style="dim")
        _console.print(line)

    def _print_pii(self, event: PIIDetectionEvent) -> None:
        line = Text()
        line.append("[StateLoom] ", style="bold cyan")
        if event.action_taken == "blocked":
            line.append("  PII BLOCKED ", style="bold red")
        else:
            line.append("  PII DETECTED ", style="bold yellow")
        mode_label = f"({event.mode})" if event.mode else ""
        line.append(f"{mode_label} ", style="dim")
        line.append(f"| type:{event.pii_type} ", style="")
        if event.pii_field:
            line.append(f"| field:{event.pii_field}", style="dim")
        _console.print(line)

    def _print_loop(self, event: LoopDetectionEvent) -> None:
        line = Text()
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  LOOP DETECTED ", style="bold red")
        line.append(f"| calls:{event.repeat_count} ", style="")
        line.append(f"| action:{event.action}", style="dim")
        _console.print(line)

    def _print_budget(self, event: BudgetEnforcementEvent) -> None:
        line = Text()
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  BUDGET ENFORCED ", style="bold red")
        line.append(f"| limit:${event.limit:.2f} ", style="")
        line.append(f"| spent:${event.spent:.2f} ", style="yellow")
        line.append(f"| action:{event.action}", style="dim")
        _console.print(line)

    def _print_routing(self, event: LocalRoutingEvent) -> None:
        line = Text()
        line.append("[StateLoom] ", style="bold cyan")
        if event.routing_success:
            line.append("  ROUTED LOCAL ", style="bold green")
            line.append(f"| {event.local_model} ", style="bold")
            line.append(f"| complexity:{event.complexity_score:.2f} ", style="dim")
            line.append(f"| reason:{event.routing_reason}", style="dim")
        else:
            line.append("  ROUTE FALLBACK ", style="bold yellow")
            line.append(f"| {event.local_model} ", style="bold")
            line.append(f"| reason:{event.routing_reason}", style="dim")
        _console.print(line)

    def _print_kill_switch(self, event: KillSwitchEvent) -> None:
        line = Text()
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  KILL SWITCH ", style="bold red")
        line.append(f"| {event.reason} ", style="")
        if event.blocked_model:
            line.append(f"| model:{event.blocked_model} ", style="dim")
        if event.blocked_provider:
            line.append(f"| provider:{event.blocked_provider}", style="dim")
        _console.print(line)

    def _print_blast_radius(self, event: BlastRadiusEvent) -> None:
        line = Text()
        line.append("[StateLoom] ", style="bold cyan")
        line.append("  BLAST RADIUS ", style="bold red")
        line.append(f"| {event.trigger} ", style="")
        line.append(f"| {event.count}/{event.threshold} ", style="yellow")
        line.append(f"| action:{event.action}", style="dim")
        if event.agent_id:
            line.append(f" | agent:{event.agent_id}", style="dim")
        _console.print(line)
