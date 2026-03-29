"""Replay safety analysis — warn about undecorated tool calls."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

from rich.console import Console
from rich.text import Text

from stateloom.replay.step import StepRecord

logger = logging.getLogger("stateloom.replay.safety")
_console = Console(stderr=True)

SAFETY_TIMEOUT_SECONDS = 10


@dataclass
class SafetyWarning:
    """A warning about a potentially unsafe replay step."""

    step: int
    message: str
    tool_name: str = ""


def analyze_replay_safety(steps: list[StepRecord], mock_until_step: int) -> list[SafetyWarning]:
    """Check for untracked tool calls in the replay range.

    Returns a list of warnings for steps that will RE-EXECUTE
    during replay because they are not decorated with @gate.tool().
    """
    warnings: list[SafetyWarning] = []

    for step_record in steps:
        if step_record.step > mock_until_step:
            break

        # Undecorated tool calls that might re-execute
        if step_record.event_type.value == "tool_call" and step_record.tool_name is None:
            warnings.append(
                SafetyWarning(
                    step=step_record.step,
                    message=(
                        f"Step {step_record.step} is an untracked tool execution "
                        f"that will RE-EXECUTE during replay."
                    ),
                    tool_name="unknown",
                )
            )

    return warnings


def display_safety_warnings(warnings: list[SafetyWarning]) -> bool:
    """Display safety warnings and wait for user confirmation.

    Returns True if the user wants to proceed, False to cancel.
    """
    if not warnings:
        return True

    _console.print()
    warning_text = Text()
    warning_text.append("[StateLoom] ", style="bold cyan")
    warning_text.append(" REPLAY WARNING: ", style="bold red")
    step_nums = ", ".join(str(w.step) for w in warnings)
    warning_text.append(
        f"Steps {step_nums} include untracked tool executions.",
        style="bold",
    )
    _console.print(warning_text)

    _console.print("  These functions will RE-EXECUTE during replay:", style="yellow")
    for w in warnings:
        _console.print(f"    step {w.step}: {w.tool_name}", style="dim")

    _console.print()
    _console.print(
        "  Decorate with @gate.tool(mutates_state=True) to mock during replay.",
        style="dim",
    )
    _console.print(
        "  Proceeding will take real actions. Ctrl+C to cancel.",
        style="bold yellow",
    )

    # Countdown
    for remaining in range(SAFETY_TIMEOUT_SECONDS, 0, -1):
        _console.print(f"  [{remaining}s]", end="\r", style="dim")
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            _console.print("\n  Replay cancelled.", style="red")
            return False

    _console.print()
    return True
