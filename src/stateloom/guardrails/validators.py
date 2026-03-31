"""Protocol and result types for pluggable guardrail validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class GuardrailResult:
    """Result from a guardrail validation check."""

    safe: bool = True
    category: str = ""
    score: float = 0.0
    raw_output: str = ""
    rule_name: str = ""
    severity: str = "medium"


@runtime_checkable
class GuardrailValidator(Protocol):
    """Protocol for pluggable guardrail validators (enterprise API integration)."""

    @property
    def name(self) -> str: ...

    def validate_input(self, messages: list[dict[str, Any]]) -> GuardrailResult: ...

    def validate_output(self, response_text: str, system_prompt: str) -> GuardrailResult: ...
