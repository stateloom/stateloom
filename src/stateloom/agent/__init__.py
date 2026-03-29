"""Managed agent definitions (Prompts-as-an-API).

Note: Enterprise agent features are now in stateloom.ee.agent.
This module provides backward compatibility.
"""

from __future__ import annotations

from stateloom.agent.models import Agent, AgentVersion, validate_slug
from stateloom.agent.prompt_file import PromptFileContent, parse_prompt_file
from stateloom.agent.resolver import AgentResolutionError, apply_agent_overrides, resolve_agent

__all__ = [
    "Agent",
    "AgentVersion",
    "AgentResolutionError",
    "PromptFileContent",
    "apply_agent_overrides",
    "parse_prompt_file",
    "resolve_agent",
    "validate_slug",
]
