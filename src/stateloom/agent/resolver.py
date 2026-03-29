"""Agent resolution and override application for proxy routing."""

from __future__ import annotations

from typing import Any

from stateloom.agent.models import Agent, AgentVersion
from stateloom.core.types import AgentStatus


class AgentResolutionError(Exception):
    """Raised when agent resolution fails."""

    def __init__(self, message: str, status_code: int = 404) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def resolve_agent(
    store: Any,
    agent_ref: str,
    vk_team_id: str,
    vk_agent_ids: list[str] | None = None,
) -> tuple[Agent, AgentVersion]:
    """Resolve an agent by slug or ID, enforcing access controls.

    Args:
        store: The store instance.
        agent_ref: Agent slug or ID (starts with "agt-").
        vk_team_id: The virtual key's team_id.
        vk_agent_ids: If non-empty, restricts access to these agent IDs.

    Returns:
        (Agent, AgentVersion) tuple.

    Raises:
        AgentResolutionError: On any resolution failure.
    """
    agent: Agent | None = None

    if agent_ref.startswith("agt-"):
        agent = store.get_agent(agent_ref)
    else:
        agent = store.get_agent_by_slug(agent_ref, vk_team_id)

    if agent is None:
        raise AgentResolutionError("Agent not found", status_code=404)

    # VK agent_ids scoping
    if vk_agent_ids and agent.id not in vk_agent_ids:
        raise AgentResolutionError("Agent not accessible with this API key", status_code=403)

    # Status checks
    if agent.status == AgentStatus.PAUSED:
        raise AgentResolutionError("Agent is paused", status_code=403)
    if agent.status == AgentStatus.ARCHIVED:
        raise AgentResolutionError("Agent has been archived", status_code=410)

    # Active version
    if not agent.active_version_id:
        raise AgentResolutionError("Agent has no active version", status_code=404)

    version = store.get_agent_version(agent.active_version_id)
    if version is None:
        raise AgentResolutionError("Agent has no active version", status_code=404)

    return agent, version


def apply_agent_overrides(
    version: AgentVersion,
    messages: list[dict[str, Any]],
    body: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Apply agent version overrides to a request.

    Args:
        version: The active agent version.
        messages: The request messages list.
        body: The full request body dict.

    Returns:
        (model, messages, extra_kwargs) tuple.
    """
    model = version.model

    # System prompt: prepend to existing or insert new
    if version.system_prompt:
        messages = list(messages)  # don't mutate original
        if messages and messages[0].get("role") == "system":
            messages = list(messages)
            messages[0] = {
                **messages[0],
                "content": version.system_prompt + "\n\n" + messages[0].get("content", ""),
            }
        else:
            messages = [{"role": "system", "content": version.system_prompt}] + messages

    # Extract extra kwargs from body
    extra_kwargs: dict[str, Any] = {}
    for key in (
        "temperature",
        "max_tokens",
        "top_p",
        "n",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "tools",
        "tool_choice",
        "response_format",
        "seed",
    ):
        if key in body:
            extra_kwargs[key] = body[key]

    # Request overrides from agent version (agent wins on conflicts)
    if version.request_overrides:
        extra_kwargs.update(version.request_overrides)

    return model, messages, extra_kwargs
