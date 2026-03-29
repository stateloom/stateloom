"""Shared fixtures for production integration tests.

Re-exports e2e fixtures and adds production-specific helpers.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_e2e.conftest import api_client, e2e_gate  # noqa: F401


@pytest.fixture
def prod_gate(e2e_gate):
    """Pre-configured gate with common production-like settings."""

    def _factory(**overrides: Any):
        defaults = {
            "cache": True,
            "pii": False,
            "console_output": False,
        }
        defaults.update(overrides)
        return e2e_gate(**defaults)

    return _factory


@pytest.fixture
def org_team_gate(e2e_gate, api_client):
    """Gate with an org and team already created, plus a dashboard client."""

    def _factory(**gate_overrides: Any):
        gate = e2e_gate(**gate_overrides)
        org = gate.create_organization(name="Acme Corp")
        team = gate.create_team(org_id=org.id, name="Engineering")
        client = api_client(gate)
        return gate, org, team, client

    return _factory


@pytest.fixture
def failing_llm_call():
    """Factory that creates a callable failing N times then succeeding."""

    def _factory(fail_count, then_response, error=None):
        if error is None:
            error = RuntimeError("LLM provider error")
        call_count = 0

        def _call():
            nonlocal call_count
            call_count += 1
            if call_count <= fail_count:
                raise error
            return then_response

        _call.call_count = lambda: call_count
        return _call

    return _factory
