"""Integration tests for multi-tenant hierarchy: org -> team -> session."""

from __future__ import annotations

import pytest

from stateloom.core.config import PIIRule, StateLoomConfig
from stateloom.core.organization import Organization, Team
from stateloom.core.types import FailureAction, OrgStatus, PIIMode, TeamStatus
from stateloom.gate import Gate


def _make_gate(**kwargs) -> Gate:
    """Create a minimal Gate for testing."""
    defaults = {
        "auto_patch": False,
        "dashboard": False,
        "console_output": False,
        "store_backend": "memory",
        "cache_enabled": False,
        "pii_enabled": False,
        "shadow_enabled": False,
    }
    defaults.update(kwargs)
    config = StateLoomConfig(**defaults)
    gate = Gate(config)
    gate._setup_middleware()
    return gate


class TestGateOrganizationCRUD:
    def test_create_organization(self):
        gate = _make_gate()
        org = gate.create_organization(name="Acme Corp", budget=100.0)
        assert org.name == "Acme Corp"
        assert org.budget == 100.0
        assert org.status == OrgStatus.ACTIVE
        # Should be in memory cache
        assert gate.get_organization(org.id) is org
        # Should be persisted in store
        stored = gate.store.get_organization(org.id)
        assert stored is not None
        gate.shutdown()

    def test_create_organization_with_id(self):
        gate = _make_gate()
        org = gate.create_organization(org_id="org-custom", name="Custom")
        assert org.id == "org-custom"
        gate.shutdown()

    def test_list_organizations(self):
        gate = _make_gate()
        gate.create_organization(name="A")
        gate.create_organization(name="B")
        orgs = gate.list_organizations()
        assert len(orgs) == 2
        gate.shutdown()

    def test_get_nonexistent_organization(self):
        gate = _make_gate()
        assert gate.get_organization("missing") is None
        gate.shutdown()


class TestGateTeamCRUD:
    def test_create_team(self):
        gate = _make_gate()
        org = gate.create_organization(name="Corp")
        team = gate.create_team(org_id=org.id, name="ML Team", budget=50.0)
        assert team.org_id == org.id
        assert team.name == "ML Team"
        assert team.budget == 50.0
        assert gate.get_team(team.id) is team
        gate.shutdown()

    def test_list_teams_by_org(self):
        gate = _make_gate()
        org1 = gate.create_organization(name="Org1")
        org2 = gate.create_organization(name="Org2")
        gate.create_team(org_id=org1.id, name="T1")
        gate.create_team(org_id=org1.id, name="T2")
        gate.create_team(org_id=org2.id, name="T3")
        teams = gate.list_teams(org_id=org1.id)
        assert len(teams) == 2
        gate.shutdown()


class TestGateSessionWithHierarchy:
    def test_session_links_to_org_team(self):
        gate = _make_gate()
        org = gate.create_organization(name="Corp")
        team = gate.create_team(org_id=org.id, name="Team")

        with gate.session(org_id=org.id, team_id=team.id) as s:
            assert s.org_id == org.id
            assert s.team_id == team.id

        gate.shutdown()

    def test_session_backward_compat(self):
        """Session without org_id/team_id still works."""
        gate = _make_gate()
        with gate.session() as s:
            assert s.org_id == ""
            assert s.team_id == ""
        gate.shutdown()


class TestGateCostPropagation:
    def test_cost_callback_propagates_to_org_and_team(self):
        gate = _make_gate()
        org = gate.create_organization(name="Corp")
        team = gate.create_team(org_id=org.id, name="Team")

        # Simulate cost callback
        gate._on_cost(org.id, team.id, 1.50, 100)

        assert org.total_cost == 1.50
        assert org.total_tokens == 100
        assert team.total_cost == 1.50
        assert team.total_tokens == 100

        gate._on_cost(org.id, team.id, 0.50, 50)
        assert org.total_cost == 2.00
        assert org.total_tokens == 150
        assert team.total_cost == 2.00
        assert team.total_tokens == 150

        gate.shutdown()

    def test_cost_callback_ignores_empty_ids(self):
        gate = _make_gate()
        # Should not raise when org_id/team_id are empty
        gate._on_cost("", "", 1.0, 100)
        gate.shutdown()


class TestGateHierarchyBudgetCheck:
    def test_team_budget_exceeded(self):
        gate = _make_gate()
        org = gate.create_organization(name="Corp")
        team = gate.create_team(org_id=org.id, name="Team", budget=10.0)
        team.total_cost = 15.0

        result = gate._check_hierarchy_budget(org.id, team.id)
        assert result is not None
        assert result == (10.0, 15.0, "team")
        gate.shutdown()

    def test_org_budget_exceeded(self):
        gate = _make_gate()
        org = gate.create_organization(name="Corp", budget=100.0)
        org.total_cost = 120.0
        team = gate.create_team(org_id=org.id, name="Team")

        result = gate._check_hierarchy_budget(org.id, team.id)
        assert result is not None
        assert result == (100.0, 120.0, "org")
        gate.shutdown()

    def test_no_budget_exceeded(self):
        gate = _make_gate()
        org = gate.create_organization(name="Corp", budget=100.0)
        team = gate.create_team(org_id=org.id, name="Team", budget=50.0)

        result = gate._check_hierarchy_budget(org.id, team.id)
        assert result is None
        gate.shutdown()

    def test_no_hierarchy(self):
        gate = _make_gate()
        result = gate._check_hierarchy_budget("", "")
        assert result is None
        gate.shutdown()


class TestGateOrgPIIRules:
    def test_get_org_pii_rules(self):
        gate = _make_gate()
        rules = [PIIRule(pattern="email", mode=PIIMode.BLOCK)]
        org = gate.create_organization(name="Corp", pii_rules=rules)

        result = gate._get_org_pii_rules(org.id)
        assert len(result) == 1
        assert result[0].pattern == "email"
        assert result[0].mode == PIIMode.BLOCK
        gate.shutdown()

    def test_get_org_pii_rules_missing_org(self):
        gate = _make_gate()
        result = gate._get_org_pii_rules("nonexistent")
        assert result == []
        gate.shutdown()


class TestGateHierarchyPersistence:
    def test_orgs_teams_loaded_from_store(self):
        """Orgs and teams should be loaded from store on Gate init."""
        gate = _make_gate()
        org = gate.create_organization(name="Corp")
        team = gate.create_team(org_id=org.id, name="Team")

        # Create a new Gate with the same store to verify loading
        gate2 = Gate(gate.config)
        gate2.store = gate.store
        gate2._load_hierarchy()

        assert gate2.get_organization(org.id) is not None
        assert gate2.get_organization(org.id).name == "Corp"
        assert gate2.get_team(team.id) is not None
        assert gate2.get_team(team.id).name == "Team"

        gate.shutdown()
