"""Tests for SQLiteStore org/team hierarchy methods."""

from __future__ import annotations

import pytest

from stateloom.core.config import PIIRule
from stateloom.core.organization import Organization, Team
from stateloom.core.session import Session
from stateloom.core.types import OrgStatus, PIIMode, TeamStatus
from stateloom.store.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(str(tmp_path / "test.db"))
    yield s
    s.close()


class TestSQLiteStoreOrganizations:
    def test_save_and_get_organization(self, store: SQLiteStore):
        org = Organization(id="org-1", name="Acme", budget=100.0)
        store.save_organization(org)
        result = store.get_organization("org-1")
        assert result is not None
        assert result.id == "org-1"
        assert result.name == "Acme"
        assert result.budget == 100.0
        assert result.status == OrgStatus.ACTIVE

    def test_get_nonexistent_organization(self, store: SQLiteStore):
        assert store.get_organization("missing") is None

    def test_list_organizations(self, store: SQLiteStore):
        store.save_organization(Organization(id="org-1", name="A"))
        store.save_organization(Organization(id="org-2", name="B"))
        orgs = store.list_organizations()
        assert len(orgs) == 2

    def test_organization_with_pii_rules(self, store: SQLiteStore):
        rules = [
            PIIRule(pattern="email", mode=PIIMode.BLOCK),
            PIIRule(pattern="ssn", mode=PIIMode.REDACT),
        ]
        org = Organization(id="org-pii", pii_rules=rules)
        store.save_organization(org)
        result = store.get_organization("org-pii")
        assert result is not None
        assert len(result.pii_rules) == 2
        assert result.pii_rules[0].pattern == "email"
        assert result.pii_rules[0].mode == PIIMode.BLOCK

    def test_organization_upsert(self, store: SQLiteStore):
        org = Organization(id="org-1", name="Before", total_cost=1.0)
        store.save_organization(org)
        org.name = "After"
        org.total_cost = 5.0
        store.save_organization(org)
        result = store.get_organization("org-1")
        assert result is not None
        assert result.name == "After"
        assert result.total_cost == 5.0

    def test_organization_metadata(self, store: SQLiteStore):
        org = Organization(id="org-m", metadata={"env": "prod", "tier": "enterprise"})
        store.save_organization(org)
        result = store.get_organization("org-m")
        assert result is not None
        assert result.metadata["env"] == "prod"


class TestSQLiteStoreTeams:
    def test_save_and_get_team(self, store: SQLiteStore):
        team = Team(id="team-1", org_id="org-1", name="ML", budget=25.0)
        store.save_team(team)
        result = store.get_team("team-1")
        assert result is not None
        assert result.id == "team-1"
        assert result.org_id == "org-1"
        assert result.name == "ML"
        assert result.budget == 25.0

    def test_get_nonexistent_team(self, store: SQLiteStore):
        assert store.get_team("missing") is None

    def test_list_teams_all(self, store: SQLiteStore):
        store.save_team(Team(id="t1", org_id="org-1"))
        store.save_team(Team(id="t2", org_id="org-2"))
        teams = store.list_teams()
        assert len(teams) == 2

    def test_list_teams_by_org(self, store: SQLiteStore):
        store.save_team(Team(id="t1", org_id="org-1"))
        store.save_team(Team(id="t2", org_id="org-1"))
        store.save_team(Team(id="t3", org_id="org-2"))
        teams = store.list_teams(org_id="org-1")
        assert len(teams) == 2


class TestSQLiteStoreSessionHierarchy:
    def test_session_with_org_team_ids(self, store: SQLiteStore):
        session = Session(id="s1", org_id="org-1", team_id="team-1")
        store.save_session(session)
        result = store.get_session("s1")
        assert result is not None
        assert result.org_id == "org-1"
        assert result.team_id == "team-1"

    def test_session_backward_compat(self, store: SQLiteStore):
        """Sessions without org_id/team_id default to empty string."""
        session = Session(id="s-old")
        store.save_session(session)
        result = store.get_session("s-old")
        assert result is not None
        assert result.org_id == ""
        assert result.team_id == ""

    def test_list_sessions_by_org_id(self, store: SQLiteStore):
        store.save_session(Session(id="s1", org_id="org-1"))
        store.save_session(Session(id="s2", org_id="org-2"))
        store.save_session(Session(id="s3", org_id="org-1"))
        result = store.list_sessions(org_id="org-1")
        assert len(result) == 2

    def test_list_sessions_by_team_id(self, store: SQLiteStore):
        store.save_session(Session(id="s1", team_id="team-a"))
        store.save_session(Session(id="s2", team_id="team-b"))
        result = store.list_sessions(team_id="team-a")
        assert len(result) == 1

    def test_save_session_with_events_includes_org_team(self, store: SQLiteStore):
        session = Session(id="s-events", org_id="org-x", team_id="team-y")
        store.save_session_with_events(session, [])
        result = store.get_session("s-events")
        assert result is not None
        assert result.org_id == "org-x"
        assert result.team_id == "team-y"


class TestSQLiteStoreHierarchyStats:
    def test_org_stats(self, store: SQLiteStore):
        store.save_session(
            Session(id="s1", org_id="org-1", total_cost=1.0, total_tokens=100, call_count=2)
        )
        store.save_session(
            Session(id="s2", org_id="org-1", total_cost=2.5, total_tokens=200, call_count=3)
        )
        store.save_session(Session(id="s3", org_id="org-2", total_cost=5.0, total_tokens=500))

        stats = store.get_org_stats("org-1")
        assert stats["org_id"] == "org-1"
        assert stats["total_sessions"] == 2
        assert stats["total_cost"] == 3.5
        assert stats["total_tokens"] == 300
        assert stats["total_calls"] == 5

    def test_team_stats(self, store: SQLiteStore):
        store.save_session(Session(id="s1", team_id="team-a", total_cost=0.5, total_tokens=50))
        store.save_session(Session(id="s2", team_id="team-b", total_cost=1.0, total_tokens=100))

        stats = store.get_team_stats("team-a")
        assert stats["team_id"] == "team-a"
        assert stats["total_sessions"] == 1
        assert stats["total_cost"] == 0.5

    def test_stats_empty(self, store: SQLiteStore):
        stats = store.get_org_stats("nonexistent")
        assert stats["total_sessions"] == 0
        assert stats["total_cost"] == 0
