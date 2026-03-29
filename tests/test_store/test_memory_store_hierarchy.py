"""Tests for MemoryStore org/team hierarchy methods."""

from __future__ import annotations

import pytest

from stateloom.core.organization import Organization, Team
from stateloom.core.session import Session
from stateloom.core.types import OrgStatus, TeamStatus
from stateloom.store.memory_store import MemoryStore


@pytest.fixture
def store():
    return MemoryStore()


class TestMemoryStoreOrganizations:
    def test_save_and_get_organization(self, store: MemoryStore):
        org = Organization(id="org-1", name="Acme")
        store.save_organization(org)
        result = store.get_organization("org-1")
        assert result is not None
        assert result.id == "org-1"
        assert result.name == "Acme"

    def test_get_nonexistent_organization(self, store: MemoryStore):
        assert store.get_organization("missing") is None

    def test_list_organizations(self, store: MemoryStore):
        org1 = Organization(id="org-1", name="First")
        org2 = Organization(id="org-2", name="Second")
        store.save_organization(org1)
        store.save_organization(org2)
        orgs = store.list_organizations()
        assert len(orgs) == 2

    def test_save_organization_upsert(self, store: MemoryStore):
        org = Organization(id="org-1", name="Before")
        store.save_organization(org)
        org.name = "After"
        store.save_organization(org)
        result = store.get_organization("org-1")
        assert result is not None
        assert result.name == "After"


class TestMemoryStoreTeams:
    def test_save_and_get_team(self, store: MemoryStore):
        team = Team(id="team-1", org_id="org-1", name="ML")
        store.save_team(team)
        result = store.get_team("team-1")
        assert result is not None
        assert result.org_id == "org-1"
        assert result.name == "ML"

    def test_get_nonexistent_team(self, store: MemoryStore):
        assert store.get_team("missing") is None

    def test_list_teams_all(self, store: MemoryStore):
        store.save_team(Team(id="t1", org_id="org-1"))
        store.save_team(Team(id="t2", org_id="org-2"))
        teams = store.list_teams()
        assert len(teams) == 2

    def test_list_teams_by_org(self, store: MemoryStore):
        store.save_team(Team(id="t1", org_id="org-1"))
        store.save_team(Team(id="t2", org_id="org-1"))
        store.save_team(Team(id="t3", org_id="org-2"))
        teams = store.list_teams(org_id="org-1")
        assert len(teams) == 2
        assert all(t.org_id == "org-1" for t in teams)


class TestMemoryStoreListSessionsFilters:
    def test_list_sessions_by_org_id(self, store: MemoryStore):
        s1 = Session(id="s1", org_id="org-1")
        s2 = Session(id="s2", org_id="org-2")
        s3 = Session(id="s3", org_id="org-1")
        store.save_session(s1)
        store.save_session(s2)
        store.save_session(s3)
        result = store.list_sessions(org_id="org-1")
        assert len(result) == 2
        assert all(s.org_id == "org-1" for s in result)

    def test_list_sessions_by_team_id(self, store: MemoryStore):
        s1 = Session(id="s1", team_id="team-a")
        s2 = Session(id="s2", team_id="team-b")
        store.save_session(s1)
        store.save_session(s2)
        result = store.list_sessions(team_id="team-a")
        assert len(result) == 1
        assert result[0].id == "s1"

    def test_list_sessions_backward_compatible(self, store: MemoryStore):
        """Existing sessions with empty org_id/team_id still work."""
        s = Session(id="s1")
        store.save_session(s)
        result = store.list_sessions()
        assert len(result) == 1


class TestMemoryStoreHierarchyStats:
    def test_org_stats(self, store: MemoryStore):
        s1 = Session(id="s1", org_id="org-1", total_cost=1.0, total_tokens=100, call_count=2)
        s2 = Session(id="s2", org_id="org-1", total_cost=2.5, total_tokens=200, call_count=3)
        s3 = Session(id="s3", org_id="org-2", total_cost=5.0, total_tokens=500)
        store.save_session(s1)
        store.save_session(s2)
        store.save_session(s3)

        stats = store.get_org_stats("org-1")
        assert stats["org_id"] == "org-1"
        assert stats["total_sessions"] == 2
        assert stats["total_cost"] == 3.5
        assert stats["total_tokens"] == 300
        assert stats["total_calls"] == 5

    def test_team_stats(self, store: MemoryStore):
        s1 = Session(id="s1", team_id="team-a", total_cost=0.5, total_tokens=50)
        s2 = Session(id="s2", team_id="team-b", total_cost=1.0, total_tokens=100)
        store.save_session(s1)
        store.save_session(s2)

        stats = store.get_team_stats("team-a")
        assert stats["team_id"] == "team-a"
        assert stats["total_sessions"] == 1
        assert stats["total_cost"] == 0.5

    def test_org_stats_empty(self, store: MemoryStore):
        stats = store.get_org_stats("nonexistent")
        assert stats["total_sessions"] == 0
        assert stats["total_cost"] == 0
