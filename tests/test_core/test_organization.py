"""Tests for Organization and Team dataclasses."""

from __future__ import annotations

import threading

import pytest

from stateloom.core.config import PIIRule
from stateloom.core.organization import Organization, Team
from stateloom.core.types import OrgStatus, PIIMode, TeamStatus


class TestOrganization:
    def test_defaults(self):
        org = Organization()
        assert org.id.startswith("org-")
        assert org.name == ""
        assert org.status == OrgStatus.ACTIVE
        assert org.budget is None
        assert org.total_cost == 0.0
        assert org.total_tokens == 0
        assert org.pii_rules == []
        assert org.metadata == {}

    def test_custom_fields(self):
        rules = [PIIRule(pattern="email", mode=PIIMode.BLOCK)]
        org = Organization(
            id="org-test",
            name="Acme Corp",
            budget=100.0,
            pii_rules=rules,
            metadata={"env": "prod"},
        )
        assert org.id == "org-test"
        assert org.name == "Acme Corp"
        assert org.budget == 100.0
        assert len(org.pii_rules) == 1
        assert org.metadata["env"] == "prod"

    def test_add_cost(self):
        org = Organization()
        org.add_cost(1.50, tokens=100)
        assert org.total_cost == 1.50
        assert org.total_tokens == 100

        org.add_cost(0.50, tokens=50)
        assert org.total_cost == 2.00
        assert org.total_tokens == 150

    def test_add_cost_thread_safety(self):
        org = Organization()
        barrier = threading.Barrier(10)

        def add():
            barrier.wait()
            for _ in range(100):
                org.add_cost(0.01, tokens=1)

        threads = [threading.Thread(target=add) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert org.total_tokens == 1000
        assert abs(org.total_cost - 10.0) < 0.001

    def test_status_enum(self):
        assert OrgStatus.ACTIVE == "active"
        assert OrgStatus.SUSPENDED == "suspended"


class TestTeam:
    def test_defaults(self):
        team = Team()
        assert team.id.startswith("team-")
        assert team.org_id == ""
        assert team.name == ""
        assert team.status == TeamStatus.ACTIVE
        assert team.budget is None
        assert team.total_cost == 0.0
        assert team.total_tokens == 0
        assert team.metadata == {}

    def test_custom_fields(self):
        team = Team(
            id="team-test",
            org_id="org-123",
            name="ML Team",
            budget=50.0,
            metadata={"project": "nlp"},
        )
        assert team.id == "team-test"
        assert team.org_id == "org-123"
        assert team.name == "ML Team"
        assert team.budget == 50.0

    def test_add_cost(self):
        team = Team()
        team.add_cost(0.75, tokens=30)
        assert team.total_cost == 0.75
        assert team.total_tokens == 30

    def test_add_cost_thread_safety(self):
        team = Team()
        barrier = threading.Barrier(10)

        def add():
            barrier.wait()
            for _ in range(100):
                team.add_cost(0.01, tokens=1)

        threads = [threading.Thread(target=add) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert team.total_tokens == 1000
        assert abs(team.total_cost - 10.0) < 0.001

    def test_status_enum(self):
        assert TeamStatus.ACTIVE == "active"
        assert TeamStatus.ARCHIVED == "archived"
