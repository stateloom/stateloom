"""Tests for PostgreSQL store.

Requires a running PostgreSQL instance. Set STATELOOM_TEST_POSTGRES_URL to enable.
Example: STATELOOM_TEST_POSTGRES_URL=postgresql://localhost:5432/stateloom_test
"""

import os
import uuid

import pytest

from stateloom.core.event import (
    KillSwitchEvent,
    LLMCallEvent,
    PIIDetectionEvent,
)
from stateloom.core.session import Session
from stateloom.core.types import EventType

pytestmark = pytest.mark.skipif(
    not os.environ.get("STATELOOM_TEST_POSTGRES_URL"),
    reason="PostgreSQL not configured (set STATELOOM_TEST_POSTGRES_URL)",
)


@pytest.fixture
def postgres_store():
    """Create a PostgresStore connected to the test database."""
    from stateloom.store.postgres_store import PostgresStore

    url = os.environ["STATELOOM_TEST_POSTGRES_URL"]
    store = PostgresStore(url=url, pool_min=1, pool_max=3)

    # Clean up tables before each test
    with store._pool.connection() as conn:
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM session_feedback")
        conn.execute("DELETE FROM experiment_assignments")
        conn.execute("DELETE FROM agent_versions")
        conn.execute("DELETE FROM agents")
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM secrets")
        conn.execute("DELETE FROM admin_locks")
        conn.execute("DELETE FROM virtual_keys")
        conn.execute("DELETE FROM experiments")
        conn.execute("DELETE FROM teams")
        conn.execute("DELETE FROM organizations")
        conn.execute("DELETE FROM sessions")
        conn.commit()

    yield store
    store.close()


def test_save_and_get_session(postgres_store):
    session = Session(id="s1", name="Test")
    session.total_cost = 0.05
    session.total_tokens = 500
    postgres_store.save_session(session)

    retrieved = postgres_store.get_session("s1")
    assert retrieved is not None
    assert retrieved.id == "s1"
    assert retrieved.name == "Test"
    assert retrieved.total_cost == 0.05
    assert retrieved.total_tokens == 500


def test_list_sessions(postgres_store):
    postgres_store.save_session(Session(id="s1"))
    postgres_store.save_session(Session(id="s2"))
    sessions = postgres_store.list_sessions()
    assert len(sessions) == 2


def test_save_and_get_llm_event(postgres_store):
    # Must save session first (foreign key)
    postgres_store.save_session(Session(id="s1"))

    event = LLMCallEvent(
        session_id="s1",
        step=1,
        provider="openai",
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost=0.00075,
        latency_ms=1234.5,
    )
    postgres_store.save_event(event)

    events = postgres_store.get_session_events("s1")
    assert len(events) == 1
    e = events[0]
    assert e.provider == "openai"
    assert e.model == "gpt-4o"
    assert e.prompt_tokens == 100
    assert e.cost == 0.00075


def test_save_and_get_pii_event(postgres_store):
    postgres_store.save_session(Session(id="s1"))
    event = PIIDetectionEvent(
        session_id="s1",
        step=1,
        pii_type="email",
        mode="audit",
        pii_field="messages[0].content",
        action_taken="logged",
    )
    postgres_store.save_event(event)

    events = postgres_store.get_session_events("s1", event_type="pii_detection")
    assert len(events) == 1
    assert events[0].pii_type == "email"


def test_cross_session_events(postgres_store):
    """get_session_events("") should return events across all sessions."""
    postgres_store.save_session(Session(id="s1"))
    postgres_store.save_session(Session(id="s2"))
    postgres_store.save_session(Session(id="s3"))
    postgres_store.save_event(
        LLMCallEvent(session_id="s1", step=1, provider="openai", model="gpt-4o")
    )
    postgres_store.save_event(
        LLMCallEvent(session_id="s2", step=1, provider="openai", model="gpt-4o")
    )
    postgres_store.save_event(
        LLMCallEvent(session_id="s3", step=1, provider="anthropic", model="claude")
    )
    events = postgres_store.get_session_events("")
    assert len(events) == 3


def test_global_stats(postgres_store):
    s1 = Session(id="s1")
    s1.total_cost = 0.10
    s1.total_tokens = 1000
    postgres_store.save_session(s1)
    stats = postgres_store.get_global_stats()
    assert stats["total_sessions"] == 1
    assert stats["total_cost"] == 0.10


def test_save_session_with_events(postgres_store):
    session = Session(id="s1")
    events = [
        LLMCallEvent(session_id="s1", step=1, provider="openai", model="gpt-4o"),
        LLMCallEvent(session_id="s1", step=2, provider="openai", model="gpt-4o"),
    ]
    postgres_store.save_session_with_events(session, events)

    retrieved = postgres_store.get_session("s1")
    assert retrieved is not None

    retrieved_events = postgres_store.get_session_events("s1")
    assert len(retrieved_events) == 2


def test_cleanup(postgres_store):
    session = Session(id="s1")
    postgres_store.save_session(session)
    postgres_store.save_event(
        LLMCallEvent(session_id="s1", step=1, provider="openai", model="gpt-4o")
    )
    # Cleanup with 0 days retention should delete everything
    deleted = postgres_store.cleanup(retention_days=0)
    assert deleted >= 1


def test_kill_switch_event_roundtrip(postgres_store):
    postgres_store.save_session(Session(id="s1"))
    event = KillSwitchEvent(
        session_id="s1",
        step=1,
        reason="test",
        message="blocked",
        matched_rule={"model": "gpt-*"},
        blocked_model="gpt-4o",
        blocked_provider="openai",
    )
    postgres_store.save_event(event)
    events = postgres_store.get_session_events("s1")
    assert len(events) == 1
    assert isinstance(events[0], KillSwitchEvent)
    assert events[0].blocked_model == "gpt-4o"


def test_child_sessions(postgres_store):
    parent = Session(id="parent-1")
    postgres_store.save_session(parent)
    child = Session(id="child-1", parent_session_id="parent-1")
    postgres_store.save_session(child)

    children = postgres_store.list_child_sessions("parent-1")
    assert len(children) == 1
    assert children[0].id == "child-1"


def test_admin_locks(postgres_store):
    postgres_store.save_admin_lock("key1", "value1", locked_by="admin", reason="test")
    lock = postgres_store.get_admin_lock("key1")
    assert lock is not None
    assert lock["value"] == "value1"

    locks = postgres_store.list_admin_locks()
    assert len(locks) == 1

    postgres_store.delete_admin_lock("key1")
    assert postgres_store.get_admin_lock("key1") is None


def test_secrets(postgres_store):
    postgres_store.save_secret("my_key", "my_value")
    assert postgres_store.get_secret("my_key") == "my_value"

    keys = postgres_store.list_secrets()
    assert "my_key" in keys

    postgres_store.delete_secret("my_key")
    assert postgres_store.get_secret("my_key") == ""


def test_organizations(postgres_store):
    from stateloom.core.organization import Organization

    org = Organization(id="org-1", name="Test Org")
    postgres_store.save_organization(org)

    retrieved = postgres_store.get_organization("org-1")
    assert retrieved is not None
    assert retrieved.name == "Test Org"

    all_orgs = postgres_store.list_organizations()
    assert len(all_orgs) == 1


def test_teams(postgres_store):
    from stateloom.core.organization import Organization, Team

    org = Organization(id="org-1", name="Test Org")
    postgres_store.save_organization(org)

    team = Team(id="team-1", org_id="org-1", name="Team A")
    postgres_store.save_team(team)

    retrieved = postgres_store.get_team("team-1")
    assert retrieved is not None
    assert retrieved.name == "Team A"

    all_teams = postgres_store.list_teams(org_id="org-1")
    assert len(all_teams) == 1


def test_experiments(postgres_store):
    from stateloom.experiment.models import Experiment

    exp = Experiment(id="exp-1", name="Test Experiment")
    postgres_store.save_experiment(exp)

    retrieved = postgres_store.get_experiment("exp-1")
    assert retrieved is not None
    assert retrieved.name == "Test Experiment"

    all_exps = postgres_store.list_experiments()
    assert len(all_exps) == 1


def test_jobs(postgres_store):
    from stateloom.core.job import Job

    job = Job(id="job-1", provider="openai", model="gpt-4o")
    postgres_store.save_job(job)

    retrieved = postgres_store.get_job("job-1")
    assert retrieved is not None

    all_jobs = postgres_store.list_jobs()
    assert len(all_jobs) == 1

    deleted = postgres_store.delete_job("job-1")
    assert deleted is True

    stats = postgres_store.get_job_stats()
    assert stats["total"] == 0


def test_upsert_session(postgres_store):
    """Verify upsert: save twice, get once."""
    s = Session(id="s1", name="v1")
    postgres_store.save_session(s)

    s.name = "v2"
    s.total_cost = 0.50
    postgres_store.save_session(s)

    retrieved = postgres_store.get_session("s1")
    assert retrieved is not None
    assert retrieved.name == "v2"
    assert retrieved.total_cost == 0.50

    # Should still be only 1 session
    assert len(postgres_store.list_sessions()) == 1


def test_purge_session(postgres_store):
    postgres_store.save_session(Session(id="s1"))
    postgres_store.save_event(
        LLMCallEvent(session_id="s1", step=1, provider="openai", model="gpt-4o")
    )
    deleted = postgres_store.purge_session("s1")
    assert deleted >= 1
    assert postgres_store.get_session("s1") is None
