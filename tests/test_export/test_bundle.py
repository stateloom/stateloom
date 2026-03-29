"""Tests for session export/import — portable JSON bundles."""

from __future__ import annotations

import gzip
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stateloom.core.config import StateLoomConfig
from stateloom.core.errors import StateLoomError
from stateloom.core.event import (
    AnyEvent,
    CacheHitEvent,
    CheckpointEvent,
    LLMCallEvent,
    PIIDetectionEvent,
    ToolCallEvent,
)
from stateloom.core.session import Session
from stateloom.core.types import EventType, SessionStatus
from stateloom.export.bundle import (
    BUNDLE_SCHEMA_VERSION,
    _scrub_event_dict,
    _scrub_text,
    export_session,
    import_session,
    validate_bundle,
    write_bundle_to_file,
)
from stateloom.store.memory_store import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(
    session_id: str = "sess-001",
    *,
    parent_session_id: str | None = None,
    status: SessionStatus = SessionStatus.COMPLETED,
    total_cost: float = 0.123,
    step_counter: int = 2,
) -> Session:
    s = Session(
        id=session_id,
        name="Test Session",
        org_id="org-1",
        team_id="team-1",
        status=status,
        total_cost=total_cost,
        step_counter=step_counter,
        parent_session_id=parent_session_id,
        metadata={"foo": "bar"},
    )
    return s


def _make_llm_event(
    session_id: str = "sess-001",
    step: int = 1,
    model: str = "gpt-4o",
    cost: float = 0.01,
    cached_response_json: str | None = None,
) -> LLMCallEvent:
    return LLMCallEvent(
        session_id=session_id,
        step=step,
        model=model,
        provider="openai",
        cost=cost,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_preview="Hello world",
        cached_response_json=cached_response_json,
    )


def _make_tool_event(
    session_id: str = "sess-001",
    step: int = 2,
) -> ToolCallEvent:
    return ToolCallEvent(
        session_id=session_id,
        step=step,
        tool_name="search",
        latency_ms=15.0,
    )


def _make_checkpoint_event(
    session_id: str = "sess-001",
    step: int = 3,
) -> CheckpointEvent:
    return CheckpointEvent(
        session_id=session_id,
        step=step,
        label="data_fetched",
        description="All data retrieved",
    )


def _populated_store(
    session_id: str = "sess-001",
    with_children: bool = False,
) -> MemoryStore:
    """Create a store with a session and events."""
    store = MemoryStore()
    session = _make_session(session_id)
    store.save_session(session)
    store.save_event(_make_llm_event(session_id, step=1))
    store.save_event(_make_tool_event(session_id, step=2))

    if with_children:
        child = _make_session("child-001", parent_session_id=session_id, step_counter=1)
        store.save_session(child)
        store.save_event(_make_llm_event("child-001", step=1, model="claude-3-opus"))

    return store


# ===========================================================================
# Export tests
# ===========================================================================


class TestExportSession:
    def test_basic_export(self):
        store = _populated_store()
        bundle = export_session(store, "sess-001")

        assert bundle["schema_version"] == "v1"
        assert "stateloom_version" in bundle
        assert "exported_at" in bundle
        assert bundle["pii_scrubbed"] is False
        assert bundle["session"]["id"] == "sess-001"
        assert len(bundle["events"]) == 2
        assert bundle["children"] == []

    def test_schema_version(self):
        store = _populated_store()
        bundle = export_session(store, "sess-001")
        assert bundle["schema_version"] == BUNDLE_SCHEMA_VERSION

    def test_session_fields_preserved(self):
        store = _populated_store()
        bundle = export_session(store, "sess-001")
        sd = bundle["session"]

        assert sd["name"] == "Test Session"
        assert sd["org_id"] == "org-1"
        assert sd["team_id"] == "team-1"
        assert sd["total_cost"] == 0.123
        assert sd["step_counter"] == 2
        assert sd["metadata"]["foo"] == "bar"

    def test_private_fields_excluded(self):
        """PrivateAttr fields (_cancelled, _lock, etc.) should not appear."""
        store = _populated_store()
        bundle = export_session(store, "sess-001")
        sd = bundle["session"]

        assert "_cancelled" not in sd
        assert "_lock" not in sd
        assert "_suspended" not in sd
        assert "_suspend_event" not in sd
        assert "_signal_payload" not in sd

    def test_event_types_preserved(self):
        store = _populated_store()
        bundle = export_session(store, "sess-001")

        event_types = [e["event_type"] for e in bundle["events"]]
        assert "llm_call" in event_types
        assert "tool_call" in event_types

    def test_cached_response_json_preserved(self):
        store = MemoryStore()
        session = _make_session()
        store.save_session(session)
        event = _make_llm_event(
            cached_response_json='{"_type": "stream", "chunks": []}',
        )
        store.save_event(event)

        bundle = export_session(store, "sess-001")
        exported_event = bundle["events"][0]
        assert exported_event["cached_response_json"] == '{"_type": "stream", "chunks": []}'

    def test_export_with_children(self):
        store = _populated_store(with_children=True)
        bundle = export_session(store, "sess-001", include_children=True)

        assert len(bundle["children"]) == 1
        child = bundle["children"][0]
        assert child["session"]["id"] == "child-001"
        assert len(child["events"]) == 1

    def test_export_without_children(self):
        store = _populated_store(with_children=True)
        bundle = export_session(store, "sess-001", include_children=False)
        assert bundle["children"] == []

    def test_export_session_not_found(self):
        store = MemoryStore()
        with pytest.raises(StateLoomError, match="Session 'nonexistent' not found"):
            export_session(store, "nonexistent")

    def test_empty_events(self):
        store = MemoryStore()
        session = _make_session(step_counter=0, total_cost=0.0)
        store.save_session(session)

        bundle = export_session(store, "sess-001")
        assert bundle["events"] == []

    def test_multiple_event_types(self):
        store = MemoryStore()
        session = _make_session()
        store.save_session(session)
        store.save_event(_make_llm_event(step=1))
        store.save_event(_make_tool_event(step=2))
        store.save_event(_make_checkpoint_event(step=3))

        bundle = export_session(store, "sess-001")
        assert len(bundle["events"]) == 3
        types = {e["event_type"] for e in bundle["events"]}
        assert types == {"llm_call", "tool_call", "checkpoint"}


# ===========================================================================
# PII scrubbing tests
# ===========================================================================


class TestPIIScrubbing:
    def test_scrub_text_replaces_matches(self):
        from stateloom.pii.scanner import PIIScanner

        scanner = PIIScanner()
        text = "Contact me at test@example.com"
        result = _scrub_text(text, scanner)
        assert "test@example.com" not in result
        assert "[PII_REDACTED:" in result

    def test_scrub_event_dict_prompt_preview(self):
        from stateloom.pii.scanner import PIIScanner

        scanner = PIIScanner()
        event_dict = {
            "event_type": "llm_call",
            "prompt_preview": "Send to test@example.com",
        }
        _scrub_event_dict(event_dict, scanner)
        assert "test@example.com" not in event_dict["prompt_preview"]

    def test_scrub_event_dict_cached_response_json(self):
        from stateloom.pii.scanner import PIIScanner

        scanner = PIIScanner()
        cached = json.dumps({"content": "Reply to test@example.com"})
        event_dict = {
            "event_type": "llm_call",
            "cached_response_json": cached,
        }
        _scrub_event_dict(event_dict, scanner)
        parsed = json.loads(event_dict["cached_response_json"])
        assert "test@example.com" not in parsed["content"]

    def test_export_with_pii_scrub(self):
        from stateloom.pii.scanner import PIIScanner

        store = MemoryStore()
        session = _make_session()
        store.save_session(session)
        event = _make_llm_event()
        event.prompt_preview = "Email: test@example.com"
        store.save_event(event)

        scanner = PIIScanner()
        bundle = export_session(
            store,
            "sess-001",
            scrub_pii=True,
            pii_scanner=scanner,
        )
        assert bundle["pii_scrubbed"] is True
        assert "test@example.com" not in bundle["events"][0]["prompt_preview"]


# ===========================================================================
# Import tests
# ===========================================================================


class TestImportSession:
    def _minimal_bundle(self, session_id: str = "sess-import") -> dict:
        session = _make_session(session_id)
        event = _make_llm_event(session_id, step=1)
        return {
            "schema_version": "v1",
            "stateloom_version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "pii_scrubbed": False,
            "session": session.model_dump(mode="json"),
            "events": [event.model_dump(mode="json")],
            "children": [],
        }

    def test_import_from_dict(self):
        store = MemoryStore()
        bundle = self._minimal_bundle()
        session = import_session(store, bundle)

        assert session.id == "sess-import"
        assert session.name == "Test Session"
        assert store.get_session("sess-import") is not None

    def test_import_events_reconstructed(self):
        store = MemoryStore()
        bundle = self._minimal_bundle()
        import_session(store, bundle)

        events = store.get_session_events("sess-import")
        assert len(events) == 1
        assert isinstance(events[0], LLMCallEvent)
        assert events[0].model == "gpt-4o"

    def test_import_from_json_file(self, tmp_path):
        store = MemoryStore()
        bundle = self._minimal_bundle()
        filepath = tmp_path / "bundle.json"
        filepath.write_text(json.dumps(bundle))

        session = import_session(store, str(filepath))
        assert session.id == "sess-import"

    def test_import_from_gzip_file(self, tmp_path):
        store = MemoryStore()
        bundle = self._minimal_bundle()
        filepath = tmp_path / "bundle.json.gz"
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(bundle, f)

        session = import_session(store, filepath)
        assert session.id == "sess-import"

    def test_import_session_id_override(self):
        store = MemoryStore()
        bundle = self._minimal_bundle()
        session = import_session(store, bundle, session_id_override="new-id-001")

        assert session.id == "new-id-001"
        assert store.get_session("new-id-001") is not None

        events = store.get_session_events("new-id-001")
        assert len(events) == 1
        assert events[0].session_id == "new-id-001"

    def test_import_collision_error(self):
        store = MemoryStore()
        bundle = self._minimal_bundle()
        import_session(store, bundle)

        with pytest.raises(StateLoomError, match="already exists"):
            import_session(store, bundle)

    def test_import_collision_with_override(self):
        store = MemoryStore()
        bundle = self._minimal_bundle()
        import_session(store, bundle)

        # Second import with override succeeds
        session = import_session(store, bundle, session_id_override="sess-import-2")
        assert session.id == "sess-import-2"

    def test_import_invalid_schema_version(self):
        bundle = {
            "schema_version": "v99",
            "session": {},
            "events": [],
        }
        store = MemoryStore()
        with pytest.raises(StateLoomError, match="Unsupported bundle schema version"):
            import_session(store, bundle)

    def test_import_missing_session_field(self):
        bundle = {
            "schema_version": "v1",
            "events": [],
        }
        store = MemoryStore()
        with pytest.raises(StateLoomError, match="missing required field 'session'"):
            import_session(store, bundle)

    def test_import_missing_events_field(self):
        bundle = {
            "schema_version": "v1",
            "session": _make_session().model_dump(mode="json"),
        }
        store = MemoryStore()
        with pytest.raises(StateLoomError, match="missing required field 'events'"):
            import_session(store, bundle)

    def test_import_file_not_found(self):
        store = MemoryStore()
        with pytest.raises(StateLoomError, match="Bundle file not found"):
            import_session(store, "/nonexistent/path/bundle.json")

    def test_import_children(self):
        store = MemoryStore()
        child_session = _make_session("child-imp", parent_session_id="sess-import")
        child_event = _make_llm_event("child-imp", step=1, model="claude-3-sonnet")

        bundle = self._minimal_bundle()
        bundle["children"] = [
            {
                "session": child_session.model_dump(mode="json"),
                "events": [child_event.model_dump(mode="json")],
                "children": [],
            }
        ]

        session = import_session(store, bundle)
        assert session.id == "sess-import"

        # Child should be imported too
        child = store.get_session("child-imp")
        assert child is not None
        assert child.parent_session_id == "sess-import"

    def test_import_children_parent_id_remapped(self):
        store = MemoryStore()
        child_session = _make_session("child-imp", parent_session_id="sess-import")
        child_event = _make_llm_event("child-imp", step=1)

        bundle = self._minimal_bundle()
        bundle["children"] = [
            {
                "session": child_session.model_dump(mode="json"),
                "events": [child_event.model_dump(mode="json")],
                "children": [],
            }
        ]

        session = import_session(store, bundle, session_id_override="remapped-parent")
        assert session.id == "remapped-parent"

        # Find child by listing sessions (its ID was auto-generated)
        all_sessions = store.list_sessions(limit=100)
        children = [s for s in all_sessions if s.parent_session_id == "remapped-parent"]
        assert len(children) == 1

    def test_import_unknown_event_type_skipped(self):
        """Events with unknown types should be skipped with a warning."""
        store = MemoryStore()
        bundle = self._minimal_bundle()
        bundle["events"].append(
            {
                "id": "unknown-evt",
                "session_id": "sess-import",
                "step": 99,
                "event_type": "completely_unknown_type",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Should not raise — unknown event is skipped
        session = import_session(store, bundle)
        events = store.get_session_events("sess-import")
        # Only the valid LLMCallEvent should be imported
        assert len(events) == 1


# ===========================================================================
# File I/O tests
# ===========================================================================


class TestWriteBundle:
    def test_write_json(self, tmp_path):
        bundle = {"schema_version": "v1", "session": {}, "events": []}
        filepath = tmp_path / "out.json"
        write_bundle_to_file(bundle, filepath)

        assert filepath.exists()
        loaded = json.loads(filepath.read_text())
        assert loaded["schema_version"] == "v1"

    def test_write_gzip(self, tmp_path):
        bundle = {"schema_version": "v1", "session": {}, "events": []}
        filepath = tmp_path / "out.json.gz"
        write_bundle_to_file(bundle, filepath)

        assert filepath.exists()
        with gzip.open(filepath, "rt") as f:
            loaded = json.load(f)
        assert loaded["schema_version"] == "v1"


# ===========================================================================
# Validation tests
# ===========================================================================


class TestValidateBundle:
    def test_valid_bundle(self):
        bundle = {
            "schema_version": "v1",
            "session": {},
            "events": [],
        }
        validate_bundle(bundle)  # Should not raise

    def test_wrong_schema_version(self):
        with pytest.raises(StateLoomError, match="Unsupported bundle schema version"):
            validate_bundle({"schema_version": "v2", "session": {}, "events": []})

    def test_missing_schema_version(self):
        with pytest.raises(StateLoomError, match="Unsupported bundle schema version"):
            validate_bundle({"session": {}, "events": []})

    def test_missing_session(self):
        with pytest.raises(StateLoomError, match="missing required field 'session'"):
            validate_bundle({"schema_version": "v1", "events": []})

    def test_missing_events(self):
        with pytest.raises(StateLoomError, match="missing required field 'events'"):
            validate_bundle({"schema_version": "v1", "session": {}})

    def test_not_a_dict(self):
        with pytest.raises(StateLoomError, match="expected a dict"):
            validate_bundle("not a dict")  # type: ignore[arg-type]


# ===========================================================================
# Roundtrip tests
# ===========================================================================


class TestRoundtrip:
    def test_export_import_roundtrip(self):
        """Export → import → export should produce equivalent bundles."""
        store1 = _populated_store()
        bundle1 = export_session(store1, "sess-001")

        store2 = MemoryStore()
        import_session(store2, bundle1)

        bundle2 = export_session(store2, "sess-001")

        # Session fields should match
        assert bundle1["session"]["id"] == bundle2["session"]["id"]
        assert bundle1["session"]["total_cost"] == bundle2["session"]["total_cost"]
        assert bundle1["session"]["step_counter"] == bundle2["session"]["step_counter"]

        # Event count should match
        assert len(bundle1["events"]) == len(bundle2["events"])

        # Event types and steps should match
        for e1, e2 in zip(bundle1["events"], bundle2["events"]):
            assert e1["event_type"] == e2["event_type"]
            assert e1["step"] == e2["step"]

    def test_roundtrip_with_children(self):
        store1 = _populated_store(with_children=True)
        bundle1 = export_session(store1, "sess-001", include_children=True)

        store2 = MemoryStore()
        import_session(store2, bundle1)

        bundle2 = export_session(store2, "sess-001", include_children=True)

        assert len(bundle1["children"]) == len(bundle2["children"])
        assert bundle1["children"][0]["session"]["id"] == bundle2["children"][0]["session"]["id"]

    def test_roundtrip_via_file(self, tmp_path):
        store1 = _populated_store()
        bundle = export_session(store1, "sess-001")
        filepath = tmp_path / "session.json"
        write_bundle_to_file(bundle, filepath)

        store2 = MemoryStore()
        session = import_session(store2, filepath)
        assert session.id == "sess-001"

    def test_roundtrip_via_gzip_file(self, tmp_path):
        store1 = _populated_store()
        bundle = export_session(store1, "sess-001")
        filepath = tmp_path / "session.json.gz"
        write_bundle_to_file(bundle, filepath)

        store2 = MemoryStore()
        session = import_session(store2, filepath)
        assert session.id == "sess-001"


# ===========================================================================
# Gate delegation tests
# ===========================================================================


class TestGateDelegation:
    def test_gate_export_session(self):
        from stateloom.gate import Gate

        config = StateLoomConfig(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
        )
        gate = Gate(config)
        gate._setup_middleware()

        session = _make_session()
        gate.store.save_session(session)
        gate.store.save_event(_make_llm_event())

        bundle = gate.export_session("sess-001")
        assert bundle["schema_version"] == "v1"
        assert bundle["session"]["id"] == "sess-001"
        assert len(bundle["events"]) == 1

    def test_gate_export_to_file(self, tmp_path):
        from stateloom.gate import Gate

        config = StateLoomConfig(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
        )
        gate = Gate(config)
        gate._setup_middleware()

        session = _make_session()
        gate.store.save_session(session)
        gate.store.save_event(_make_llm_event())

        filepath = tmp_path / "exported.json"
        bundle = gate.export_session("sess-001", str(filepath))
        assert filepath.exists()
        loaded = json.loads(filepath.read_text())
        assert loaded["session"]["id"] == "sess-001"

    def test_gate_import_session(self):
        from stateloom.gate import Gate

        config = StateLoomConfig(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
        )
        gate = Gate(config)
        gate._setup_middleware()

        bundle = {
            "schema_version": "v1",
            "stateloom_version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "pii_scrubbed": False,
            "session": _make_session("gate-import").model_dump(mode="json"),
            "events": [_make_llm_event("gate-import").model_dump(mode="json")],
            "children": [],
        }

        session = gate.import_session(bundle)
        assert session.id == "gate-import"
        assert gate.store.get_session("gate-import") is not None

    def test_gate_export_with_pii_scrub(self):
        from stateloom.gate import Gate

        config = StateLoomConfig(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
            pii_enabled=True,
        )
        gate = Gate(config)
        gate._setup_middleware()

        session = _make_session()
        gate.store.save_session(session)
        event = _make_llm_event()
        event.prompt_preview = "Email: test@example.com"
        gate.store.save_event(event)

        bundle = gate.export_session("sess-001", scrub_pii=True)
        assert bundle["pii_scrubbed"] is True
        assert "test@example.com" not in bundle["events"][0]["prompt_preview"]


# ===========================================================================
# Dashboard API tests
# ===========================================================================


class TestDashboardAPI:
    @pytest.fixture
    def gate(self):
        from stateloom.gate import Gate

        config = StateLoomConfig(
            auto_patch=False,
            dashboard=False,
            console_output=False,
            store_backend="memory",
        )
        g = Gate(config)
        g._setup_middleware()
        return g

    @pytest.fixture
    def client(self, gate):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from stateloom.dashboard.api import create_api_router

        app = FastAPI()
        app.include_router(create_api_router(gate), prefix="/api/v1")
        return TestClient(app)

    def test_export_endpoint_200(self, gate, client):
        session = _make_session()
        gate.store.save_session(session)
        gate.store.save_event(_make_llm_event())

        resp = client.get("/api/v1/sessions/sess-001/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_version"] == "v1"
        assert data["session"]["id"] == "sess-001"
        assert "Content-Disposition" in resp.headers

    def test_export_endpoint_404(self, client):
        resp = client.get("/api/v1/sessions/nonexistent/export")
        assert resp.status_code == 404

    def test_export_endpoint_with_children(self, gate, client):
        session = _make_session()
        gate.store.save_session(session)
        child = _make_session("child-api", parent_session_id="sess-001")
        gate.store.save_session(child)

        resp = client.get("/api/v1/sessions/sess-001/export?include_children=true")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["children"]) == 1

    def test_import_endpoint_200(self, client):
        bundle = {
            "schema_version": "v1",
            "stateloom_version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "pii_scrubbed": False,
            "session": _make_session("api-import").model_dump(mode="json"),
            "events": [_make_llm_event("api-import").model_dump(mode="json")],
            "children": [],
        }

        resp = client.post("/api/v1/sessions/import", json={"bundle": bundle})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "imported"
        assert data["session_id"] == "api-import"

    def test_import_endpoint_400_invalid_schema(self, client):
        bundle = {
            "schema_version": "v99",
            "session": {},
            "events": [],
        }
        resp = client.post("/api/v1/sessions/import", json={"bundle": bundle})
        assert resp.status_code == 400

    def test_import_endpoint_400_collision(self, gate, client):
        # Pre-populate a session
        session = _make_session("collision-sess")
        gate.store.save_session(session)

        bundle = {
            "schema_version": "v1",
            "stateloom_version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "pii_scrubbed": False,
            "session": _make_session("collision-sess").model_dump(mode="json"),
            "events": [],
            "children": [],
        }
        resp = client.post("/api/v1/sessions/import", json={"bundle": bundle})
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"]

    def test_import_endpoint_with_override(self, gate, client):
        # Pre-populate
        session = _make_session("existing-sess")
        gate.store.save_session(session)

        bundle = {
            "schema_version": "v1",
            "stateloom_version": "0.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "pii_scrubbed": False,
            "session": _make_session("existing-sess").model_dump(mode="json"),
            "events": [],
            "children": [],
        }
        resp = client.post(
            "/api/v1/sessions/import",
            json={
                "bundle": bundle,
                "session_id_override": "override-sess",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "override-sess"


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_deeply_nested_children_capped(self):
        """Recursion should be capped at _MAX_CHILD_DEPTH."""
        from stateloom.export.bundle import _MAX_CHILD_DEPTH

        store = MemoryStore()
        parent_id = "root"
        session = _make_session(parent_id)
        store.save_session(session)

        # Create a chain of children deeper than the limit
        current_id = parent_id
        for i in range(_MAX_CHILD_DEPTH + 2):
            child_id = f"child-{i}"
            child = _make_session(child_id, parent_session_id=current_id, step_counter=0)
            store.save_session(child)
            current_id = child_id

        # Export should not raise — just silently truncates
        bundle = export_session(store, "root", include_children=True)
        # The root should export, and children are truncated at some depth

    def test_export_import_large_event_count(self):
        """Handles sessions with many events."""
        store = MemoryStore()
        session = _make_session(step_counter=100)
        store.save_session(session)
        for i in range(100):
            store.save_event(_make_llm_event(step=i + 1))

        bundle = export_session(store, "sess-001")
        assert len(bundle["events"]) == 100

        store2 = MemoryStore()
        import_session(store2, bundle)
        events = store2.get_session_events("sess-001", limit=10000)
        assert len(events) == 100

    def test_import_preserves_event_discriminator(self):
        """AnyEvent discriminated union correctly picks subclass."""
        store = MemoryStore()
        session = _make_session()
        store.save_session(session)
        store.save_event(
            CacheHitEvent(
                session_id="sess-001",
                step=1,
                original_model="gpt-4o",
                saved_cost=0.05,
            )
        )
        store.save_event(
            PIIDetectionEvent(
                session_id="sess-001",
                step=2,
                pii_type="email",
                mode="redact",
            )
        )

        bundle = export_session(store, "sess-001")

        store2 = MemoryStore()
        import_session(store2, bundle)
        events = store2.get_session_events("sess-001")
        assert len(events) == 2
        types = {type(e) for e in events}
        assert CacheHitEvent in types
        assert PIIDetectionEvent in types
