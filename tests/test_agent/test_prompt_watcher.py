"""Tests for prompt file watcher."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from stateloom.agent.prompt_watcher import PromptWatcher
from stateloom.core.config import StateLoomConfig
from stateloom.core.types import AgentStatus
from stateloom.gate import Gate


def _make_gate(tmp_path: Path) -> Gate:
    """Create a Gate with memory store for testing."""
    config = StateLoomConfig(
        auto_patch=False,
        dashboard=False,
        console_output=False,
        store_backend="memory",
        cache_enabled=False,
    )
    gate = Gate(config)
    gate._setup_middleware()
    return gate


class TestPromptWatcherSync:
    """Test the core sync logic (no watchdog needed)."""

    def test_new_file_creates_agent(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        # Create a prompt file
        f = prompts_dir / "support-bot.md"
        f.write_text(
            "---\nmodel: gpt-4o\ntemperature: 0.3\n---\nYou are a helpful support agent.\n"
        )

        watcher.scan()

        agents = gate.list_agents(team_id=watcher._local_team_id)
        assert len(agents) == 1
        assert agents[0].slug == "support-bot"
        assert agents[0].metadata["source"] == "file"

        # Check version was created
        versions = gate.store.list_agent_versions(agents[0].id)
        assert len(versions) == 1
        assert versions[0].model == "gpt-4o"
        assert versions[0].system_prompt == "You are a helpful support agent."
        assert versions[0].request_overrides.get("temperature") == 0.3

    def test_modified_file_creates_new_version(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        f = prompts_dir / "my-agent.md"
        f.write_text("---\nmodel: gpt-4o\n---\nVersion 1\n")
        watcher.scan()

        agents = gate.list_agents(team_id=watcher._local_team_id)
        assert len(agents) == 1
        agent_id = agents[0].id

        # Modify file
        f.write_text("---\nmodel: gpt-4o\n---\nVersion 2\n")
        watcher.scan()

        versions = gate.store.list_agent_versions(agent_id)
        assert len(versions) == 2
        # Active version should be v2
        agent = gate.get_agent(agent_id)
        active = gate.store.get_agent_version(agent.active_version_id)
        assert active.system_prompt == "Version 2"

    def test_unchanged_file_noop(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        f = prompts_dir / "my-agent.md"
        f.write_text("---\nmodel: gpt-4o\n---\nStable prompt\n")
        watcher.scan()

        agents = gate.list_agents(team_id=watcher._local_team_id)
        agent_id = agents[0].id
        versions_before = gate.store.list_agent_versions(agent_id)

        # Scan again — no changes
        watcher.scan()
        versions_after = gate.store.list_agent_versions(agent_id)
        assert len(versions_after) == len(versions_before)

    def test_deleted_file_archives_agent(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        f = prompts_dir / "my-agent.md"
        f.write_text("---\nmodel: gpt-4o\n---\nPrompt\n")
        watcher.scan()

        agents = gate.list_agents(team_id=watcher._local_team_id)
        assert len(agents) == 1
        agent_id = agents[0].id

        # Delete file
        f.unlink()
        watcher.scan()

        agent = gate.get_agent(agent_id)
        assert agent.status == AgentStatus.ARCHIVED

    def test_invalid_filename_skipped(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        # Invalid slug (too short)
        f = prompts_dir / "ab.md"
        f.write_text("---\nmodel: gpt-4o\n---\nPrompt\n")
        watcher.scan()

        agents = gate.list_agents(team_id=watcher._local_team_id)
        assert len(agents) == 0

    def test_api_created_agent_not_overwritten(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        # Create agent via API first (no "source" metadata)
        gate.create_agent(
            slug="my-agent",
            team_id=watcher._local_team_id,
            model="gpt-4o",
            system_prompt="API prompt",
            org_id=watcher._local_org_id,
        )

        # Now create a file with the same slug
        f = prompts_dir / "my-agent.md"
        f.write_text("---\nmodel: gpt-4o\n---\nFile prompt\n")
        watcher.scan()

        # Agent should still have the API prompt (only 1 version)
        agents = gate.list_agents(team_id=watcher._local_team_id)
        assert len(agents) == 1
        versions = gate.store.list_agent_versions(agents[0].id)
        assert len(versions) == 1
        assert versions[0].system_prompt == "API prompt"

    def test_parse_error_recorded_in_status(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        # Create a file that will cause a parse error by mocking
        f = prompts_dir / "bad-file.md"
        f.write_text("---\nmodel: gpt-4o\n---\nValid prompt\n")

        # Simulate error by making _sync_file raise
        original_sync = watcher._sync_file

        def _broken_sync(path, slug):
            raise RuntimeError("Test error")

        watcher._sync_file = _broken_sync
        watcher.scan()

        status = watcher.get_status()
        assert len(status["recent_errors"]) > 0
        assert "Test error" in status["recent_errors"][0]["error"]

    def test_scan_idempotent(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        f = prompts_dir / "my-agent.md"
        f.write_text("---\nmodel: gpt-4o\n---\nPrompt\n")

        watcher.scan()
        watcher.scan()  # Second scan should be no-op

        agents = gate.list_agents(team_id=watcher._local_team_id)
        assert len(agents) == 1
        versions = gate.store.list_agent_versions(agents[0].id)
        assert len(versions) == 1

    def test_local_org_team_auto_created(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
        )
        watcher._ensure_local_hierarchy()

        assert watcher._local_org_id != ""
        assert watcher._local_team_id != ""

        # Calling again should be idempotent
        org_id = watcher._local_org_id
        team_id = watcher._local_team_id
        watcher._ensure_local_hierarchy()
        assert watcher._local_org_id == org_id
        assert watcher._local_team_id == team_id

    def test_txt_without_default_model_skipped(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="",  # No default model
        )
        watcher._ensure_local_hierarchy()

        f = prompts_dir / "txt-agent.txt"
        f.write_text("You are a helpful assistant.")
        watcher.scan()

        agents = gate.list_agents(team_id=watcher._local_team_id)
        assert len(agents) == 0

    def test_empty_file_skipped(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        f = prompts_dir / "empty-agent.md"
        f.write_text("")
        watcher.scan()

        agents = gate.list_agents(team_id=watcher._local_team_id)
        assert len(agents) == 0

    def test_get_status(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        f = prompts_dir / "my-agent.md"
        f.write_text("---\nmodel: gpt-4o\n---\nPrompt\n")
        watcher.scan()

        status = watcher.get_status()
        assert status["enabled"] is True
        assert status["tracked_files"] == 1
        assert "my-agent" in status["tracked_slugs"]
        assert status["local_team_id"] == watcher._local_team_id

    def test_startup_seeds_from_store(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        # First watcher creates an agent
        watcher1 = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher1._ensure_local_hierarchy()

        f = prompts_dir / "my-agent.md"
        f.write_text("---\nmodel: gpt-4o\n---\nPrompt\n")
        watcher1.scan()

        # Second watcher should seed from store
        watcher2 = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher2._local_org_id = watcher1._local_org_id
        watcher2._local_team_id = watcher1._local_team_id
        watcher2._seed_hashes()

        assert "my-agent" in watcher2._file_hashes

        # Scanning with same content should be a no-op
        watcher2.scan()
        agents = gate.list_agents(team_id=watcher2._local_team_id)
        assert len(agents) == 1
        versions = gate.store.list_agent_versions(agents[0].id)
        assert len(versions) == 1  # No new version created

    def test_stop_cleanly(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
        )
        # stop() without start() should be fine
        watcher.stop()
        assert not watcher._started

    def test_binary_files_ignored(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        # Create non-supported files
        (prompts_dir / "image.png").write_bytes(b"\x89PNG")
        (prompts_dir / "data.json").write_text("{}")
        watcher.scan()

        agents = gate.list_agents(team_id=watcher._local_team_id)
        assert len(agents) == 0

    def test_deleted_file_then_recreated(self, tmp_path: Path) -> None:
        """Delete a file, then recreate it — agent should be reactivated."""
        gate = _make_gate(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        watcher = PromptWatcher(
            gate=gate,
            prompts_dir=prompts_dir,
            default_model="gpt-4o",
        )
        watcher._ensure_local_hierarchy()

        f = prompts_dir / "my-agent.md"
        f.write_text("---\nmodel: gpt-4o\n---\nFirst version\n")
        watcher.scan()

        agents = gate.list_agents(team_id=watcher._local_team_id)
        agent_id = agents[0].id

        # Delete
        f.unlink()
        watcher.scan()
        assert gate.get_agent(agent_id).status == AgentStatus.ARCHIVED

        # Recreate with new content
        f.write_text("---\nmodel: gpt-4o\n---\nSecond version\n")
        watcher.scan()

        agent = gate.get_agent(agent_id)
        assert agent.status == AgentStatus.ACTIVE
        versions = gate.store.list_agent_versions(agent_id)
        assert len(versions) == 2
