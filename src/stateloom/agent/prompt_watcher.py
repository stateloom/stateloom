"""Watchdog-based file watcher for prompt file → agent sync."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stateloom.agent.prompt_file import (
    content_hash,
    parse_prompt_file,
    slug_from_filename,
)

if TYPE_CHECKING:
    from stateloom.gate import Gate

logger = logging.getLogger("stateloom")

_SUPPORTED_EXTENSIONS = frozenset({".md", ".yaml", ".yml", ".txt"})

# Sentinel org/team for file-based agents.
_LOCAL_ORG_NAME = "_local"
_LOCAL_TEAM_NAME = "_local"


class PromptWatcher:
    """Watches a directory for prompt files and syncs them as managed agents.

    Uses the ``watchdog`` library for event-driven filesystem monitoring.
    Falls back to periodic polling if watchdog is not installed.
    """

    def __init__(
        self,
        gate: Gate,
        prompts_dir: Path,
        poll_interval: float = 2.0,
        default_model: str = "",
        auto_activate: bool = True,
    ) -> None:
        self._gate = gate
        self._prompts_dir = prompts_dir
        self._poll_interval = poll_interval
        self._default_model = default_model
        self._auto_activate = auto_activate

        # slug → content SHA-256
        self._file_hashes: dict[str, str] = {}
        # Recent errors for status reporting
        self._recent_errors: list[dict[str, Any]] = []
        self._max_errors = 20

        # Debounce timers: slug → Timer
        self._debounce_timers: dict[str, threading.Timer] = {}
        self._debounce_lock = threading.Lock()
        self._debounce_delay = 0.5  # seconds

        # watchdog observer
        self._observer: Any = None
        self._started = False

        # Team/org IDs for file-based agents
        self._local_org_id: str = ""
        self._local_team_id: str = ""

    def start(self) -> None:
        """Create prompts dir if needed, run initial scan, start watchdog."""
        if self._started:
            return

        # Ensure directory exists
        self._prompts_dir.mkdir(parents=True, exist_ok=True)

        # Auto-create _local org + team
        self._ensure_local_hierarchy()

        # Seed hashes from existing file-sourced agents in the store
        self._seed_hashes()

        # Initial scan
        self.scan()

        # Start watchdog observer
        try:
            from watchdog.observers import Observer

            handler = _PromptEventHandler(self)
            self._observer = Observer()
            self._observer.schedule(handler, str(self._prompts_dir), recursive=False)
            self._observer.daemon = True
            self._observer.start()
            logger.info(
                "Prompt watcher started: %s (%d files tracked)",
                self._prompts_dir,
                len(self._file_hashes),
            )
        except ImportError:
            logger.warning(
                "watchdog not installed — prompt watcher using initial scan only. "
                "Install with: pip install watchdog"
            )

        self._started = True

    def stop(self) -> None:
        """Stop the watchdog observer."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        # Cancel pending debounce timers
        with self._debounce_lock:
            for timer in self._debounce_timers.values():
                timer.cancel()
            self._debounce_timers.clear()

        self._started = False

    def scan(self) -> None:
        """Full directory reconciliation — idempotent."""
        if not self._prompts_dir.is_dir():
            return

        seen_slugs: set[str] = set()

        for path in self._prompts_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue

            slug = slug_from_filename(path)
            if slug is None:
                continue

            seen_slugs.add(slug)
            try:
                self._sync_file(path, slug)
            except Exception as exc:
                self._record_error(path, str(exc))

        # Handle deletions: tracked slugs no longer present on disk
        removed_slugs = set(self._file_hashes.keys()) - seen_slugs
        for slug in removed_slugs:
            try:
                self._handle_deletion(slug)
            except Exception as exc:
                self._record_error(self._prompts_dir / slug, str(exc))

    def get_status(self) -> dict[str, Any]:
        """Return watcher status for the dashboard API."""
        return {
            "enabled": True,
            "prompts_dir": str(self._prompts_dir),
            "tracked_files": len(self._file_hashes),
            "tracked_slugs": sorted(self._file_hashes.keys()),
            "local_team_id": self._local_team_id,
            "local_org_id": self._local_org_id,
            "observer_alive": (self._observer.is_alive() if self._observer is not None else False),
            "recent_errors": list(self._recent_errors),
        }

    # --- Internal ---

    def _ensure_local_hierarchy(self) -> None:
        """Create _local org + _local team if they don't exist."""
        org = self._gate.create_organization(name=_LOCAL_ORG_NAME)
        self._local_org_id = org.id

        team = self._gate.create_team(org.id, name=_LOCAL_TEAM_NAME)
        self._local_team_id = team.id

    def _seed_hashes(self) -> None:
        """Seed _file_hashes from existing file-sourced agents in the store."""
        try:
            agents = self._gate.list_agents(team_id=self._local_team_id)
            for agent in agents:
                meta = agent.metadata or {}
                if meta.get("source") == "file" and "content_hash" in meta:
                    self._file_hashes[agent.slug] = meta["content_hash"]
        except Exception:
            logger.debug("Failed to seed file hashes from store", exc_info=True)

    def _sync_file(self, path: Path, slug: str) -> None:
        """Sync a single prompt file to the agent store."""
        # Parse file
        content = parse_prompt_file(path, default_model=self._default_model)
        if not content.system_prompt:
            logger.debug("Skipping empty prompt file: %s", path)
            return

        # .txt files without a model are unusable
        if path.suffix.lower() == ".txt" and not content.model:
            logger.warning("Skipping %s: .txt file with no default_model configured", path)
            return

        # Compute content hash
        file_hash = content_hash(path)
        if self._file_hashes.get(slug) == file_hash:
            return  # No change

        # Check existing agent
        existing = self._gate.get_agent_by_slug(slug, self._local_team_id)

        if existing is not None:
            # Don't overwrite API-created agents
            if (existing.metadata or {}).get("source") != "file":
                logger.warning(
                    "Skipping %s: agent '%s' exists but was not created from a file",
                    path,
                    slug,
                )
                return

            # Check if archived — reactivate
            from stateloom.core.types import AgentStatus

            if existing.status == AgentStatus.ARCHIVED:
                self._gate.update_agent(
                    existing.id,
                    status=AgentStatus.ACTIVE.value,
                    metadata={
                        **existing.metadata,
                        "content_hash": file_hash,
                        "source_path": str(path),
                    },
                )

            # Create new version
            version = self._gate.create_agent_version(
                existing.id,
                model=content.model,
                system_prompt=content.system_prompt,
                request_overrides=content.request_overrides or None,
                budget_per_session=content.budget_per_session,
                metadata={"content_hash": file_hash, "source_path": str(path)},
                created_by="prompt_watcher",
            )

            # Auto-activate
            if self._auto_activate:
                self._gate.activate_agent_version(existing.id, version.id)

            # Update metadata on agent
            self._gate.update_agent(
                existing.id,
                name=content.name or existing.name,
                description=content.description or existing.description,
                metadata={
                    **(existing.metadata or {}),
                    "source": "file",
                    "source_path": str(path),
                    "content_hash": file_hash,
                },
            )
        else:
            # Create new agent
            self._gate.create_agent(
                slug=slug,
                team_id=self._local_team_id,
                name=content.name or slug,
                description=content.description,
                model=content.model,
                system_prompt=content.system_prompt,
                request_overrides=content.request_overrides or None,
                budget_per_session=content.budget_per_session,
                metadata={
                    "source": "file",
                    "source_path": str(path),
                    "content_hash": file_hash,
                    **(content.metadata or {}),
                },
                created_by="prompt_watcher",
                org_id=self._local_org_id,
            )

        self._file_hashes[slug] = file_hash
        logger.info("Synced prompt file: %s → agent '%s'", path.name, slug)

    def _handle_deletion(self, slug: str) -> None:
        """Archive an agent whose prompt file has been deleted."""
        agent = self._gate.get_agent_by_slug(slug, self._local_team_id)
        if agent is None:
            self._file_hashes.pop(slug, None)
            return

        if (agent.metadata or {}).get("source") != "file":
            self._file_hashes.pop(slug, None)
            return

        from stateloom.core.types import AgentStatus

        if agent.status != AgentStatus.ARCHIVED:
            self._gate.archive_agent(agent.id)
            logger.info("Archived agent '%s' (prompt file deleted)", slug)

        self._file_hashes.pop(slug, None)

    def _debounced_sync(self, path: Path, slug: str) -> None:
        """Schedule a debounced sync for a file."""
        with self._debounce_lock:
            existing = self._debounce_timers.get(slug)
            if existing is not None:
                existing.cancel()

            timer = threading.Timer(
                self._debounce_delay,
                self._safe_sync_file,
                args=(path, slug),
            )
            timer.daemon = True
            self._debounce_timers[slug] = timer
            timer.start()

    def _safe_sync_file(self, path: Path, slug: str) -> None:
        """Sync a file, catching all exceptions."""
        try:
            if path.is_file():
                self._sync_file(path, slug)
            else:
                self._handle_deletion(slug)
        except Exception as exc:
            self._record_error(path, str(exc))

        with self._debounce_lock:
            self._debounce_timers.pop(slug, None)

    def _record_error(self, path: Path, error: str) -> None:
        """Record a parse/sync error for status reporting."""
        logger.warning("Prompt watcher error for %s: %s", path, error)
        self._recent_errors.append(
            {
                "path": str(path),
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Cap error list
        if len(self._recent_errors) > self._max_errors:
            self._recent_errors = self._recent_errors[-self._max_errors :]


class _PromptEventHandler:
    """Watchdog event handler for prompt file changes.

    Implements the watchdog FileSystemEventHandler interface as a duck-type
    so watchdog is not required at import time.
    """

    def __init__(self, watcher: PromptWatcher) -> None:
        self._watcher = watcher

    def dispatch(self, event: Any) -> None:
        """Route watchdog events to the appropriate handler."""
        if event.is_directory:
            return

        event_type = event.event_type
        if event_type == "created":
            self.on_created(event)
        elif event_type == "modified":
            self.on_modified(event)
        elif event_type == "deleted":
            self.on_deleted(event)
        elif event_type == "moved":
            self.on_moved(event)

    def on_created(self, event: Any) -> None:
        self._handle_change(Path(event.src_path))

    def on_modified(self, event: Any) -> None:
        self._handle_change(Path(event.src_path))

    def on_deleted(self, event: Any) -> None:
        path = Path(event.src_path)
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return
        slug = slug_from_filename(path)
        if slug is None:
            return
        self._watcher._debounced_sync(path, slug)

    def on_moved(self, event: Any) -> None:
        # Old name → archive
        old_path = Path(event.src_path)
        if old_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
            old_slug = slug_from_filename(old_path)
            if old_slug is not None:
                self._watcher._debounced_sync(old_path, old_slug)

        # New name → create/update
        new_path = Path(event.dest_path)
        self._handle_change(new_path)

    def _handle_change(self, path: Path) -> None:
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            return
        slug = slug_from_filename(path)
        if slug is None:
            return
        self._watcher._debounced_sync(path, slug)
