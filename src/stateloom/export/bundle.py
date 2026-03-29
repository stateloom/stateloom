"""Session export/import — portable JSON bundles.

Export a session (with events and optional children) as a standalone JSON
file.  Import it on another machine for replay or debugging.
"""

from __future__ import annotations

import gzip
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from stateloom._version import __version__
from stateloom.core.errors import StateLoomError
from stateloom.core.event import AnyEvent, Event
from stateloom.core.session import Session

if TYPE_CHECKING:
    from stateloom.pii.scanner import PIIScanner
    from stateloom.store.base import Store

logger = logging.getLogger("stateloom.export")

BUNDLE_SCHEMA_VERSION = "v1"

_event_adapter: TypeAdapter[Event] = TypeAdapter(AnyEvent)

# Maximum recursion depth for child sessions to prevent infinite loops.
_MAX_CHILD_DEPTH = 10


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_session(
    store: Store,
    session_id: str,
    *,
    include_children: bool = False,
    scrub_pii: bool = False,
    pii_scanner: PIIScanner | None = None,
) -> dict[str, Any]:
    """Export a session as a portable JSON bundle dict.

    Args:
        store: The store backend to read from.
        session_id: ID of the session to export.
        include_children: If True, recursively export child sessions.
        scrub_pii: If True, redact PII from event fields.
        pii_scanner: Scanner instance (required when scrub_pii=True).

    Returns:
        Bundle dict ready for serialization or ``write_bundle_to_file()``.
    """
    session = store.get_session(session_id)
    if session is None:
        raise StateLoomError(f"Session '{session_id}' not found")

    events = store.get_session_events(session_id, limit=10_000)

    session_dict = session.model_dump(mode="json")
    event_dicts = [e.model_dump(mode="json") for e in events]

    if scrub_pii and pii_scanner is not None:
        for ed in event_dicts:
            _scrub_event_dict(ed, pii_scanner)

    bundle: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "stateloom_version": __version__,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "pii_scrubbed": scrub_pii,
        "session": session_dict,
        "events": event_dicts,
        "children": [],
    }

    if include_children:
        bundle["children"] = _export_children(
            store,
            session_id,
            scrub_pii=scrub_pii,
            pii_scanner=pii_scanner,
            depth=0,
            visited=set(),
        )

    return bundle


def _export_children(
    store: Store,
    parent_id: str,
    *,
    scrub_pii: bool,
    pii_scanner: PIIScanner | None,
    depth: int,
    visited: set[str],
) -> list[dict[str, Any]]:
    """Recursively export child sessions."""
    if depth >= _MAX_CHILD_DEPTH:
        return []

    children = store.list_child_sessions(parent_id)
    result: list[dict[str, Any]] = []

    for child in children:
        if child.id in visited:
            continue
        visited.add(child.id)

        child_events = store.get_session_events(child.id, limit=10_000)
        child_dict = child.model_dump(mode="json")
        child_event_dicts = [e.model_dump(mode="json") for e in child_events]

        if scrub_pii and pii_scanner is not None:
            for ed in child_event_dicts:
                _scrub_event_dict(ed, pii_scanner)

        result.append(
            {
                "session": child_dict,
                "events": child_event_dicts,
                "children": _export_children(
                    store,
                    child.id,
                    scrub_pii=scrub_pii,
                    pii_scanner=pii_scanner,
                    depth=depth + 1,
                    visited=visited,
                ),
            }
        )

    return result


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def import_session(
    store: Store,
    source: str | Path | dict[str, Any],
    *,
    session_id_override: str | None = None,
) -> Session:
    """Import a session bundle into the store.

    Args:
        source: A bundle dict, a file path (str/Path) to a ``.json`` or
            ``.json.gz`` bundle file.
        session_id_override: If set, replaces the session ID (and all event
            references) to avoid collisions.

    Returns:
        The imported ``Session`` object.
    """
    if isinstance(source, dict):
        bundle = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise StateLoomError(f"Bundle file not found: '{path}'")
        if path.suffix == ".gz" or path.name.endswith(".json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                bundle = json.load(f)
        else:
            with open(path, encoding="utf-8") as f:
                bundle = json.load(f)
    else:
        raise StateLoomError(
            f"Unsupported source type: {type(source).__name__}. Expected dict, str, or Path."
        )

    validate_bundle(bundle)

    return _import_node(
        store,
        bundle,
        session_id_override=session_id_override,
        parent_id_override=None,
        depth=0,
        visited=set(),
    )


def _import_node(
    store: Store,
    node: dict[str, Any],
    *,
    session_id_override: str | None,
    parent_id_override: str | None,
    depth: int,
    visited: set[str],
) -> Session:
    """Import a single session node (session + events + children)."""
    if depth > _MAX_CHILD_DEPTH:
        raise StateLoomError(
            f"Import recursion depth exceeded ({_MAX_CHILD_DEPTH}). "
            "Possible circular child references."
        )

    session = Session.model_validate(node["session"])
    original_id = session.id

    if original_id in visited:
        raise StateLoomError(
            f"Circular reference detected: session '{original_id}' already imported."
        )
    visited.add(original_id)

    # Collision check / ID remapping
    new_id = session_id_override or session.id
    if session_id_override:
        session.id = new_id
    else:
        existing = store.get_session(session.id)
        if existing is not None:
            raise StateLoomError(f"Session '{session.id}' already exists. Use session_id_override.")

    # Update parent reference if remapped
    if parent_id_override is not None:
        session.parent_session_id = parent_id_override

    # Deserialize events (fail-open per event)
    events: list[Event] = []
    for event_dict in node.get("events", []):
        try:
            # Remap session_id on event to match (possibly remapped) session
            event_dict["session_id"] = new_id
            event = _event_adapter.validate_python(event_dict)
            events.append(event)
        except Exception:
            logger.warning(
                "Skipping unrecognized event during import: %s",
                event_dict.get("event_type", "unknown"),
            )

    store.save_session_with_events(session, events)

    # Recursively import children
    for child_node in node.get("children", []):
        # Generate a new child ID if the parent was remapped
        child_id_override = None
        if session_id_override:
            child_id_override = uuid.uuid4().hex[:12]

        _import_node(
            store,
            child_node,
            session_id_override=child_id_override,
            parent_id_override=new_id,
            depth=depth + 1,
            visited=visited,
        )

    return session


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_bundle_to_file(bundle: dict[str, Any], path: str | Path) -> None:
    """Write a bundle dict to a JSON or gzipped JSON file.

    Args:
        bundle: The bundle dict from ``export_session()``.
        path: Output file path.  Use ``.json.gz`` for gzip compression.
    """
    path = Path(path)
    content = json.dumps(bundle, indent=2, ensure_ascii=False)

    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_bundle(bundle: dict[str, Any]) -> None:
    """Validate a bundle dict. Raises ``StateLoomError`` on failure."""
    if not isinstance(bundle, dict):
        raise StateLoomError("Invalid bundle: expected a dict")

    version = bundle.get("schema_version")
    if version != BUNDLE_SCHEMA_VERSION:
        raise StateLoomError(
            f"Unsupported bundle schema version: '{version}'. Expected '{BUNDLE_SCHEMA_VERSION}'."
        )

    for field in ("session", "events"):
        if field not in bundle:
            raise StateLoomError(f"Invalid bundle: missing required field '{field}'")


# ---------------------------------------------------------------------------
# PII scrubbing helpers
# ---------------------------------------------------------------------------


def _scrub_text(text: str, scanner: PIIScanner) -> str:
    """Replace PII matches in *text* with redaction placeholders."""
    matches = scanner.scan(text)
    if not matches:
        return text

    # Sort by start position descending so replacements don't shift indices.
    matches.sort(key=lambda m: m.start, reverse=True)
    chars = list(text)
    for m in matches:
        placeholder = f"[PII_REDACTED:{m.pattern_name}]"
        chars[m.start : m.end] = list(placeholder)
    return "".join(chars)


def _scrub_event_dict(event_dict: dict[str, Any], scanner: PIIScanner) -> None:
    """In-place scrub PII-sensitive fields in an event dict."""
    # Simple string fields
    for key in ("prompt_preview", "cloud_preview", "local_preview", "comment", "description"):
        if key in event_dict and isinstance(event_dict[key], str) and event_dict[key]:
            event_dict[key] = _scrub_text(event_dict[key], scanner)

    # cached_response_json: parse → scrub text values → re-serialize
    crj = event_dict.get("cached_response_json")
    if crj and isinstance(crj, str):
        try:
            parsed = json.loads(crj)
            _scrub_json_value(parsed, scanner)
            event_dict["cached_response_json"] = json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass


def _scrub_json_value(obj: Any, scanner: PIIScanner) -> None:
    """Recursively scrub string values inside a parsed JSON structure."""
    if isinstance(obj, dict):
        for key in obj:
            if isinstance(obj[key], str):
                obj[key] = _scrub_text(obj[key], scanner)
            else:
                _scrub_json_value(obj[key], scanner)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                obj[i] = _scrub_text(item, scanner)
            else:
                _scrub_json_value(item, scanner)
