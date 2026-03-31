"""Prompt file parser — extract agent config from .md, .yaml/.yml, and .txt files."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateloom.agent.models import validate_slug

logger = logging.getLogger("stateloom")

# Frontmatter keys that map to PromptFileContent fields (not metadata).
_KNOWN_KEYS = frozenset(
    {
        "model",
        "temperature",
        "max_tokens",
        "top_p",
        "stop",
        "response_format",
        "seed",
        "budget_per_session",
        "description",
        "name",
    }
)

# Keys that go into request_overrides (provider kwargs).
_OVERRIDE_KEYS = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "stop",
        "response_format",
        "seed",
    }
)


@dataclass
class PromptFileContent:
    """Parsed content from a prompt file."""

    model: str = ""
    system_prompt: str = ""
    request_overrides: dict[str, Any] = field(default_factory=dict)
    budget_per_session: float | None = None
    description: str = ""
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_prompt_file(path: Path, default_model: str = "") -> PromptFileContent:
    """Parse a prompt file and return its content.

    Dispatches by file extension:
    - ``.md``: YAML frontmatter + body as system_prompt
    - ``.yaml`` / ``.yml``: structured YAML
    - ``.txt``: entire content as system_prompt
    """
    ext = path.suffix.lower()
    if ext == ".md":
        return _parse_markdown(path, default_model)
    if ext in (".yaml", ".yml"):
        return _parse_yaml(path, default_model)
    if ext == ".txt":
        return _parse_text(path, default_model)
    return PromptFileContent()


def _parse_markdown(path: Path, default_model: str) -> PromptFileContent:
    """Parse a Markdown file with optional YAML frontmatter."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return PromptFileContent()

    # Check for frontmatter delimiters
    if text.startswith("---\n") or text.startswith("---\r\n"):
        # Find closing delimiter
        end_idx = text.find("\n---", 1)
        if end_idx == -1:
            # No closing delimiter — treat entire file as system_prompt
            logger.warning("No closing frontmatter delimiter in %s, treating as plain text", path)
            return PromptFileContent(
                system_prompt=text.strip(),
                model=default_model,
            )

        frontmatter_str = text[4:end_idx]  # skip leading "---\n"
        body = text[end_idx + 4 :].strip()  # skip "\n---" + newline

        try:
            import yaml  # type: ignore[import-untyped]

            fm = yaml.safe_load(frontmatter_str)
        except Exception:
            logger.warning("Invalid YAML frontmatter in %s, treating as plain text", path)
            return PromptFileContent(
                system_prompt=text.strip(),
                model=default_model,
            )

        if not isinstance(fm, dict):
            # Frontmatter parsed but isn't a dict
            return PromptFileContent(
                system_prompt=body or text.strip(),
                model=default_model,
            )

        return _build_from_dict(fm, body, default_model)

    # No frontmatter — entire file is system_prompt
    return PromptFileContent(
        system_prompt=text.strip(),
        model=default_model,
    )


def _parse_yaml(path: Path, default_model: str) -> PromptFileContent:
    """Parse a full YAML file."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return PromptFileContent()

    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        logger.warning("Invalid YAML in %s", path)
        return PromptFileContent()

    if not isinstance(data, dict):
        return PromptFileContent()

    system_prompt = str(data.pop("system_prompt", ""))
    overrides = data.pop("request_overrides", None)

    result = _build_from_dict(data, system_prompt, default_model)

    # YAML files can specify request_overrides explicitly
    if isinstance(overrides, dict):
        result.request_overrides.update(overrides)

    return result


def _parse_text(path: Path, default_model: str) -> PromptFileContent:
    """Parse a plain text file — entire content is the system prompt."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return PromptFileContent()
    return PromptFileContent(
        system_prompt=text,
        model=default_model,
    )


def _build_from_dict(fm: dict[str, Any], body: str, default_model: str) -> PromptFileContent:
    """Build PromptFileContent from a frontmatter/YAML dict and body text."""
    overrides: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    model = str(fm.get("model", "")) or default_model
    budget = fm.get("budget_per_session")
    description = str(fm.get("description", ""))
    name = str(fm.get("name", ""))

    for key, value in fm.items():
        if key in _OVERRIDE_KEYS:
            overrides[key] = value
        elif key not in _KNOWN_KEYS:
            metadata[key] = value

    return PromptFileContent(
        model=model,
        system_prompt=body,
        request_overrides=overrides,
        budget_per_session=float(budget) if budget is not None else None,
        description=description,
        name=name,
        metadata=metadata,
    )


def slug_from_filename(path: Path) -> str | None:
    """Extract a valid agent slug from a filename stem.

    Returns None if the stem is not a valid slug.
    """
    stem = path.stem.lower()
    if validate_slug(stem):
        return stem
    return None


def content_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
