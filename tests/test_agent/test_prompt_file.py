"""Tests for prompt file parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from stateloom.agent.prompt_file import (
    PromptFileContent,
    content_hash,
    parse_prompt_file,
    slug_from_filename,
)


class TestParseMarkdown:
    """Tests for .md file parsing."""

    def test_md_with_frontmatter(self, tmp_path: Path) -> None:
        f = tmp_path / "support-bot.md"
        f.write_text(
            "---\n"
            "model: gpt-4o\n"
            "temperature: 0.3\n"
            "max_tokens: 4096\n"
            "budget_per_session: 2.0\n"
            "description: Customer support agent\n"
            "---\n"
            "You are a helpful customer support agent.\n"
        )
        result = parse_prompt_file(f)
        assert result.model == "gpt-4o"
        assert result.system_prompt == "You are a helpful customer support agent."
        assert result.request_overrides["temperature"] == 0.3
        assert result.request_overrides["max_tokens"] == 4096
        assert result.budget_per_session == 2.0
        assert result.description == "Customer support agent"

    def test_md_without_frontmatter(self, tmp_path: Path) -> None:
        f = tmp_path / "simple.md"
        f.write_text("You are a helpful assistant.\nBe kind.\n")
        result = parse_prompt_file(f, default_model="gpt-3.5-turbo")
        assert result.system_prompt == "You are a helpful assistant.\nBe kind."
        assert result.model == "gpt-3.5-turbo"
        assert result.request_overrides == {}

    def test_md_invalid_frontmatter_fallback(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.md"
        f.write_text("---\ninvalid: [yaml: broken\n---\nSome prompt text\n")
        result = parse_prompt_file(f, default_model="fallback-model")
        # Should treat entire file as system_prompt
        assert "invalid" in result.system_prompt or "Some prompt text" in result.system_prompt
        assert result.model == "fallback-model"

    def test_md_no_closing_delimiter(self, tmp_path: Path) -> None:
        f = tmp_path / "noclosing.md"
        f.write_text("---\nmodel: gpt-4o\nSome prompt without closing delimiter\n")
        result = parse_prompt_file(f, default_model="default")
        # No closing delimiter — treated as plain text
        assert result.model == "default"
        assert "model: gpt-4o" in result.system_prompt

    def test_md_frontmatter_with_name(self, tmp_path: Path) -> None:
        f = tmp_path / "named.md"
        f.write_text("---\nmodel: claude-3-haiku\nname: My Agent\n---\nYou are a named agent.\n")
        result = parse_prompt_file(f)
        assert result.name == "My Agent"
        assert result.model == "claude-3-haiku"

    def test_md_unrecognized_keys_go_to_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "custom.md"
        f.write_text(
            "---\nmodel: gpt-4o\ncustom_key: custom_value\nteam: engineering\n---\nPrompt text.\n"
        )
        result = parse_prompt_file(f)
        assert result.metadata["custom_key"] == "custom_value"
        assert result.metadata["team"] == "engineering"
        assert "model" not in result.metadata

    def test_md_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.md"
        f.write_text("")
        result = parse_prompt_file(f)
        assert result.system_prompt == ""
        assert result.model == ""


class TestParseYaml:
    """Tests for .yaml/.yml file parsing."""

    def test_yaml_full_structure(self, tmp_path: Path) -> None:
        f = tmp_path / "agent.yaml"
        f.write_text(
            "model: gpt-4o\n"
            "system_prompt: You are an assistant.\n"
            "request_overrides:\n"
            "  temperature: 0.5\n"
            "  max_tokens: 2000\n"
            "budget_per_session: 1.5\n"
            "description: A YAML agent\n"
        )
        result = parse_prompt_file(f)
        assert result.model == "gpt-4o"
        assert result.system_prompt == "You are an assistant."
        assert result.request_overrides["temperature"] == 0.5
        assert result.request_overrides["max_tokens"] == 2000
        assert result.budget_per_session == 1.5
        assert result.description == "A YAML agent"

    def test_yml_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "agent.yml"
        f.write_text("model: claude-3-opus\nsystem_prompt: Be helpful.\n")
        result = parse_prompt_file(f)
        assert result.model == "claude-3-opus"
        assert result.system_prompt == "Be helpful."

    def test_yaml_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.yaml"
        f.write_text("")
        result = parse_prompt_file(f)
        assert result.system_prompt == ""


class TestParseText:
    """Tests for .txt file parsing."""

    def test_txt_plain_text(self, tmp_path: Path) -> None:
        f = tmp_path / "simple.txt"
        f.write_text("You are a helpful assistant.")
        result = parse_prompt_file(f, default_model="gpt-4o-mini")
        assert result.system_prompt == "You are a helpful assistant."
        assert result.model == "gpt-4o-mini"
        assert result.request_overrides == {}

    def test_txt_no_default_model(self, tmp_path: Path) -> None:
        f = tmp_path / "nomodel.txt"
        f.write_text("You are a helpful assistant.")
        result = parse_prompt_file(f)
        assert result.system_prompt == "You are a helpful assistant."
        assert result.model == ""

    def test_txt_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = parse_prompt_file(f)
        assert result.system_prompt == ""


class TestSlugFromFilename:
    """Tests for slug extraction from filenames."""

    def test_valid_slug(self, tmp_path: Path) -> None:
        assert slug_from_filename(tmp_path / "support-bot.md") == "support-bot"
        assert slug_from_filename(tmp_path / "my-agent-v2.yaml") == "my-agent-v2"
        assert slug_from_filename(tmp_path / "abc.txt") == "abc"

    def test_invalid_slug_too_short(self, tmp_path: Path) -> None:
        assert slug_from_filename(tmp_path / "ab.md") is None

    def test_invalid_slug_leading_hyphen(self, tmp_path: Path) -> None:
        assert slug_from_filename(tmp_path / "-bad.md") is None

    def test_invalid_slug_uppercase(self, tmp_path: Path) -> None:
        # slug_from_filename lowercases the stem, then validates
        # "AB" lowered is "ab" which is only 2 chars → invalid
        assert slug_from_filename(tmp_path / "AB.md") is None

    def test_invalid_slug_underscore(self, tmp_path: Path) -> None:
        assert slug_from_filename(tmp_path / "bad_name.md") is None


class TestContentHash:
    """Tests for content hashing."""

    def test_consistent_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("hello world")
        h1 = content_hash(f)
        h2 = content_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.md"
        f1.write_text("hello")
        f2 = tmp_path / "b.md"
        f2.write_text("world")
        assert content_hash(f1) != content_hash(f2)
