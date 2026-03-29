"""Tests for the Cohere adapter."""

from __future__ import annotations

from types import SimpleNamespace

from stateloom.core.types import Provider
from stateloom.intercept.adapters.cohere_adapter import CohereAdapter


class TestCohereAdapterUnit:
    """Unit tests for CohereAdapter methods (no cohere install needed)."""

    def _adapter(self):
        return CohereAdapter()

    def test_name(self):
        assert self._adapter().name == Provider.COHERE
        assert self._adapter().name == "cohere"

    def test_method_label(self):
        assert self._adapter().method_label == "chat"

    def test_get_patch_targets(self):
        targets = self._adapter().get_patch_targets()
        # Returns [] if cohere not installed, or 2 targets if it is
        assert len(targets) in (0, 2)

    def test_extract_model_from_kwargs(self):
        adapter = self._adapter()
        result = adapter.extract_model(None, (), {"model": "command-a-03-2025"})
        assert result == "command-a-03-2025"

    def test_extract_model_default(self):
        assert self._adapter().extract_model(None, (), {}) == "unknown"

    def test_extract_tokens(self):
        """Cohere V2 uses usage.tokens.input_tokens/output_tokens (float)."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                tokens=SimpleNamespace(
                    input_tokens=100.0,
                    output_tokens=50.0,
                )
            )
        )
        assert self._adapter().extract_tokens(response) == (100, 50, 150)

    def test_extract_tokens_no_usage(self):
        assert self._adapter().extract_tokens(SimpleNamespace()) == (0, 0, 0)

    def test_extract_tokens_none_values(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                tokens=SimpleNamespace(
                    input_tokens=None,
                    output_tokens=None,
                )
            )
        )
        assert self._adapter().extract_tokens(response) == (0, 0, 0)

    def test_extract_stream_tokens_message_end(self):
        """Token usage arrives in the message-end streaming event."""
        chunk = SimpleNamespace(
            type="message-end",
            delta=SimpleNamespace(
                usage=SimpleNamespace(
                    tokens=SimpleNamespace(
                        input_tokens=30.0,
                        output_tokens=15.0,
                    )
                )
            ),
        )
        acc: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        result = self._adapter().extract_stream_tokens(chunk, acc)
        assert result == {"prompt_tokens": 30, "completion_tokens": 15}

    def test_extract_stream_tokens_content_delta_ignored(self):
        """Non-end events should not update token counts."""
        chunk = SimpleNamespace(type="content-delta")
        acc: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        result = self._adapter().extract_stream_tokens(chunk, acc)
        assert result == {"prompt_tokens": 0, "completion_tokens": 0}

    def test_apply_system_prompt_insert(self):
        adapter = self._adapter()
        kwargs: dict = {"messages": [{"role": "user", "content": "hello"}]}
        adapter.apply_system_prompt(kwargs, "You are helpful.")
        assert kwargs["messages"][0] == {
            "role": "system",
            "content": "You are helpful.",
        }
        assert len(kwargs["messages"]) == 2

    def test_apply_system_prompt_replace(self):
        adapter = self._adapter()
        kwargs: dict = {
            "messages": [
                {"role": "system", "content": "old"},
                {"role": "user", "content": "hello"},
            ]
        }
        adapter.apply_system_prompt(kwargs, "new")
        assert kwargs["messages"][0]["content"] == "new"
        assert len(kwargs["messages"]) == 2

    def test_get_instance_targets_v2_client(self):
        """V2 clients have 'V2' in class name."""

        class FakeClientV2:
            pass

        client = FakeClientV2()
        targets = self._adapter().get_instance_targets(client)
        assert len(targets) == 1
        assert targets[0][1] == "chat"

    def test_get_instance_targets_v1_with_v2_accessor(self):
        """V1 clients with .v2 property."""
        v2 = SimpleNamespace()
        client = SimpleNamespace(v2=v2)
        targets = self._adapter().get_instance_targets(client)
        assert len(targets) == 1
        assert targets[0][0] is v2
        assert targets[0][1] == "chat"

    def test_get_instance_targets_unknown(self):
        client = SimpleNamespace()
        assert self._adapter().get_instance_targets(client) == []


class TestProviderEnum:
    def test_cohere_in_provider_enum(self):
        assert Provider.COHERE == "cohere"
        assert Provider.COHERE.value == "cohere"
