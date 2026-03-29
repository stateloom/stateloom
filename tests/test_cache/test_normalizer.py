"""Tests for RequestNormalizer — hash normalization of dynamic content."""

from __future__ import annotations

import pytest

from stateloom.cache.normalizer import RequestNormalizer


class TestRequestNormalizer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.normalizer = RequestNormalizer()

    def test_strips_uuid(self):
        kwargs = {
            "messages": [
                {"role": "user", "content": "Process request 550e8400-e29b-41d4-a716-446655440000"}
            ]
        }
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "550e8400" not in result["messages"][0]["content"]
        assert "<DYN>" in result["messages"][0]["content"]

    def test_strips_iso_timestamp(self):
        kwargs = {"messages": [{"role": "user", "content": "Event at 2024-03-17T14:30:00Z"}]}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "2024-03-17T14:30:00Z" not in result["messages"][0]["content"]
        assert "<DYN>" in result["messages"][0]["content"]

    def test_strips_iso_timestamp_with_offset(self):
        kwargs = {
            "messages": [{"role": "user", "content": "Event at 2024-03-17T14:30:00.123+05:30"}]
        }
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "2024-03-17T14:30:00.123+05:30" not in result["messages"][0]["content"]

    def test_strips_unix_timestamp_10_digit(self):
        kwargs = {"messages": [{"role": "user", "content": "Created at 1710000000 epoch"}]}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "1710000000" not in result["messages"][0]["content"]
        assert "<DYN>" in result["messages"][0]["content"]

    def test_strips_unix_timestamp_13_digit(self):
        kwargs = {"messages": [{"role": "user", "content": "Time 1710000000123 ms"}]}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "1710000000123" not in result["messages"][0]["content"]

    def test_strips_date_iso_format(self):
        kwargs = {"messages": [{"role": "user", "content": "Report for 2024-03-17 summary"}]}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "2024-03-17" not in result["messages"][0]["content"]

    def test_strips_date_us_format(self):
        kwargs = {"messages": [{"role": "user", "content": "Report for 03/17/2024 summary"}]}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "03/17/2024" not in result["messages"][0]["content"]

    def test_strips_date_eu_format(self):
        kwargs = {"messages": [{"role": "user", "content": "Report for 17.03.2024 summary"}]}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "17.03.2024" not in result["messages"][0]["content"]

    def test_strips_long_hex_id(self):
        kwargs = {
            "messages": [
                {"role": "user", "content": "Trace 507f1f77bcf86cd799439011abcd1234 found"}
            ]
        }
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "507f1f77bcf86cd799439011abcd1234" not in result["messages"][0]["content"]
        assert "<DYN>" in result["messages"][0]["content"]

    def test_does_not_strip_short_hex(self):
        # 23 hex chars is below the 24 threshold
        kwargs = {"messages": [{"role": "user", "content": "Code abc0123456789abcdef0123"}]}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "abc0123456789abcdef0123" in result["messages"][0]["content"]

    def test_preserves_non_content_fields(self):
        kwargs = {
            "model": "gpt-4",
            "temperature": 0.7,
            "messages": [
                {"role": "user", "content": "Request 550e8400-e29b-41d4-a716-446655440000"}
            ],
        }
        result = self.normalizer.normalize_kwargs(kwargs)
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7
        assert result["messages"][0]["role"] == "user"

    def test_does_not_mutate_original(self):
        original_content = "Request 550e8400-e29b-41d4-a716-446655440000"
        kwargs = {"messages": [{"role": "user", "content": original_content}]}
        self.normalizer.normalize_kwargs(kwargs)
        assert kwargs["messages"][0]["content"] == original_content

    def test_normalizes_structured_content_blocks(self):
        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "ID: 550e8400-e29b-41d4-a716-446655440000"},
                    ],
                }
            ]
        }
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "<DYN>" in result["messages"][0]["content"][0]["text"]

    def test_normalizes_gemini_contents_string(self):
        kwargs = {"contents": "Event at 2024-03-17T14:30:00Z please process"}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "<DYN>" in result["contents"]

    def test_normalizes_gemini_contents_list_strings(self):
        kwargs = {"contents": ["Event 2024-03-17T14:30:00Z", "normal text"]}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "<DYN>" in result["contents"][0]
        assert result["contents"][1] == "normal text"

    def test_normalizes_gemini_contents_list_dicts(self):
        kwargs = {"contents": [{"text": "ID: 507f1f77bcf86cd799439011abcd1234"}]}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "<DYN>" in result["contents"][0]["text"]

    def test_normalizes_anthropic_system_prompt(self):
        kwargs = {
            "system": "Date: 2024-03-17 context",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = self.normalizer.normalize_kwargs(kwargs)
        assert "<DYN>" in result["system"]

    def test_multiple_dynamic_values_in_one_message(self):
        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Process 550e8400-e29b-41d4-a716-446655440000 "
                        "at 2024-03-17T14:30:00Z with trace "
                        "507f1f77bcf86cd799439011abcd1234"
                    ),
                }
            ]
        }
        result = self.normalizer.normalize_kwargs(kwargs)
        content = result["messages"][0]["content"]
        assert content.count("<DYN>") == 3

    def test_same_content_different_dynamics_produce_same_result(self):
        kwargs1 = {
            "messages": [
                {"role": "user", "content": "Process request 550e8400-e29b-41d4-a716-446655440000"}
            ]
        }
        kwargs2 = {
            "messages": [
                {"role": "user", "content": "Process request aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}
            ]
        }
        r1 = self.normalizer.normalize_kwargs(kwargs1)
        r2 = self.normalizer.normalize_kwargs(kwargs2)
        assert r1["messages"][0]["content"] == r2["messages"][0]["content"]

    def test_custom_patterns(self):
        normalizer = RequestNormalizer(custom_patterns=[r"REQ-\d+"])
        kwargs = {"messages": [{"role": "user", "content": "Process REQ-12345 immediately"}]}
        result = normalizer.normalize_kwargs(kwargs)
        assert "REQ-12345" not in result["messages"][0]["content"]
        assert "<DYN>" in result["messages"][0]["content"]

    def test_empty_messages(self):
        kwargs = {"messages": []}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert result["messages"] == []

    def test_no_messages_key(self):
        kwargs = {"model": "gpt-4"}
        result = self.normalizer.normalize_kwargs(kwargs)
        assert result == {"model": "gpt-4"}
