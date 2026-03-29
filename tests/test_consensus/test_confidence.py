"""Tests for adapter-aware confidence extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from stateloom.consensus.confidence import extract_confidence, get_confidence_instruction
from stateloom.consensus.prompts import DEFAULT_CONFIDENCE_INSTRUCTION


class TestGetConfidenceInstruction:
    def test_default_instruction(self):
        result = get_confidence_instruction()
        assert result == DEFAULT_CONFIDENCE_INSTRUCTION

    def test_empty_provider_returns_default(self):
        result = get_confidence_instruction("")
        assert result == DEFAULT_CONFIDENCE_INSTRUCTION

    def test_adapter_instruction_used(self):
        mock_adapter = MagicMock()
        mock_adapter.confidence_instruction.return_value = "Custom instruction"
        with patch(
            "stateloom.intercept.provider_registry.get_adapter",
            return_value=mock_adapter,
        ):
            result = get_confidence_instruction("openai")
        assert result == "Custom instruction"

    def test_adapter_returning_empty_falls_back(self):
        mock_adapter = MagicMock()
        mock_adapter.confidence_instruction.return_value = ""
        with patch(
            "stateloom.intercept.provider_registry.get_adapter",
            return_value=mock_adapter,
        ):
            result = get_confidence_instruction("openai")
        assert result == DEFAULT_CONFIDENCE_INSTRUCTION

    def test_adapter_not_found_falls_back(self):
        with patch(
            "stateloom.intercept.provider_registry.get_adapter",
            return_value=None,
        ):
            result = get_confidence_instruction("unknown_provider")
        assert result == DEFAULT_CONFIDENCE_INSTRUCTION


class TestExtractConfidence:
    def test_standard_format(self):
        text = "The answer is 42. [Confidence: 0.95]"
        assert extract_confidence(text) == 0.95

    def test_case_insensitive(self):
        text = "Hello [confidence: 0.80]"
        assert extract_confidence(text) == 0.80

    def test_zero_confidence(self):
        assert extract_confidence("[Confidence: 0.00]") == 0.0

    def test_one_confidence(self):
        assert extract_confidence("[Confidence: 1.00]") == 1.0

    def test_clamp_above_one(self):
        assert extract_confidence("[Confidence: 1.50]") == 1.0

    def test_missing_confidence_returns_default(self):
        assert extract_confidence("No confidence here") == 0.5

    def test_adapter_extraction_used(self):
        mock_adapter = MagicMock()
        mock_adapter.extract_confidence.return_value = 0.77
        with patch(
            "stateloom.intercept.provider_registry.get_adapter",
            return_value=mock_adapter,
        ):
            result = extract_confidence("some text", provider="openai")
        assert result == 0.77

    def test_adapter_returning_none_falls_back_to_regex(self):
        mock_adapter = MagicMock()
        mock_adapter.extract_confidence.return_value = None
        with patch(
            "stateloom.intercept.provider_registry.get_adapter",
            return_value=mock_adapter,
        ):
            result = extract_confidence("[Confidence: 0.65]", provider="openai")
        assert result == 0.65

    def test_adapter_exception_falls_back_to_regex(self):
        mock_adapter = MagicMock()
        mock_adapter.extract_confidence.side_effect = RuntimeError("boom")
        with patch(
            "stateloom.intercept.provider_registry.get_adapter",
            return_value=mock_adapter,
        ):
            result = extract_confidence("[Confidence: 0.42]", provider="openai")
        assert result == 0.42
