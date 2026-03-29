"""Tests for NER-based PII detector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stateloom.pii.ner_detector import NERDetector, NERMatch


class TestNERDetectorFallback:
    """Tests when GLiNER is not installed."""

    def test_unavailable_when_gliner_missing(self):
        """Detector gracefully handles missing gliner."""
        with patch("stateloom.pii.ner_detector._GLINER_AVAILABLE", False):
            detector = NERDetector()
            # Force re-init since _GLINER_AVAILABLE is checked at init time
            detector._initialized = False
            detector._failed = False
            assert detector.detect("My name is John Doe") == []

    def test_is_available_false_when_gliner_missing(self):
        with patch("stateloom.pii.ner_detector._GLINER_AVAILABLE", False):
            detector = NERDetector()
            assert detector.is_available is False


class TestNERDetectorWithMock:
    """Tests with mocked GLiNER model."""

    def _make_detector_with_mock(self, entities: list[dict]) -> tuple[NERDetector, MagicMock]:
        """Create a detector with a pre-loaded mock model."""
        detector = NERDetector()
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = entities
        detector._model = mock_model
        detector._initialized = True
        return detector, mock_model

    def test_detect_person(self):
        entities = [{"label": "person", "text": "John Doe", "start": 11, "end": 19, "score": 0.95}]
        detector, mock = self._make_detector_with_mock(entities)
        results = detector.detect("My name is John Doe")
        assert len(results) == 1
        assert results[0].entity_type == "person"
        assert results[0].text == "John Doe"
        assert results[0].start == 11
        assert results[0].end == 19
        assert results[0].score == 0.95

    def test_detect_multiple_entities(self):
        entities = [
            {"label": "person", "text": "John Doe", "start": 11, "end": 19, "score": 0.95},
            {"label": "location", "text": "New York", "start": 29, "end": 37, "score": 0.88},
        ]
        detector, _ = self._make_detector_with_mock(entities)
        results = detector.detect("My name is John Doe and I live in New York")
        assert len(results) == 2
        assert results[0].entity_type == "person"
        assert results[1].entity_type == "location"

    def test_detect_empty_text(self):
        detector, mock = self._make_detector_with_mock([])
        results = detector.detect("")
        assert results == []

    def test_detect_no_entities(self):
        detector, _ = self._make_detector_with_mock([])
        results = detector.detect("The weather is nice today")
        assert results == []

    def test_model_called_with_correct_args(self):
        detector, mock = self._make_detector_with_mock([])
        detector.detect("test text")
        mock.predict_entities.assert_called_once_with(
            "test text",
            ["person", "location", "organization", "address"],
            threshold=0.5,
        )

    def test_custom_labels_and_threshold(self):
        detector = NERDetector(
            labels=["person", "phone_number"],
            threshold=0.7,
        )
        mock_model = MagicMock()
        mock_model.predict_entities.return_value = []
        detector._model = mock_model
        detector._initialized = True

        detector.detect("test text")
        mock_model.predict_entities.assert_called_once_with(
            "test text",
            ["person", "phone_number"],
            threshold=0.7,
        )

    def test_detect_handles_exception_gracefully(self):
        detector, mock = self._make_detector_with_mock([])
        mock.predict_entities.side_effect = RuntimeError("model error")
        results = detector.detect("some text")
        assert results == []

    def test_missing_score_defaults_to_zero(self):
        entities = [{"label": "person", "text": "Jane", "start": 0, "end": 4}]
        detector, _ = self._make_detector_with_mock(entities)
        results = detector.detect("Jane")
        assert results[0].score == 0.0


class TestNERDetectorLazyInit:
    """Tests for lazy initialization behavior."""

    def test_not_initialized_on_construction(self):
        detector = NERDetector()
        assert detector._initialized is False
        assert detector._model is None

    def test_lazy_init_sets_failed_on_load_error(self):
        with patch("stateloom.pii.ner_detector._GLINER_AVAILABLE", True):
            with patch("stateloom.pii.ner_detector.GLiNER", create=True) as mock_gliner:
                mock_gliner.from_pretrained.side_effect = Exception("download failed")
                detector = NERDetector()
                result = detector._lazy_init()
                assert result is False
                assert detector._failed is True
                assert detector._initialized is True
                assert detector.is_available is False

    def test_lazy_init_succeeds(self):
        with patch("stateloom.pii.ner_detector._GLINER_AVAILABLE", True):
            with patch("stateloom.pii.ner_detector.GLiNER", create=True) as mock_gliner:
                mock_model = MagicMock()
                mock_gliner.from_pretrained.return_value = mock_model
                detector = NERDetector(model_name="test-model")
                result = detector._lazy_init()
                assert result is True
                assert detector._model is mock_model
                mock_gliner.from_pretrained.assert_called_once_with("test-model")

    def test_lazy_init_called_only_once(self):
        with patch("stateloom.pii.ner_detector._GLINER_AVAILABLE", True):
            with patch("stateloom.pii.ner_detector.GLiNER", create=True) as mock_gliner:
                mock_gliner.from_pretrained.return_value = MagicMock()
                detector = NERDetector()
                detector._lazy_init()
                detector._lazy_init()
                detector._lazy_init()
                assert mock_gliner.from_pretrained.call_count == 1

    def test_failed_init_not_retried(self):
        with patch("stateloom.pii.ner_detector._GLINER_AVAILABLE", True):
            with patch("stateloom.pii.ner_detector.GLiNER", create=True) as mock_gliner:
                mock_gliner.from_pretrained.side_effect = Exception("fail")
                detector = NERDetector()
                detector._lazy_init()
                assert detector._failed is True
                # Second call should not retry
                result = detector._lazy_init()
                assert result is False
                assert mock_gliner.from_pretrained.call_count == 1


class TestScannerNERIntegration:
    """Tests for PIIScanner + NER integration."""

    def test_scanner_with_ner_detects_person(self):
        from stateloom.pii.scanner import PIIScanner

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            NERMatch(entity_type="person", text="John Doe", start=11, end=19, score=0.95)
        ]

        scanner = PIIScanner(ner_detector=mock_detector)
        matches = scanner.scan("My name is John Doe")
        ner_matches = [m for m in matches if m.pattern_name == "ner_person"]
        assert len(ner_matches) == 1
        assert ner_matches[0].matched_text == "John Doe"

    def test_ner_deduplicates_with_regex(self):
        """NER match overlapping a regex match is skipped."""
        from stateloom.pii.scanner import PIIScanner

        mock_detector = MagicMock()
        # NER detects "123-45-6789" as an address (overlaps with SSN regex)
        mock_detector.detect.return_value = [
            NERMatch(entity_type="address", text="123-45-6789", start=5, end=16, score=0.7)
        ]

        scanner = PIIScanner(ner_detector=mock_detector)
        matches = scanner.scan("SSN: 123-45-6789")
        # Should have the regex SSN match, not the NER address match
        ssn_matches = [m for m in matches if m.pattern_name == "ssn"]
        ner_matches = [m for m in matches if m.pattern_name.startswith("ner_")]
        assert len(ssn_matches) == 1
        assert len(ner_matches) == 0

    def test_ner_adds_non_overlapping_matches(self):
        """NER matches that don't overlap with regex are added."""
        from stateloom.pii.scanner import PIIScanner

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            NERMatch(entity_type="person", text="John", start=0, end=4, score=0.9)
        ]

        scanner = PIIScanner(ner_detector=mock_detector)
        matches = scanner.scan("John emailed alice@test.com")
        person_matches = [m for m in matches if m.pattern_name == "ner_person"]
        email_matches = [m for m in matches if m.pattern_name == "email"]
        assert len(person_matches) == 1
        assert len(email_matches) == 1

    def test_scanner_without_ner_unchanged(self):
        """Scanner works exactly as before when no NER detector is provided."""
        from stateloom.pii.scanner import PIIScanner

        scanner = PIIScanner()
        matches = scanner.scan("My name is John Doe")
        # No regex matches for a plain name
        assert len(matches) == 0

    def test_ner_failure_falls_back_to_regex_only(self):
        """If NER raises, regex results are still returned."""
        from stateloom.pii.scanner import PIIScanner

        mock_detector = MagicMock()
        mock_detector.detect.side_effect = RuntimeError("NER failed")

        scanner = PIIScanner(ner_detector=mock_detector)
        matches = scanner.scan("Contact john@example.com")
        assert len(matches) == 1
        assert matches[0].pattern_name == "email"
