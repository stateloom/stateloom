"""Tests for the curated model catalog."""

from __future__ import annotations

from stateloom.local.models import MODEL_CATALOG


class TestModelCatalog:
    def test_catalog_not_empty(self):
        assert len(MODEL_CATALOG) > 0

    def test_all_entries_have_required_fields(self):
        required = {"model", "size_gb", "description", "tier", "parameters"}
        for entry in MODEL_CATALOG:
            assert required.issubset(entry.keys()), f"Missing keys in {entry['model']}"

    def test_valid_tiers(self):
        valid = {"ultra-light", "light", "medium", "heavy"}
        for entry in MODEL_CATALOG:
            assert entry["tier"] in valid, f"Invalid tier: {entry['tier']}"

    def test_sizes_positive(self):
        for entry in MODEL_CATALOG:
            assert entry["size_gb"] > 0
