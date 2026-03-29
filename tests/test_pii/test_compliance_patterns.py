"""Tests for EU/HIPAA/CCPA PII patterns."""

from __future__ import annotations

import re

from stateloom.pii.patterns import (
    PATTERN_GROUPS,
    PII_PATTERNS,
    resolve_pattern_names,
)


class TestVATPattern:
    def test_german_vat(self):
        assert PII_PATTERNS["vat_id"].regex.search("DE123456789")

    def test_french_vat(self):
        assert PII_PATTERNS["vat_id"].regex.search("FR12345678901")

    def test_invalid_vat(self):
        assert not PII_PATTERNS["vat_id"].regex.search("123456789")


class TestIBANPattern:
    def test_german_iban(self):
        assert PII_PATTERNS["iban"].regex.search("DE89370400440532013000")

    def test_british_iban(self):
        assert PII_PATTERNS["iban"].regex.search("GB29NWBK60161331926819")

    def test_invalid_iban(self):
        assert not PII_PATTERNS["iban"].regex.search("1234567890")


class TestNationalIDEU:
    def test_valid_format(self):
        assert PII_PATTERNS["national_id_eu"].regex.search("12.34.56-789.01-23")

    def test_invalid_format(self):
        assert not PII_PATTERNS["national_id_eu"].regex.search("123456789")


class TestPhoneEU:
    def test_german_phone(self):
        assert PII_PATTERNS["phone_eu"].regex.search("+49301234567")

    def test_french_phone(self):
        assert PII_PATTERNS["phone_eu"].regex.search("+33123456789")

    def test_invalid_no_plus(self):
        assert not PII_PATTERNS["phone_eu"].regex.search("49301234567")


class TestMedicalRecordNumber:
    def test_valid_mrn(self):
        assert PII_PATTERNS["medical_record_number"].regex.search("MRN: 123456")

    def test_mrn_longer(self):
        assert PII_PATTERNS["medical_record_number"].regex.search("MRN: 1234567890")

    def test_invalid_mrn(self):
        assert not PII_PATTERNS["medical_record_number"].regex.search("MRN: 12")


class TestHealthPlanID:
    def test_valid_hpi(self):
        assert PII_PATTERNS["health_plan_id"].regex.search("HPI: 1234567890")

    def test_invalid_hpi(self):
        assert not PII_PATTERNS["health_plan_id"].regex.search("HPI: 123")


class TestNPI:
    def test_valid_npi(self):
        assert PII_PATTERNS["npi"].regex.search("NPI: 1234567890")

    def test_invalid_npi_short(self):
        assert not PII_PATTERNS["npi"].regex.search("NPI: 123456")


class TestDateOfBirth:
    def test_valid_dob(self):
        assert PII_PATTERNS["date_of_birth"].regex.search("DOB: 01/15/1990")

    def test_invalid_dob(self):
        assert not PII_PATTERNS["date_of_birth"].regex.search("DOB: 1990")


class TestCaliforniaDL:
    def test_valid_ca_dl(self):
        assert PII_PATTERNS["california_dl"].regex.search("A1234567")

    def test_invalid_ca_dl(self):
        assert not PII_PATTERNS["california_dl"].regex.search("12345678")


class TestPatternGroups:
    def test_phone_group_includes_eu(self):
        assert "phone_eu" in PATTERN_GROUPS["phone"]
        assert "phone_us" in PATTERN_GROUPS["phone"]

    def test_eu_pii_group(self):
        assert "vat_id" in PATTERN_GROUPS["eu_pii"]
        assert "iban" in PATTERN_GROUPS["eu_pii"]
        assert "national_id_eu" in PATTERN_GROUPS["eu_pii"]

    def test_phi_group(self):
        assert "medical_record_number" in PATTERN_GROUPS["phi"]
        assert "health_plan_id" in PATTERN_GROUPS["phi"]
        assert "npi" in PATTERN_GROUPS["phi"]
        assert "date_of_birth" in PATTERN_GROUPS["phi"]

    def test_resolve_phi_group(self):
        resolved = resolve_pattern_names(["phi"])
        assert "medical_record_number" in resolved
        assert "npi" in resolved
