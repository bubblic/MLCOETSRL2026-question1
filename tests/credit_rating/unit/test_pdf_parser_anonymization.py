"""Tests for anonymization integration in PdfAnnualReportParser."""

from __future__ import annotations

from typing import Dict, Optional
from unittest.mock import patch

import pytest

from credit_rating.config.settings import CreditRatingSettings
from credit_rating.ingestion.pdf_parser import PdfAnnualReportParser


_SAMPLE_PAGES: Dict[int, str] = {
    0: (
        "Item 1A. Risk Factors\n"
        "Evergrande Group faces significant liquidity risks. "
        "In 2022, the company reported substantial losses. "
        "CEO Hui Ka Yan has been under investigation."
    ),
    1: (
        "Item 7. Management's Discussion and Analysis\n"
        "Evergrande's revenue declined sharply in 2022 compared to 2021. "
        "The Board of Directors met in Shenzhen to discuss restructuring."
    ),
}


@pytest.fixture
def parser_with_entities() -> PdfAnnualReportParser:
    """Parser with known Evergrande entities, NER disabled."""
    return PdfAnnualReportParser(
        settings=CreditRatingSettings(
            anonymize_text_for_llm=True,
            anonymize_use_ner=False,
            anonymize_years=True,
        ),
        known_entities=[
            {
                "type": "ORG",
                "names": ["Evergrande Group", "Evergrande"],
            },
            {
                "type": "PERSON",
                "names": ["Hui Ka Yan"],
            },
        ],
    )


class TestAnonymizationIntegration:
    """Verify EntityAnonymizer is wired into PdfAnnualReportParser."""

    @patch(
        "credit_rating.ingestion.pdf_parser._extract_pages",
        return_value=_SAMPLE_PAGES,
    )
    def test_company_name_replaced(self, mock_pages, parser_with_entities):
        """Company names should be replaced with [ORG_N] placeholders."""
        report = parser_with_entities.parse("/fake/evergrande.pdf")
        assert "Evergrande" not in report.mda_text
        assert "[ORG_" in report.mda_text

    @patch(
        "credit_rating.ingestion.pdf_parser._extract_pages",
        return_value=_SAMPLE_PAGES,
    )
    def test_person_name_replaced(self, mock_pages, parser_with_entities):
        """Person names should be replaced with [PERSON_N] placeholders."""
        report = parser_with_entities.parse("/fake/evergrande.pdf")
        assert "Hui Ka Yan" not in report.risk_factors_text
        assert "[PERSON_" in report.risk_factors_text

    @patch(
        "credit_rating.ingestion.pdf_parser._extract_pages",
        return_value=_SAMPLE_PAGES,
    )
    def test_years_replaced(self, mock_pages, parser_with_entities):
        """Absolute years should be replaced with FY_T markers."""
        report = parser_with_entities.parse("/fake/evergrande.pdf")
        assert "2022" not in report.mda_text
        assert "FY_T" in report.mda_text

    @patch(
        "credit_rating.ingestion.pdf_parser._extract_pages",
        return_value=_SAMPLE_PAGES,
    )
    def test_section_finding_works_on_anonymized_text(
        self, mock_pages, parser_with_entities,
    ):
        """Section headers should still be found after anonymization."""
        report = parser_with_entities.parse("/fake/evergrande.pdf")
        assert report.mda_text != ""
        assert report.risk_factors_text != ""

    @patch(
        "credit_rating.ingestion.pdf_parser._extract_pages",
        return_value=_SAMPLE_PAGES,
    )
    def test_anonymizer_stored_for_deanonymization(
        self, mock_pages, parser_with_entities,
    ):
        """The anonymizer instance should be accessible after parse()."""
        parser_with_entities.parse("/fake/evergrande.pdf")
        anon = parser_with_entities.last_anonymizer
        assert anon is not None
        assert len(anon.entity_map()) > 0

    @patch(
        "credit_rating.ingestion.pdf_parser._extract_pages",
        return_value=_SAMPLE_PAGES,
    )
    def test_deanonymize_restores_original(
        self, mock_pages, parser_with_entities,
    ):
        """De-anonymization should restore the original company name."""
        report = parser_with_entities.parse("/fake/evergrande.pdf")
        anon = parser_with_entities.last_anonymizer
        restored = anon.deanonymize(report.mda_text)
        assert "Evergrande" in restored

    @patch(
        "credit_rating.ingestion.pdf_parser._extract_pages",
        return_value=_SAMPLE_PAGES,
    )
    def test_anonymization_disabled(self, mock_pages):
        """When disabled, text should pass through unchanged."""
        parser = PdfAnnualReportParser(
            settings=CreditRatingSettings(anonymize_text_for_llm=False),
        )
        report = parser.parse("/fake/evergrande.pdf")
        assert "Evergrande" in report.mda_text
        assert parser.last_anonymizer is None

    @patch(
        "credit_rating.ingestion.pdf_parser._extract_pages",
        return_value=_SAMPLE_PAGES,
    )
    def test_ticker_used_as_fallback_known_entity(self, mock_pages):
        """When no known_entities given, ticker is used as minimal ORG."""
        parser = PdfAnnualReportParser(
            settings=CreditRatingSettings(
                anonymize_text_for_llm=True,
                anonymize_use_ner=False,
                anonymize_years=False,
            ),
        )
        report = parser.parse("/fake/EVERGRANDE.pdf")
        assert parser.last_anonymizer is not None
        entity_map = parser.last_anonymizer.entity_map()
        assert any("EVERGRANDE" in v for v in entity_map.values())
