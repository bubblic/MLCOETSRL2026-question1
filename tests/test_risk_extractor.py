"""Integration tests for the full risk extraction pipeline."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from financial_forecast.extraction.risk.risk_categories import RiskCategory
from financial_forecast.extraction.risk.risk_extractor import RiskWarningsExtractor


MOCK_PAGES = {i: f"Text of page {i}" for i in range(1, 31)}

MOCK_FLAG_RESPONSE = [
    {"page": 5, "category": "going_concern", "reason": "doubt"},
    {"page": 10, "category": "auditor_opinion", "reason": "qualified"},
    {"page": 15, "category": "debt_covenants", "reason": "breach"},
]

MOCK_EXTRACTION_RESPONSE = {
    "opinion_type": "qualified",
    "qualification_basis": "going concern",
    "emphasis_of_matter": None,
    "going_concern_language": "material uncertainty",
}

MOCK_SYNTHESIS_TEXT = (
    "RISK RATING: HIGH\n\n"
    "Top risks:\n"
    "1. Going concern\n"
    "2. Qualified opinion\n"
    "3. Covenant breach"
)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client that returns stage-appropriate responses."""
    client = Mock()

    call_count = {"n": 0}

    def ask_json_side_effect(**kwargs):
        call_count["n"] += 1
        prompt = kwargs.get("prompt", "")
        # Stage 1: flagging prompts contain "Categories to detect"
        if "Categories to detect" in prompt:
            return MOCK_FLAG_RESPONSE
        # Stage 2: extraction prompts contain "CONTENT:"
        return MOCK_EXTRACTION_RESPONSE

    client.ask_json.side_effect = ask_json_side_effect
    client.ask_text.return_value = MOCK_SYNTHESIS_TEXT
    return client


class TestExtractOnePdfFullPipeline:
    @patch(
        "financial_forecast.extraction.risk.risk_extractor.BasePdfExtractor._extract_pages",
        return_value=MOCK_PAGES,
    )
    def test_full_pipeline_writes_output(self, mock_extract, mock_llm_client, tmp_path):
        extractor = RiskWarningsExtractor(
            llm_client=mock_llm_client,
            chunk_size=30,
            category_workers=1,
        )

        pdf_path = tmp_path / "test_report.pdf"
        pdf_path.touch()
        output_dir = tmp_path / "output"

        extractor._extract_one_pdf(pdf_path, output_dir)

        output_file = output_dir / "test_report.risk-warnings.llm.json"
        assert output_file.exists()

        with output_file.open() as f:
            result = json.load(f)

        assert "flagged_pages" in result
        assert "category_results" in result
        assert "risk_memo" in result
        assert "usage" in result
        assert result["risk_memo"]["risk_memo"] == MOCK_SYNTHESIS_TEXT

    @patch(
        "financial_forecast.extraction.risk.risk_extractor.BasePdfExtractor._extract_pages",
        return_value=MOCK_PAGES,
    )
    def test_usage_tracker_records_stages(self, mock_extract, mock_llm_client, tmp_path):
        extractor = RiskWarningsExtractor(
            llm_client=mock_llm_client,
            chunk_size=30,
            category_workers=1,
        )

        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        output_dir = tmp_path / "output"

        extractor._extract_one_pdf(pdf_path, output_dir)

        output_file = output_dir / "test.risk-warnings.llm.json"
        with output_file.open() as f:
            result = json.load(f)

        usage = result["usage"]
        assert usage["total_calls"] > 0
        assert "flagging" in usage["stages"]
        assert "synthesis" in usage["stages"]


class TestExtractOnePdfNoFlags:
    @patch(
        "financial_forecast.extraction.risk.risk_extractor.BasePdfExtractor._extract_pages",
        return_value=MOCK_PAGES,
    )
    def test_no_flagged_pages(self, mock_extract, tmp_path):
        client = Mock()
        client.ask_json.return_value = []  # No flags
        client.ask_text.return_value = "No risks identified."

        extractor = RiskWarningsExtractor(
            llm_client=client,
            chunk_size=30,
            category_workers=1,
        )

        pdf_path = tmp_path / "clean_report.pdf"
        pdf_path.touch()
        output_dir = tmp_path / "output"

        extractor._extract_one_pdf(pdf_path, output_dir)

        output_file = output_dir / "clean_report.risk-warnings.llm.json"
        with output_file.open() as f:
            result = json.load(f)

        assert result["flagged_pages"] == {}
        assert result["category_results"] == {}
        assert result["risk_memo"]["risk_memo"] == "No risks identified."


class TestOutputFileNaming:
    @patch(
        "financial_forecast.extraction.risk.risk_extractor.BasePdfExtractor._extract_pages",
        return_value=MOCK_PAGES,
    )
    def test_output_follows_convention(self, mock_extract, mock_llm_client, tmp_path):
        extractor = RiskWarningsExtractor(
            llm_client=mock_llm_client,
            chunk_size=30,
            category_workers=1,
        )

        pdf_path = tmp_path / "ar2022.pdf"
        pdf_path.touch()
        output_dir = tmp_path / "output"

        extractor._extract_one_pdf(pdf_path, output_dir)

        expected = output_dir / "ar2022.risk-warnings.llm.json"
        assert expected.exists()


class TestCustomCategories:
    @patch(
        "financial_forecast.extraction.risk.risk_extractor.BasePdfExtractor._extract_pages",
        return_value=MOCK_PAGES,
    )
    def test_subset_of_categories(self, mock_extract, tmp_path):
        client = Mock()
        # Only return flags for going_concern
        client.ask_json.side_effect = lambda **kwargs: (
            [{"page": 5, "category": "going_concern", "reason": "test"}]
            if "Categories to detect" in kwargs.get("prompt", "")
            else {"disclosure_text": "test"}
        )
        client.ask_text.return_value = "Memo."

        subset = [RiskCategory.GOING_CONCERN, RiskCategory.DEBT_COVENANTS]
        extractor = RiskWarningsExtractor(
            llm_client=client,
            categories=subset,
            chunk_size=30,
            category_workers=1,
        )

        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        output_dir = tmp_path / "output"

        extractor._extract_one_pdf(pdf_path, output_dir)

        output_file = output_dir / "test.risk-warnings.llm.json"
        with output_file.open() as f:
            result = json.load(f)

        # Only going_concern should be in results (debt_covenants wasn't flagged)
        assert "going_concern" in result["category_results"]


class TestRunSingleFile:
    @patch(
        "financial_forecast.extraction.risk.risk_extractor.BasePdfExtractor._extract_pages",
        return_value=MOCK_PAGES,
    )
    def test_single_pdf_path(self, mock_extract, mock_llm_client, tmp_path):
        extractor = RiskWarningsExtractor(
            llm_client=mock_llm_client,
            chunk_size=30,
            category_workers=1,
        )

        pdf_path = tmp_path / "single.pdf"
        pdf_path.touch()
        output_dir = tmp_path / "output"

        # Pass file path directly (not directory)
        extractor.run(
            input_path=str(pdf_path),
            output_dir=str(output_dir),
        )

        expected = output_dir / "single.risk-warnings.llm.json"
        assert expected.exists()
