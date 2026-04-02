"""Tests for FinancialStatementExtractor.

Mock-heavy to keep tests fast and deterministic while validating
control flow and JSON output shaping.

Run:
    python -m pytest -q tests/test_extraction.py
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from financial_forecast.extraction.financial_statement_extractor import (
    FinancialStatementExtractor,
    SUPPLEMENTARY_EXTRACTION_MAX_PAGES,
)
from financial_forecast.extraction.statement_config import StatementType


@pytest.fixture
def sample_pages():
    """Representative extracted PDF pages keyed by page number."""
    return {
        1: "Balance sheet header\nCash and cash equivalents ...",
        2: "Continuation of primary statement",
        10: "Supplementary note A",
        11: "Supplementary note B",
    }


@pytest.fixture
def sample_primary_extraction():
    """Representative primary extraction payload."""
    return {
        "table_name": "Consolidated Balance Sheet",
        "rows": [{"line_item": "Cash", "value": "100"}],
    }


@pytest.fixture
def mock_client():
    """Reusable LLM client test double."""
    return Mock()


@pytest.fixture
def extractor(mock_client):
    """Extractor with default config and mock client."""
    return FinancialStatementExtractor(
        queries=[StatementType.BALANCE_SHEET],
        llm_client=mock_client,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_stores_config(mock_client):
    ext = FinancialStatementExtractor(
        queries=[StatementType.BALANCE_SHEET, StatementType.INCOME_STATEMENT],
        llm_client=mock_client,
        batch_size=50,
        parameters={"temperature": 0.5},
        max_workers=2,
    )
    assert ext.queries == [StatementType.BALANCE_SHEET, StatementType.INCOME_STATEMENT]
    assert ext.batch_size == 50
    assert ext.parameters == {"temperature": 0.5}
    assert ext.max_workers == 2


def test_constructor_default_parameters(mock_client):
    ext = FinancialStatementExtractor(queries=[], llm_client=mock_client)
    assert ext.parameters == {"temperature": 0, "top_k": 1}


# ---------------------------------------------------------------------------
# run() validation
# ---------------------------------------------------------------------------


def test_run_raises_on_missing_path(extractor):
    with pytest.raises(FileNotFoundError):
        extractor.run("does_not_exist_at_all")


def test_run_raises_on_empty_dir(extractor, tmp_path):
    with pytest.raises(FileNotFoundError, match="No PDF files found"):
        extractor.run(str(tmp_path))


# ---------------------------------------------------------------------------
# _extract_table
# ---------------------------------------------------------------------------


def test_extract_table_requires_pages(extractor, sample_pages):
    with pytest.raises(ValueError, match="No pages selected"):
        extractor._extract_table(
            query_label="Consolidated Balance Sheet",
            page_numbers=[],
            pages=sample_pages,
        )


def test_extract_table_uses_raw_response(extractor, mock_client, sample_pages):
    mock_client.ask_json.return_value = {"raw_response": '{"parsed": true}'}
    with patch(
        "financial_forecast.extraction.base_pdf_extractor.normalize_llm_response",
        return_value={"parsed": True},
    ):
        result = extractor._extract_table(
            query_label="Consolidated Balance Sheet",
            page_numbers=[1, 2],
            pages=sample_pages,
        )
    assert result == {"parsed": True}
    mock_client.ask_json.assert_called_once()


# ---------------------------------------------------------------------------
# _extract_supplementary
# ---------------------------------------------------------------------------


def test_extract_supplementary_no_pages(extractor, mock_client):
    result = extractor._extract_supplementary(
        statement_type=StatementType.BALANCE_SHEET,
        primary_extraction={},
        page_numbers=[],
        pages={},
    )
    assert result == {}
    mock_client.ask_json.assert_not_called()


def test_extract_supplementary_multiple_chunks(
    mock_client,
    sample_pages,
    sample_primary_extraction,
    monkeypatch,
):
    monkeypatch.setattr(
        "financial_forecast.extraction.financial_statement_extractor."
        "SUPPLEMENTARY_EXTRACTION_MAX_PAGES",
        2,
    )
    mock_client.ask_json.side_effect = [
        {"raw_response": "Chunk 1 data"},
        {"raw_response": "Chunk 2 data"},
    ]
    ext = FinancialStatementExtractor(
        queries=[],
        llm_client=mock_client,
    )
    result = ext._extract_supplementary(
        statement_type=StatementType.BALANCE_SHEET,
        primary_extraction=sample_primary_extraction,
        page_numbers=[1, 2, 10],
        pages=sample_pages,
    )
    assert result == {"chunk_1": "Chunk 1 data", "chunk_2": "Chunk 2 data"}
    assert mock_client.ask_json.call_count == 2


def test_extract_supplementary_single_chunk(
    extractor,
    mock_client,
    sample_pages,
    sample_primary_extraction,
):
    mock_client.ask_json.return_value = {"raw_response": "Single chunk"}
    result = extractor._extract_supplementary(
        statement_type=StatementType.BALANCE_SHEET,
        primary_extraction=sample_primary_extraction,
        page_numbers=[1, 2],
        pages=sample_pages,
    )
    assert result == {"chunk_1": "Single chunk"}


# ---------------------------------------------------------------------------
# Full pipeline (_extract_one_pdf)
# ---------------------------------------------------------------------------


def test_extract_one_pdf_writes_json(
    tmp_path,
    mock_client,
    sample_pages,
    sample_primary_extraction,
):
    input_pdf = tmp_path / "company_report.pdf"
    input_pdf.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "out"

    ext = FinancialStatementExtractor(
        queries=[StatementType.BALANCE_SHEET],
        llm_client=mock_client,
    )

    with (
        patch(
            "financial_forecast.extraction.base_pdf_extractor."
            "extract_text_pdfplumber",
            return_value=sample_pages,
        ),
        patch(
            "financial_forecast.extraction.base_pdf_extractor."
            "select_pages_with_llm",
            side_effect=[[1, 2], [10, 11]],
        ),
    ):
        mock_client.ask_json.side_effect = [
            sample_primary_extraction,
            {"raw_response": "Supplementary data here"},
        ]
        ext._extract_one_pdf(input_pdf, output_dir)

    output_file = output_dir / "company_report.consolidated-balance-sheet.llm.json"
    assert output_file.exists()
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["query"] == "Consolidated Balance Sheet"
    assert payload["statement_type"] == "consolidated-balance-sheet"
    assert payload["selected_pages"]["primary_statement_pages"] == [1, 2]
