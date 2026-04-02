"""Tests for TaxAnomalyExtractor.

Mock-heavy to validate control flow and JSON output shaping without
invoking real LLM calls or PDF extraction.

Run:
    python -m pytest -q tests/test_tax_anomalies.py
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from financial_forecast.extraction.tax_anomaly_extractor import (
    TaxAnomalyExtractor,
    DEFAULT_SELECTION_PROMPT,
    DEFAULT_EXTRACTION_PROMPT,
)


@pytest.fixture
def sample_pages():
    """Representative page dictionary keyed by 1-based page number."""
    return {
        5: "Income taxes note text 1",
        6: "Income taxes note text 2",
    }


@pytest.fixture
def mock_client():
    return Mock()


@pytest.fixture
def extractor(mock_client):
    return TaxAnomalyExtractor(llm_client=mock_client)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_stores_config(mock_client):
    ext = TaxAnomalyExtractor(
        llm_client=mock_client,
        batch_size=50,
        parameters={"temperature": 0.5},
        max_workers=2,
    )
    assert ext.llm_client is mock_client
    assert ext.batch_size == 50
    assert ext.parameters == {"temperature": 0.5}
    assert ext.max_workers == 2


def test_constructor_default_prompts(mock_client):
    ext = TaxAnomalyExtractor(llm_client=mock_client)
    assert ext.selection_prompt == DEFAULT_SELECTION_PROMPT
    assert ext.extraction_prompt == DEFAULT_EXTRACTION_PROMPT


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
# _extract_tax_json
# ---------------------------------------------------------------------------


def test_extract_tax_json_empty_pages(extractor):
    result = extractor._extract_tax_json(page_numbers=[], pages={})
    assert result == {
        "current_tax_year": None,
        "tax_onetime_amount": None,
        "tax_onetime_note": None,
        "amount_scale": None,
    }


def test_extract_tax_json_uses_raw_response(
    extractor,
    mock_client,
    sample_pages,
):
    mock_client.ask_json.return_value = {
        "raw_response": '{"tax_onetime_amount": 1.2}',
    }
    with patch(
        "financial_forecast.extraction.base_pdf_extractor.normalize_llm_response",
        return_value={
            "current_tax_year": 2024,
            "tax_onetime_amount": 1.2,
            "tax_onetime_note": "One-time charge",
        },
    ):
        result = extractor._extract_tax_json([5, 6], sample_pages)

    assert result["tax_onetime_amount"] == 1.2
    mock_client.ask_json.assert_called_once()


def test_extract_tax_json_fallback_when_extraction_returns_none(
    extractor,
    mock_client,
    sample_pages,
):
    mock_client.ask_json.return_value = {
        "raw_response": "some text",
        "tax_onetime_amount": 2.5,
        "tax_onetime_note": "Note",
    }
    with patch(
        "financial_forecast.extraction.base_pdf_extractor.normalize_llm_response",
        return_value={
            "raw_response": "some text",
            "tax_onetime_amount": 2.5,
            "tax_onetime_note": "Note",
        },
    ):
        result = extractor._extract_tax_json([5, 6], sample_pages)

    assert result["tax_onetime_amount"] == 2.5


def test_extract_tax_json_partial_keys(
    extractor,
    mock_client,
    sample_pages,
):
    mock_client.ask_json.return_value = {
        "tax_onetime_amount": 1.0,
    }
    result = extractor._extract_tax_json([5, 6], sample_pages)
    assert result["tax_onetime_amount"] == 1.0


# ---------------------------------------------------------------------------
# Full pipeline (_extract_one_pdf)
# ---------------------------------------------------------------------------


def test_extract_one_pdf_writes_json(
    tmp_path,
    mock_client,
    sample_pages,
):
    input_pdf = tmp_path / "apple_2024.pdf"
    input_pdf.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "extracted_text"

    ext = TaxAnomalyExtractor(llm_client=mock_client)

    with (
        patch(
            "financial_forecast.extraction.base_pdf_extractor."
            "extract_text_pdfplumber",
            return_value=sample_pages,
        ),
        patch(
            "financial_forecast.extraction.base_pdf_extractor." "select_pages_with_llm",
            return_value=[5, 6],
        ),
    ):
        mock_client.ask_json.return_value = {
            "tax_onetime_amount": 1.1,
            "tax_onetime_note": "Discrete charge",
        }
        ext._extract_one_pdf(input_pdf, output_dir)

    out_file = output_dir / "apple_2024.tax-anomalies.llm.json"
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["selected_pages"] == [5, 6]
    assert payload["extraction"]["tax_onetime_amount"] == 1.1
