"""Tests for llm_extract_tax_anomalies_from_pdf.

These tests are intentionally mock-heavy and file-light to validate logic and
control flow without invoking real LLM calls or PDF extraction engines.


Run the script by:
python -m pytest -q test_llm_extract_tax_anomalies_from_pdf.py
"""

import argparse
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import llm_extract_tax_anomalies_from_pdf as module


@pytest.fixture
def sample_pages():
    """Representative page dictionary keyed by 1-based page number."""
    return {
        5: "Income taxes note text",
        6: "Contingencies note text",
    }


@pytest.fixture
def valid_args(tmp_path):
    """A reusable argparse namespace with valid values."""
    input_dir = tmp_path / "annual_reports"
    input_dir.mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(
        output_dir=str(tmp_path / "extracted_text"),
        input_dir=str(input_dir),
        input_file=None,
        query="Income Taxes and Contingencies",
        batch_size=100,
        endpoint="https://example.endpoint",
        parameters='{"temperature": 0, "top_k": 1}',
        selection_prompt=module.DEFAULT_SELECTION_PROMPT,
        extraction_prompt=module.DEFAULT_EXTRACTION_PROMPT,
    )


def test_parse_parameters_none_returns_empty_dict():
    assert module.parse_parameters(None) == {}


def test_parse_parameters_valid_object_json():
    parsed = module.parse_parameters('{"temperature": 0, "top_k": 1}')
    assert parsed == {"temperature": 0, "top_k": 1}


def test_parse_parameters_non_object_raises():
    with pytest.raises(ValueError, match="JSON object"):
        module.parse_parameters('["invalid"]')


def test_resolve_endpoint_prefers_override():
    assert (
        module.resolve_endpoint(" https://override.example ")
        == "https://override.example"
    )


def test_resolve_endpoint_reads_env(monkeypatch):
    monkeypatch.setenv("AZURE_DEEPSEEK_ENDPOINT", "https://env.example")
    assert module.resolve_endpoint(None) == "https://env.example"


def test_resolve_endpoint_missing_raises(monkeypatch):
    monkeypatch.delenv("AZURE_DEEPSEEK_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="Missing endpoint"):
        module.resolve_endpoint(None)


def test_extract_tax_json_with_llm_empty_pages_returns_null_payload():
    result = module.extract_tax_json_with_llm(
        client=Mock(),
        parameters={},
        page_numbers=[],
        pages={},
        prompt_template="Pages:\n{pages}",
    )
    assert result == {
        "tax_onetime_amount": None,
        "tax_onetime_note": None,
        "tax_contingency_amount": None,
        "tax_contingency_note": None,
    }


def test_extract_tax_json_with_llm_uses_raw_response_extraction(
    sample_pages, monkeypatch
):
    mock_client = Mock()
    mock_client.ask_json.return_value = {"raw_response": '{"tax_onetime_amount": 1.2}'}
    monkeypatch.setattr(
        module,
        "extract_json_from_text",
        Mock(
            return_value={
                "tax_onetime_amount": 1.2,
                "tax_onetime_note": "One-time tax item",
                "tax_contingency_amount": 0.4,
                "tax_contingency_note": "Potential future exposure",
            }
        ),
    )

    result = module.extract_tax_json_with_llm(
        client=mock_client,
        parameters={"temperature": 0},
        page_numbers=[5, 6],
        pages=sample_pages,
        prompt_template="Tax extraction.\nPages:\n{pages}",
    )

    assert result["tax_onetime_amount"] == 1.2
    assert result["tax_contingency_amount"] == 0.4
    mock_client.ask_json.assert_called_once()


def test_run_pipeline_writes_expected_output_file(tmp_path, sample_pages, monkeypatch):
    input_pdf = tmp_path / "apple_2024.pdf"
    input_pdf.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "extracted_text"

    mock_client = Mock()
    monkeypatch.setitem(
        module.EXTRACTORS, "pdfplumber", Mock(return_value=sample_pages)
    )
    monkeypatch.setattr(module, "AzureLLMClient", Mock(return_value=mock_client))
    monkeypatch.setattr(module, "select_pages_with_llm", Mock(return_value=[5, 6]))
    monkeypatch.setattr(
        module,
        "extract_tax_json_with_llm",
        Mock(
            return_value={
                "tax_onetime_amount": 1.1,
                "tax_onetime_note": "Discrete charge",
                "tax_contingency_amount": 0.3,
                "tax_contingency_note": "Uncertain tax position",
            }
        ),
    )

    module.run_pipeline(
        input_file=input_pdf,
        output_dir=output_dir,
        query="Income taxes and contingencies",
        batch_size=100,
        endpoint="https://example.endpoint",
        parameters={"temperature": 0},
        selection_prompt=module.DEFAULT_SELECTION_PROMPT,
        extraction_prompt=module.DEFAULT_EXTRACTION_PROMPT,
    )

    out_file = output_dir / "apple_2024.tax-anomalies-contingencies.llm.json"
    assert out_file.exists()

    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload["selected_pages"] == [5, 6]
    assert payload["extraction"]["tax_onetime_amount"] == 1.1
    assert payload["query"] == "Income taxes and contingencies"


def test_main_with_single_input_file_invokes_run_pipeline(
    monkeypatch, tmp_path, valid_args
):
    input_pdf = tmp_path / "single_report.pdf"
    input_pdf.write_bytes(b"%PDF-1.4")
    valid_args.input_file = str(input_pdf)

    monkeypatch.setattr(module, "parse_args", Mock(return_value=valid_args))
    monkeypatch.setattr(
        module, "parse_parameters", Mock(return_value={"temperature": 0})
    )
    monkeypatch.setattr(
        module, "resolve_endpoint", Mock(return_value="https://endpoint")
    )
    mocked_run_pipeline = Mock()
    monkeypatch.setattr(module, "run_pipeline", mocked_run_pipeline)

    module.main()

    mocked_run_pipeline.assert_called_once()
    kwargs = mocked_run_pipeline.call_args.kwargs
    assert kwargs["input_file"] == input_pdf
    assert kwargs["query"] == valid_args.query


def test_main_raises_when_no_pdfs_found(monkeypatch, valid_args):
    valid_args.input_file = None
    monkeypatch.setattr(module, "parse_args", Mock(return_value=valid_args))
    monkeypatch.setattr(module, "parse_parameters", Mock(return_value={}))
    monkeypatch.setattr(
        module, "resolve_endpoint", Mock(return_value="https://endpoint")
    )
    monkeypatch.setattr(Path, "glob", Mock(return_value=[]))

    with pytest.raises(FileNotFoundError, match="No PDF files found"):
        module.main()
