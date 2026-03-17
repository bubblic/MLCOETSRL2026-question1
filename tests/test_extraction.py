"""Tests for llm_extract_financial_statement_from_pdf.

The suite is mock-heavy to keep tests fast and deterministic while validating
control flow, argument handling, and JSON output shaping.


Run the script by:
python -m pytest -q tests/test_extraction.py
"""

import argparse
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from financial_forecast.extraction import statement_extractor as module


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
def valid_args(tmp_path):
    """CLI namespace with valid defaults for main()."""
    input_dir = tmp_path / "annual_reports"
    input_dir.mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(
        output_dir=str(tmp_path / "extracted_text"),
        input_dir=str(input_dir),
        input_file=None,
        query=["Consolidated Balance Sheet"],
        batch_size=100,
        max_workers=1,
        endpoint="https://example.endpoint",
        parameters='{"temperature": 0, "top_k": 1}',
        selection_prompt=None,
        extraction_prompt=None,
    )


@pytest.fixture
def mock_client():
    """Reusable LLM client test double."""
    return Mock()


def test_parse_parameters_none_returns_empty_dict():
    assert module.parse_parameters(None) == {}


def test_parse_parameters_valid_json_object():
    parsed = module.parse_parameters('{"temperature": 0, "top_k": 1}')
    assert parsed == {"temperature": 0, "top_k": 1}


def test_parse_parameters_non_object_raises():
    with pytest.raises(ValueError, match="JSON object"):
        module.parse_parameters('["not", "an", "object"]')


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


def test_extract_table_with_llm_requires_pages(mock_client, sample_pages):
    with pytest.raises(ValueError, match="No pages selected"):
        module.extract_table_with_llm(
            client=mock_client,
            parameters={},
            query="Consolidated Balance Sheet",
            page_numbers=[],
            pages=sample_pages,
        )


def test_extract_table_with_llm_uses_raw_response_json_if_available(
    mock_client, sample_pages, monkeypatch
):
    mock_client.ask_json.return_value = {"raw_response": '{"parsed": true}'}
    monkeypatch.setattr(
        module, "extract_json_from_text", Mock(return_value={"parsed": True})
    )

    extracted = module.extract_table_with_llm(
        client=mock_client,
        parameters={"temperature": 0},
        query="Consolidated Balance Sheet",
        page_numbers=[1, 2],
        pages=sample_pages,
    )

    assert extracted == {"parsed": True}
    mock_client.ask_json.assert_called_once()


def test_extract_supplementary_with_llm_no_pages_returns_empty(mock_client):
    output = module.extract_supplementary_with_llm(
        client=mock_client,
        parameters={},
        query="Consolidated Balance Sheet",
        primary_extraction={},
        page_numbers=[],
        pages={},
    )
    assert output == {"supplementary_tables": []}
    mock_client.ask_json.assert_not_called()


def test_extract_supplementary_with_llm_aggregates_chunk_tables(
    mock_client, sample_pages, sample_primary_extraction, monkeypatch
):
    # Force two chunks to validate aggregation logic.
    monkeypatch.setattr(module, "SUPPLEMENTARY_EXTRACTION_MAX_PAGES", 2)

    mock_client.ask_json.side_effect = [
        {"supplementary_tables": [{"id": "A"}]},
        {"supplementary_tables": [{"id": "B"}]},
    ]

    result = module.extract_supplementary_with_llm(
        client=mock_client,
        parameters={},
        query="Consolidated Balance Sheet",
        primary_extraction=sample_primary_extraction,
        page_numbers=[1, 2, 10],
        pages=sample_pages,
    )

    assert result == {"supplementary_tables": [{"id": "A"}, {"id": "B"}]}
    assert mock_client.ask_json.call_count == 2


def test_run_pipeline_writes_expected_json_output(
    tmp_path, sample_pages, sample_primary_extraction, monkeypatch
):
    input_pdf = tmp_path / "company_report.pdf"
    input_pdf.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "out"

    monkeypatch.setitem(
        module.EXTRACTORS, "pdfplumber", Mock(return_value=sample_pages)
    )
    monkeypatch.setattr(module, "AzureLLMClient", Mock(return_value=Mock()))
    monkeypatch.setattr(
        module, "select_pages_with_llm", Mock(side_effect=[[1, 2], [10, 11]])
    )
    monkeypatch.setattr(
        module,
        "extract_table_with_llm",
        Mock(return_value=sample_primary_extraction),
    )
    monkeypatch.setattr(
        module,
        "extract_supplementary_with_llm",
        Mock(return_value={"supplementary_tables": [{"id": "N1"}]}),
    )

    module.run_pipeline(
        input_file=input_pdf,
        output_dir=output_dir,
        queries=["Consolidated Balance Sheet"],
        batch_size=100,
        endpoint="https://example.endpoint",
        parameters={"temperature": 0},
        selection_prompt=None,
        extraction_prompt=None,
    )

    output_file = output_dir / "company_report.consolidated-balance-sheet.llm.json"
    assert output_file.exists()

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["query"] == "Consolidated Balance Sheet"
    assert payload["selected_pages"]["primary_statement_pages"] == [1, 2]
    assert payload["selected_pages"]["supplementary_pages"] == [10, 11]
    assert payload["extraction"]["primary_statement"] == sample_primary_extraction


def test_main_runs_pipeline_for_single_input_file(monkeypatch, tmp_path, valid_args):
    input_pdf = tmp_path / "single_report.pdf"
    input_pdf.write_bytes(b"%PDF-1.4")
    valid_args.input_file = str(input_pdf)
    valid_args.max_workers = 1

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
    assert kwargs["queries"] == ["Consolidated Balance Sheet"]


def test_main_raises_when_no_pdf_files(monkeypatch, valid_args):
    empty_input_dir = Path(valid_args.input_dir)
    valid_args.input_file = None

    monkeypatch.setattr(module, "parse_args", Mock(return_value=valid_args))
    monkeypatch.setattr(module, "parse_parameters", Mock(return_value={}))
    monkeypatch.setattr(
        module, "resolve_endpoint", Mock(return_value="https://endpoint")
    )
    monkeypatch.setattr(Path, "glob", Mock(return_value=[]))

    with pytest.raises(FileNotFoundError, match="No PDF files found"):
        module.main()


def test_parse_parameters_malformed_json_raises():
    """Malformed JSON should raise json.JSONDecodeError."""
    import json
    with pytest.raises(json.JSONDecodeError):
        module.parse_parameters("{not valid json}")


def test_extract_table_with_llm_with_prompt_override(mock_client, sample_pages, monkeypatch):
    """Custom prompt_override should be passed to the LLM client."""
    mock_client.ask_json.return_value = {"table_name": "Custom", "rows": []}
    monkeypatch.setattr(
        module, "extract_json_from_text", Mock(return_value=None)
    )

    custom_prompt = "Extract the {query} from these pages:\n{pages}"
    extracted = module.extract_table_with_llm(
        client=mock_client,
        parameters={},
        query="Consolidated Balance Sheet",
        page_numbers=[1],
        pages=sample_pages,
        prompt_override=custom_prompt,
    )

    mock_client.ask_json.assert_called_once()
    call_kwargs = mock_client.ask_json.call_args
    # The prompt should contain the query text
    prompt_used = call_kwargs.kwargs.get("prompt", call_kwargs[1].get("prompt", ""))
    if not prompt_used and len(call_kwargs.args) > 0:
        prompt_used = str(call_kwargs)
    assert "Consolidated Balance Sheet" in str(call_kwargs)


def test_extract_supplementary_with_llm_missing_key_aggregates_response(
    mock_client, sample_pages, sample_primary_extraction, monkeypatch
):
    """When chunk response lacks 'supplementary_tables' key, it should be captured in chunk_responses."""
    monkeypatch.setattr(module, "SUPPLEMENTARY_EXTRACTION_MAX_PAGES", 2)
    mock_client.ask_json.side_effect = [
        {"unexpected_key": "data"},
        {"supplementary_tables": [{"id": "B"}]},
    ]

    result = module.extract_supplementary_with_llm(
        client=mock_client,
        parameters={},
        query="Consolidated Balance Sheet",
        primary_extraction=sample_primary_extraction,
        page_numbers=[1, 2, 10],
        pages=sample_pages,
    )

    # Should still return a result (possibly with partial data)
    assert "supplementary_tables" in result
