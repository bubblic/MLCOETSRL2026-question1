"""LLM-driven tax anomaly and contingency extraction from Form 10-K PDFs.

This module identifies relevant pages in a 10-K filing (Item 7 MD&A and
Item 8 financial statement notes covering income taxes and commitments/
contingencies), then uses a reasoning LLM to extract structured JSON
describing one-time tax anomalies and future tax contingencies.

Example usage::

    python -m financial_forecast.extraction.tax_anomaly_extractor \\
        --input-file "./annual_reports/google_2024.pdf"

    python -m financial_forecast.extraction.tax_anomaly_extractor \\
        --input-dir ./annual_reports
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from financial_forecast.clients.azure_llm_client import AzureLLMClient
from financial_forecast.extraction.page_identifier import extract_json_from_text, select_pages_with_llm
from financial_forecast.extraction.pdf_extractor import extract_text_pdfplumber

WORKSPACE_DIR = Path(__file__).resolve().parent
ANNUAL_REPORTS_DIR = WORKSPACE_DIR / "annual_reports"
OUTPUT_DIR = WORKSPACE_DIR / "extracted_text"

EXTRACTORS = {
    "pdfplumber": extract_text_pdfplumber,
}

DEFAULT_SELECTION_QUERY = (
    "Item 7 Management's Discussion and Analysis (MD&A), and Item 8 Financial "
    'Statements and Supplementary Data sections specifically covering "Income Taxes" '
    'and "Commitments and Contingencies".'
)

DEFAULT_SELECTION_PROMPT = (
    "You are given pages from a Form 10-K annual report.\n"
    "Identify all pages that contain any of the following:\n"
    "1) Item 7: Management's Discussion and Analysis (MD&A)\n"
    "2) Item 8: Financial Statements and Supplementary Data notes for:\n"
    "   - Income Taxes\n"
    "   - Commitments and Contingencies\n\n"
    "Include pages with headings, substantive discussion, tables, and continuations of "
    "these sections.\n"
    'Return ONLY valid JSON with this exact shape: {{"pages": [<page_number>, ...]}}.\n'
    'If none match, return {{"pages": []}}.\n\n'
    "Query: {query}\n\n"
    "Pages:\n{pages}"
)

DEFAULT_EXTRACTION_PROMPT = (
    "You are an expert financial analyst. Your task is to analyze the provided excerpts "
    "from a company's Form 10-K (MD&A and Financial Footnotes) and extract data regarding "
    "one-time tax anomalies and future tax contingencies.\n\n"
    "Carefully evaluate the text for the following:\n\n"
    "Current Year Anomalies: Identify any massive, non-recurring, discrete tax charges "
    "or benefits that heavily skewed the current year's net income (e.g., finalized state "
    "aid decisions, sudden impacts from new tax legislation).\n\n"
    "Future Contingencies: Identify any quantified maximum tax exposures, unreserved tax "
    "liabilities, or significant unrecognized tax benefits (UTBs) that management indicates "
    "could be resolved or assessed in future years (e.g., tax funds held in escrow pending "
    "appeal, estimated UTB decreases in the next 12 months).\n\n"
    "You must respond ONLY with a valid JSON object using the exact schema below. Do not "
    "include any markdown formatting, preamble, or postscript. If a specific data point is "
    "not explicitly mentioned or cannot be reliably quantified from the text, output null "
    "for that field.\n\n"
    "JSON Schema:\n"
    "{{\n"
    '  "tax_onetime_amount": <number in billions or null>,\n'
    '  "tax_onetime_note": "<string explaining the anomaly or null>",\n'
    '  "tax_contingency_amount": <number in billions or null>,\n'
    '  "tax_contingency_note": "<string explaining the future reserve or null>"\n'
    "}}\n\n"
    "Pages:\n{pages}"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tax anomaly extraction pipeline.

    Returns:
        Parsed ``argparse.Namespace`` with all CLI options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Use an LLM to identify relevant 10-K pages (Item 7/Item 8 tax-related "
            "content) and extract anomaly/contingency JSON."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory to write extracted JSON files.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(ANNUAL_REPORTS_DIR),
        help="Directory containing annual report PDFs.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Single PDF file to extract instead of a whole directory.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_SELECTION_QUERY,
        help="Query text used for page selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Pages per prompt to the LLM.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="LLM endpoint override. Defaults to AZURE_DEEPSEEK_ENDPOINT.",
    )
    parser.add_argument(
        "--parameters",
        default='{"temperature": 0, "top_k": 1}',
        help="JSON string of parameters to send to the extraction call.",
    )
    parser.add_argument(
        "--selection-prompt",
        default=DEFAULT_SELECTION_PROMPT,
        help="Optional override for the page selection prompt template.",
    )
    parser.add_argument(
        "--extraction-prompt",
        default=DEFAULT_EXTRACTION_PROMPT,
        help="Optional override for the extraction prompt template.",
    )
    return parser.parse_args()


def parse_parameters(raw: str | None) -> dict[str, object]:
    """Decode a JSON string into a parameter dictionary.

    Args:
        raw: JSON-encoded string, or ``None``.

    Returns:
        Decoded dictionary, or empty dict when *raw* is ``None``.

    Raises:
        ValueError: If the decoded value is not a JSON object.
    """
    if raw is None:
        return {}
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("Parameters must decode to a JSON object.")
    return loaded


def resolve_endpoint(override: str | None) -> str:
    """Resolve the LLM endpoint from an override or environment variable.

    Args:
        override: Explicit endpoint string, or ``None`` to fall back to
            the ``AZURE_DEEPSEEK_ENDPOINT`` environment variable.

    Returns:
        Resolved endpoint URL.

    Raises:
        ValueError: If no endpoint can be determined.
    """
    endpoint = (override or os.getenv("AZURE_DEEPSEEK_ENDPOINT", "")).strip()
    if not endpoint:
        raise ValueError("Missing endpoint. Set AZURE_DEEPSEEK_ENDPOINT.")
    return endpoint


def extract_tax_json_with_llm(
    client: AzureLLMClient,
    parameters: dict[str, object],
    page_numbers: list[int],
    pages: dict[int, str | None],
    prompt_template: str,
) -> dict[str, object]:
    """Extract tax anomaly/contingency JSON from the selected pages.

    Args:
        client: Configured Azure LLM client.
        parameters: Extra parameters forwarded to the LLM call.
        page_numbers: Pages identified as containing tax-related content.
        pages: Full page-text mapping from the PDF.
        prompt_template: Prompt template with a ``{pages}`` placeholder.

    Returns:
        Dictionary with ``tax_onetime_amount``, ``tax_onetime_note``,
        ``tax_contingency_amount``, and ``tax_contingency_note`` keys.
    """
    if not page_numbers:
        return {
            "tax_onetime_amount": None,
            "tax_onetime_note": None,
            "tax_contingency_amount": None,
            "tax_contingency_note": None,
        }
    joined_pages = []
    for page_num in page_numbers:
        joined_pages.append(f"Page {page_num}:\n{(pages.get(page_num) or '').strip()}")
    pages_text = "\n\n---\n\n".join(joined_pages)
    prompt = prompt_template.format(pages=pages_text)
    response = client.ask_json(
        message="gen-ai-response",
        prompt=prompt,
        parameters=parameters,
        reasoning=True,
    )
    if "raw_response" in response:
        extracted = extract_json_from_text(str(response["raw_response"]))
        if extracted:
            response = extracted

    return {
        "tax_onetime_amount": response.get("tax_onetime_amount"),
        "tax_onetime_note": response.get("tax_onetime_note"),
        "tax_contingency_amount": response.get("tax_contingency_amount"),
        "tax_contingency_note": response.get("tax_contingency_note"),
    }


def run_pipeline(
    input_file: Path,
    output_dir: Path,
    query: str,
    batch_size: int,
    endpoint: str,
    parameters: dict[str, object],
    selection_prompt: str,
    extraction_prompt: str,
) -> None:
    """Execute the tax anomaly extraction pipeline for one PDF.

    Args:
        input_file: Path to the 10-K PDF.
        output_dir: Directory for output JSON files.
        query: Query text for page selection.
        batch_size: Pages per LLM prompt during page selection.
        endpoint: Azure LLM endpoint URL.
        parameters: Extra parameters forwarded to the LLM.
        selection_prompt: Page-selection prompt template.
        extraction_prompt: Extraction prompt template.
    """
    extractor = EXTRACTORS["pdfplumber"]
    pages = extractor(str(input_file))
    client = AzureLLMClient(endpoint=endpoint)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selecting pages for {input_file.name}...")
    selected_pages = select_pages_with_llm(
        client=client,
        parameters=parameters,
        pages=pages,
        query=query,
        batch_size=batch_size,
        is_financial_statement=False,
        prompt_override=selection_prompt,
    )
    print(f"Selected pages: {selected_pages}")

    extraction = extract_tax_json_with_llm(
        client=client,
        parameters=parameters,
        page_numbers=selected_pages,
        pages=pages,
        prompt_template=extraction_prompt,
    )
    result = {
        "query": query,
        "selected_pages": selected_pages,
        "extraction": extraction,
    }
    output_path = output_dir / f"{input_file.stem}.tax-anomalies-contingencies.llm.json"
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(result, file_handle, ensure_ascii=False, indent=2)
        file_handle.write("\n")
    print(extraction)
    print(f"Wrote {output_path}")


def main() -> None:
    """CLI entry point for the tax anomaly extraction pipeline."""
    args = parse_args()
    parameters = parse_parameters(args.parameters)
    endpoint = resolve_endpoint(args.endpoint)

    if args.input_file:
        pdf_files = [Path(args.input_file)]
    else:
        pdf_files = sorted(Path(args.input_dir).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found.")

    for pdf_path in pdf_files:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {pdf_path}")
        run_pipeline(
            input_file=pdf_path,
            output_dir=Path(args.output_dir),
            query=args.query,
            batch_size=args.batch_size,
            endpoint=endpoint,
            parameters=parameters,
            selection_prompt=args.selection_prompt,
            extraction_prompt=args.extraction_prompt,
        )


if __name__ == "__main__":
    main()
