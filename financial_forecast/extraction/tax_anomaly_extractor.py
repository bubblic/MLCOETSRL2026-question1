"""Tax anomaly extraction from Form 10-K PDFs.

Provides :class:`TaxAnomalyExtractor`, which extends
:class:`BasePdfExtractor` to identify relevant 10-K pages and extract
structured JSON describing one-time tax anomalies, in billions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from financial_forecast.extraction.base_pdf_extractor import BasePdfExtractor
from financial_forecast.clients.protocols import LLMClient
import time


DEFAULT_SELECTION_PROMPT = (
    "You are given pages from a Form 10-K annual report.\n"
    "Identify all pages that contain any of the following:\n"
    "1) Item 7: Management's Discussion and Analysis (MD&A)\n"
    "2) Item 8: Financial Statements and Supplementary Data notes for:\n"
    "   - Income Taxes\n"
    "Include pages with headings, substantive discussion, tables, and "
    "continuations of these sections.\n"
    'Return ONLY valid JSON with this exact shape: {{"pages": [<page_number>, ...]}}.\n'
    'If none match, return {{"pages": []}}.\n\n'
    "Pages:\n{pages}"
)

DEFAULT_EXTRACTION_PROMPT = (
    "You are an expert financial analyst. Your task is to analyze the provided "
    "excerpts from a company's Form 10-K (MD&A and Financial Footnotes) and "
    "extract data regarding one-time tax anomalies.\n\n"
    "Carefully evaluate the text for the following:\n\n"
    "Current Year Anomalies: Identify any massive, non-recurring, discrete tax "
    "charges or benefits ASSESSED CURRENT TAX YEAR that skewed the current year's net income (e.g., "
    "finalized state aid decisions, sudden impacts from new tax legislation), not something that happened in a previous tax year that the author uses to explain the difference of this year's tax from a previous year's. Positive for tax paid by company, negative for tax reduction or benefit for the company.\n\n"
    "For better accuracy, try to extract the associated quantity from a text from a table (or ordered list of values since the provided text is extracted from pdf using a simple extraction tool) instead of from just sentences, paying attention to how such quantity would be labeled in the table (e.g., 'Impacts of the Act' field). You must respond ONLY with a valid JSON object using the exact schema below. "
    "Do not include any markdown formatting, preamble, or postscript. If a specific "
    "data point is not explicitly mentioned or cannot be reliably quantified from "
    "the text, output null for that field.\n\n"
    "JSON Schema:\n"
    "{{\n"
    '  "current_tax_year": <number>,\n'
    '  "tax_onetime_amount": <number in billions or null>,\n'
    '  "tax_onetime_note": "<string explaining the anomaly or null>",\n'
    "}}\n\n"
    "Pages:\n{pages}"
)


class TaxAnomalyExtractor(BasePdfExtractor):
    """Extract tax anomalies from 10-K PDFs.

    Args:
        llm_client: Configured :class:`LLMClient`.
        selection_prompt: Prompt template for page selection.
        extraction_prompt: Prompt template for tax extraction
            (with ``{pages}`` placeholder).
        batch_size: Pages per LLM prompt during page selection.
        parameters: Extra parameters forwarded to the LLM.
        max_workers: Number of PDFs to process in parallel.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        selection_prompt: str = DEFAULT_SELECTION_PROMPT,
        extraction_prompt: str = DEFAULT_EXTRACTION_PROMPT,
        batch_size: int = 100,
        parameters: Optional[Dict] = None,
        max_workers: int = 9,
    ):
        super().__init__(
            llm_client=llm_client,
            batch_size=batch_size,
            parameters=parameters,
            max_workers=max_workers,
        )
        self.selection_prompt = selection_prompt
        self.extraction_prompt = extraction_prompt

    def _extract_one_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
    ) -> None:
        """Process one PDF for tax anomaly extraction."""
        pages = self._extract_pages(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Selecting pages for {pdf_path.name}...")
        selected_pages = self._select_pages(
            pages,
            "",
            is_financial_statement=False,
            prompt_override=self.selection_prompt,
        )
        print(f"Selected pages: {selected_pages}")

        extraction = self._extract_tax_json(selected_pages, pages)

        result = {
            "selected_pages": selected_pages,
            "extraction": extraction,
        }
        self._write_json(
            output_dir,
            pdf_path,
            "tax-anomalies",
            result,
        )

    def _extract_tax_json(
        self,
        page_numbers: List[int],
        pages: Dict[int, Optional[str]],
    ) -> dict:
        """Extract tax anomaly/contingency JSON from selected pages."""
        if not page_numbers:
            return {
                "current_tax_year": None,
                "tax_onetime_amount": None,
                "tax_onetime_note": None,
                "amount_scale": None,
            }

        pages_text = self._format_pages(page_numbers, pages)
        prompt = self.extraction_prompt.format(pages=pages_text)

        max_retries = 3
        for attempt in range(max_retries):
            response = self._call_llm_with_fallback(prompt)
            if response.get("current_tax_year") is not None:
                break
            print(
                f"  Retry {attempt + 1}/{max_retries}: "
                f"current_tax_year was null, waiting 5s..."
            )
            time.sleep(5)

        return {
            "current_tax_year": response.get("current_tax_year"),
            "tax_onetime_amount": response.get("tax_onetime_amount"),
            "tax_onetime_note": response.get("tax_onetime_note"),
            "amount_scale": 1e9,
        }
