"""Financial statement extraction from annual report PDFs.

Provides :class:`FinancialStatementExtractor`, which extends
:class:`BasePdfExtractor` with multi-query extraction of primary
financial tables and supplementary disclosures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from financial_forecast.extraction.base_pdf_extractor import BasePdfExtractor
from financial_forecast.extraction.statement_config import (
    STATEMENT_CONFIGS,
    StatementType,
)
from financial_forecast.clients.protocols import LLMClient

SUPPLEMENTARY_EXTRACTION_MAX_PAGES = 50


def _get_supplementary_context(statement_type: StatementType) -> str:
    """Return a formatted string of financial elements for the statement type.

    Derives human-readable labels from the field names in STATEMENT_CONFIGS.
    """
    fields = STATEMENT_CONFIGS[statement_type]["fields"]
    readable = ", ".join(f.replace("_", " ") for f in fields)
    return (
        f"Pay extra attention to supplementary disclosures related to these "
        f"financial elements: {readable}."
    )


class FinancialStatementExtractor(BasePdfExtractor):
    """Extract financial statements from annual report PDFs.

    For each statement type, performs a two-phase extraction:
    1. Primary table extraction from LLM-selected pages.
    2. Supplementary disclosure extraction from related pages.

    Args:
        queries: Statement types to extract.
        llm_client: Configured :class:`LLMClient`.
        batch_size: Pages per LLM prompt during page selection.
        parameters: Extra parameters forwarded to the LLM.
        max_workers: Number of PDFs to process in parallel.
    """

    def __init__(
        self,
        queries: List[StatementType],
        llm_client: LLMClient,
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
        self.queries = queries

    def _extract_one_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
    ) -> None:
        """Process one PDF through all queries."""
        pages = self._extract_pages(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        for statement_type in self.queries:
            query_label = statement_type.value.replace("-", " ").title()

            print(
                f"Step 1/2 - selecting primary pages for "
                f"{pdf_path.name} - {query_label}..."
            )
            primary_pages = self._select_pages(
                pages,
                query_label,
                is_financial_statement=True,
            )
            print(f"Primary statement pages: {primary_pages}")

            primary_extraction = self._extract_table(
                query_label,
                primary_pages,
                pages,
            )

            supplementary_query = self._build_supplementary_page_query(
                statement_type,
                primary_extraction,
            )
            print(
                f"Step 2/2 - selecting supplementary pages for "
                f"{pdf_path.name} - {query_label}..."
            )
            supplementary_pages = self._select_pages(
                pages,
                supplementary_query,
                is_financial_statement=False,
            )
            print(f"Supplementary pages: {supplementary_pages}")

            supplementary_extraction = self._extract_supplementary(
                statement_type,
                primary_extraction,
                supplementary_pages,
                pages,
            )

            result = {
                "query": query_label,
                "statement_type": statement_type.value,
                "selected_pages": {
                    "primary_statement_pages": primary_pages,
                    "supplementary_pages": supplementary_pages,
                },
                "extraction": {
                    "primary_statement": primary_extraction,
                    "supplementary": supplementary_extraction,
                },
            }
            self._write_json(
                output_dir,
                pdf_path,
                statement_type.value,
                result,
            )

    def _extract_table(
        self,
        query_label: str,
        page_numbers: List[int],
        pages: Dict[int, Optional[str]],
    ) -> dict:
        """Extract a financial table from selected pages."""
        if not page_numbers:
            raise ValueError("No pages selected for extraction.")
        pages_text = self._format_pages(page_numbers, pages)
        prompt = (
            f"Find the table that corresponds to {query_label} and "
            f"output it in a nice tabular form (in English) from the "
            f"following pages data.\nPages:\n{pages_text}"
        )
        return self._call_llm_with_fallback(prompt)

    def _extract_supplementary(
        self,
        statement_type: StatementType,
        primary_extraction: dict,
        page_numbers: List[int],
        pages: Dict[int, Optional[str]],
    ) -> dict:
        """Extract supplementary disclosure tables in chunks.

        Returns a flat dict with ``chunk_1``, ``chunk_2``, ... keys,
        each containing the raw LLM response for that chunk.
        """
        if not page_numbers:
            return {}

        query_label = statement_type.value.replace("-", " ").title()
        elements_context = _get_supplementary_context(statement_type)
        result = {}
        primary_context = json.dumps(
            primary_extraction,
            ensure_ascii=False,
            indent=2,
        )

        chunk_idx = 0
        for start in range(
            0,
            len(page_numbers),
            SUPPLEMENTARY_EXTRACTION_MAX_PAGES,
        ):
            chunk_idx += 1
            chunk = page_numbers[start : start + SUPPLEMENTARY_EXTRACTION_MAX_PAGES]
            pages_text = self._format_pages(chunk, pages)
            prompt = (
                "You are extracting supplementary financial disclosures "
                "from annual report pages.\n"
                f"Primary statement: {query_label}\n"
                "Primary statement extraction context:\n"
                f"{primary_context}\n\n"
                f"\n{elements_context}\n"
                "Infer the line items from the primary statement context and the financial elements "
                "above. Extract supplementary tables related to those "
                "inferred line items (for example, breakdowns/expansions/"
                "schedules/notes such as an expanded 'Other Income' "
                "table). You can be generous with the supplementary "
                "tables you extract since it is better to have redundant information than "
                "not have necessary information.\n"
                "Return in a nice tabular format (in English).\n"
                'If none are found, return "No supplementary tables '
                'found".\n\n'
                f"Pages:\n{pages_text}"
            )
            parsed = self._call_llm_with_fallback(prompt)
            raw = parsed.get("raw_response", parsed)
            result[f"chunk_{chunk_idx}"] = raw

        return result

    @staticmethod
    def _build_supplementary_page_query(
        statement_type: StatementType,
        primary_extraction: dict,
    ) -> str:
        """Build query for locating supplementary disclosure pages."""
        query_label = statement_type.value.replace("-", " ").title()
        primary_context = json.dumps(
            primary_extraction,
            ensure_ascii=False,
        )
        elements_context = _get_supplementary_context(statement_type)
        return (
            f"Supplementary disclosures, breakdowns, expansions, and "
            f"note tables for {query_label}.\n "
            f"Use this extracted primary statement and the following list of financial elements as context to infer "
            f"relevant line items and find related supplementary pages: "
            f"{primary_context}\n\n"
            f"{elements_context}"
        )
