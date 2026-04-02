"""Risk warnings extraction from annual report PDFs.

Provides :class:`RiskWarningsExtractor`, which extends
:class:`BasePdfExtractor` to identify risk-relevant pages, extract
structured findings per risk category, and synthesise a risk memo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from financial_forecast.clients.protocols import LLMClient
from financial_forecast.extraction.base_pdf_extractor import BasePdfExtractor
from financial_forecast.extraction.risk.category_extraction import (
    extract_all_categories,
)
from financial_forecast.extraction.risk.page_flagger import flag_all_pages
from financial_forecast.extraction.risk.risk_categories import (
    DEFAULT_CHUNK_SIZE,
    RiskCategory,
)
from financial_forecast.extraction.risk.synthesiser import synthesise_risk_memo
from financial_forecast.extraction.risk.usage_tracker import UsageTracker


class RiskWarningsExtractor(BasePdfExtractor):
    """Extract risk warnings from annual report PDFs.

    Pipeline stages:
        0. PDF text extraction (inherited from BasePdfExtractor)
        1. Coarse page flagging across all chunks
        2. Category-specific extraction (parallel across categories)
        3. Risk memo synthesis

    Args:
        llm_client: Configured :class:`LLMClient`.
        categories: Risk categories to detect.  Defaults to all 9.
        chunk_size: Pages per chunk during Stage 1 flagging.
        batch_size: Pages per LLM prompt (inherited, for page selection).
        parameters: Extra parameters forwarded to the LLM.
        max_workers: Number of PDFs to process in parallel.
        category_workers: Parallelism for Stage 2 category extraction.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        categories: Optional[List[RiskCategory]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        batch_size: int = 100,
        parameters: Optional[Dict] = None,
        max_workers: int = 9,
        category_workers: int = 4,
    ):
        super().__init__(
            llm_client=llm_client,
            batch_size=batch_size,
            parameters=parameters,
            max_workers=max_workers,
        )
        self.categories = categories or list(RiskCategory)
        self.chunk_size = chunk_size
        self.category_workers = category_workers

    def _extract_one_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
    ) -> None:
        """Process one PDF through the full risk extraction pipeline."""
        tracker = UsageTracker()

        # Stage 0: Text extraction (inherited)
        print(f"Stage 0: Parsing {pdf_path.name}...")
        pages = self._extract_pages(pdf_path)
        print(f"Extracted {len(pages)} pages")

        # Stage 1: Flag pages
        print("Stage 1: Flagging risk-relevant pages...")
        flagged = flag_all_pages(
            pages,
            self.categories,
            self.llm_client,
            self.parameters,
            self.chunk_size,
            tracker,
        )
        flagged_summary = {
            cat.value: page_nums for cat, page_nums in flagged.items()
        }
        print(f"Flagged pages: {flagged_summary}")

        # Stage 2: Category extraction (parallel)
        print("Stage 2: Extracting category-specific findings...")
        findings = extract_all_categories(
            flagged,
            pages,
            self.llm_client,
            self.parameters,
            tracker,
            self.category_workers,
        )
        print(
            f"Extracted findings for {len(findings)} categories"
        )

        # Stage 3: Synthesis
        print("Stage 3: Synthesising risk memo...")
        memo = synthesise_risk_memo(
            findings, self.llm_client, self.parameters, tracker
        )

        # Write output
        result = {
            "flagged_pages": flagged_summary,
            "category_results": findings,
            "risk_memo": memo,
            "usage": tracker.summary(),
        }
        self._write_json(output_dir, pdf_path, "risk-warnings", result)
        tracker.print_summary()
