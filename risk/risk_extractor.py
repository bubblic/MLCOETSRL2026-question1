"""Risk warnings extraction from annual report PDFs.

Provides :class:`RiskWarningsExtractor`, which extends
:class:`BasePdfExtractor` to identify risk-relevant pages, extract
structured findings per risk category, and synthesise a risk memo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from risk.anonymizer import EntityAnonymizer
from financial_forecast.clients.protocols import LLMClient
from financial_forecast.extraction.base_pdf_extractor import BasePdfExtractor
from risk.category_extraction import (
    extract_all_categories,
)
from risk.page_flagger import flag_all_pages
from risk.risk_categories import (
    DEFAULT_CHUNK_SIZE,
    RiskCategory,
)
from risk.synthesiser import synthesise_risk_memo
from risk.usage_tracker import UsageTracker


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
        known_entities: Optional entity list for anonymization.
            Each entry is a dict with ``"type"`` and ``"names"`` keys.
        use_ner: Use spacy NER for automatic entity detection.
        anonymize_years: Replace absolute years with relative markers.
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
        known_entities: Optional[List[Dict[str, Any]]] = None,
        use_ner: bool = True,
        anonymize_years: bool = True,
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
        self.known_entities = known_entities
        self.use_ner = use_ner
        self.anonymize_years = anonymize_years

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

        # Stage 0.5: Entity anonymization
        anonymizer = EntityAnonymizer(
            known_entities=self.known_entities,
            use_ner=self.use_ner,
            anonymize_years=self.anonymize_years,
        )
        pages = anonymizer.anonymize_pages(pages)
        print(f"Anonymized {len(anonymizer.entity_map())} entities")

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

        # Write output (de-anonymize findings and memo)
        result = {
            "flagged_pages": flagged_summary,
            "category_results": anonymizer.deanonymize(findings),
            "risk_memo": anonymizer.deanonymize(memo),
            "entity_map": anonymizer.entity_map(),
            "usage": tracker.summary(),
        }
        self._write_json(output_dir, pdf_path, "risk-warnings", result)
        tracker.print_summary()
