"""Abstract base class for LLM-driven PDF extraction.

Provides shared infrastructure for extracting structured data from
annual report PDFs: text extraction, LLM page selection, LLM calls
with raw_response fallback, page formatting, and file/directory handling.

Subclasses implement :meth:`_extract_one_pdf` with domain-specific logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Dict, List, Optional

from financial_forecast.extraction.page_identifier import (
    select_pages_with_llm,
    normalize_llm_response,
)
from financial_forecast.clients.protocols import LLMClient
from financial_forecast.extraction.pdf_extractor import extract_text_pdfplumber


class BasePdfExtractor(ABC):
    """Abstract base for PDF extraction pipelines.

    Handles file vs directory input, parallel execution, PDF text
    extraction, LLM page selection, and common utilities.  Subclasses
    implement :meth:`_extract_one_pdf` with their specific extraction logic.

    Args:
        llm_client: Configured :class:`LLMClient`.
        batch_size: Pages per LLM prompt during page selection.
        parameters: Extra parameters forwarded to the LLM.
        max_workers: Number of PDFs to process in parallel.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        batch_size: int = 100,
        parameters: Optional[Dict] = None,
        max_workers: int = 9,
    ):
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.parameters = parameters or {"temperature": 0, "top_k": 1}
        self.max_workers = max_workers

    def run(
        self,
        input_path: str,
        output_dir: str = "extracted_text",
    ) -> None:
        """Extract from a single PDF or a directory of PDFs.

        Args:
            input_path: Path to a PDF file or directory of PDFs.
            output_dir: Directory for output JSON files.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        if input_path.is_file():
            self._extract_one_pdf(input_path, output_dir)
            return

        if input_path.is_dir():
            pdf_files = sorted(input_path.glob("*.pdf"))
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in {input_path}")
            if self.max_workers <= 1:
                for pdf_path in pdf_files:
                    self._extract_one_pdf(pdf_path, output_dir)
                return

            with ThreadPoolExecutor(
                max_workers=self.max_workers,
            ) as executor:
                futures = {
                    executor.submit(
                        self._extract_one_pdf,
                        pdf_path,
                        output_dir,
                    ): pdf_path
                    for pdf_path in pdf_files
                }
                for future in as_completed(futures):
                    pdf_path = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed processing {pdf_path}: {exc}"
                        ) from exc
            return

        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    @abstractmethod
    def _extract_one_pdf(
        self,
        pdf_path: Path,
        output_dir: Path,
    ) -> None:
        """Process one PDF. Subclasses implement domain-specific logic."""

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _extract_pages(self, pdf_path: Path) -> Dict[int, Optional[str]]:
        """Extract text from all pages of a PDF."""
        return extract_text_pdfplumber(str(pdf_path))

    def _select_pages(
        self,
        pages: Dict[int, Optional[str]],
        query: str,
        is_financial_statement: bool = False,
        prompt_override: Optional[str] = None,
    ) -> List[int]:
        """Identify relevant pages using the LLM."""
        return select_pages_with_llm(
            client=self.llm_client,
            parameters=self.parameters,
            pages=pages,
            query=query,
            batch_size=self.batch_size,
            is_financial_statement=is_financial_statement,
            prompt_override=prompt_override,
        )

    def _call_llm_with_fallback(self, prompt: str) -> dict:
        """Call the LLM and try to extract JSON from raw_response if present."""
        response = self.llm_client.ask_json(
            message="gen-ai-response",
            prompt=prompt,
            parameters=self.parameters,
            reasoning=True,
        )
        return normalize_llm_response(response)

    @staticmethod
    def _format_pages(
        page_numbers: List[int],
        pages: Dict[int, Optional[str]],
    ) -> str:
        """Format selected pages into a single text block."""
        joined = []
        for page_num in page_numbers:
            joined.append(
                f"Page {page_num}:\n" f"{(pages.get(page_num) or '').strip()}"
            )
        return "\n\n---\n\n".join(joined)

    @staticmethod
    def _write_json(
        output_dir: Path,
        pdf_path: Path,
        slug: str,
        result: dict,
    ) -> None:
        """Write extraction result as JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{pdf_path.stem}.{slug}.llm.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"Wrote {output_path}")
