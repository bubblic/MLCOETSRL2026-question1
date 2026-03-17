"""PDF text extraction utilities.

This module wraps third-party PDF libraries to provide a uniform
page-level text extraction interface used by downstream LLM pipelines.
"""

from __future__ import annotations

from typing import Dict, Optional

import pdfplumber


def extract_text_pdfplumber(pdf_path: str) -> Dict[int, Optional[str]]:
    """Extract text from every page of a PDF using pdfplumber.

    Args:
        pdf_path: Filesystem path to the PDF file.

    Returns:
        Dictionary mapping zero-based page numbers to the extracted text
        string for that page (may be ``None`` if a page contains no
        extractable text).
    """
    with pdfplumber.open(pdf_path) as pdf:
        return {
            page_num: page.extract_text() for page_num, page in enumerate(pdf.pages)
        }
