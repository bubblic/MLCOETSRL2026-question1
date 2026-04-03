"""Stage 1: Coarse page flagging for risk-relevant content.

Chunks all PDF pages and sends each chunk to the LLM to identify
pages belonging to any of the 9 risk categories.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from financial_forecast.clients.protocols import LLMClient
from financial_forecast.extraction.page_identifier import normalize_llm_response
from risk.models import FlaggedPage
from risk.prompts import PAGE_FLAG_PROMPT
from risk.risk_categories import (
    ALL_CATEGORIES,
    RiskCategory,
)
from risk.usage_tracker import UsageTracker


@dataclass(frozen=True)
class PageChunk:
    """An immutable group of consecutive PDF pages for LLM processing.

    Attributes:
        start: First page number in this chunk.
        end: Last page number in this chunk.
        text: Concatenated page text with ``[PAGE X]`` markers.
    """

    start: int
    end: int
    text: str


def chunk_pages(
    pages: Dict[int, Optional[str]],
    chunk_size: int,
) -> List[PageChunk]:
    """Split pages into chunks with ``[PAGE X]`` markers.

    Args:
        pages: Mapping of page numbers to extracted text.
        chunk_size: Maximum pages per chunk.

    Returns:
        List of :class:`PageChunk` instances.
    """
    page_nums = sorted(p for p in pages if pages[p] is not None)
    if not page_nums:
        return []
    chunks: List[PageChunk] = []
    for i in range(0, len(page_nums), chunk_size):
        batch = page_nums[i : i + chunk_size]
        combined = "\n\n".join(
            f"[PAGE {p}]\n{(pages[p] or '').strip()}" for p in batch
        )
        chunks.append(PageChunk(start=batch[0], end=batch[-1], text=combined))
    return chunks


def flag_pages_in_chunk(
    chunk: PageChunk,
    llm_client: LLMClient,
    parameters: Dict[str, Any],
    categories: List[str],
    tracker: UsageTracker,
) -> List[FlaggedPage]:
    """Send one chunk to the LLM and parse flagged pages.

    Handles both ``[{...}]`` and ``{"pages": [...]}`` response formats.
    """
    prompt = PAGE_FLAG_PROMPT.format(
        start=chunk.start,
        end=chunk.end,
        categories=json.dumps(categories),
        text=chunk.text,
    )
    response = llm_client.ask_json(
        message="gen-ai-response",
        prompt=prompt,
        parameters=parameters,
        reasoning=True,
    )
    tracker.record("flagging", prompt, json.dumps(response))
    return _parse_flag_response(response, categories)


def _parse_flag_response(
    response: Any,
    valid_categories: List[str],
) -> List[FlaggedPage]:
    """Parse the LLM response into a list of FlaggedPage."""
    items: List[Any] = []

    # Normalize raw_response fallback before parsing structure.
    if isinstance(response, dict) and "raw_response" in response:
        normalized = normalize_llm_response(response)
        if normalized is not response:
            return _parse_flag_response(normalized, valid_categories)
        return []

    if isinstance(response, list):
        items = response
    elif isinstance(response, dict):
        for key in ("pages", "results", "flagged_pages"):
            if key in response and isinstance(response[key], list):
                items = response[key]
                break

    flagged: List[FlaggedPage] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", ""))
        if category not in valid_categories:
            continue
        try:
            flagged.append(
                FlaggedPage(
                    page=int(item["page"]),
                    category=category,
                    reason=str(item.get("reason", "")),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return flagged


def flag_all_pages(
    pages: Dict[int, Optional[str]],
    categories: List[RiskCategory],
    llm_client: LLMClient,
    parameters: Dict[str, Any],
    chunk_size: int,
    tracker: UsageTracker,
) -> Dict[RiskCategory, List[int]]:
    """Flag all pages across all chunks, grouped by category.

    Returns:
        Mapping from each :class:`RiskCategory` to sorted, deduplicated
        page numbers.
    """
    category_values = [cat.value for cat in categories]
    chunks = chunk_pages(pages, chunk_size)

    by_category: Dict[RiskCategory, set] = defaultdict(set)
    for chunk in chunks:
        print(
            f"Flagging pages {chunk.start}-{chunk.end}..."
        )
        flagged = flag_pages_in_chunk(
            chunk, llm_client, parameters, category_values, tracker
        )
        for fp in flagged:
            try:
                cat = RiskCategory(fp.category)
                by_category[cat].add(fp.page)
            except ValueError:
                continue

    return {cat: sorted(pgs) for cat, pgs in by_category.items()}
