"""LLM-assisted page identification for annual report PDFs.

This module sends batches of PDF page text to a large language model and
asks it to identify which pages contain content matching a given query
(e.g. a specific financial statement).  Results are returned as sorted
lists of page numbers.
"""

from __future__ import annotations

import json
import re

from financial_forecast.clients.protocols import LLMClient


def select_pages_with_llm(
    client: LLMClient,
    parameters: dict[str, object],
    pages: dict[int, str | None],
    query: str,
    batch_size: int,
    is_financial_statement: bool,
    prompt_override: str | None = None,
) -> list[int]:
    """Identify relevant pages by querying the LLM in batches.

    Args:
        client: Configured Azure LLM client instance.
        parameters: Extra parameters forwarded to the LLM call.
        pages: Mapping of page numbers to their extracted text.
        query: Natural-language description of the target content.
        batch_size: Maximum number of pages to include in a single prompt.
        is_financial_statement: If ``True``, prompts are tailored for
            tabular financial-statement detection.
        prompt_override: Optional format-string template that will receive
            ``{query}`` and ``{pages}`` placeholders.

    Returns:
        Sorted list of page numbers identified as relevant.
    """
    valid_pages = set(pages.keys())
    selected_pages: set[int] = set()
    for batch in chunk_pages(sorted(valid_pages), batch_size):
        batch_pages = {page_num: pages[page_num] for page_num in batch}
        pages_text = build_page_blocks(batch_pages)
        prompt = (
            prompt_override.format(query=query, pages=pages_text)
            if prompt_override
            else build_selection_prompt(query, pages_text, is_financial_statement)
        )
        print(prompt[:2000])
        response = client.ask_json(
            message="gen-ai-response",
            prompt=prompt,
            parameters={},  # This has to be blank for non-reasoning model.
            reasoning=False,
        )
        print(response)
        payload = response
        if "raw_response" in response:
            extracted = extract_json_from_text(str(response["raw_response"]))
            if extracted:
                payload = extracted
        batch_selected = normalize_pages(payload, valid_pages)
        selected_pages.update(batch_selected)
    return sorted(selected_pages)


def chunk_pages(page_numbers: list[int], batch_size: int) -> list[list[int]]:
    """Split page numbers into fixed-size batches.

    Args:
        page_numbers: Sorted list of page numbers.
        batch_size: Maximum batch size (must be positive).

    Returns:
        List of page-number sublists.

    Raises:
        ValueError: If *batch_size* is not positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [
        page_numbers[idx : idx + batch_size]
        for idx in range(0, len(page_numbers), batch_size)
    ]


def build_page_blocks(batch_pages: dict[int, str]) -> str:
    """Format a batch of pages into a single prompt-ready string.

    Args:
        batch_pages: Mapping of page numbers to their text content.

    Returns:
        Formatted string with page blocks separated by ``---``.
    """
    page_blocks = []
    for page_num in sorted(batch_pages):
        text = (batch_pages[page_num] or "").strip()
        page_blocks.append(f"Page {page_num}:\n{text}")
    return "\n\n---\n\n".join(page_blocks)


def normalize_llm_response(response: dict[str, object]) -> dict[str, object]:
    """Normalize an LLM response by extracting JSON from ``raw_response``.

    If the response contains a ``raw_response`` key whose value embeds
    a JSON object, extract and return it.  Otherwise return the original
    response unchanged.

    Args:
        response: Parsed response dict from ``LLMClient.ask_json()``.

    Returns:
        The normalized response dict.
    """
    if not isinstance(response, dict) or "raw_response" not in response:
        return response
    extracted = extract_json_from_text(str(response["raw_response"]))
    if extracted is not None:
        return extracted
    return response


def extract_json_from_text(text: str) -> dict[str, object] | None:
    """Extract the first JSON object found in free-form text.

    Args:
        text: Arbitrary string that may contain an embedded JSON object.

    Returns:
        Parsed dictionary if a valid JSON object is found, otherwise
        ``None``.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def normalize_pages(payload: dict[str, object], valid_pages: set[int]) -> list[int]:
    """Normalise and validate page numbers returned by the LLM.

    Accepts both ``"pages"`` and ``"page_numbers"`` keys.  String-encoded
    integers are coerced, and values outside *valid_pages* are silently
    dropped.

    Args:
        payload: LLM response dictionary.
        valid_pages: Set of acceptable page numbers.

    Returns:
        Sorted, deduplicated list of valid page numbers.
    """
    raw = payload.get("pages") or payload.get("page_numbers")
    if not isinstance(raw, list):
        return []
    normalized = []
    for item in raw:
        if isinstance(item, int) and item in valid_pages:
            normalized.append(item)
        elif isinstance(item, str) and item.isdigit():
            page_num = int(item)
            if page_num in valid_pages:
                normalized.append(page_num)
    return sorted(set(normalized))


def build_selection_prompt(
    query: str, pages_text: str, is_financial_statement: bool
) -> str:
    """Build the default page-selection prompt for the LLM.

    Args:
        query: Target content description.
        pages_text: Pre-formatted page blocks.
        is_financial_statement: Whether the query targets a tabular
            financial statement.

    Returns:
        Prompt string ready to send to the LLM.
    """
    table_or_information = "table" if is_financial_statement else "information"
    table_or_line_items = (
        "table or line items" if is_financial_statement else "information"
    )
    prompt = (
        "You are given multiple pages from an annual report. "
        f"Identify which pages contain the {table_or_information} "
        "for the requested query. "
        'Return ONLY JSON in the shape: {"pages": [<page_number>, ...]}. '
        f"Include all pages that contain the {table_or_line_items}. "
        'If none, return {"pages": []}.\n\n'
        f"Query: {query}\n\n"
        f"Pages:\n{pages_text}"
    )
    return prompt
