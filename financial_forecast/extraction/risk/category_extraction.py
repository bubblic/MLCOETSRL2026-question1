"""Stage 2: Category-specific risk extraction.

For each risk category with flagged pages, sends the page text to the
LLM with a category-specific prompt and validates the response.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from financial_forecast.clients.protocols import LLMClient
from financial_forecast.extraction.page_identifier import normalize_llm_response
from financial_forecast.extraction.risk.models import CATEGORY_MODELS
from financial_forecast.extraction.risk.prompts import EXTRACTION_PROMPTS
from financial_forecast.extraction.risk.risk_categories import RiskCategory
from financial_forecast.extraction.risk.usage_tracker import UsageTracker


def extract_category(
    category: RiskCategory,
    page_numbers: List[int],
    pages: Dict[int, Optional[str]],
    llm_client: LLMClient,
    parameters: Dict[str, Any],
    tracker: UsageTracker,
) -> Dict[str, Any]:
    """Extract risk findings for one category from flagged pages.

    Returns:
        Dict with ``category`` and ``extraction`` keys.
    """
    if not page_numbers:
        return {"category": category.value, "extraction": None}

    pages_text = "\n\n".join(
        f"[PAGE {p}]\n{(pages.get(p) or '').strip()}"
        for p in page_numbers
        if p in pages
    )
    if not pages_text.strip():
        return {"category": category.value, "extraction": None}

    prompt_template = EXTRACTION_PROMPTS.get(category)
    if prompt_template is None:
        return {"category": category.value, "extraction": None}

    prompt = prompt_template.format(pages=pages_text)

    response = llm_client.ask_json(
        message="gen-ai-response",
        prompt=prompt,
        parameters=parameters,
        reasoning=True,
    )
    tracker.record("extraction", prompt, json.dumps(response))

    extraction = normalize_llm_response(response)

    # Validate through the category's Pydantic model when available.
    # On validation failure, fall back to the raw LLM response so the
    # pipeline is never broken by an unexpected LLM output shape.
    adapter = CATEGORY_MODELS.get(category)
    if adapter is not None:
        try:
            validated = adapter.validate_python(extraction)
            extraction = adapter.dump_python(validated, mode="python")
        except ValidationError as exc:
            print(
                f"Warning: Pydantic validation failed for "
                f"{category.value}, using raw response. "
                f"Errors: {exc.error_count()}"
            )

    return {"category": category.value, "extraction": extraction}


def extract_all_categories(
    flagged_pages: Dict[RiskCategory, List[int]],
    pages: Dict[int, Optional[str]],
    llm_client: LLMClient,
    parameters: Dict[str, Any],
    tracker: UsageTracker,
    max_workers: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """Extract findings for all flagged categories in parallel.

    Returns:
        Dict keyed by category value string, each containing
        ``category`` and ``extraction`` fields.
    """
    if not flagged_pages:
        return {}

    results: Dict[str, Dict[str, Any]] = {}

    if max_workers <= 1:
        for category, page_nums in flagged_pages.items():
            result = extract_category(
                category, page_nums, pages, llm_client, parameters, tracker
            )
            results[category.value] = result
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_category,
                category,
                page_nums,
                pages,
                llm_client,
                parameters,
                tracker,
            ): category
            for category, page_nums in flagged_pages.items()
        }
        for future in as_completed(futures):
            category = futures[future]
            try:
                result = future.result()
                results[category.value] = result
            except Exception as exc:
                print(
                    f"Warning: extraction failed for "
                    f"{category.value}: {exc}"
                )
                results[category.value] = {
                    "category": category.value,
                    "extraction": None,
                    "error": str(exc),
                }

    return results
