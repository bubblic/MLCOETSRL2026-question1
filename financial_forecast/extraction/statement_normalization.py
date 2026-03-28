"""Normalization, parsing, and prompt construction helpers.

This module provides utilities for loading raw statement text, building LLM
prompts, converting loosely-formatted numeric values, and normalizing the
period payloads returned by the language model into a consistent schema.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from financial_forecast.extraction.statement_config import StatementType


def load_raw_statement(path: Path) -> str:
    """Load the statement text payload from a source JSON file."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    extraction = payload.get("extraction", {})
    if isinstance(extraction, str):
        return extraction
    return str(extraction)


def build_prompt(
    company_id: str,
    statement_type: StatementType,
    required_fields: List[str],
    statement_and_supplementary_tables: str,
) -> str:
    """Build the LLM prompt for one financial statement extraction."""
    statement_label = statement_type.value.replace("-", " ").title()
    fields_schema = ",\n".join(
        [
            f'        "{field}": [{{"source_label": number}}]'
            for field in required_fields
        ]
    )
    prompt = (
        "You are a financial statement extraction engine.\n"
        f"Extract normalized {statement_label} values from the statement and supplementary tables.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "company_id": "string",\n'
        '  "periods": [\n'
        "    {\n"
        '      "year": "string",\n'
        '      "currency": "string",\n'
        '      "scale": number,\n'
        '      "values": {\n'
        f"{fields_schema}\n"
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "notes": string\n'
        "}\n\n"
        "Rules:\n"
        "1) Use every period/column available in the statement.\n"
        "2) Return an array of maps that make up each field where each map is {source_label: number} (source_label = actual label from the statement; no commas, no currency symbols, no percent signs in numbers).\n"
        "3) Ensure that the elements in the array correctly make up the total value of the field.\n"
        "4) If multiple elements need to be combined to make up a field, return all of them individually in an array.\n"
        "5) If a value(s) cannot be directly mapped to a field, try to map element(s) to the corresponding field by taking into account the industry the company is in, and note your reasoning in the notes field. If you cannot find a match, return an empty array.\n"
        "6) If there are multiple close synonyms, use best accounting match.\n"
        "7) For year, report the year of the period only.\n"
        "8) For currency, report its formal 3-letter acronym.\n"
        "9) For scale, report the scale of the values in the statement. For example, if the values are in millions, the scale should be 1E6.\n"
        "10) For total_operating_cost, list all elements that reflect the total operational costs incurred in generating income and are included in standard practice for the industry the company is in.\n"
        "11) Generally, numbers in parentheses are negative.\n"
        "12) For income_tax_expense, interest_expenses, and total_operating_cost, the sign convention is the opposite: if an element reduces income, it should be positive; and if it increases income, it should be negative.\n"
        "13) For short_term_market_securities, these are liquid, unrestricted debt or equity investments intended to be sold in the short term, e.g., U.S. Treasuries, commercial paper, money market funds, publicly traded equities, short-term investments.\n"
        "14) Return JSON only.\n\n"
        f"company_id: {company_id}\n\n"
        "statement and supplementary tables:\n"
        f"{statement_and_supplementary_tables}\n"
    )
    return prompt


def to_float_or_none(value: Any) -> Optional[float]:
    """Convert loosely formatted numeric values to float, else return None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"null", "none", "na", "n/a", "-", "--", "\u2014"}:
        return None

    negative = False
    if raw.startswith("(") and raw.endswith(")"):
        negative = True
        raw = raw[1:-1].strip()

    cleaned = re.sub(r"[^0-9.\-]", "", raw)
    if cleaned.count(".") > 1:
        return None
    if cleaned in {"", "-", "."}:
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    if negative:
        parsed = -abs(parsed)
    return parsed


def to_labeled_float_list(value: Any) -> List[Dict[str, float]]:
    """Normalize LLM field output into a list of labeled numeric entries."""
    if isinstance(value, list):
        normalized: List[Dict[str, float]] = []
        fallback_idx = 1
        for item in value:
            if isinstance(item, dict):
                for raw_key, raw_value in item.items():
                    key = str(raw_key).strip()
                    if not key:
                        continue
                    parsed = to_float_or_none(raw_value)
                    if parsed is not None:
                        normalized.append({key: parsed})
                continue

            parsed_scalar = to_float_or_none(item)
            if parsed_scalar is not None:
                normalized.append({f"unlabeled_value_{fallback_idx}": parsed_scalar})
                fallback_idx += 1
        return normalized

    if isinstance(value, dict):
        normalized_dict_items: List[Dict[str, float]] = []
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key:
                continue
            parsed = to_float_or_none(raw_value)
            if parsed is not None:
                normalized_dict_items.append({key: parsed})
        return normalized_dict_items

    parsed_scalar = to_float_or_none(value)
    if parsed_scalar is None:
        return []
    return [{"unlabeled_value_1": parsed_scalar}]


def normalize_periods(periods: Any, required_fields: List[str]) -> List[Dict[str, Any]]:
    """Normalize LLM period payloads to a consistent schema."""
    if not isinstance(periods, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for period_item in periods:
        if not isinstance(period_item, dict):
            continue
        period_label = str(period_item.get("year", "")).strip()
        currency_label = str(period_item.get("currency", "")).strip()
        scale_label = period_item.get("scale", 1)
        values = period_item.get("values", {})
        if not isinstance(values, dict):
            values = {}

        normalized_values: Dict[str, List[Dict[str, float]]] = {}
        for field in required_fields:
            normalized_values[field] = to_labeled_float_list(values.get(field))

        normalized.append(
            {
                "year": period_label,
                "currency": currency_label,
                "scale": scale_label,
                "values": normalized_values,
            }
        )
    return normalized


def sum_labeled_entries(items: Any) -> float:
    """Sum numeric values from a list of single-key label-to-number dictionaries."""
    if not isinstance(items, list):
        return 0.0
    total = 0.0
    for entry in items:
        if not isinstance(entry, dict):
            continue
        for value in entry.values():
            parsed = to_float_or_none(value)
            if parsed is not None:
                total += parsed
    return total


def statement_period_to_single_values(
    period: Dict[str, Any], fields: List[str]
) -> Dict[str, float]:
    """Flatten one statement period into scaled single numeric values."""
    values = period.get("values", {})
    if not isinstance(values, dict):
        values = {}
    scale = to_float_or_none(period.get("scale"))
    if scale is None:
        scale = 1.0

    single_values: Dict[str, float] = {}
    for field in fields:
        single_values[field] = sum_labeled_entries(values.get(field)) * scale
    return single_values


def sanitize_filename(value: str) -> str:
    """Convert arbitrary text into a filesystem-safe filename."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"
