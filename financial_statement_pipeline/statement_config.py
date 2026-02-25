"""Shared configuration and filename helpers for statement processing."""

from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0"

RunMap = Dict[str, Dict[str, Dict[str, Any]]]
ExtractionCounts = Tuple[int, int]

STATEMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "consolidated-balance-sheet": {
        "suffix": ".consolidated-balance-sheet.llm.json",
        "output_suffix": ".consolidated-balance-sheet.normalized.json",
        "fields": [
            "cash_and_cash_equivalents",
            "short_term_market_securities",
            "total_accounts_receivable",
            "total_current_liabilities",
            "total_debt_short_term_and_long_term",
            "total_equity",
            "total_assets",
        ],
    },
    "consolidated-income-statement": {
        "suffix": ".consolidated-income-statement.llm.json",
        "output_suffix": ".consolidated-income-statement.normalized.json",
        "fields": [
            "total_revenue",
            "total_operating_cost",
            "net_income",
            "taxes",
            "interest_expenses",
        ],
    },
    "consolidated-cash-flow-statement": {
        "suffix": ".consolidated-cash-flow-statement.llm.json",
        "output_suffix": ".consolidated-cash-flow-statement.normalized.json",
        "fields": [
            "depreciation_and_amortization",
        ],
    },
}


def parse_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Parse statement filename into company ID and statement type."""
    for statement_type, config in STATEMENT_CONFIGS.items():
        suffix = config["suffix"]
        if filename.endswith(suffix):
            company_id = filename[: -len(suffix)].strip()
            if not company_id:
                return None
            return company_id, statement_type
    return None


def parse_normalized_filename(filename: str) -> Optional[Tuple[str, str]]:
    """Parse normalized filename into company ID and statement type."""
    for statement_type, config in STATEMENT_CONFIGS.items():
        output_suffix = config["output_suffix"]
        if filename.endswith(output_suffix):
            company_id = filename[: -len(output_suffix)].strip()
            if not company_id:
                return None
            return company_id, statement_type
    return None


def output_filename_for_source(source_filename: str, statement_type: str) -> str:
    """Build normalized output filename from a source filename."""
    config = STATEMENT_CONFIGS[statement_type]
    suffix = config["suffix"]
    output_suffix = config["output_suffix"]
    if source_filename.endswith(suffix):
        base = source_filename[: -len(suffix)]
        return f"{base}{output_suffix}"
    return f"{source_filename}.normalized.json"


def all_statement_fields() -> List[str]:
    """Return all configured statement fields without duplicates."""
    fields: List[str] = []
    for config in STATEMENT_CONFIGS.values():
        for field in config["fields"]:
            if field not in fields:
                fields.append(field)
    return fields
