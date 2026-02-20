"""
Extract normalized balance-sheet fields from `extracted_text` using Azure DeepSeek.

This script reads files named:
    <company_id>.consolidated-balance-sheet.llm.json

For each file, it sends the extracted raw statement text to Azure DeepSeek via
`AzureDeepSeekClient` and writes one normalized JSON output file containing:
    - total assets
    - advance payments for purchases
    - account receivables
    - inventory
    - cash
    - investment in securities
    - total liabilities
    - total non-current liabilities
    - total current liabilities
    - short-term debt
    - long-term debt
    - account payables_and_or_other_liabilities
    - advance payments from sales
    - total equity
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from azure_balance_sheet_model import AzureDeepSeekClient


REQUIRED_FIELDS = [
    "total_assets",
    "advance_payments_for_purchases",
    "accounts_receivable",
    "inventory",
    "cash",
    "investment_in_securities",
    "total_liabilities",
    "total_non_current_liabilities",
    "total_current_liabilities",
    "accounts_payable_and_or_other_liabilities",
    "advance_payments_from_sales",
    "short_term_debt",
    "long_term_debt",
    "total_equity",
]


def parse_filename(filename: str) -> Optional[str]:
    suffix = ".consolidated-balance-sheet.llm.json"
    if not filename.endswith(suffix):
        return None
    company_id = filename[: -len(suffix)].strip()
    if not company_id:
        return None
    return company_id


def output_filename_for_source(source_filename: str) -> str:
    suffix = ".consolidated-balance-sheet.llm.json"
    if source_filename.endswith(suffix):
        base = source_filename[: -len(suffix)]
        return f"{base}.consolidated-balance-sheet.normalized.json"
    return f"{source_filename}.normalized.json"


def load_raw_statement(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    extraction = payload.get("extraction", {})
    raw = extraction.get("raw_response", "")
    if isinstance(raw, str):
        return raw
    return str(raw)


def build_prompt(company_id: str, statement_text: str) -> str:
    return (
        "You are a financial statement extraction engine.\n"
        "Extract normalized consolidated balance sheet values from the statement text.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "company_id": "string",\n'
        '  "periods": [\n'
        "    {\n"
        '      "period": "string",\n'
        '      "values": {\n'
        '        "total_assets": number|null,\n'
        '        "advance_payments_for_purchases": number|null,\n'
        '        "accounts_receivable": number|null,\n'
        '        "inventory": number|null,\n'
        '        "cash": number|null,\n'
        '        "investment_in_securities": number|null,\n'
        '        "total_liabilities": number|null,\n'
        '        "total_non_current_liabilities": number|null,\n'
        '        "total_current_liabilities": number|null,\n'
        '        "short_term_debt": number|null,\n'
        '        "long_term_debt": number|null,\n'
        '        "accounts_payable_and_or_other_liabilities": number|null,\n'
        '        "advance_payments_from_sales": number|null,\n'
        '        "total_equity": number|null\n'
        "      }\n"
        "    }\n"
        "  ],\n"
        '  "notes": string\n'
        "}\n\n"
        "Rules:\n"
        "1) Use every period/column available in the statement.\n"
        "2) Return numeric values only (no commas, no currency symbols, no percent signs).\n"
        "3) If a value cannot be directly mapped to a field, try to map number(s) to the corresponding field by taking into account the industry the company is in, and note what you did in notes field. If you cannot find a match, return null.\n"
        "4) If there are multiple close synonyms, use best accounting match.\n"
        "5) For parentheses negatives, return negative numbers.\n"
        "6) Keep the period text exactly as shown in the statement.\n"
        "7) Return JSON only.\n\n"
        f"company_id: {company_id}\n\n"
        "statement_text:\n"
        f"{statement_text}\n"
    )


def to_float_or_none(value: Any) -> Optional[float]:
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
    if lowered in {"null", "none", "na", "n/a", "-", "--", "—"}:
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


def normalize_periods(periods: Any) -> List[Dict[str, Any]]:
    if not isinstance(periods, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for period_item in periods:
        if not isinstance(period_item, dict):
            continue
        period_label = str(period_item.get("period", "")).strip()
        values = period_item.get("values", {})
        if not isinstance(values, dict):
            values = {}

        normalized_values: Dict[str, Optional[float]] = {}
        for field in REQUIRED_FIELDS:
            normalized_values[field] = to_float_or_none(values.get(field))

        normalized.append(
            {
                "period": period_label,
                "values": normalized_values,
            }
        )
    return normalized


def extract_one_statement(
    client: AzureDeepSeekClient,
    company_id: str,
    statement_text: str,
    parameters: Dict[str, Any],
    message: str,
) -> Dict[str, Any]:
    prompt = build_prompt(company_id=company_id, statement_text=statement_text)
    result = client.ask_json(
        message=message,
        prompt=prompt,
        parameters=parameters,
        reasoning=True,
    )
    print(result)
    normalized_periods = normalize_periods(result.get("periods"))
    return {
        "company_id": str(result.get("company_id", company_id)),
        "statement_type": "consolidated-balance-sheet",
        "periods": normalized_periods,
        "notes": str(result.get("notes", "")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract normalized balance-sheet fields from extracted_text "
            "using Azure DeepSeek."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="extracted_text",
        help="Directory containing *.consolidated-balance-sheet.llm.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="deepseek_balance_sheets",
        help="Output directory to write one normalized JSON per balance sheet file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Maximum tokens for model response.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--message",
        default="gen-ai-response",
        help="Message field sent to the Azure DeepSeek endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input dir: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.consolidated-balance-sheet.llm.json"))
    if not files:
        raise ValueError(f"No consolidated balance sheet files found in: {input_dir}")

    client = AzureDeepSeekClient()
    parameters = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
    }

    wrote = 0
    failed = 0
    for idx, path in enumerate(files, start=1):
        company_id = parse_filename(path.name)
        if not company_id:
            continue
        statement_text = load_raw_statement(path)
        print(f"[{idx}/{len(files)}] Extracting {path.name}")
        try:
            extracted = extract_one_statement(
                client=client,
                company_id=company_id,
                statement_text=statement_text,
                parameters=parameters,
                message=args.message,
            )
            extracted["source_file"] = str(path)
            single_payload = {
                "schema_version": "1.0",
                "fields": REQUIRED_FIELDS,
                "statement": extracted,
            }
            output_path = output_dir / output_filename_for_source(path.name)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(single_payload, f, ensure_ascii=False, indent=2)
            wrote += 1
            print(f"  -> Wrote {output_path}")
        except Exception as exc:
            failed += 1
            print(f"  -> Failed for {path.name}: {exc}")

    print(f"Finished. Wrote {wrote} file(s) to {output_dir}. Failed: {failed}")


if __name__ == "__main__":
    main()
