"""
Extract normalized financial statement fields from `extracted_text` using Azure DeepSeek.

This script reads files named:
    <company_id>.consolidated-balance-sheet.llm.json
    <company_id>.consolidated-income-statement.llm.json
    <company_id>.consolidated-cash-flow-statement.llm.json

For each file, it sends the extracted raw statement text to Azure DeepSeek and writes
one normalized JSON output file for the matching statement type.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from azure_llm_client import AzureLLMClient


STATEMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "consolidated-balance-sheet": {
        "suffix": ".consolidated-balance-sheet.llm.json",
        "output_suffix": ".consolidated-balance-sheet.normalized.json",
        "fields": [
            "cash_and_cash_equivalents",
            "marketable_securities",
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
            "revenue",
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
    for statement_type, config in STATEMENT_CONFIGS.items():
        suffix = config["suffix"]
        if filename.endswith(suffix):
            company_id = filename[: -len(suffix)].strip()
            if not company_id:
                return None
            return company_id, statement_type
    return None


def output_filename_for_source(source_filename: str, statement_type: str) -> str:
    config = STATEMENT_CONFIGS[statement_type]
    suffix = config["suffix"]
    output_suffix = config["output_suffix"]
    if source_filename.endswith(suffix):
        base = source_filename[: -len(suffix)]
        return f"{base}{output_suffix}"
    return f"{source_filename}.normalized.json"


def load_raw_statement(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    extraction = payload.get("extraction", {})
    if isinstance(extraction, str):
        return extraction
    return str(extraction)


def build_prompt(
    company_id: str,
    statement_type: str,
    required_fields: List[str],
    statement_and_supplementary_tables: str,
) -> str:
    fields_schema = ",\n".join(
        [
            f'        "{field}": [{{"source_label": number}}]'
            for field in required_fields
        ]
    )
    prompt = (
        "You are a financial statement extraction engine.\n"
        f"Extract normalized {statement_type} values from the statement and supplementary tables.\n\n"
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
        "2) Return an array of maps for each field where each map is {source_label: number} where source_label is the direct copy of the label from the statement and the number is the value of the element (no commas, no currency symbols, no percent signs in numbers).\n"
        "3) If multiple elements need to be combined to make up a field, return all of them individually in an array.\n"
        "4) If a value(s) cannot be directly mapped to a field, try to map element(s) to the corresponding field by taking into account the industry the company is in, and note your reasoning in the notes field. If you cannot find a match, return an empty array.\n"
        "5) If there are multiple close synonyms, use best accounting match.\n"
        "6) For year, report the year of the period only.\n"
        "7) For currency, report its formal 3-letter acronym.\n"
        "8) For scale, report the scale of the values in the statement. For example, if the values are in millions, the scale should be 1E6.\n"
        "9) For total_operating_cost, list all elements that should be included in standard practice for the industry the company is in.\n"
        "10) For taxes, it is the tax assessed on the income. If a tax belongs to the cost of revenue, it should be part of total_operating_cost.\n"
        "11) For taxes, interest_expenses, and total_operating_cost, the sign convention should be such that if an element REDUCES income, it should be POSITIVE, and if it INCREASES income, it should be NEGATIVE. Otherwise, generally, numbers in parentheses are negative.\n"
        "12) For marketable_securities, it includes all current liquid assets (excluding cash and cash equivalents) that can be easily converted to cash.\n"
        "13) Return JSON only.\n\n"
        f"company_id: {company_id}\n\n"
        "statement and supplementary tables:\n"
        f"{statement_and_supplementary_tables}\n"
    )
    return prompt


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


def to_float_list(value: Any) -> List[float]:
    if isinstance(value, list):
        normalized: List[float] = []
        for item in value:
            parsed = to_float_or_none(item)
            if parsed is not None:
                normalized.append(parsed)
        return normalized

    parsed_scalar = to_float_or_none(value)
    if parsed_scalar is None:
        return []
    return [parsed_scalar]


def to_labeled_float_list(value: Any) -> List[Dict[str, float]]:
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

            # Backward compatibility if the model still emits a bare number array.
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


def extract_one_statement(
    client: AzureLLMClient,
    company_id: str,
    statement_type: str,
    required_fields: List[str],
    statement_and_supplementary_tables: str,
    parameters: Dict[str, Any],
    message: str,
) -> Dict[str, Any]:
    prompt = build_prompt(
        company_id=company_id,
        statement_type=statement_type,
        required_fields=required_fields,
        statement_and_supplementary_tables=statement_and_supplementary_tables,
    )
    result = client.ask_json(
        message=message,
        prompt=prompt,
        parameters=parameters,
        reasoning=True,
    )
    print(result)
    normalized_periods = normalize_periods(
        result.get("periods"), required_fields=required_fields
    )
    return {
        "company_id": str(result.get("company_id", company_id)),
        "statement_type": statement_type,
        "periods": normalized_periods,
        "notes": str(result.get("notes", "")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract normalized financial-statement fields from extracted_text "
            "using Azure DeepSeek."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="extracted_text",
        help=(
            "Directory containing *.consolidated-balance-sheet.llm.json, "
            "*.consolidated-income-statement.llm.json, and "
            "*.consolidated-cash-flow-statement.llm.json files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="deepseek_financial_statements",
        help="Output directory to write one normalized JSON per statement file.",
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=9,
        help=(
            "Number of statement files to process in parallel. "
            "Set to 1 to run sequentially."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input dir: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.llm.json"))
    if not files:
        raise ValueError(f"No statement files found in: {input_dir}")

    parameters = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
    }

    def process_file(idx: int, path: Path) -> str:
        parsed = parse_filename(path.name)
        if not parsed:
            return "skipped"
        company_id, statement_type = parsed
        required_fields = STATEMENT_CONFIGS[statement_type]["fields"]
        statement_and_supplementary_tables = load_raw_statement(path)
        print(f"[{idx}/{len(files)}] Extracting {path.name} ({statement_type})")
        try:
            client = AzureLLMClient()
            extracted = extract_one_statement(
                client=client,
                company_id=company_id,
                statement_type=statement_type,
                required_fields=required_fields,
                statement_and_supplementary_tables=statement_and_supplementary_tables,
                parameters=parameters,
                message=args.message,
            )
            extracted["source_file"] = str(path)
            single_payload = {
                "schema_version": "1.0",
                "fields": required_fields,
                "statement": extracted,
            }
            output_path = output_dir / output_filename_for_source(
                path.name, statement_type=statement_type
            )
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(single_payload, f, ensure_ascii=False, indent=2)
            print(f"  -> Wrote {output_path}")
            return "wrote"
        except Exception as exc:
            print(f"  -> Failed for {path.name}: {exc}")
            return "failed"

    wrote = 0
    failed = 0
    if args.max_workers <= 1:
        for idx, path in enumerate(files, start=1):
            status = process_file(idx, path)
            if status == "wrote":
                wrote += 1
            elif status == "failed":
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_file, idx, path)
                for idx, path in enumerate(files, start=1)
            ]
            for future in as_completed(futures):
                status = future.result()
                if status == "wrote":
                    wrote += 1
                elif status == "failed":
                    failed += 1

    print(f"Finished. Wrote {wrote} file(s) to {output_dir}. Failed: {failed}")


if __name__ == "__main__":
    main()
