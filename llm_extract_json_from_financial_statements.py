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
        "2) Return an array of maps that make up each field where each map is {source_label: number} (source_label = actual label from the statement; no commas, no currency symbols, no percent signs in numbers).\n"
        "3) Ensure that the elements in the array correctly make up the total value of the field.\n"
        "4) If multiple elements need to be combined to make up a field, return all of them individually in an array.\n"
        "5) If a value(s) cannot be directly mapped to a field, try to map element(s) to the corresponding field by taking into account the industry the company is in, and note your reasoning in the notes field. If you cannot find a match, return an empty array.\n"
        "6) If there are multiple close synonyms, use best accounting match.\n"
        "7) For year, report the year of the period only.\n"
        "8) For currency, report its formal 3-letter acronym.\n"
        "9) For scale, report the scale of the values in the statement. For example, if the values are in millions, the scale should be 1E6.\n"
        "10) For total_operating_cost, list all elements that should be included in standard practice for the industry the company is in.\n"
        "11) For taxes, it is the tax assessed on the income. If a tax belongs to the cost of revenue, it should be part of total_operating_cost.\n"
        "12) Generally, numbers in parentheses are negative.\n"
        "13) For taxes, interest_expenses, and total_operating_cost, the sign convention is the opposite: if an element reduces income, it should be positive; and if it increases income, it should be negative.\n"
        "14) For marketable_securities, these are most liquid, unrestricted debt or equity investments intended to be sold in the near term (e.g., U.S. Treasuries, commercial paper, money market funds, publicly traded equities)\n"
        "15) Return JSON only.\n\n"
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


def parse_normalized_filename(filename: str) -> Optional[Tuple[str, str]]:
    for statement_type, config in STATEMENT_CONFIGS.items():
        output_suffix = config["output_suffix"]
        if filename.endswith(output_suffix):
            company_id = filename[: -len(output_suffix)].strip()
            if not company_id:
                return None
            return company_id, statement_type
    return None


def sum_labeled_entries(items: Any) -> float:
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


def safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def sum_if_all_present(*values: Optional[float]) -> Optional[float]:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def statement_period_to_single_values(period: Dict[str, Any], fields: List[str]) -> Dict[str, float]:
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


def calculate_financial_ratios(
    flat_values: Dict[str, Optional[float]]
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    revenue = flat_values.get("total_revenue")
    operating_cost = flat_values.get("total_operating_cost")
    cash_and_cash_equivalents = flat_values.get("cash_and_cash_equivalents")
    marketable_securities = flat_values.get("marketable_securities")
    total_accounts_receivable = flat_values.get("total_accounts_receivable")
    total_current_liabilities = flat_values.get("total_current_liabilities")
    total_debt = flat_values.get("total_debt_short_term_and_long_term")
    total_equity = flat_values.get("total_equity")
    total_assets = flat_values.get("total_assets")
    net_income = flat_values.get("net_income")
    taxes = flat_values.get("taxes")
    interest_expenses = flat_values.get("interest_expenses")
    depreciation_and_amortization = flat_values.get("depreciation_and_amortization")

    ebit = sum_if_all_present(net_income, interest_expenses, taxes)
    ebitda = sum_if_all_present(ebit, depreciation_and_amortization)
    quick_assets = sum_if_all_present(
        cash_and_cash_equivalents, marketable_securities, total_accounts_receivable
    )
    debt_plus_equity = sum_if_all_present(total_debt, total_equity)

    ratios = {
        "cost_to_income_ratio": safe_divide(operating_cost, revenue),
        "quick_ratio": safe_divide(quick_assets, total_current_liabilities),
        "debt_to_equity_ratio": safe_divide(total_debt, total_equity),
        "debt_to_assets_ratio": safe_divide(total_debt, total_assets),
        "debt_to_capital_ratio": safe_divide(total_debt, debt_plus_equity),
        "debt_to_ebitda_ratio": safe_divide(total_debt, ebitda),
        "interest_coverage_ratio": safe_divide(ebit, interest_expenses),
    }
    derived_values = {
        "ebit": ebit,
        "ebitda": ebitda,
        "quick_assets": quick_assets,
        "debt_plus_equity": debt_plus_equity,
    }
    return derived_values, ratios


def compute_ratios_from_normalized_files(input_dir: Path, output_file: Path) -> int:
    files = sorted(input_dir.glob("*.normalized.json"))
    if not files:
        raise ValueError(f"No normalized files found in: {input_dir}")

    company_periods: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for path in files:
        parsed = parse_normalized_filename(path.name)
        if not parsed:
            continue
        filename_company_id, statement_type = parsed
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        statement = payload.get("statement", {})
        if not isinstance(statement, dict):
            continue
        company_id = str(statement.get("company_id", filename_company_id)).strip() or filename_company_id
        periods = statement.get("periods", [])
        if not isinstance(periods, list):
            continue

        fields = payload.get("fields")
        if not isinstance(fields, list):
            fields = STATEMENT_CONFIGS[statement_type]["fields"]
        fields = [str(field) for field in fields]

        company_years = company_periods.setdefault(company_id, {})
        for period in periods:
            if not isinstance(period, dict):
                continue
            year = str(period.get("year", "")).strip()
            if not year:
                continue
            year_record = company_years.setdefault(
                year,
                {
                    "currency": "",
                    "field_values": {},
                    "present_fields": set(),
                },
            )
            currency = str(period.get("currency", "")).strip()
            if currency:
                year_record["currency"] = currency

            single_values = statement_period_to_single_values(period, fields=fields)
            year_record["field_values"].update(single_values)
            year_record["present_fields"].update(fields)

    companies_output: List[Dict[str, Any]] = []
    total_periods = 0
    for company_id in sorted(company_periods):
        periods_out: List[Dict[str, Any]] = []
        for year in sorted(company_periods[company_id]):
            record = company_periods[company_id][year]
            present_fields = record.get("present_fields", set())
            if not isinstance(present_fields, set):
                present_fields = set()
            field_values: Dict[str, Optional[float]] = {}
            for config in STATEMENT_CONFIGS.values():
                for field in config["fields"]:
                    if field in present_fields:
                        field_values[field] = float(record["field_values"].get(field, 0.0))
                    else:
                        field_values[field] = None
            derived_values, ratios = calculate_financial_ratios(field_values)
            periods_out.append(
                {
                    "year": year,
                    "currency": record.get("currency", ""),
                    "field_values": field_values,
                    "derived_values": derived_values,
                    "ratios": ratios,
                }
            )
            total_periods += 1
        companies_output.append({"company_id": company_id, "periods": periods_out})

    output_payload = {
        "schema_version": "1.0",
        "source_dir": str(input_dir),
        "companies": companies_output,
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)
    print(
        f"Finished ratios. Wrote {len(companies_output)} company file(s), {total_periods} period(s) to {output_file}"
    )
    return total_periods


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
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip LLM extraction and only run ratio calculation (if enabled).",
    )
    parser.add_argument(
        "--compute-ratios",
        action="store_true",
        help="Compute single-value fields and financial ratios from *.normalized.json files.",
    )
    parser.add_argument(
        "--ratios-input-dir",
        default=None,
        help=(
            "Input directory containing *.normalized.json files. "
            "Defaults to --output-dir."
        ),
    )
    parser.add_argument(
        "--ratios-output-file",
        default="financial_ratios.json",
        help="Output JSON file path for computed ratios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.skip_extraction:
        input_dir = Path(args.input_dir)
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
        print(f"Extraction finished. Wrote {wrote} file(s) to {output_dir}. Failed: {failed}")

    if args.compute_ratios:
        ratios_input_dir = Path(args.ratios_input_dir) if args.ratios_input_dir else output_dir
        if not ratios_input_dir.exists() or not ratios_input_dir.is_dir():
            raise ValueError(f"Invalid ratios input dir: {ratios_input_dir}")
        ratios_output_file = Path(args.ratios_output_file)
        if not ratios_output_file.is_absolute():
            ratios_output_file = ratios_input_dir / ratios_output_file
        ratios_output_file.parent.mkdir(parents=True, exist_ok=True)
        compute_ratios_from_normalized_files(
            input_dir=ratios_input_dir,
            output_file=ratios_output_file,
        )


if __name__ == "__main__":
    main()
