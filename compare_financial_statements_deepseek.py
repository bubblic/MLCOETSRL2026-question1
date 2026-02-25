"""
Compare extracted financial statements against backup statements with DeepSeek.
This is to test how robust the DeepSeek model is to multiple extraction runs.

For each matching statement file in `extracted_text` and `extracted_text_backup`,
this script sends exactly one pair to Azure DeepSeek and asks it to identify
discrepancies. Results are flattened into a CSV file.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from azure_llm_client import AzureLLMClient


SUPPORTED_STATEMENTS = {
    "consolidated-balance-sheet",
    "consolidated-cash-flow-statement",
    "consolidated-income-statement",
}


def parse_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse '<company>.<statement>.llm.json' into (company, statement).
    """
    if not filename.endswith(".llm.json"):
        return None
    stem = filename[: -len(".llm.json")]
    parts = stem.rsplit(".", 1)
    if len(parts) != 2:
        return None
    company_id, statement_type = parts
    if statement_type not in SUPPORTED_STATEMENTS:
        return None
    return company_id, statement_type


def load_raw_statement(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    extraction = data.get("extraction", {})
    raw = extraction.get("raw_response", "")
    if not isinstance(raw, str):
        return str(raw)
    return raw


def build_prompt(
    company_id: str,
    statement_type: str,
    extracted_statement: str,
    backup_statement: str,
) -> str:
    return (
        "You are a financial statement comparator.\n"
        "Compare these two versions of the same statement and report discrepancies.\n\n"
        "Rules:\n"
        "1) Treat semantically equivalent formatting differences as NO discrepancy "
        "(markdown style, bolding, spacing, bullet style, punctuation).\n"
        "2) Report ONLY meaningful differences: missing line items, changed numeric "
        "values, changed years/period labels, changed sign, changed units/currency, "
        "or materially altered wording.\n"
        "3) If there are no meaningful differences, return has_discrepancy=false and "
        "an empty discrepancies array.\n"
        "4) Return valid JSON only.\n\n"
        "Return schema:\n"
        "{\n"
        '  "company_id": "string",\n'
        '  "statement_type": "string",\n'
        '  "has_discrepancy": true|false,\n'
        '  "discrepancies": [\n'
        "    {\n"
        '      "category": "value_mismatch|missing_line_item|period_mismatch|'
        'unit_or_currency_mismatch|wording_mismatch|other",\n'
        '      "line_item": "string",\n'
        '      "field_or_period": "string",\n'
        '      "extracted_text_value": "string",\n'
        '      "backup_text_value": "string",\n'
        '      "explanation": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"company_id: {company_id}\n"
        f"statement_type: {statement_type}\n\n"
        "version_a_name: extracted_text\n"
        "version_a_statement:\n"
        f"{extracted_statement}\n\n"
        "version_b_name: extracted_text_backup\n"
        "version_b_statement:\n"
        f"{backup_statement}\n"
    )


def compare_one_pair(
    client: AzureLLMClient,
    company_id: str,
    statement_type: str,
    extracted_statement: str,
    backup_statement: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = build_prompt(
        company_id=company_id,
        statement_type=statement_type,
        extracted_statement=extracted_statement,
        backup_statement=backup_statement,
    )
    result = client.ask_json(
        message="gen-ai-response",
        prompt=prompt,
        parameters=parameters,
        reasoning=True,
    )
    if "has_discrepancy" not in result:
        result["has_discrepancy"] = bool(result.get("discrepancies"))
    if "discrepancies" not in result or not isinstance(
        result.get("discrepancies"), list
    ):
        result["discrepancies"] = []
    result.setdefault("company_id", company_id)
    result.setdefault("statement_type", statement_type)
    return result


def flatten_for_csv(
    model_result: Dict[str, Any],
    file_name: str,
    extracted_path: Path,
    backup_path: Path,
) -> List[Dict[str, str]]:
    company_id = str(model_result.get("company_id", ""))
    statement_type = str(model_result.get("statement_type", ""))
    has_discrepancy = bool(model_result.get("has_discrepancy", False))
    discrepancies = model_result.get("discrepancies", [])
    if not isinstance(discrepancies, list):
        discrepancies = []

    common = {
        "company_id": company_id,
        "statement_type": statement_type,
        "file_name": file_name,
        "extracted_path": str(extracted_path),
        "backup_path": str(backup_path),
        "has_discrepancy": str(has_discrepancy).lower(),
    }

    if not has_discrepancy or len(discrepancies) == 0:
        return [
            {
                **common,
                "category": "none",
                "line_item": "",
                "field_or_period": "",
                "extracted_text_value": "",
                "backup_text_value": "",
                "explanation": "No discrepancy found.",
            }
        ]

    rows: List[Dict[str, str]] = []
    for d in discrepancies:
        if not isinstance(d, dict):
            continue
        rows.append(
            {
                **common,
                "category": str(d.get("category", "other")),
                "line_item": str(d.get("line_item", "")),
                "field_or_period": str(d.get("field_or_period", "")),
                "extracted_text_value": str(d.get("extracted_text_value", "")),
                "backup_text_value": str(d.get("backup_text_value", "")),
                "explanation": str(d.get("explanation", "")),
            }
        )

    if not rows:
        rows.append(
            {
                **common,
                "category": "other",
                "line_item": "",
                "field_or_period": "",
                "extracted_text_value": "",
                "backup_text_value": "",
                "explanation": "Model returned malformed discrepancies payload.",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare extracted financial statements with backup statements using "
            "Azure DeepSeek and output discrepancy rows to CSV."
        )
    )
    parser.add_argument(
        "--extracted-dir",
        default="extracted_text",
        help="Directory containing primary extracted statement JSON files.",
    )
    parser.add_argument(
        "--backup-dir",
        default="extracted_text_backup",
        help="Directory containing backup extracted statement JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        default="deepseek_statement_discrepancies.csv",
        help="Output CSV file path.",
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
    args = parser.parse_args()

    extracted_dir = Path(args.extracted_dir)
    backup_dir = Path(args.backup_dir)
    output_csv = Path(args.output_csv)

    if not extracted_dir.exists() or not extracted_dir.is_dir():
        raise ValueError(f"Invalid extracted dir: {extracted_dir}")
    if not backup_dir.exists() or not backup_dir.is_dir():
        raise ValueError(f"Invalid backup dir: {backup_dir}")

    extracted_files = sorted(extracted_dir.glob("*.llm.json"))
    backup_lookup = {p.name: p for p in backup_dir.glob("*.llm.json")}

    client = AzureLLMClient()
    parameters = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
    }

    csv_rows: List[Dict[str, str]] = []
    total_pairs = 0

    for extracted_path in extracted_files:
        parsed = parse_filename(extracted_path.name)
        if parsed is None:
            continue
        company_id, statement_type = parsed
        backup_path = backup_lookup.get(extracted_path.name)
        if backup_path is None:
            csv_rows.append(
                {
                    "company_id": company_id,
                    "statement_type": statement_type,
                    "file_name": extracted_path.name,
                    "extracted_path": str(extracted_path),
                    "backup_path": "",
                    "has_discrepancy": "true",
                    "category": "missing_line_item",
                    "line_item": "",
                    "field_or_period": "",
                    "extracted_text_value": "present",
                    "backup_text_value": "missing file",
                    "explanation": "Missing matching file in extracted_text_backup.",
                }
            )
            continue

        extracted_statement = load_raw_statement(extracted_path)
        backup_statement = load_raw_statement(backup_path)
        total_pairs += 1
        print(f"[{total_pairs}] Comparing {extracted_path.name}")

        model_result = compare_one_pair(
            client=client,
            company_id=company_id,
            statement_type=statement_type,
            extracted_statement=extracted_statement,
            backup_statement=backup_statement,
            parameters=parameters,
        )
        csv_rows.extend(
            flatten_for_csv(
                model_result=model_result,
                file_name=extracted_path.name,
                extracted_path=extracted_path,
                backup_path=backup_path,
            )
        )

    fieldnames = [
        "company_id",
        "statement_type",
        "file_name",
        "extracted_path",
        "backup_path",
        "has_discrepancy",
        "category",
        "line_item",
        "field_or_period",
        "extracted_text_value",
        "backup_text_value",
        "explanation",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"Finished. Compared {total_pairs} paired statements and wrote "
        f"{len(csv_rows)} row(s) to {output_csv}"
    )


if __name__ == "__main__":
    main()
