"""LLM extraction workflow for financial statement normalization."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from azure_llm_client import AzureLLMClient

from .statement_config import (
    SCHEMA_VERSION,
    ExtractionCounts,
    STATEMENT_CONFIGS,
    output_filename_for_source,
    parse_filename,
)
from .statement_normalization import build_prompt, load_raw_statement, normalize_periods
from .statement_runs import get_next_run_number


def extract_one_statement(
    client: AzureLLMClient,
    company_id: str,
    statement_type: str,
    required_fields: List[str],
    statement_and_supplementary_tables: str,
    parameters: Dict[str, Any],
    message: str,
) -> Dict[str, Any]:
    """Extract and normalize one statement using the configured LLM client."""
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
    normalized_periods = normalize_periods(
        result.get("periods"), required_fields=required_fields
    )
    return {
        "company_id": str(result.get("company_id", company_id)),
        "statement_type": statement_type,
        "periods": normalized_periods,
        "notes": str(result.get("notes", "")),
    }


def build_model_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    """Build model parameter payload sent to the LLM endpoint."""
    return {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
    }


def process_statement_file(
    idx: int,
    total_files: int,
    path: Path,
    output_dir: Path,
    parameters: Dict[str, Any],
    message: str,
) -> str:
    """Process one raw statement file and write normalized output."""
    parsed = parse_filename(path.name)
    if not parsed:
        return "skipped"

    company_id, statement_type = parsed
    required_fields = STATEMENT_CONFIGS[statement_type]["fields"]
    statement_and_supplementary_tables = load_raw_statement(path)
    print(f"[{idx}/{total_files}] Extracting {path.name} ({statement_type})")

    try:
        client = AzureLLMClient()
        extracted = extract_one_statement(
            client=client,
            company_id=company_id,
            statement_type=statement_type,
            required_fields=required_fields,
            statement_and_supplementary_tables=statement_and_supplementary_tables,
            parameters=parameters,
            message=message,
        )
        extracted["source_file"] = str(path)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "fields": required_fields,
            "statement": extracted,
        }
        output_path = output_dir / output_filename_for_source(
            path.name, statement_type=statement_type
        )
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  -> Wrote {output_path}")
        return "wrote"
    except Exception as exc:
        print(f"  -> Failed for {path.name}: {exc}")
        return "failed"


def run_extraction_once(
    input_dir: Path,
    output_dir: Path,
    parameters: Dict[str, Any],
    message: str,
    max_workers: int,
) -> ExtractionCounts:
    """Run one extraction pass over all source statement files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.llm.json"))
    if not files:
        raise ValueError(f"No statement files found in: {input_dir}")

    wrote = 0
    failed = 0
    if max_workers == 1:
        for idx, path in enumerate(files, start=1):
            status = process_statement_file(
                idx=idx,
                total_files=len(files),
                path=path,
                output_dir=output_dir,
                parameters=parameters,
                message=message,
            )
            if status == "wrote":
                wrote += 1
            elif status == "failed":
                failed += 1
        return wrote, failed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_statement_file,
                idx,
                len(files),
                path,
                output_dir,
                parameters,
                message,
            )
            for idx, path in enumerate(files, start=1)
        ]
        for future in as_completed(futures):
            status = future.result()
            if status == "wrote":
                wrote += 1
            elif status == "failed":
                failed += 1
    return wrote, failed


def run_extraction_pipeline(args: argparse.Namespace) -> None:
    """Run extraction workflow for single-run or multi-run mode."""
    if args.skip_extraction:
        return

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input dir: {input_dir}")

    parameters = build_model_parameters(args)
    if args.num_extraction_runs == 1:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        wrote, failed = run_extraction_once(
            input_dir=input_dir,
            output_dir=output_dir,
            parameters=parameters,
            message=args.message,
            max_workers=args.max_workers,
        )
        print(
            f"Extraction finished. Wrote {wrote} file(s) to {output_dir}. Failed: {failed}"
        )
        return

    runs_root = Path(args.runs_output_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    start_run_number = get_next_run_number(
        runs_root=runs_root,
        append_runs=not args.no_append_runs,
    )
    for run_idx in range(1, args.num_extraction_runs + 1):
        run_number = start_run_number + run_idx - 1
        run_dir = runs_root / f"run_{run_number:02d}"
        print(
            f"Starting extraction run {run_idx}/{args.num_extraction_runs} "
            f"(folder run_{run_number:02d}): {run_dir}"
        )
        wrote, failed = run_extraction_once(
            input_dir=input_dir,
            output_dir=run_dir,
            parameters=parameters,
            message=args.message,
            max_workers=args.max_workers,
        )
        print(
            f"Run {run_idx}/{args.num_extraction_runs} (run_{run_number:02d}) finished. "
            f"Wrote {wrote} file(s) to {run_dir}. Failed: {failed}"
        )
