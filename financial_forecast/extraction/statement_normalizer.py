"""LLM-driven normalization of extracted financial statements.

Provides :class:`StatementNormalizer`, which reads raw extracted statement
files (``*.llm.json``), sends them to an LLM for field normalization,
and writes structured output (``*.normalized.json``).
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from financial_forecast.clients.azure_llm_client import AzureLLMClient
from financial_forecast.extraction.statement_config import (
    SCHEMA_VERSION,
    ExtractionCounts,
    STATEMENT_CONFIGS,
    StatementType,
    output_filename_for_source,
    parse_filename,
)
from financial_forecast.extraction.statement_normalization import (
    build_prompt,
    load_raw_statement,
    normalize_periods,
)
from financial_forecast.extraction.statement_runs import get_next_run_number


class StatementNormalizer:
    """Normalize raw extracted statements into structured JSON.

    Args:
        input_dir: Directory containing ``*.llm.json`` files.
        output_dir: Directory for normalized output files.
        temperature: LLM temperature parameter.
        max_tokens: Maximum tokens for LLM response.
        top_k: Top-k sampling parameter.
        max_workers: Number of files to process in parallel.
        message: Message field sent to the LLM endpoint.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        temperature: float = 0.0,
        max_tokens: int = 100000,
        top_k: int = 1,
        max_workers: int = 9,
        message: str = "gen-ai-response",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.parameters = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
        }
        self.max_workers = max_workers
        self.message = message

    def run(self) -> ExtractionCounts:
        """Normalize all ``*.llm.json`` files in input_dir.

        Returns:
            Tuple ``(wrote, failed)`` counts.
        """
        return self._run_once(self.input_dir, self.output_dir)

    def run_multi(
        self,
        num_runs: int,
        runs_output_dir: str,
        append: bool = True,
    ) -> None:
        """Run multiple normalization passes for median aggregation.

        Args:
            num_runs: Number of extraction runs to execute.
            runs_output_dir: Root directory for ``run_01/``, ``run_02/``, ...
            append: If ``True``, continue numbering from existing runs.
        """
        runs_root = Path(runs_output_dir)
        runs_root.mkdir(parents=True, exist_ok=True)
        start_run_number = get_next_run_number(
            runs_root=runs_root,
            append_runs=append,
        )
        for run_idx in range(1, num_runs + 1):
            run_number = start_run_number + run_idx - 1
            run_dir = runs_root / f"run_{run_number:02d}"
            print(
                f"Starting extraction run {run_idx}/{num_runs} "
                f"(folder run_{run_number:02d}): {run_dir}"
            )
            wrote, failed = self._run_once(self.input_dir, run_dir)
            print(
                f"Run {run_idx}/{num_runs} (run_{run_number:02d}) "
                f"finished. Wrote {wrote} file(s). Failed: {failed}"
            )

    def _run_once(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> ExtractionCounts:
        """Run one extraction pass over all source statement files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(input_dir.glob("*.llm.json"))
        if not files:
            raise ValueError(f"No statement files found in: {input_dir}")

        wrote = 0
        failed = 0

        if self.max_workers == 1:
            for idx, path in enumerate(files, start=1):
                status = self._process_one_file(
                    idx,
                    len(files),
                    path,
                    output_dir,
                )
                if status == "wrote":
                    wrote += 1
                elif status == "failed":
                    failed += 1
            return wrote, failed

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._process_one_file,
                    idx,
                    len(files),
                    path,
                    output_dir,
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

    def _process_one_file(
        self,
        idx: int,
        total_files: int,
        path: Path,
        output_dir: Path,
    ) -> str:
        """Process one raw statement file and write normalized output."""
        parsed = parse_filename(path.name)
        if not parsed:
            return "skipped"

        company_id, statement_type = parsed
        required_fields = STATEMENT_CONFIGS[statement_type]["fields"]
        statement_and_supplementary_tables = load_raw_statement(path)
        print(f"[{idx}/{total_files}] Extracting {path.name} " f"({statement_type})")

        try:
            extracted = self._extract_one_statement(
                company_id,
                statement_type,
                required_fields,
                statement_and_supplementary_tables,
            )
            extracted["source_file"] = str(path)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "fields": required_fields,
                "statement": extracted,
            }
            output_path = output_dir / output_filename_for_source(
                path.name,
                statement_type=statement_type,
            )
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"  -> Wrote {output_path}")
            return "wrote"
        except Exception as exc:
            print(f"  -> Failed for {path.name}: {exc}")
            return "failed"

    def _extract_one_statement(
        self,
        company_id: str,
        statement_type: StatementType,
        required_fields: List[str],
        statement_and_supplementary_tables: str,
    ) -> Dict[str, Any]:
        """Extract and normalize one statement using the LLM."""
        prompt = build_prompt(
            company_id=company_id,
            statement_type=statement_type,
            required_fields=required_fields,
            statement_and_supplementary_tables=(statement_and_supplementary_tables),
        )
        client = AzureLLMClient()
        result = client.ask_json(
            message=self.message,
            prompt=prompt,
            parameters=self.parameters,
            reasoning=True,
        )
        normalized_periods = normalize_periods(
            result.get("periods"),
            required_fields=required_fields,
        )
        return {
            "company_id": str(result.get("company_id", company_id)),
            "statement_type": statement_type.value,
            "periods": normalized_periods,
            "notes": str(result.get("notes", "")),
        }
