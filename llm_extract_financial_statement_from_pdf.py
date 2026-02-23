"""
Example usage:
  python llm_extract_financial_statement_from_pdf.py --input-file ./annual_reports/alibaba_2025.pdf \\
    --query "Consolidated Balance Sheet" --query "Consolidated Income Statement"

Env vars:
  AZURE_DEEPSEEK_ENDPOINT for both page selection and table extraction
"""

# python llm_extract_financial_statement_from_pdf.py --input-file "./annual_reports/alibaba_2025.pdf"  --query "Consolidated Balance Sheet" --query "Consolidated Income Statement"
# python llm_extract_financial_statement_from_pdf.py --input-file "./annual_reports/2023 General Motors Annual Report .pdf" --query "Consolidated Balance Sheet" --query "Consolidated Income Statement"
# python llm_extract_financial_statement_from_pdf.py --input-file "./annual_reports/lvmh_dec_2024.pdf" --query "Consolidated Balance Sheet" --query "Consolidated Income Statement"

## To extract all three statements from all annual reports in the annual_reports directory, run the following in Terminal:
# python llm_extract_financial_statement_from_pdf.py --input-dir ./annual_reports --query "Consolidated Balance Sheet" --query "Consolidated Income Statement" --query "Consolidated Cash Flow Statement"


import argparse
import json
import os
import re
from pathlib import Path
from llm_pdf_pages_identifier import select_pages_with_llm, extract_json_from_text

from azure_llm_client import AzureLLMClient
from pdf_extractor.pdfplumber import extract_text_pdfplumber

WORKSPACE_DIR = Path(__file__).resolve().parent
ANNUAL_REPORTS_DIR = WORKSPACE_DIR / "annual_reports"
OUTPUT_DIR = WORKSPACE_DIR / "extracted_text"

EXTRACTORS = {
    "pdfplumber": extract_text_pdfplumber,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a long-context LLM to locate target table pages, then extract the "
            "table."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory to write extracted JSON files.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(ANNUAL_REPORTS_DIR),
        help="Directory containing annual report PDFs.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Single PDF file to extract instead of a whole directory.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Target table name to extract (repeatable).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Pages per prompt to the LLM.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="LLM endpoint override. Defaults to AZURE_DEEPSEEK_ENDPOINT.",
    )
    parser.add_argument(
        "--parameters",
        default='{"temperature": 0, "top_k": 1}',
        help="JSON string of parameters to send to the LLM.",
    )
    parser.add_argument(
        "--selection-prompt",
        default=None,
        help="Optional override for the page selection prompt template.",
    )
    parser.add_argument(
        "--extraction-prompt",
        default=None,
        help="Optional override for the table extraction prompt template.",
    )
    return parser.parse_args()


def parse_parameters(raw: str | None) -> dict[str, object]:
    if raw is None:
        return {}
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("Parameters must decode to a JSON object.")
    return loaded


def resolve_endpoint(override: str | None) -> str:
    endpoint = (override or os.getenv("AZURE_DEEPSEEK_ENDPOINT", "")).strip()
    if not endpoint:
        raise ValueError("Missing endpoint. Set AZURE_DEEPSEEK_ENDPOINT.")
    return endpoint


def build_extraction_prompt(query: str, pages_text: str) -> str:
    return (
        f"Find the table that corresponds to {query} and output it in a nice tabular form from the following pages data.\n"
        f"Pages:\n{pages_text}"
    )


def extract_table_with_llm(
    client: AzureLLMClient,
    parameters: dict[str, object],
    query: str,
    page_numbers: list[int],
    pages: dict[int, str | None],
    prompt_override: str | None = None,
) -> dict[str, object]:
    if not page_numbers:
        raise ValueError("No pages selected for extraction.")
    joined_pages = []
    for page_num in page_numbers:
        joined_pages.append(f"Page {page_num}:\n{(pages[page_num] or '').strip()}")
    pages_text = "\n\n---\n\n".join(joined_pages)
    prompt = (
        prompt_override.format(query=query, pages=pages_text)
        if prompt_override
        else build_extraction_prompt(query, pages_text)
    )
    response = client.ask_json(
        message="gen-ai-response", prompt=prompt, parameters=parameters, reasoning=True
    )
    if "raw_response" in response:
        extracted = extract_json_from_text(str(response["raw_response"]))
        if extracted:
            return extracted
    return response


def run_pipeline(
    input_file: Path,
    output_dir: Path,
    queries: list[str],
    batch_size: int,
    endpoint: str,
    parameters: dict[str, object],
    selection_prompt: str | None,
    extraction_prompt: str | None,
) -> None:
    extractor = EXTRACTORS["pdfplumber"]
    pages = extractor(str(input_file))

    client = AzureLLMClient(endpoint=endpoint)
    extraction_parameters = dict(parameters)

    output_dir.mkdir(parents=True, exist_ok=True)

    for query in queries:
        print(f"Selecting pages for {input_file.name} - {query}...")
        selected_pages = select_pages_with_llm(
            client=client,
            parameters=parameters,
            pages=pages,
            query=query,
            batch_size=batch_size,
            is_financial_statement=True,
            prompt_override=selection_prompt,
        )
        print(f"Selected pages: {selected_pages}")
        extracted = extract_table_with_llm(
            client=client,
            parameters=extraction_parameters,
            query=query,
            page_numbers=selected_pages,
            pages=pages,
            prompt_override=extraction_prompt,
        )
        result = {
            "query": query,
            "selected_pages": selected_pages,
            "extraction": extracted,
        }
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-") or "query"
        output_path = output_dir / f"{input_file.stem}.{slug}.llm.json"
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(result, file_handle, ensure_ascii=False, indent=2)
            file_handle.write("\n")
        print(extracted)
        print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    queries = args.query or ["Consolidated Balance Sheet"]
    parameters = parse_parameters(args.parameters)
    endpoint = resolve_endpoint(args.endpoint)

    if args.input_file:
        pdf_files = [Path(args.input_file)]
    else:
        pdf_files = sorted(Path(args.input_dir).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found.")

    for pdf_path in pdf_files:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {pdf_path}")
        run_pipeline(
            input_file=pdf_path,
            output_dir=Path(args.output_dir),
            queries=queries,
            batch_size=args.batch_size,
            endpoint=endpoint,
            parameters=parameters,
            selection_prompt=args.selection_prompt,
            extraction_prompt=args.extraction_prompt,
        )


if __name__ == "__main__":
    main()
