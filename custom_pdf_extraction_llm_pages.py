"""
Example usage:
  python custom_pdf_extraction_llm_pages.py --input-file ./annual_reports/alibaba_2025.pdf \\
    --query "Consolidated Balance Sheet" --query "Consolidated Income Statement"

Env vars:
  AZURE_DEEPSEEK_ENDPOINT for both page selection and table extraction
"""

# python custom_pdf_extraction_llm_pages.py --input-file "./annual_reports/alibaba_2025.pdf"  --query "Consolidated Balance Sheet" --query "Consolidated Income Statement"
# python custom_pdf_extraction_llm_pages.py --input-file "./annual_reports/2023 General Motors Annual Report .pdf" --query "Consolidated Balance Sheet" --query "Consolidated Income Statement"
# python custom_pdf_extraction_llm_pages.py --input-file "./annual_reports/lvmh_dec_2024.pdf" --query "Consolidated Balance Sheet" --query "Consolidated Income Statement"

# python custom_pdf_extraction_llm_pages.py --input-dir ./annual_reports --query "Consolidated Balance Sheet" --query "Consolidated Income Statement"


import argparse
import json
import os
import re
from pathlib import Path

from azure_balance_sheet_model import AzureDeepSeekClient
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
        "--extractor-name",
        choices=EXTRACTORS.keys(),
        default="pdfplumber",
        help="Which PDF text extractor to use.",
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
        "--message",
        default="gen-ai-response",
        help="Message field sent to the LLM endpoint.",
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


def chunk_pages(page_numbers: list[int], batch_size: int) -> list[list[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [
        page_numbers[idx : idx + batch_size]
        for idx in range(0, len(page_numbers), batch_size)
    ]


def build_page_blocks(batch_pages: dict[int, str]) -> str:
    page_blocks = []
    for page_num in sorted(batch_pages):
        text = (batch_pages[page_num] or "").strip()
        page_blocks.append(f"Page {page_num}:\n{text}")
    return "\n\n---\n\n".join(page_blocks)


def build_selection_prompt(query: str, pages_text: str) -> str:
    prompt = (
        "You are given multiple pages from an annual report. "
        "Identify which pages contain the table for the requested query. "
        'Return ONLY JSON in the shape: {"pages": [<page_number>, ...]}. '
        "Include all pages that contain the table or line items. "
        'If none, return {"pages": []}.\n\n'
        f"Query: {query}\n\n"
        f"Pages:\n{pages_text}"
    )
    return prompt


def build_extraction_prompt(query: str, pages_text: str) -> str:
    return (
        f"Find the table that corresponds to {query} and output it in a nice tabular form from the following pages data.\n"
        f"Pages:\n{pages_text}"
    )


def extract_json_from_text(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def normalize_pages(payload: dict[str, object], valid_pages: set[int]) -> list[int]:
    raw = payload.get("pages") or payload.get("page_numbers")
    if not isinstance(raw, list):
        return []
    normalized = []
    for item in raw:
        if isinstance(item, int) and item in valid_pages:
            normalized.append(item)
        elif isinstance(item, str) and item.isdigit():
            page_num = int(item)
            if page_num in valid_pages:
                normalized.append(page_num)
    return sorted(set(normalized))


def select_pages_with_llm(
    client: AzureDeepSeekClient,
    message: str,
    parameters: dict[str, object],
    pages: dict[int, str | None],
    query: str,
    batch_size: int,
    prompt_override: str | None = None,
) -> list[int]:
    valid_pages = set(pages.keys())
    selected_pages: set[int] = set()
    for batch in chunk_pages(sorted(valid_pages), batch_size):
        batch_pages = {page_num: pages[page_num] for page_num in batch}
        pages_text = build_page_blocks(batch_pages)
        prompt = (
            prompt_override.format(query=query, pages=pages_text)
            if prompt_override
            else build_selection_prompt(query, pages_text)
        )
        print(prompt[:2000])
        response = client.ask_json(
            message=message, prompt=prompt, parameters={}, reasoning=False
        )
        print(response)
        payload = response
        if "raw_response" in response:
            extracted = extract_json_from_text(str(response["raw_response"]))
            if extracted:
                payload = extracted
        batch_selected = normalize_pages(payload, valid_pages)
        selected_pages.update(batch_selected)
    return sorted(selected_pages)


def extract_table_with_llm(
    client: AzureDeepSeekClient,
    message: str,
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
        message=message, prompt=prompt, parameters=parameters, reasoning=True
    )
    if "raw_response" in response:
        extracted = extract_json_from_text(str(response["raw_response"]))
        if extracted:
            return extracted
    return response


def run_pipeline(
    input_file: Path,
    extractor_name: str,
    output_dir: Path,
    queries: list[str],
    batch_size: int,
    endpoint: str,
    message: str,
    parameters: dict[str, object],
    selection_prompt: str | None,
    extraction_prompt: str | None,
) -> None:
    extractor = EXTRACTORS[extractor_name]
    pages = extractor(str(input_file))

    client = AzureDeepSeekClient(endpoint=endpoint)
    extraction_parameters = dict(parameters)

    output_dir.mkdir(parents=True, exist_ok=True)

    for query in queries:
        print(f"Selecting pages for {input_file.name} - {query}...")
        selected_pages = select_pages_with_llm(
            client=client,
            message=message,
            parameters=parameters,
            pages=pages,
            query=query,
            batch_size=batch_size,
            prompt_override=selection_prompt,
        )
        print(f"Selected pages: {selected_pages}")
        extracted = extract_table_with_llm(
            client=client,
            message=message,
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
            extractor_name=args.extractor_name,
            output_dir=Path(args.output_dir),
            queries=queries,
            batch_size=args.batch_size,
            endpoint=endpoint,
            message=args.message,
            parameters=parameters,
            selection_prompt=args.selection_prompt,
            extraction_prompt=args.extraction_prompt,
        )


if __name__ == "__main__":
    main()
