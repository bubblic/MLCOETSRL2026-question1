import argparse
import json
import math
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

from azure_balance_sheet_model import AzureDeepSeekClient
from pdf_extractor.pdfplumber import extract_text_pdfplumber
from pdf_extractor.pymupdf4llm import extract_text_pymupdf4llm

WORKSPACE_DIR = Path(__file__).resolve().parent
ANNUAL_REPORTS_DIR = WORKSPACE_DIR / "annual_reports"
OUTPUT_DIR = WORKSPACE_DIR / "extracted_text"

EXTRACTORS = {
    "pdfplumber": extract_text_pdfplumber,
    # "pymupdf4llm": extract_text_pymupdf4llm,  # DOES NOT WORK WELL.
}


# python custom_pdf_extraction.py --input-file ./annual_reports/alibaba_2025.pdf
# python custom_pdf_extraction.py --run-balance-sheet --input-file ./annual_reports/alibaba_2025.pdf
# python custom_pdf_extraction.py --run-balance-sheet --input-file ./annual_reports/lvmh_dec_2024.pdf
# python custom_pdf_extraction.py --run-balance-sheet --input-file './annual_reports/2023 General Motors Annual Report .pdf'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from annual reports using a selectable extractor."
    )
    parser.add_argument(
        "--extractor-name",
        choices=EXTRACTORS.keys(),
        default="pdfplumber",
        help="Which extractor to use.",
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
        "--embedding-model",
        default="google/embeddinggemma-300m",
        help="SentenceTransformer model to use for embeddings.",
    )
    parser.add_argument(
        "--run-balance-sheet",
        action="store_true",
        help="Run the consolidated balance sheet extraction pipeline.",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help="Azure DeepSeek endpoint override. Defaults to AZURE_DEEPSEEK_ENDPOINT.",
    )
    parser.add_argument(
        "--deepseek-message",
        default="gen-ai-response",
        help="Message field sent to the Azure DeepSeek endpoint.",
    )
    parser.add_argument(
        "--deepseek-parameters",
        default='{"temperature": 0, "max_tokens": 10000, "top_k": 1}',
        help="JSON string of parameters to send to DeepSeek.",
    )
    parser.add_argument(
        "--rewrite-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite queries for retrieval using DeepSeek.",
    )
    return parser.parse_args()


def vectorize_pages(pages: dict[int, str | None], model_name: str) -> dict[str, object]:
    model = SentenceTransformer(model_name)
    page_items = []
    for page_num in sorted(pages):
        text = pages[page_num] or ""
        print("Page " + str(page_num) + ": " + text[:50])
        embedding = model.encode([text])[0].tolist()
        page_items.append({"page_num": page_num, "text": text, "embedding": embedding})
    return {"model": model_name, "pages": page_items}


def embed_pages(
    pages: dict[int, str | None], model: SentenceTransformer
) -> list[dict[str, object]]:
    page_items = []
    for page_num in sorted(pages):
        text = pages[page_num] or ""
        embedding = model.encode([text])[0].tolist()
        page_items.append({"page_num": page_num, "text": text, "embedding": embedding})
    return page_items


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def parse_parameters(raw: str | None) -> dict[str, object]:
    if raw is None:
        return {}
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("Parameters must decode to a JSON object.")
    return loaded


def detect_language_heuristic(pages: dict[int, str | None]) -> str:
    sample_texts = []
    for page_num in sorted(pages)[:10]:
        text = (pages[page_num] or "").strip()
        if text:
            sample_texts.append(text)
    if not sample_texts:
        return "unknown"

    combined = " ".join(sample_texts)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", combined))
    latin_count = len(re.findall(r"[A-Za-z]", combined))
    total = cjk_count + latin_count
    if total == 0:
        return "unknown"
    if cjk_count / total >= 0.2:
        return "zh"
    if latin_count / total >= 0.6:
        return "en"
    return "unknown"


def detect_language(
    pages: dict[int, str | None],
    client: AzureDeepSeekClient,
    message: str,
    parameters: dict[str, object],
) -> str:
    sample_texts = []
    for page_num in sorted(pages):
        text = (pages[page_num] or "").strip()
        if text:
            sample_texts.append(text)
        if len(sample_texts) >= 10:
            break
    if not sample_texts:
        return "unknown"

    combined = "\n\n".join(sample_texts)
    combined = combined[:4000]
    prompt = (
        "Detect the primary language of the provided text. "
        'Return ONLY JSON like {"language": "en"}.\n'
        "Use ISO 639-1 codes when possible (e.g., en, zh, ja, ko). "
        'If mixed or unclear, return {"language": "unknown"}.\n\n'
        f"Text:\n{combined}"
    )
    try:
        response = client.ask_json(
            message=message, prompt=prompt, parameters=parameters
        )
        raw_language = response.get("language", "unknown")
        if isinstance(raw_language, str):
            normalized = raw_language.strip().lower()
            if re.fullmatch(r"[a-z]{2}", normalized):
                return normalized
            if normalized == "unknown":
                return "unknown"
    except Exception as exc:
        print(f"Language detection failed, using heuristic. Error: {exc}")
    return detect_language_heuristic(pages)


def rewrite_query_with_llm(
    client: AzureDeepSeekClient,
    query: str,
    document_language: str,
    toc_context: str,
    message: str,
    parameters: dict[str, object],
) -> str:
    prompt = (
        "You are helping find the exact heading keywords used in this document. "
        "Use the provided table-of-contents context to return the exact phrase "
        "used in the document for the requested query.\n"
        f"Query: {query}\n"
        "If the query language and document language are different, find the best translation that matches the exact "
        "document wording from the TOC/context. If no matching heading is found, "
        "return the best translation used in similar financial reports. Return only the rewritten query.\n\n"
        f"TOC/context:\n{toc_context}\n\n"
    )
    print(f"Rewrite query prompt: {prompt}")
    try:
        rewritten = client.ask_text(
            message=message, prompt=prompt, parameters=parameters
        )
    except Exception as exc:
        print(f"Query rewrite failed, using original query. Error: {exc}")
        return query
    cleaned = rewritten.strip().strip('"')
    return cleaned or query


def build_toc_context(pages: dict[int, str | None], max_pages: int = 5) -> str:
    sample_texts = []
    for page_num in sorted(pages)[:max_pages]:
        text = (pages[page_num] or "").strip()
        if text:
            sample_texts.append(text)
    if not sample_texts:
        return ""
    return "\n\n".join(sample_texts)[:6000]


def rank_pages_by_query(
    pages: dict[int, str | None],
    model: SentenceTransformer,
    query: str,
    rewritten_query: str | None = None,
) -> list[tuple[int, float]]:
    query_text = rewritten_query or query
    query_embedding = model.encode([query_text])[0].tolist()
    page_items = embed_pages(pages, model)
    ranked = []
    for item in page_items:
        print(f"Comparing with Page {item['page_num']}: {item['text'][:50]}")
        score = (
            0.0
            if not item["text"].strip()  # Assign zero score if the page is empty
            else cosine_similarity(query_embedding, item["embedding"])
        )
        print(f"Score: {score}")
        ranked.append((int(item["page_num"]), score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def deepseek_page_contains_query(
    client: AzureDeepSeekClient,
    query: str,
    page_text: str,
    message: str,
    parameters: dict[str, object],
) -> bool:
    prompt = (
        "You are given a single page of a financial report.\n"
        'Return ONLY valid JSON like {"contains": true} or {"contains": false}.\n'
        f"The value must be true only if the page contains {query} "
        "table or line items. If unsure, return false.\n\n"
        f"Page text:\n{page_text}"
    )
    print(f"DeepSeek prompt: {prompt}")
    response = client.ask_json(message=message, prompt=prompt, parameters=parameters)
    contains = response.get("contains")
    print(f"DeepSeek response: {response}")
    return bool(contains is True)


def deepseek_format_balance_sheet(
    client: AzureDeepSeekClient,
    balance_sheet_text: str,
    message: str,
    parameters: dict[str, object],
) -> str:
    prompt = (
        "Can you output the following in a nice tabular form:\n\n"
        f"{balance_sheet_text}\n\n"
        "Return only the table."
    )
    return client.ask_text(message=message, prompt=prompt, parameters=parameters)


def run_balance_sheet_pipeline(
    input_file: str,
    extractor_name: str,
    embedding_model: str,
    azure_endpoint: str | None,
    deepseek_message: str,
    deepseek_parameters: dict[str, object],
    rewrite_queries: bool,
    query: str,
) -> str:
    extractor = EXTRACTORS[extractor_name]
    pdf_path = Path(input_file)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {pdf_path}")

    pages = extractor(str(pdf_path))
    model = SentenceTransformer(embedding_model)
    client = AzureDeepSeekClient(endpoint=azure_endpoint)
    document_language = detect_language(
        pages,
        client=client,
        message=deepseek_message,
        parameters=deepseek_parameters,
    )
    toc_context = build_toc_context(pages)

    rewritten_query = None
    if rewrite_queries:
        rewritten_query = rewrite_query_with_llm(
            client=client,
            query=query,
            document_language=document_language,
            toc_context=toc_context,
            message=deepseek_message,
            parameters=deepseek_parameters,
        )
        print(f"Rewritten query: {rewritten_query}")
    ranked = rank_pages_by_query(
        pages,
        model,
        query,
        rewritten_query=rewritten_query,
    )
    # ranked = rank_pages_by_query(pages, model, "合併資產負債表")

    if not ranked:
        raise ValueError("No pages found to rank.")

    top_page = ranked[0][0]
    print(f"Top page: {top_page}")
    page_numbers = sorted(pages)
    if top_page not in page_numbers:
        raise ValueError("Top-ranked page not found in extracted pages.")

    start_index = page_numbers.index(top_page)
    balance_sheet_pages: list[int] = []
    for page_num in page_numbers[start_index:]:
        page_text = pages[page_num] or ""
        if deepseek_page_contains_query(
            client,
            query=query,
            page_text=page_text,
            message=deepseek_message,
            parameters=deepseek_parameters,
        ):
            balance_sheet_pages.append(page_num)
        else:
            break

    if not balance_sheet_pages:
        raise ValueError("DeepSeek did not find any balance sheet pages.")

    joined_text = "\n".join(pages[page_num] or "" for page_num in balance_sheet_pages)
    table_text = deepseek_format_balance_sheet(
        client,
        balance_sheet_text=joined_text,
        message=deepseek_message,
        parameters=deepseek_parameters,
    )

    print(f"Balance sheet pages: {balance_sheet_pages}")
    print(table_text)
    return table_text


def extract_reports(
    input_dir: Path,
    output_dir: Path,
    extractor_name: str,
    input_file: str | None,
    embedding_model: str,
) -> None:
    extractor = EXTRACTORS[extractor_name]
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_file:
        pdf_path = Path(input_file)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {pdf_path}")
        pdf_files = [pdf_path]
    else:
        pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")

    for pdf_path in pdf_files:
        print(f"Extracting {pdf_path.name} with {extractor_name}...")
        extracted = extractor(str(pdf_path))
        vectorized = vectorize_pages(extracted, embedding_model)
        output_path = output_dir / f"{pdf_path.stem}.{extractor_name}.text.json"
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(extracted, file_handle, ensure_ascii=False, indent=2)
            file_handle.write("\n")

        output_path = output_dir / f"{pdf_path.stem}.{extractor_name}.embedding.json"
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(vectorized, file_handle, ensure_ascii=False, indent=2)
            file_handle.write("\n")

        print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    deepseek_parameters = parse_parameters(args.deepseek_parameters)
    query = "Consolidated Income Statement"

    if args.run_balance_sheet:
        if not args.input_file:
            raise ValueError("--run-balance-sheet requires --input-file.")
        print(f"Running balance sheet pipeline for {args.input_file}...")
        run_balance_sheet_pipeline(
            input_file=args.input_file,
            extractor_name=args.extractor_name,
            embedding_model=args.embedding_model,
            azure_endpoint=args.azure_endpoint,
            deepseek_message=args.deepseek_message,
            deepseek_parameters=deepseek_parameters,
            rewrite_queries=args.rewrite_queries,
            query=query,
        )
    else:
        print(f"Extracting reports for {args.input_dir}...")
        extract_reports(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            extractor_name=args.extractor_name,
            input_file=args.input_file,
            embedding_model=args.embedding_model,
        )


if __name__ == "__main__":
    main()
