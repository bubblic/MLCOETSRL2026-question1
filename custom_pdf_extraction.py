import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from annual reports using a selectable extractor."
    )
    parser.add_argument(
        "--extractor",
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
    extract_reports(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        extractor_name=args.extractor,
        input_file=args.input_file,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
