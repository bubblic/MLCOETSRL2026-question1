import pymupdf4llm


def extract_text_pymupdf4llm(pdf_path):
    """
    Extracts text from a PDF and returns a list of dictionaries
    containing the page number and markdown-formatted text content.
    """
    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    return {page_num: page_content for page_num, page_content in pages}
