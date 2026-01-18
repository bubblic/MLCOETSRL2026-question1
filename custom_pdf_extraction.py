import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF and returns a list of dictionaries
    containing the page number and the raw text content.
    """
    doc = fitz.open(pdf_path)
    pages_content = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")  # "text" preserves simple layout
        pages_content.append({"page": page_num + 1, "content": text})

    return pages_content


# Example usage:
# data = extract_text_from_pdf("annual_report.pdf")
