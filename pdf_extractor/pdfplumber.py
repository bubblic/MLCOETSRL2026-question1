import pdfplumber


def extract_text_pdfplumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return {
            page_num: page.extract_text() for page_num, page in enumerate(pdf.pages)
        }
