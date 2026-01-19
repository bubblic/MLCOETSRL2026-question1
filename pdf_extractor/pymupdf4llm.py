import pymupdf4llm

### DOES NOT WORK WELL.


def extract_text_pymupdf4llm(pdf_path):
    """
    Extracts text from a PDF and returns a list of dictionaries
    containing the page number and markdown-formatted text content.
    """
    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    # output first 10 items
    print(pages[25])
    # output length of array
    print(len(pages))

    extracted = {}
    for item in pages:
        if isinstance(item, dict):
            page_num = item.get("page")
            if page_num is None:
                page_num = item.get("page_number")
            content = (
                item.get("text")
                if "text" in item
                else item.get("markdown", item.get("content", ""))
            )
        else:
            page_num, content = item
        extracted[page_num] = content
    return extracted
