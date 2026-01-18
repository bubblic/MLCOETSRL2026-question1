import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv

import requests

load_dotenv()

workspace_dir = Path(__file__).resolve().parent
CHATPDF_API_KEY = os.getenv("CHATPDF_API_KEY", "").strip()
CHATPDF_UPLOAD_URL = "https://api.chatpdf.com/v1/sources/add-file"
CHATPDF_CHAT_URL = "https://api.chatpdf.com/v1/chats/message"
CHATPDF_CACHE_PATH = Path(__file__).resolve().parent / "chatpdf_sources.json"


def file_sha256(file_path: Path) -> str:
    hash_obj = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def load_source_cache() -> dict:
    if not CHATPDF_CACHE_PATH.exists():
        return {"files": {}}
    try:
        with CHATPDF_CACHE_PATH.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except json.JSONDecodeError:
        return {"files": {}}
    if "files" not in data or not isinstance(data["files"], dict):
        return {"files": {}}
    return data


def save_source_cache(cache: dict) -> None:
    with CHATPDF_CACHE_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(cache, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")


def get_cached_source_id(file_path: str) -> str | None:
    cache = load_source_cache()
    full_path = workspace_dir / file_path
    file_hash = file_sha256(full_path)
    entry = cache["files"].get(file_hash)
    if not entry:
        return None
    return entry.get("sourceId")


def store_source_id(file_path: str, source_id: str) -> None:
    cache = load_source_cache()
    full_path = workspace_dir / file_path
    file_hash = file_sha256(full_path)
    cache["files"][file_hash] = {
        "sourceId": source_id,
        "filename": file_path,
    }
    save_source_cache(cache)


def upload_pdf(file_path: str) -> str:
    if not CHATPDF_API_KEY:
        raise RuntimeError("CHATPDF_API_KEY env var is required.")

    full_path = workspace_dir / file_path

    with full_path.open("rb") as file_handle:
        files = [("file", (full_path.name, file_handle, "application/octet-stream"))]
        headers = {"x-api-key": CHATPDF_API_KEY}
        response = requests.post(
            CHATPDF_UPLOAD_URL, headers=headers, files=files, timeout=60
        )

    if response.status_code != 200:
        raise RuntimeError(f"Upload failed ({response.status_code}): {response.text}")

    return response.json()["sourceId"]


def chat_with_pdf(source_id: str, prompt: str) -> str:
    if not CHATPDF_API_KEY:
        raise RuntimeError("CHATPDF_API_KEY env var is required.")

    headers = {"x-api-key": CHATPDF_API_KEY, "Content-Type": "application/json"}
    payload = {
        "referenceSources": True,
        "sourceId": source_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    print(f'Chatting with {source_id} with prompt: "{prompt[:100]}..."')
    response = requests.post(
        CHATPDF_CHAT_URL, headers=headers, json=payload, timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(f"Chat failed ({response.status_code}): {response.text}")

    return response.json()["content"]


def main() -> None:
    annual_report_filename = "2023 General Motors Annual Report .pdf"
    annual_report_filename = "alibaba_2025.pdf"
    # annual_report_filename = "lvmh_dec_2024.pdf"

    pdf_path = "annual_reports/" + annual_report_filename

    # First, check if the file exists.
    full_path = workspace_dir / pdf_path
    if not full_path.exists():
        raise FileNotFoundError(f"PDF not found at {full_path}")

    # Then, check if the file is cached.
    source_id = get_cached_source_id(pdf_path)
    if not source_id:
        # If not cached, upload the file.
        print(f"Uploading {pdf_path} to ChatPDF...")
        source_id = upload_pdf(pdf_path)
        print(f"Uploaded {pdf_path} to ChatPDF with source ID {source_id}")
        # Then, store the source ID in the cache.
        store_source_id(pdf_path, source_id)
        print(f"Stored source ID {source_id} in cache for {pdf_path}")

    # Then, chat with the PDF.
    prompt_income_statement = (
        "Extract the income statement "
        "from the condensed consolidated income statement section in the annual report. Return ONLY valid JSON with this shape:\n"
        "{\n"
        '   "years_ended": [\n'
        '      {"year": <number>, "data": {"<line item>": <number|string>, ...}},\n'
        "      ...\n"
        # '      {"year": <number>, "data": {...}},\n'
        # '      {"year": <number>, "data": {...}}\n'
        "   ],\n"
        '   "reference_page": <number>,\n'
        '   "currency": <string>\n'
        "}\n"
        "Use numbers only (no commas or currency symbols). Use null when the value "
        "is not available. Do not include any commentary outside the JSON."
    )

    prompt_balance_sheet_page = "Identify the page range where the consolidated balance sheet is located. If you see multiple ranges, return the one with the most data. Return the actual page numbers in the pdf. Return numbers only."

    def get_prompt_balance_sheet(page_range: str) -> str:
        return (
            # "First, identify the page range or section header for the Consolidated Balance Sheet. Then, extract only the table data from that section, excluding any explanatory comments or footnotes."
            f"Extract the balance sheet ONLY from page(s): {page_range}. "
            # f"Extract the balance sheet ONLY from page 26. "
            # "Extract the balance sheet "
            # "from the condensed consolidated balance sheet section in the annual report. "
            "Return ONLY valid JSON with this shape, translating any non-English text to English:\n"
            "{\n"
            '   "years_ended": [\n'
            "      {\n"
            '         "year": <number>, '
            '         "data": {\n'
            '            "assets": {\n'
            '               "total": <number>,\n'
            '               "current": {\n '
            '                  "total": <number>,\n'
            '                  "<line item>": <number>,\n'
            "                  ...\n"
            "               },\n"
            '               "non_current": {\n'
            '                  "total": <number>,\n'
            '                  "<line item>": <number>,\n'
            "                  ...\n"
            "               }\n"
            "            },\n"
            '            "liabilities": {\n'
            '               "current": {\n '
            '                  "total": <number>,\n'
            '                  "<line item>": <number>,\n'
            "                  ...\n"
            "               },\n"
            '               "non_current": {\n'
            '                  "total": <number>,\n'
            '                  "<line item>": <number>,\n'
            "                  ...\n"
            "               }\n"
            "            },\n"
            '            "equity": {\n'
            '               "total": <number>,\n'
            '               "<line item>": <number>,\n'
            "               ...\n"
            "            }\n"
            "         }\n"
            "      },\n"
            "      ...\n"
            "   ],\n"
            '   "reference_page": <number>,\n'
            '   "currency": <string>\n'
            "}\n"
            "Use numbers only (no commas or currency symbols). Use null when the value "
            "is not available. Do not include any commentary outside the JSON."
        )

    prompt_balance_sheet_simple = "Extract the consolidated balance sheet for all years listed in table. Return in JSON format."

    # result_income_statement = chat_with_pdf(source_id, prompt_income_statement)
    page_range = chat_with_pdf(source_id, prompt_balance_sheet_page)
    print(f"Page range identified: {page_range}")
    result_balance_sheet = chat_with_pdf(
        source_id, get_prompt_balance_sheet(page_range)
    )
    # result_balance_sheet_simple = chat_with_pdf(source_id, prompt_balance_sheet_simple)
    try:
        # parsed_income_statement = json.loads(result_income_statement)
        parsed_balance_sheet = json.loads(result_balance_sheet)
        # parsed_balance_sheet_simple = json.loads(result_balance_sheet_simple)
        # print(json.dumps(parsed_income_statement, indent=2))
        print(json.dumps(parsed_balance_sheet, indent=2))
        # print(json.dumps(parsed_balance_sheet_simple, indent=2))
    except json.JSONDecodeError:
        # print(result_income_statement)
        print(result_balance_sheet)
        # print(result_balance_sheet_simple)


if __name__ == "__main__":
    main()


# example = {
#     "title": "CONSOLIDATED INCOME STATEMENTS (In millions, except per share amounts)",
#     "years_ended_december_31": [
#         {
#             "year": 2023,
#             "data": {
#                 "Net sales and revenue": {
#                     "Automotive": 157658,
#                     "GM Financial": 14184,
#                     "Total net sales and revenue (Note 3)": 171842,
#                 },
#                 "Costs and expenses": {
#                     "Automotive and other cost of sales": 141330,
#                     "GM Financial interest, operating and other expenses": 11374,
#                     "Automotive and other selling, general and administrative expense": 9840,
#                     "Total costs and expenses": 162544,
#                 },
#                 "Operating income (loss)": 9298,
#                 "Automotive interest expense": 911,
#                 "Interest income and other non-operating income, net (Note 19)": 1537,
#                 "Equity income (loss) (Note 8)": 480,
#                 "Income (loss) before income taxes": 10403,
#                 "Income tax expense (benefit) (Note 17)": 563,
#                 "Net income (loss)": 9840,
#                 "Net loss (income) attributable to noncontrolling interests": 287,
#                 "Net income (loss) attributable to stockholders": 10127,
#                 "Net income (loss) attributable to common stockholders": 10022,
#                 "Earnings per share (Note 21)": {
#                     "Basic earnings per common share": 7.35,
#                     "Weighted-average common shares outstanding – basic": 1364,
#                     "Diluted earnings per common share": 7.32,
#                     "Weighted-average common shares outstanding – diluted": 1369,
#                 },
#             },
#         },
#         {
#             "year": 2022,
#             "data": {
#                 "Net sales and revenue": {
#                     "Automotive": 143975,
#                     "GM Financial": 12760,
#                     "Total net sales and revenue (Note 3)": 156735,
#                 },
#                 "Costs and expenses": {
#                     "Automotive and other cost of sales": 126892,
#                     "GM Financial interest, operating and other expenses": 8862,
#                     "Automotive and other selling, general and administrative expense": 10667,
#                     "Total costs and expenses": 146421,
#                 },
#                 "Operating income (loss)": 10315,
#                 "Automotive interest expense": 987,
#                 "Interest income and other non-operating income, net (Note 19)": 1432,
#                 "Equity income (loss) (Note 8)": 837,
#                 "Income (loss) before income taxes": 11597,
#                 "Income tax expense (benefit) (Note 17)": 1888,
#                 "Net income (loss)": 9708,
#                 "Net loss (income) attributable to noncontrolling interests": 226,
#                 "Net income (loss) attributable to stockholders": 9934,
#                 "Net income (loss) attributable to common stockholders": 8915,
#                 "Earnings per share (Note 21)": {
#                     "Basic earnings per common share": 6.17,
#                     "Weighted-average common shares outstanding – basic": 1445,
#                     "Diluted earnings per common share": 6.13,
#                     "Weighted-average common shares outstanding – diluted": 1454,
#                 },
#             },
#         },
#         {
#             "year": 2021,
#             "data": {
#                 "Net sales and revenue": {
#                     "Automotive": 113590,
#                     "GM Financial": 13414,
#                     "Total net sales and revenue (Note 3)": 127004,
#                 },
#                 "Costs and expenses": {
#                     "Automotive and other cost of sales": 100544,
#                     "GM Financial interest, operating and other expenses": 8582,
#                     "Automotive and other selling, general and administrative expense": 8554,
#                     "Total costs and expenses": 117680,
#                 },
#                 "Operating income (loss)": 9324,
#                 "Automotive interest expense": 950,
#                 "Interest income and other non-operating income, net (Note 19)": 3041,
#                 "Equity income (loss) (Note 8)": 1301,
#                 "Income (loss) before income taxes": 12716,
#                 "Income tax expense (benefit) (Note 17)": 2771,
#                 "Net income (loss)": 9945,
#                 "Net loss (income) attributable to noncontrolling interests": 74,
#                 "Net income (loss) attributable to stockholders": 10019,
#                 "Net income (loss) attributable to common stockholders": 9837,
#                 "Earnings per share (Note 21)": {
#                     "Basic earnings per common share": 6.78,
#                     "Weighted-average common shares outstanding – basic": 1451,
#                     "Diluted earnings per common share": 6.70,
#                     "Weighted-average common shares outstanding – diluted": 1468,
#                 },
#             },
#         },
#     ],
#     "reference": " ",
# }
