from azure_llm_client import AzureLLMClient
import re
import json


def select_pages_with_llm(
    client: AzureLLMClient,
    parameters: dict[str, object],
    pages: dict[int, str | None],
    query: str,
    batch_size: int,
    is_financial_statement: bool,
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
            else build_selection_prompt(query, pages_text, is_financial_statement)
        )
        print(prompt[:2000])
        response = client.ask_json(
            message="gen-ai-response",
            prompt=prompt,
            parameters={},  # This has to be blank for non-reasoning model.
            reasoning=False,
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


def build_selection_prompt(
    query: str, pages_text: str, is_financial_statement: bool
) -> str:
    table_or_information = "table" if is_financial_statement else "information"
    table_or_line_items = (
        "table or line items" if is_financial_statement else "information"
    )
    prompt = (
        "You are given multiple pages from an annual report. "
        f"Identify which pages contain the {table_or_information} "
        "for the requested query. "
        'Return ONLY JSON in the shape: {"pages": [<page_number>, ...]}. '
        f"Include all pages that contain the {table_or_line_items}. "
        'If none, return {"pages": []}.\n\n'
        f"Query: {query}\n\n"
        f"Pages:\n{pages_text}"
    )
    return prompt
