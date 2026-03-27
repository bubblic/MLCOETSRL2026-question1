"""Extract financial statements from annual report PDFs.

Stage 1 of the extraction pipeline: uses an LLM to identify relevant
pages in each PDF, then extracts primary financial tables and
supplementary disclosures.

Usage:
    python run_pdf_extraction.py
"""

from financial_forecast.extraction.financial_statement_extractor import (
    FinancialStatementExtractor,
)
from financial_forecast.clients.azure_llm_client import AzureLLMClient

if __name__ == "__main__":

    extractor = FinancialStatementExtractor(
        queries=[
            "Consolidated Balance Sheet",
            "Consolidated Income Statement",
            "Consolidated Cash Flow Statement",
        ],
        llm_client=AzureLLMClient(),
    )

    extractor.run(
        input_dir="./annual_reports/for_financial_statements",
        output_dir="extracted_text/financial_statements",
    )
