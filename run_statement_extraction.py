"""Extract financial statements from annual report PDFs.

Stage 1 of the extraction pipeline: uses an LLM to identify relevant
pages in each PDF, then extracts primary financial tables and
supplementary disclosures.

Usage:
    python run_statement_extraction.py
"""

from financial_forecast.extraction.financial_statement_extractor import (
    FinancialStatementExtractor,
)
from financial_forecast.extraction.statement_config import StatementType
from financial_forecast.clients.azure_llm_client import AzureLLMClient

if __name__ == "__main__":

    extractor = FinancialStatementExtractor(
        queries=[
            StatementType.BALANCE_SHEET,
            StatementType.INCOME_STATEMENT,
            StatementType.CASH_FLOW,
        ],
        llm_client=AzureLLMClient(),
    )

    extractor.run(
        input_path="./annual_reports/for_financial_statements/alibaba_2025.pdf",  # can be a file or a folder
        output_dir="./extracted_text/financial_statements",
    )
