"""Run the full extraction pipeline: PDF → normalized JSON → financial ratios.

Chains the three extraction stages:
1. Extract financial statements from PDFs
2. Normalize extracted statements via LLM
3. Calculate financial ratios from normalized data

Usage:
    python run_pdf_to_ratios_pipeline.py
"""

from financial_forecast.extraction.financial_statement_extractor import (
    FinancialStatementExtractor,
)
from financial_forecast.extraction.statement_config import StatementType
from financial_forecast.extraction.statement_normalizer import StatementNormalizer
from financial_forecast.extraction.statement_ratios import RatioCalculator
from financial_forecast.clients.azure_llm_client import AzureLLMClient

if __name__ == "__main__":

    pdf_dir = f"./annual_reports/for_financial_statements"
    extracted_dir = f"./extracted_text/financial_statements"
    normalized_dir = f"./extracted_json/financial_statements"

    # Stage 1: PDF → raw extracted JSON
    extractor = FinancialStatementExtractor(
        queries=[
            StatementType.BALANCE_SHEET,
            StatementType.INCOME_STATEMENT,
            StatementType.CASH_FLOW,
        ],
        llm_client=AzureLLMClient(),
    )
    extractor.run(
        input_path=pdf_dir,
        output_dir=extracted_dir,
    )

    # Stage 2: Raw JSON → normalized JSON
    normalizer = StatementNormalizer(
        input_dir=extracted_dir,
        output_dir=normalized_dir,
    )
    normalizer.run()

    # Stage 3: Normalized JSON → financial ratios
    calculator = RatioCalculator(
        input_dir=normalized_dir,
        output_file="financial_ratios.json",
    )
    calculator.run()
