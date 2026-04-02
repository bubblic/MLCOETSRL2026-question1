"""Extract risk warnings from annual report PDFs.

Uses an LLM to flag risk-relevant pages, extract structured findings
per risk category, and synthesise a professional risk memo.

Usage:
    python run_risk_extraction.py
"""

from financial_forecast.extraction.risk.risk_extractor import RiskWarningsExtractor
from financial_forecast.clients.azure_llm_client import AzureLLMClient

if __name__ == "__main__":

    extractor = RiskWarningsExtractor(
        llm_client=AzureLLMClient(),
    )

    company = "evergrande"

    extractor.run(
        input_path=f"./annual_reports/for_risk_warnings/{company}",
        output_dir=f"./extracted_json/risk_warnings/{company}",
    )
