"""Extract tax anomalies from 10-K PDFs.

Uses an LLM to identify relevant pages (Item 7/8 tax-related content)
and extract structured JSON with one-time tax charges and future
contingency amounts.

Usage:
    python run_tax_anomaly_extraction.py
"""

from financial_forecast.extraction.tax_anomaly_extractor import TaxAnomalyExtractor
from financial_forecast.clients.azure_llm_client import AzureLLMClient

if __name__ == "__main__":

    extractor = TaxAnomalyExtractor(
        llm_client=AzureLLMClient(),
    )

    company = "aapl"

    extractor.run(
        input_path=f"./annual_reports/for_tax_anomalies/{company}",
        output_dir=f"./extracted_json/tax_anomalies/{company}",
    )
