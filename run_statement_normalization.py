"""Normalize raw extracted statements into structured JSON.

Stage 2 of the extraction pipeline: reads raw ``*.llm.json`` files,
sends them to an LLM for field normalization, and writes structured
``*.normalized.json`` output.

Usage:
    python run_statement_normalization.py
"""

from financial_forecast.extraction.statement_normalizer import StatementNormalizer

if __name__ == "__main__":

    normalizer = StatementNormalizer(
        input_dir="./extracted_text/financial_statements/alibaba",
        output_dir="./extracted_json/financial_statements/alibaba",
    )

    normalizer.run()
