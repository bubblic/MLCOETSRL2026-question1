"""Calculate financial ratios from normalized statement data.

Stage 3 of the extraction pipeline: reads ``*.normalized.json`` files
and computes derived metrics and financial ratios.

Usage:
    python run_ratio_calculation.py
"""

from financial_forecast.extraction.statement_ratios import RatioCalculator

if __name__ == "__main__":

    calculator = RatioCalculator(
        input_dir="./extracted_json/financial_statements/alibaba",
        output_file="financial_ratios.json",
    )

    calculator.run()
