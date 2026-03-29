"""Run multiple normalization passes for hallucination measurement.

Runs the LLM normalization N times on the same input, then computes
median-aggregated ratios and hallucination rates across runs.

Usage:
    python run_statement_normalization_multi.py
"""

from financial_forecast.extraction.statement_normalizer import StatementNormalizer
from financial_forecast.extraction.statement_ratios import MedianRatioCalculator

if __name__ == "__main__":

    input_dir = "./extracted_text/financial_statements/alibaba"
    runs_output_dir = "./extracted_json/financial_statements_multiruns/alibaba"

    # Stage 1: Run normalization N times
    normalizer = StatementNormalizer(
        input_dir=input_dir,
        output_dir=runs_output_dir,
    )
    normalizer.run_multi(
        num_runs=30,
        runs_output_dir=runs_output_dir,
    )

    # Stage 2: Compute median ratios + hallucination analysis
    calculator = MedianRatioCalculator(
        runs_output_dir=runs_output_dir,
        plot_distributions=True,
        hallucination_output_file="hallucination_rates.json",
    )
    calculator.run()
