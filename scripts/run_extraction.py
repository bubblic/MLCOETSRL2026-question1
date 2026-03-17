"""Run the LLM-powered financial statement extraction pipeline.

Wraps the extraction CLI from the financial_forecast.extraction module.

Example:
    python -m scripts.run_extraction --input-dir ./annual_reports \\
        --query "Consolidated Balance Sheet"
"""

from financial_forecast.extraction.statement_cli import main


if __name__ == "__main__":
    main()
