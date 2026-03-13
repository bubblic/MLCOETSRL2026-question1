"""Run the financial statement extraction pipeline.

Example:
    python llm_extract_json_from_financial_statements.py \
      --input-dir extracted_text \
      --num-extraction-runs 3 \
      --ratios-aggregation median \
      --runs-output-dir deepseek_financial_statements_runs \
      --max-workers 9 \
      --plot-distributions \
      --hallucination-output-file hallucination_rates.json \
      --hallucination-top-k 10

    Hallucination analysis runs automatically when --ratios-aggregation=median.
    Use --skip-hallucination-report to disable it.
"""

from financial_statement_pipeline.statement_cli import main


if __name__ == "__main__":
    main()
