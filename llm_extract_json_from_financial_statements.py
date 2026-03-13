"""Run the financial statement extraction pipeline.

Example:
    python llm_extract_json_from_financial_statements.py \
      --input-dir extracted_text \
      --num-extraction-runs 3 \
      --ratios-aggregation median \ 
      --runs-output-dir deepseek_financial_statements_runs \
      --max-workers 9 \
      --plot-distributions
"""

from financial_statement_pipeline.statement_cli import main


if __name__ == "__main__":
    main()
