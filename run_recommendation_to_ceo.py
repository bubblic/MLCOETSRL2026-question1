"""Send a forecast report to the LLM for CEO/CFO strategic recommendations.

Reads the self-contained JSON artifact produced by
:meth:`ForecastPipeline.run` (which includes historical and forecast
markdown tables, model metadata, and a reference to the trained
parameters file) and sends it to the Azure DeepSeek reasoning model
for capital-structure and capital-allocation analysis.

Inputs:
    - ``training_results/adv_policies_w_bayesianopex_taxanomalies/forecast_report.json``
      (produced by the training pipeline)

LLM Configuration:
    - Model: DeepSeek-V3.2 (Azure-hosted reasoning model)
    - temperature=0, top_k=1 (greedy decoding for minimal hallucination)
    - max_tokens=100,000

Usage::

    python run_recommendation_to_ceo.py
"""

import json

from financial_forecast.reporting.advisor import DeepseekCEOAdvisor


if __name__ == "__main__":

    # -- Step 1: Load the forecast report JSON --
    report_path = (
        "training_results/adv_policies_w_bayesianopex_taxanomalies/forecast_report.json"
    )

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print(f"Company:             {report['company']}")
    print(f"Generated at:        {report['generated_at']}")
    print(f"Parameters file:     {report['parameters_path']}")
    print(f"Forecast years:      {report['forecast_years']}")
    print(f"Monte Carlo samples: {report['n_monte_carlo_samples']}")
    print()

    # -- Step 2: Configure the LLM advisor (greedy decoding) --
    advisor = DeepseekCEOAdvisor(
        message="gen-ai-response",
        parameters={
            "temperature": 0,
            "max_tokens": 100000,
            "top_k": 1,
        },
    )

    # -- Step 3: Build prompt and send to LLM --
    prompt = advisor.build_prompt(
        historical_table=report["historical_table"],
        forecast_table=report["forecast_table"],
    )

    print("Sending forecast to LLM for CEO recommendations...\n")
    response = advisor.recommend(prompt)
    print(response)
