"""Send a forecast report JSON to the LLM for CEO/CFO recommendations.

Reads the JSON artifact produced by ForecastPipeline.run() and sends
the historical + forecast tables to the Azure reasoning model.

Usage:
    python run_recommendation_to_ceo.py
"""

import json

from financial_forecast.reporting.advisor import DeepseekCEOAdvisor


if __name__ == "__main__":

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

    advisor = DeepseekCEOAdvisor(
        message="gen-ai-response",
        parameters={
            "temperature": 0,
            "max_tokens": 100000,
            "top_k": 1,
        },
    )

    prompt = advisor.build_prompt(
        historical_table=report["historical_table"],
        forecast_table=report["forecast_table"],
    )

    print("Sending forecast to LLM for CEO recommendations...\n")
    response = advisor.recommend(prompt)
    print(response)
