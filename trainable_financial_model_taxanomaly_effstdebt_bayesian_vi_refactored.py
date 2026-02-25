"""Run the refactored trainable financial model pipeline.

Examples:
    python trainable_financial_model_taxanomaly_effstdebt_bayesian_vi_refactored.py
"""

import numpy as np

from historical_data import get_apple_historical_data
from financial_model_pipeline.pipeline import run_training_and_forecast


def main() -> None:
    """Run training and forecasting with explicit runtime configuration."""
    historical_data = get_apple_historical_data()
    sales_hist = historical_data["sales"]

    n_forecast_years = 10
    yearly_deltas = np.diff(sales_hist)
    avg_linear_growth = np.mean(yearly_deltas)
    sales_forecast_usd = np.array(
        [sales_hist[-1] + avg_linear_growth * idx for idx in range(n_forecast_years)],
        dtype=np.float64,
    )
    inflation_forecast = np.full(n_forecast_years, 0.03, dtype=np.float64)

    run_training_and_forecast(
        historical_data,
        sales_forecast_usd,
        inflation_forecast,
        use_trained_parameters=False,
        parameters_path="trained_parameters_include_tax_anomalies.npz",
        use_inflation=True,
        include_tax_anomalies=True,
        simple_policy_epochs=25000,
        structural_epochs=20000,
        monte_carlo_samples=1000,
    )


if __name__ == "__main__":
    main()
