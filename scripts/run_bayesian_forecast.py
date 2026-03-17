"""Run the Bayesian financial model training and forecast pipeline.

Trains the BayesianFinancialModel on Apple historical data (FY2018-FY2024),
then runs a 10-year Monte Carlo forecast.

Example:
    python -m scripts.run_bayesian_forecast
"""

import tensorflow as tf

from financial_forecast.data.historical_data import get_apple_historical_data
from financial_forecast.training.pipeline import run_training_and_forecast


def main() -> None:
    """Run training and forecasting with explicit runtime configuration."""
    historical_data = get_apple_historical_data()
    sales_hist = historical_data["sales"]
    inflation_hist = historical_data["inflation"]

    n_forecast_years = 10
    yearly_deltas = sales_hist[1:] - sales_hist[:-1]
    avg_linear_growth = tf.reduce_mean(yearly_deltas)
    sales_forecast_usd = tf.constant(
        [
            float(sales_hist[-1] + avg_linear_growth * idx)
            for idx in range(n_forecast_years)
        ],
        dtype=tf.float64,
    )
    inflation_forecast = tf.concat(
        [
            inflation_hist[-1:],
            tf.fill([n_forecast_years - 1], tf.constant(0.03, dtype=tf.float64)),
        ],
        axis=0,
    )

    run_training_and_forecast(
        historical_data,
        sales_forecast_usd,
        inflation_forecast,
        use_trained_parameters=False,
        parameters_path="new_trained_parameters.npz",
        use_inflation=True,
        include_tax_anomalies=True,
        simple_policy_epochs=25000,
        structural_epochs=20000,
        monte_carlo_samples=1000,
    )


if __name__ == "__main__":
    main()
