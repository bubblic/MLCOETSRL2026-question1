"""Run the refactored trainable financial model pipeline.

Examples:
    python trainable_financial_model_taxanomaly_effstdebt_bayesian_vi_refactored.py
"""

import tensorflow as tf

from historical_data import get_apple_historical_data
from financial_model_pipeline.pipeline import run_training_and_forecast


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
    # First element uses last historical inflation; rest default to 0.03
    inflation_forecast = tf.concat(
        [
            inflation_hist[-1:],
            tf.fill([n_forecast_years - 1], tf.constant(0.03, dtype=tf.float64)),
        ],
        axis=0,
    )

    # Model is trained on historical data up to the second-to-last year, and actually the last year of the historical data is the start of the forecasted years. This is so that we can use the last historical year data for testing the model's forecasting ability.
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
