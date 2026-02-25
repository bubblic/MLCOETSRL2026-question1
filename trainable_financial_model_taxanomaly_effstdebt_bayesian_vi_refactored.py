"""Run the refactored trainable financial model pipeline.

Examples:
    python trainable_financial_model_taxanomaly_effstdebt_bayesian_vi_refactored.py
"""

from financial_model_pipeline.pipeline import run_training_and_forecast


def main() -> None:
    """Run training and forecasting with default settings."""
    run_training_and_forecast(
        use_trained_parameters=False,
        parameters_path="trained_parameters_include_tax_anomalies.npz",
        use_inflation=True,
        include_tax_anomalies=True,
    )


if __name__ == "__main__":
    main()
