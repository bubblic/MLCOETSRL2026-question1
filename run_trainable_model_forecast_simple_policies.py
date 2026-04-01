"""Trainable model with simple (non-trend) policies — all defaults."""

from financial_forecast.experiment import ExperimentConfig, run_experiment

if __name__ == "__main__":
    run_experiment(
        ExperimentConfig(
            company="aapl",
            parameters_save_path="trained_parameters_simple_policies.npz",
        )
    )
