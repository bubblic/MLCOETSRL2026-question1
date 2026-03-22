"""Run the financial model training and forecast pipeline
+ Bayesian OpEx model
+ tax anomalies information.

Wires together TrainableFinancialModel + PolicyTrainer + StructuralTrainer
and runs the full pipeline (train, Monte Carlo forecast, plot).

Usage:
    python run_trainable_model_forecast_w_bayesian_opex_taxanomaly.py
"""

from financial_forecast.data.loader import HistoricalDataLoader
from financial_forecast.models.trainable_financial_model import TrainableFinancialModel
from financial_forecast.models.opex import BayesianOpEx
from financial_forecast.inference.trajectory_simulator import MonteCarloSimulator
from financial_forecast.training.policy_trainer import PolicyTrainer
from financial_forecast.training.structural_trainer import StructuralTrainer
from financial_forecast.training.pipeline import ForecastPipeline

if __name__ == "__main__":
    historical_data = HistoricalDataLoader(
        "aapl", include_inflation=True, include_tax_onetime=True
    )

    model = TrainableFinancialModel(
        opex_module=BayesianOpEx(),
        trajectory_simulator=MonteCarloSimulator(n_samples=1000),
        tax_anomalies=historical_data.tax_onetime_payments,
    )

    ForecastPipeline(
        model=model,
        trainers=[PolicyTrainer(epochs=25000), StructuralTrainer(epochs=20000)],
        financial_statements=historical_data.financial_statements,
        inflation=historical_data.inflation,
        test_years=1,  # hold out last N years for testing
        forecast_years=10,  # includes test_years
        sales_forecast_usd=None,  # default: use linear extrapolation of the historical average annual delta
        inflation_forecast=None,  # default: 3% inflation
        parameters_save_path="trained_parameters_bayesianopex_taxanomalies.npz",
        use_trained_parameters=False,
    ).run()

    # Plot OpEx fit diagnostics (Bayesian-specific)
    model.opex_module.plot_fit()
    model.opex_module.plot_fit(use_gaussian_ci=True)
