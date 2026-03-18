"""Run the Bayesian financial model training and forecast pipeline.

Wires together BayesianFinancialModel + PolicyTrainer + StructuralTrainer
and runs the full pipeline (train, Monte Carlo forecast, plot).

Usage:
    python run_bayesian_forecast.py
"""

from financial_forecast.data.historical_data import get_apple_historical_data
from financial_forecast.models.bayesian_model import BayesianFinancialModel
from financial_forecast.training import PolicyTrainer, StructuralTrainer
from financial_forecast.training.pipeline import ForecastPipeline

if __name__ == "__main__":
    ForecastPipeline(
        model=BayesianFinancialModel(base_year=2018),
        trainers=[PolicyTrainer(epochs=25000), StructuralTrainer(epochs=20000)],
        historical_data=get_apple_historical_data(),
        forecast_years=10,  # Including the last year in historical data
        sales_forecast_usd=None,  # default: use linear extrapolation of the historical average annual delta
        inflation_forecast=None,  # default: 3% inflation
        monte_carlo_samples=1000,
        parameters_save_path="new_trained_parameters.npz",
        use_trained_parameters=False,
        use_inflation=True,
        include_tax_anomalies=True,
    ).run()
