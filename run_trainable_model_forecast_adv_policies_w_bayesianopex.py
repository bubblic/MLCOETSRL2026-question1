"""Run the financial model with BayesianOpEx + all advanced policies.

Usage:
    python run_trainable_model_forecast_adv_policies_w_bayesianopex.py
"""

from financial_forecast.data.loader import HistoricalDataLoader
from financial_forecast.models.trainable_financial_model import TrainableFinancialModel
from financial_forecast.models.opex import BayesianOpEx
from financial_forecast.inference.trajectory_simulator import MonteCarloSimulator
from financial_forecast.models.liquidity import TrendLiquidityPolicy
from financial_forecast.models.dividends import LintnerDividendPolicy
from financial_forecast.models.buyback import BaselineBuybackPolicy
from financial_forecast.models.purchases import TrendCostRatioPolicy
from financial_forecast.models.debt import TrendDebtPolicy
from financial_forecast.models.capex import CapexPolicy
from financial_forecast.models.working_capital import WorkingCapitalPolicy
from financial_forecast.training.policy_trainer import PolicyTrainer
from financial_forecast.training.structural_trainer import StructuralTrainer
from financial_forecast.training.pipeline import ForecastPipeline
from financial_forecast.inference.forecast_driver_models import (
    LinearSalesForecast,
    ConstantInflationForecast,
)
import tensorflow as tf


if __name__ == "__main__":

    tf.random.set_seed(42)

    data = HistoricalDataLoader(
        "aapl",
        include_inflation=True,
    )

    model = TrainableFinancialModel(
        opex_module=BayesianOpEx(),
        trajectory_simulator=MonteCarloSimulator(n_samples=1000),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
    )

    model.prepare(
        financial_statements=data.financial_statements,
        inflation=data.inflation,
        test_years=1,
    )

    model.train(
        trainers=[PolicyTrainer(epochs=25000), StructuralTrainer(epochs=20000)],
        parameters_save_path="trained_parameters_adv_policies_w_bayesianopex.npz",
    )

    ForecastPipeline(
        model,
        sales_forecast=LinearSalesForecast(
            data.financial_statements["sales"],
            forecast_years=10,
        ),
        inflation_forecast=ConstantInflationForecast(
            data.inflation,
            forecast_years=10,
        ),
    ).run()

    # Plot OpEx fit diagnostics (Bayesian-specific)
    model.opex_module.plot_fit()
    model.opex_module.plot_fit(use_gaussian_ci=True)
