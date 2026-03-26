"""Run the financial model with all advanced policies.

Uses trend-based liquidity, Lintner dividends, baseline buybacks,
trend cost ratio, and trend debt — but with deterministic (SimpleOpEx).

Usage:
    python run_trainable_model_forecast_adv_policies.py
"""

from financial_forecast.data.loader import HistoricalDataLoader
from financial_forecast.models.trainable_financial_model import TrainableFinancialModel
from financial_forecast.models.opex import SimpleOpEx
from financial_forecast.inference.trajectory_simulator import DeterministicSimulator
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
import tensorflow as tf


if __name__ == "__main__":

    tf.random.set_seed(42)

    data = HistoricalDataLoader(
        "aapl",
        include_inflation=True,
    )

    model = TrainableFinancialModel(
        opex_module=SimpleOpEx(),
        trajectory_simulator=DeterministicSimulator(),
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
        forecast_years=10,
        test_years=1,
    )

    model.train(
        trainers=[PolicyTrainer(epochs=25000), StructuralTrainer(epochs=20000)],
        parameters_save_path="trained_parameters_adv_policies.npz",
    )

    ForecastPipeline(model).run()
