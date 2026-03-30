"""Run the trainable financial model with advanced policies.

Trains policy and structural parameters via gradient descent, then
runs a deterministic 10-year forecast. Uses trend-based policies for
all modules but keeps deterministic OpEx (SimpleOpEx).

Policies:
    - OpEx: SimpleOpEx (deterministic linear)
    - Liquidity: TrendLiquidityPolicy (time-varying cash/IMS targets)
    - Dividends: LintnerDividendPolicy (partial-adjustment model)
    - Buybacks: BaselineBuybackPolicy (depreciation-scaled baseline)
    - Purchases: TrendCostRatioPolicy (time-varying cost ratio)
    - Debt: TrendDebtPolicy (logit-linear ST debt + NCL decay)
    - Tax: SimpleTax (flat effective rate)

Training:
    - PolicyTrainer: 25,000 epochs
    - StructuralTrainer: 20,000 epochs

Outputs:
    - Trained parameters saved to ``trained_parameters_adv_policies.npz``
    - Forecast plots saved to ``training_results/``
    - Forecast report JSON saved to ``training_results/forecast_report.json``

Usage::

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
from financial_forecast.models.tax import SimpleTax
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

    # -- Step 1: Load historical financial data --
    data = HistoricalDataLoader(
        "aapl",
        include_inflation=True,
    )

    # -- Step 2: Build model with advanced trend-based policies --
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
        tax_module=SimpleTax(),
    )

    # -- Step 3: Prepare model (scale data, initialize parameters) --
    model.prepare(
        financial_statements=data.financial_statements,
        inflation=data.inflation,
        test_years=1,
    )

    # -- Step 4: Train policy and structural parameters --
    model.train(
        trainers=[PolicyTrainer(epochs=25000), StructuralTrainer(epochs=20000)],
        parameters_save_path="trained_parameters_adv_policies.npz",
    )

    # -- Step 5: Run forecast pipeline (simulate, plot, export JSON) --
    ForecastPipeline(
        model,
        data=data,
        sales_forecast=LinearSalesForecast(
            data.financial_statements["sales"],
            forecast_years=10,
        ),
        inflation_forecast=ConstantInflationForecast(
            data.inflation,
            forecast_years=10,
        ),
    ).run()
