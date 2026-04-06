"""Run a simple Pareja (2009) balance sheet forecast -- no training.

Demonstrates the Cash Budget construction by running a deterministic
forward simulation with policy parameters set from historical averages.
No gradient-based optimization is performed; this is a pure forward
projection of the balance sheet using the Pareja framework equations.

Policies:
    - OpEx: SimpleOpEx (deterministic linear)
    - Liquidity: CashTargetPolicy (fixed cash-to-sales target)
    - Dividends: SimpleDividendPolicy (constant payout ratio)
    - Buybacks: SimpleBuybackPolicy (depreciation multiple)
    - Purchases: StaticCostRatioPolicy (fixed cost-of-revenue ratio)
    - Debt: SimpleDebtPolicy (deficit-driven, no trend)
    - Tax: SimpleTax (flat effective rate)

Outputs:
    - Forecast plots saved to ``training_results/``
    - Forecast report JSON saved to ``training_results/forecast_report.json``

Usage::

    python run_simple_model_forecast.py
"""

import tensorflow as tf

from financial_forecast.data.loader import HistoricalDataLoader
from financial_forecast.models.base import BaseFinancialModel
from financial_forecast.models.opex import SimpleOpEx
from financial_forecast.inference.trajectory_simulator import DeterministicSimulator
from financial_forecast.models.liquidity import CashTargetPolicy
from financial_forecast.models.dividends import SimpleDividendPolicy
from financial_forecast.models.buyback import SimpleBuybackPolicy
from financial_forecast.models.purchases import StaticCostRatioPolicy
from financial_forecast.models.debt import SimpleDebtPolicy
from financial_forecast.models.capex import CapexPolicy
from financial_forecast.models.working_capital import WorkingCapitalPolicy
from financial_forecast.models.tax import SimpleTax
from financial_forecast.inference.pipeline import ForecastPipeline
from financial_forecast.inference.forecast_driver_models import (
    LinearSalesForecast,
    ConstantInflationForecast,
)


if __name__ == "__main__":

    tf.random.set_seed(42)

    # -- Step 1: Load historical financial data --
    data = HistoricalDataLoader("aapl", include_inflation=True)

    # -- Step 2: Build model with simple (non-trainable) policies --
    model = BaseFinancialModel(
        opex_module=SimpleOpEx(),
        trajectory_simulator=DeterministicSimulator(),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=CashTargetPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
        purchases_policy=StaticCostRatioPolicy(),
        debt_policy=SimpleDebtPolicy(),
        tax_module=SimpleTax(),
    )

    # -- Step 3: Prepare model (scale data, initialize parameters) --
    model.prepare(
        financial_statements=data.financial_statements,
        inflation=data.inflation,
    )

    # -- Step 4: Run forecast pipeline (simulate, plot, export JSON) --
    ForecastPipeline(
        model,
        data=data,
        sales_forecast=LinearSalesForecast(
            data.financial_statements["sales"],
            forecast_years=4,
        ),
        inflation_forecast=ConstantInflationForecast(
            data.inflation,
            forecast_years=4,
        ),
    ).run()
