"""Run a simple Pareja (2009) balance sheet forecast -- no training.

Demonstrates the Cash Budget construction by running a deterministic
forward simulation with policy parameters set from historical averages.
No gradient-based optimization is performed; this is a pure forward
projection of the balance sheet using the Pareja framework equations.

Usage:
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
from financial_forecast.training.pipeline import ForecastPipeline
from financial_forecast.inference.forecast_driver_models import (
    LinearSalesForecast,
    ConstantInflationForecast,
)


if __name__ == "__main__":

    tf.random.set_seed(42)

    data = HistoricalDataLoader("aapl", include_inflation=True)

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
    )

    model.prepare(
        financial_statements=data.financial_statements,
        inflation=data.inflation,
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
