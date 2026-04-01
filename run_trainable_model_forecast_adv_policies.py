"""Trainable model with advanced trend-based policies, deterministic OpEx."""

from financial_forecast.experiment import ExperimentConfig, run_experiment
from financial_forecast.models.liquidity import TrendLiquidityPolicy
from financial_forecast.models.dividends import LintnerDividendPolicy
from financial_forecast.models.buyback import BaselineBuybackPolicy
from financial_forecast.models.purchases import TrendCostRatioPolicy
from financial_forecast.models.debt import TrendDebtPolicy

if __name__ == "__main__":
    run_experiment(
        ExperimentConfig(
            company="aapl",
            liquidity_policy=TrendLiquidityPolicy(),
            dividend_policy=LintnerDividendPolicy(),
            buyback_policy=BaselineBuybackPolicy(),
            purchases_policy=TrendCostRatioPolicy(),
            debt_policy=TrendDebtPolicy(),
            parameters_save_path="trained_parameters_adv_policies.npz",
        )
    )
