"""Trainable model with BayesianOpEx + advanced trend-based policies."""

from financial_forecast.experiment import ExperimentConfig, run_experiment
from financial_forecast.models.opex import BayesianOpEx
from financial_forecast.inference.trajectory_simulator import MonteCarloSimulator
from financial_forecast.models.liquidity import TrendLiquidityPolicy
from financial_forecast.models.dividends import LintnerDividendPolicy
from financial_forecast.models.buyback import BaselineBuybackPolicy
from financial_forecast.models.purchases import TrendCostRatioPolicy
from financial_forecast.models.debt import TrendDebtPolicy

if __name__ == "__main__":
    run_experiment(
        ExperimentConfig(
            company="aapl",
            opex_module=BayesianOpEx(),
            trajectory_simulator=MonteCarloSimulator(n_samples=1000),
            liquidity_policy=TrendLiquidityPolicy(),
            dividend_policy=LintnerDividendPolicy(),
            buyback_policy=BaselineBuybackPolicy(),
            purchases_policy=TrendCostRatioPolicy(),
            debt_policy=TrendDebtPolicy(),
            parameters_save_path="trained_parameters_adv_policies_w_bayesianopex.npz",
        )
    )
