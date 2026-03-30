"""Run the trainable financial model with BayesianOpEx, tax anomalies, and advanced policies.

The most complete model configuration: combines Bayesian OpEx (variational
inference with Monte Carlo sampling), LLM-extracted one-time tax anomalies,
and all advanced trend-based policies.

Policies:
    - OpEx: BayesianOpEx (variational inference, stochastic sampling)
    - Liquidity: TrendLiquidityPolicy (time-varying cash/IMS targets)
    - Dividends: LintnerDividendPolicy (partial-adjustment model)
    - Buybacks: BaselineBuybackPolicy (depreciation-scaled baseline)
    - Purchases: TrendCostRatioPolicy (time-varying cost ratio)
    - Debt: TrendDebtPolicy (logit-linear ST debt + NCL decay)
    - Tax: TaxWithAnomalies (flat rate + LLM-extracted one-time payments)

Training:
    - PolicyTrainer: 25,000 epochs (includes VI for BayesianOpEx)
    - StructuralTrainer: 20,000 epochs

Outputs:
    - Trained parameters saved to
      ``trained_parameters_adv_policies_w_bayesianopex_taxanomaly.npz``
    - Monte Carlo forecast plots saved to ``training_results/``
    - OpEx fit diagnostics (Monte Carlo and Gaussian CI)
    - Forecast report JSON saved to ``training_results/forecast_report.json``

Usage::

    python run_trainable_model_forecast_adv_policies_w_bayesianopex_taxanomaly.py
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
from financial_forecast.models.tax import TaxWithAnomalies
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

    company = "aapl"

    # -- Step 1: Load historical data with tax anomalies from extracted JSON --
    data = HistoricalDataLoader(
        company,
        include_inflation=True,
        tax_anomaly_dir=f"./extracted_json/tax_anomalies/{company}",
    )

    # -- Step 2: Build model with BayesianOpEx + tax anomalies + advanced policies --
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
        tax_module=TaxWithAnomalies(data.tax_onetime_payments),
    )

    # -- Step 3: Prepare model (scale data, initialize parameters, fit VI) --
    model.prepare(
        financial_statements=data.financial_statements,
        inflation=data.inflation,
        test_years=1,
    )

    # -- Step 4: Train policy and structural parameters --
    model.train(
        trainers=[PolicyTrainer(epochs=25000), StructuralTrainer(epochs=20000)],
        parameters_save_path="trained_parameters_adv_policies_w_bayesianopex_taxanomaly.npz",
    )

    # -- Step 5: Run Monte Carlo forecast pipeline (simulate, plot, export JSON) --
    ForecastPipeline(
        model,
        data=data,
        sales_forecast=LinearSalesForecast(
            data.financial_statements["sales"],
            forecast_years=8,
        ),
        inflation_forecast=ConstantInflationForecast(
            data.inflation,
            forecast_years=8,
        ),
    ).run()

    # -- Step 6: Plot OpEx fit diagnostics (Bayesian-specific) --
    model.opex_module.plot_fit()
    model.opex_module.plot_fit(use_gaussian_ci=True)
