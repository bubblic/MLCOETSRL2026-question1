"""Configuration-driven experiment runner for trainable financial models.

Provides :class:`ExperimentConfig` and :func:`run_experiment` to eliminate
duplication across the ``run_trainable_model*.py`` entry-point scripts.
Each script becomes a thin config declaration.

Example::

    from financial_forecast.experiment import ExperimentConfig, run_experiment

    run_experiment(ExperimentConfig(company="aapl"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import tensorflow as tf

from financial_forecast.data.loader import HistoricalDataLoader
from financial_forecast.models.trainable_financial_model import TrainableFinancialModel
from financial_forecast.models.opex import OpExModule, SimpleOpEx
from financial_forecast.models.capex import CapexPolicy
from financial_forecast.models.working_capital import WorkingCapitalPolicy
from financial_forecast.models.liquidity import LiquidityPolicy, CashTargetPolicy
from financial_forecast.models.dividends import DividendPolicy, SimpleDividendPolicy
from financial_forecast.models.buyback import BuybackPolicy, SimpleBuybackPolicy
from financial_forecast.models.purchases import PurchasesPolicy, StaticCostRatioPolicy
from financial_forecast.models.debt import DebtPolicy, SimpleDebtPolicy
from financial_forecast.models.tax import SimpleTax, TaxWithAnomalies
from financial_forecast.inference.trajectory_simulator import (
    TrajectorySimulator,
    DeterministicSimulator,
)
from financial_forecast.training.policy_trainer import PolicyTrainer
from financial_forecast.training.structural_trainer import StructuralTrainer
from financial_forecast.inference.pipeline import ForecastPipeline
from financial_forecast.inference.forecast_driver_models import (
    LinearSalesForecast,
    ConstantInflationForecast,
)


@dataclass
class ExperimentConfig:
    """Declares a complete model training + forecast experiment.

    All policy fields default to their simple (non-trend) implementations.
    Override individual fields to use advanced policies.
    """

    # Data
    company: str = "aapl"
    include_inflation: bool = True
    tax_anomaly_dir: Optional[str] = None

    # Model composition (constructed instances)
    opex_module: OpExModule = field(default_factory=SimpleOpEx)
    trajectory_simulator: TrajectorySimulator = field(
        default_factory=DeterministicSimulator,
    )
    capex_policy: CapexPolicy = field(default_factory=CapexPolicy)
    working_capital: WorkingCapitalPolicy = field(default_factory=WorkingCapitalPolicy)
    liquidity_policy: LiquidityPolicy = field(default_factory=CashTargetPolicy)
    dividend_policy: DividendPolicy = field(default_factory=SimpleDividendPolicy)
    buyback_policy: BuybackPolicy = field(default_factory=SimpleBuybackPolicy)
    purchases_policy: PurchasesPolicy = field(default_factory=StaticCostRatioPolicy)
    debt_policy: DebtPolicy = field(default_factory=SimpleDebtPolicy)

    # Training
    policy_epochs: int = 25000
    structural_epochs: int = 20000
    parameters_save_path: str = "trained_parameters.npz"
    use_trained_parameters: bool = False

    # Forecast
    forecast_years: int = 10
    test_years: int = 1
    show_plot: bool = False

    # Reproducibility
    seed: int = 42


def run_experiment(config: ExperimentConfig) -> None:
    """Execute a complete training + forecast experiment.

    1. Set random seed for reproducibility.
    2. Load historical data (with optional tax anomalies).
    3. Build model from config policy choices.
    4. Prepare model (scale data, initialize parameters).
    5. Train or load parameters.
    6. Run forecast pipeline (simulate, plot, export JSON).

    If the OpEx module is stochastic, OpEx fit diagnostics are plotted
    after the forecast.
    """
    tf.random.set_seed(config.seed)

    # Load data
    data = HistoricalDataLoader(
        config.company,
        include_inflation=config.include_inflation,
        tax_anomaly_dir=config.tax_anomaly_dir,
    )

    # Build tax module: TaxWithAnomalies if anomaly dir was provided
    if config.tax_anomaly_dir is not None:
        tax_module: SimpleTax = TaxWithAnomalies(data.tax_onetime_payments)
    else:
        tax_module = SimpleTax()

    # Build model
    model = TrainableFinancialModel(
        opex_module=config.opex_module,
        trajectory_simulator=config.trajectory_simulator,
        capex_policy=config.capex_policy,
        working_capital=config.working_capital,
        liquidity_policy=config.liquidity_policy,
        dividend_policy=config.dividend_policy,
        buyback_policy=config.buyback_policy,
        purchases_policy=config.purchases_policy,
        debt_policy=config.debt_policy,
        tax_module=tax_module,
    )

    # Prepare
    model.prepare(
        financial_statements=data.financial_statements,
        inflation=data.inflation,
        test_years=config.test_years,
    )

    # Train
    model.train(
        policy_trainer=PolicyTrainer(epochs=config.policy_epochs),
        structural_trainer=StructuralTrainer(epochs=config.structural_epochs),
        parameters_save_path=config.parameters_save_path,
        use_trained_parameters=config.use_trained_parameters,
    )

    # Forecast
    ForecastPipeline(
        model,
        data=data,
        sales_forecast=LinearSalesForecast(
            data.financial_statements["sales"],
            forecast_years=config.forecast_years,
        ),
        inflation_forecast=ConstantInflationForecast(
            data.inflation,
            forecast_years=config.forecast_years,
        ),
        show_plot=config.show_plot,
    ).run()

    # OpEx fit diagnostics (Bayesian-specific)
    if config.opex_module.is_stochastic:
        model.opex_module.plot_fit(show_plot=config.show_plot)
        model.opex_module.plot_fit(
            use_gaussian_ci=True,
            show_plot=config.show_plot,
        )
