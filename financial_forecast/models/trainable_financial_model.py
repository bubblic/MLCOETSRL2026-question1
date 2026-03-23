"""Trainable financial model — Pareja (2009) Cash Budget construction.

Composes three financial statement modules:

- :class:`~financial_forecast.models.balance_sheet.BalanceSheetModel`
- :class:`~financial_forecast.models.income_statement.IncomeStatementModel`
- :class:`~financial_forecast.models.cash_budget.CashBudgetModel`

Plus pluggable OpEx and tax modules.
"""

import tensorflow as tf

from financial_forecast.models.base import BaseFinancialModel
from financial_forecast.models.balance_sheet import BalanceSheetModel
from financial_forecast.models.income_statement import IncomeStatementModel
from financial_forecast.models.cash_budget import CashBudgetModel
from financial_forecast.models.tax import SimpleTax, TaxWithAnomalies
from financial_forecast.serialization.parameter_io import (
    save_parameters as _save_parameters,
    load_parameters as _load_parameters,
)
from financial_forecast.inference.state_index import RECURRENT_KEYS, DIAGNOSTIC_KEYS
from financial_forecast.inference.trajectory_simulator import (
    MonteCarloSimulator,
    DeterministicSimulator,
)


class TrainableFinancialModel(BaseFinancialModel):
    """Bayesian financial model using variational inference for OpEx estimation.

    Combines deterministic policy parameters with Bayesian variational
    inference for operating expenses. Structural parameters (interest rates,
    maturity, financing mix) form a third trainable layer.

    All constrained parameters use bijector-based reparameterization
    (Softplus for non-negative, Sigmoid for [0,1]-bounded, Chain for
    lower-bounded).

    Attributes:
        base_year: The fiscal year corresponding to ``t = 0`` in all
            logit-linear time-trend parameters.  Set by the pipeline
            from the historical data's ``"years"`` field.
    """

    def __init__(
        self,
        opex_module,
        trajectory_simulator,
        capex_policy,
        working_capital,
        liquidity_policy,
        dividend_policy,
        buyback_policy,
        purchases_policy,
        debt_policy,
        tax_anomalies=None,
        name=None,
    ):
        """Create a new trainable financial model.

        Args:
            opex_module: OpEx module (``SimpleOpEx`` or ``BayesianOpEx``).
            trajectory_simulator: Trajectory simulator
                (``DeterministicSimulator`` or ``MonteCarloSimulator``).
            capex_policy: Capital expenditure policy (``CapexPolicy``).
            working_capital: Working capital ratios
                (``WorkingCapitalPolicy``).
            liquidity_policy: Liquidity allocation
                (``SimpleLiquidityPolicy`` or ``TrendLiquidityPolicy``).
            dividend_policy: Dividend policy
                (``SimpleDividendPolicy`` or ``LintnerDividendPolicy``).
            buyback_policy: Buyback policy
                (``SimpleBuybackPolicy`` or ``BaselineBuybackPolicy``).
            purchases_policy: Purchases/cost ratio
                (``StaticCostRatioPolicy`` or ``TrendCostRatioPolicy``).
            debt_policy: Debt financing policy
                (``SimpleDebtPolicy`` or ``TrendDebtPolicy``).
            tax_anomalies: Optional dict of one-time tax amounts by year.
                If ``None``, uses ``SimpleTax``.
            name: Optional name for the underlying ``tf.Module``.
        """
        self.amount_scale = None
        self.base_year = None
        self.opex_module = opex_module
        self.balance_sheet = BalanceSheetModel(
            capex_policy=capex_policy,
            working_capital=working_capital,
            liquidity_policy=liquidity_policy,
            purchases_policy=purchases_policy,
        )
        self.income_statement = IncomeStatementModel()
        self.cash_budget = CashBudgetModel(
            debt_policy=debt_policy,
            dividend_policy=dividend_policy,
            buyback_policy=buyback_policy,
        )
        self.trajectory_simulator = trajectory_simulator
        if tax_anomalies is not None:
            self.tax_module = TaxWithAnomalies(tax_anomalies)
        else:
            self.tax_module = SimpleTax()

        super().__init__(name=name)

    def prepare_for_training(self, years, amount_scale):
        """Configure data-derived model settings.

        Called by the pipeline after computing ``amount_scale``.

        Args:
            years: 1-D tensor of fiscal year labels.
            amount_scale: USD-to-scaled-units conversion factor.
            scaled_sales: 1-D tensor of historical sales (already scaled).
            scaled_opex: 1-D tensor of historical OpEx (already scaled).
            inflation: 1-D tensor of annual inflation rates (or zeros).
        """
        self.base_year = int(years[0])
        self.amount_scale = amount_scale

    # ------------------------------------------------------------------
    # Abstract method implementation (required by BaseFinancialModel)
    # ------------------------------------------------------------------

    def _initialize_parameters(self) -> None:
        """No-op -- all parameters are owned by sub-modules."""

    # ------------------------------------------------------------------
    # Serialization (delegates to serialization module)
    # ------------------------------------------------------------------

    def save_parameters(self, path):
        """Save all model parameters to an .npz file.

        Args:
            path: Filesystem path for the output ``.npz`` file.
        """
        _save_parameters(self, path)

    def load_parameters(self, path):
        """Load model parameters from an .npz file.

        Args:
            path: Filesystem path to the ``.npz`` parameter file.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        _load_parameters(self, path)

    # ------------------------------------------------------------------
    # Forecast (overrides base class template)
    # ------------------------------------------------------------------

    @tf.function
    def forecast_step(
        self,
        state,
        inputs,
        use_mean_opex=False,
    ):
        """Advance the financial state by one period (graph-compiled).

        Thin wrapper around :meth:`forecast_step_compiled` that converts
        between the dict-based interface and packed tensor representation.

        Args:
            state: Dict at *t-1* with balance-sheet entries.
            inputs: Dict with ``sales_t``, ``year``, ``cum_inflation``.
            use_mean_opex: Use posterior mean (no sampling/noise).

        Returns:
            Dict mapping output keys to scalar tensors for period *t*.
        """
        state_tensor = tf.expand_dims(
            tf.stack([tf.cast(state[k], tf.float64) for k in RECURRENT_KEYS]),
            0,
        )
        sales_t = tf.reshape(inputs["sales_t"], [1])
        opex = self.opex_module.predict(
            sales_t,
            inputs["cum_inflation"],
            use_mean=use_mean_opex,
        )

        _, diagnostics = self.forecast_step_compiled(
            state_tensor,
            sales_t,
            inputs["year"],
            opex,
        )

        return {key: diagnostics[0, i] for i, key in enumerate(DIAGNOSTIC_KEYS)}

    def forecast_step_compiled(self, state, sales_t, year, opex):
        """Batched single-period forecast -- single source of truth.

        Orchestrates the three financial statement components:
        balance sheet, income statement, and cash budget.

        Args:
            state: ``[n_samples, 14]`` recurrent state tensor.
            sales_t: ``[n_samples]`` sales for this period.
            year: Scalar float64 calendar year.
            opex: ``[n_samples]`` pre-computed operating expenses.

        Returns:
            Tuple ``(new_state, diagnostics)`` where *new_state* has shape
            ``[n_samples, 14]`` and *diagnostics* has shape
            ``[n_samples, 27]``.
        """
        time_index = year - tf.constant(float(self.base_year), dtype=tf.float64)

        assets = self.balance_sheet.evolve_assets(state, sales_t, time_index)
        income = self.income_statement.calculate_income(
            state,
            assets,
            sales_t,
            opex,
            self.tax_module,
            year,
        )
        financing = self.cash_budget.manage_liquidity(
            state,
            assets,
            income,
            sales_t,
            time_index,
        )
        return self.cash_budget.assemble_state(
            state,
            assets,
            income,
            financing,
        )
