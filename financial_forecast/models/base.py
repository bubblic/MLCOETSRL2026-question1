"""Composable financial forecasting model — Pareja (2009) Cash Budget.

Provides :class:`BaseFinancialModel`, a concrete model that composes three
financial statement modules with pluggable policy modules:

- :class:`~financial_forecast.models.balance_sheet.BalanceSheetModel`
- :class:`~financial_forecast.models.income_statement.IncomeStatementModel`
- :class:`~financial_forecast.models.cash_budget.CashBudgetModel`

All forecast logic lives here.  :class:`TrainableFinancialModel` extends
this class with training and serialization support.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import tensorflow as tf

from financial_forecast.models.balance_sheet import BalanceSheetModel
from financial_forecast.models.income_statement import IncomeStatementModel
from financial_forecast.models.cash_budget import CashBudgetModel
from financial_forecast.models.tax import SimpleTax, TaxWithAnomalies
from financial_forecast.inference.state_index import RECURRENT_KEYS, DIAGNOSTIC_KEYS


_REQUIRED_KEYS = {
    "sales",
    "purchases",
    "cogs",
    "nca",
    "depreciation",
    "advance_payments_purchases",
    "accounts_receivable",
    "accounts_payable",
    "advance_payments_sales",
    "cash",
    "ims",
    "inventory",
    "current_liabilities",
    "non_current_liabilities",
    "equity",
    "net_income",
    "dividends",
    "stock_buyback",
    "opex",
    "tax",
    "current_lt_debt",
    "interest_payment",
    "ms_return",
}

_SUPPLEMENTAL_KEYS = {"effective_st_debt"}


class BaseFinancialModel(tf.Module):
    """Composable Pareja (2009) Cash Budget financial model.

    Accepts pluggable policy modules for CapEx, working capital, liquidity,
    dividends, buybacks, purchases, debt, OpEx, and tax.  The
    :meth:`forecast_step` and :meth:`forecast_step_compiled` methods
    implement the full single-period forecast pipeline used by both
    forward simulation and gradient-based training.

    Attributes:
        base_year: The fiscal year corresponding to ``t = 0`` in all
            logit-linear time-trend parameters.
        amount_scale: USD-to-scaled-units conversion factor.
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
        self.amount_scale = None
        self.base_year = None
        self.balance_sheet = BalanceSheetModel(
            capex_policy=capex_policy,
            working_capital=working_capital,
            purchases_policy=purchases_policy,
        )
        self.income_statement = IncomeStatementModel(opex_module=opex_module)
        self.cash_budget = CashBudgetModel(
            liquidity_policy=liquidity_policy,
            debt_policy=debt_policy,
            dividend_policy=dividend_policy,
            buyback_policy=buyback_policy,
        )
        self.trajectory_simulator = trajectory_simulator
        if tax_anomalies is not None:
            self.tax_module = TaxWithAnomalies(tax_anomalies)
        else:
            self.tax_module = SimpleTax()

        # Populated by prepare()
        self._d: Dict[str, tf.Tensor] = {}
        self._s: Dict[str, tf.Tensor] = {}
        self._initial_state: Optional[Dict[str, tf.Tensor]] = None
        self._test_years: int = 0

        super().__init__(name=name)

    @property
    def opex_module(self):
        """Convenience accessor for the OpEx module owned by the income statement."""
        return self.income_statement.opex_module

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare(
        self,
        financial_statements: Mapping[str, tf.Tensor],
        inflation: Optional[tf.Tensor] = None,
        test_years: int = 0,
    ) -> None:
        """Ingest historical data, configure sub-modules, build initial state.

        Args:
            financial_statements: Historical financial series in USD.
            inflation: Optional 1-D annual inflation rates.
            test_years: Historical years held out for testing.
        """
        raw = dict(financial_statements)
        self._test_years = test_years

        # Validate
        missing = sorted(_REQUIRED_KEYS.difference(raw.keys()))
        if missing:
            raise ValueError(
                "financial_statements missing required keys: " + ", ".join(missing)
            )

        d = self._d
        for k in _REQUIRED_KEYS:
            d[k] = raw[k]

        n_hist = len(d["sales"])
        d["inflation"] = (
            inflation if inflation is not None else tf.zeros(n_hist, dtype=tf.float64)
        )
        d["effective_st_debt"] = (
            d["current_liabilities"]
            - d["accounts_payable"]
            - d["advance_payments_sales"]
            - d["current_lt_debt"]
        )

        # Scale to billions
        mean_sales = float(tf.reduce_mean(d["sales"]))
        scale = 10 ** int(tf.math.floor(tf.math.log(mean_sales) / tf.math.log(10.0)))
        for key in _REQUIRED_KEYS | _SUPPLEMENTAL_KEYS:
            self._s[key] = d[key] / scale

        # Configure model
        self.base_year = int(raw["years"][0])
        self.amount_scale = scale

        # Initial state: the year just before the test window
        self._initial_state = self._build_state_from_index(-(test_years + 1))

        # Initialize parameters from historical averages
        self._init_parameters_from_data()

    def _init_parameters_from_data(self) -> None:
        """Initialize all policy parameters from historical averages.

        Delegates to each sub-module's ``init_from_data(s)`` method.
        For the simple model this provides the final parameter values;
        for the trainable model these serve as better starting points
        for gradient descent.
        """
        s = self._s
        self.balance_sheet.capex_policy.init_from_data(s)
        self.balance_sheet.working_capital.init_from_data(s)
        self.balance_sheet.purchases_policy.init_from_data(s)
        self.cash_budget.liquidity_policy.init_from_data(s)
        self.cash_budget.dividend_policy.init_from_data(s)
        self.cash_budget.buyback_policy.init_from_data(s)
        self.cash_budget.debt_policy.init_from_data(s)
        self.opex_module.init_from_data(s)
        self.income_statement.init_from_data(s)

    def _build_state_from_index(self, index: int) -> Dict[str, tf.Tensor]:
        """Build a state dict from scaled historical data at *index*."""
        s = self._s
        f64 = lambda v: tf.constant(float(v), dtype=tf.float64)
        return {
            "nca": f64(s["nca"][index]),
            "advance_payments_purchases": f64(s["advance_payments_purchases"][index]),
            "accounts_receivable": f64(s["accounts_receivable"][index]),
            "inventory": f64(s["inventory"][index]),
            "cash": f64(s["cash"][index]),
            "investment_in_market_securities": f64(s["ims"][index]),
            "accounts_payable": f64(s["accounts_payable"][index]),
            "advance_payments_sales": f64(s["advance_payments_sales"][index]),
            "effective_st_debt": f64(s["effective_st_debt"][index]),
            "current_lt_debt": f64(s["current_lt_debt"][index]),
            "non_current_liabilities": f64(s["non_current_liabilities"][index]),
            "equity": f64(s["equity"][index]),
            "net_income": f64(s["net_income"][index]),
            "dividends": f64(s["dividends"][index]),
        }

    # ------------------------------------------------------------------
    # Forecast (single-step evolution)
    # ------------------------------------------------------------------

    @tf.function
    def forecast_step(
        self,
        state,
        inputs,
        use_mean_opex=True,
    ):
        """Advance the financial state by one period (graph-compiled).

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
        _, diagnostics = self.forecast_step_compiled(
            state_tensor,
            sales_t,
            inputs["year"],
            inputs["cum_inflation"],
            use_mean_opex,
        )
        return {key: diagnostics[0, i] for i, key in enumerate(DIAGNOSTIC_KEYS)}

    def forecast_step_compiled(
        self,
        state,
        sales_t,
        year,
        cum_inflation,
        use_mean_opex=True,
    ):
        """Batched single-period forecast -- single source of truth.

        Args:
            state: ``[n_samples, 14]`` recurrent state tensor.
            sales_t: ``[n_samples]`` sales for this period.
            year: Scalar float64 calendar year.
            cum_inflation: Scalar cumulative inflation factor.
            use_mean_opex: If ``True``, use deterministic/mean OpEx.

        Returns:
            Tuple ``(new_state, diagnostics)``.
        """
        time_index = year - tf.constant(
            float(self.base_year),
            dtype=tf.float64,
        )
        assets = self.balance_sheet.evolve_assets(state, sales_t, time_index)
        income = self.income_statement.calculate_income(
            state,
            assets,
            sales_t,
            cum_inflation,
            self.tax_module,
            year,
            use_mean_opex,
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
