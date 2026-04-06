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
from financial_forecast.models.opex import OpExModule
from financial_forecast.models.capex import CapexPolicy
from financial_forecast.models.working_capital import WorkingCapitalPolicy
from financial_forecast.models.liquidity import LiquidityPolicy
from financial_forecast.models.dividends import DividendPolicy
from financial_forecast.models.buyback import BuybackPolicy
from financial_forecast.models.purchases import PurchasesPolicy
from financial_forecast.models.debt import DebtPolicy
from financial_forecast.models.tax import SimpleTax
from financial_forecast.inference.trajectory_simulator import TrajectorySimulator
from financial_forecast.inference.state_index import RECURRENT_KEYS, DIAGNOSTIC_KEYS
from financial_forecast.types import (
    RecurrentState,
    ForecastInputs,
    HistoricalTrainingData,
    validate_recurrent_state,
)


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
        opex_module: OpExModule,
        trajectory_simulator: TrajectorySimulator,
        capex_policy: CapexPolicy,
        working_capital: WorkingCapitalPolicy,
        liquidity_policy: LiquidityPolicy,
        dividend_policy: DividendPolicy,
        buyback_policy: BuybackPolicy,
        purchases_policy: PurchasesPolicy,
        debt_policy: DebtPolicy,
        tax_module: SimpleTax,
        name: Optional[str] = None,
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
        self.tax_module = tax_module

        # Populated by prepare()
        self._historical_data: Dict[str, tf.Tensor] = {}
        self._scaled_data: Dict[str, tf.Tensor] = {}
        self._initial_state: Optional[Dict[str, tf.Tensor]] = None
        self._test_years: int = 0

        super().__init__(name=name)

    @property
    def opex_module(self) -> OpExModule:
        """Convenience accessor for the OpEx module owned by the income statement."""
        return self.income_statement.opex_module

    @property
    def historical_data(self) -> Dict[str, tf.Tensor]:
        """Raw historical financial data in USD (read-only)."""
        return dict(self._historical_data)

    @property
    def scaled_data(self) -> Dict[str, tf.Tensor]:
        """Historical data scaled to model units (read-only)."""
        return dict(self._scaled_data)

    @property
    def initial_state(self) -> RecurrentState:
        """Balance-sheet state at the forecast start point."""
        return dict(self._initial_state)

    @property
    def test_years(self) -> int:
        """Number of historical years held out for testing."""
        return self._test_years

    # ------------------------------------------------------------------
    # Trainable variable aggregation
    # ------------------------------------------------------------------

    @property
    def policy_trainable_variables(self) -> list:
        """All variables optimized in the policy training phase."""
        return [
            *self.balance_sheet.trainable_variables,
            *self.tax_module.trainable_variables,
            *self.cash_budget.liquidity_policy.trainable_variables,
            *self.cash_budget.debt_policy.policy_trainable_variables,
            *self.cash_budget.dividend_policy.trainable_variables,
            *self.cash_budget.buyback_policy.trainable_variables,
            *self.opex_module.trainable_variables,
        ]

    @property
    def structural_trainable_variables(self) -> list:
        """All variables optimized in the structural training phase."""
        return [
            *self.income_statement.structural_trainable_variables,
            *self.cash_budget.debt_policy.structural_trainable_variables,
        ]

    # ------------------------------------------------------------------
    # Loss aggregation (raw, unscaled)
    # ------------------------------------------------------------------

    def compute_policy_losses(
        self,
        data: HistoricalTrainingData,
    ) -> Dict[str, tf.Tensor]:
        """Compute all raw (unscaled) policy loss components.

        Delegates to each sub-module's ``loss()`` method with
        ``scale=1.0``.  The trainer is responsible for scaling.

        Args:
            data: Historical training data (scaled).

        Returns:
            Dict mapping loss names to scalar tensors.
        """
        _one = tf.constant(1.0, dtype=tf.float64)

        # Alignment: growth/depr use [1:] vs [:-1]
        delta_nca = data.nca[1:] - data.nca[:-1]
        depr_aligned = data.depreciation[1:]
        sales_aligned = data.sales[1:]
        nca_prev = data.nca[:-1]

        # Alignment: Lintner dividend smoothing
        div_aligned = data.dividends[1:]
        ni_prev = data.net_income[:-1]
        div_prev = data.dividends[:-1]

        # Derived: cumulative inflation and time indices
        cum_inf = tf.math.cumprod(1 + data.inflation)
        time_indices = tf.cast(data.years, tf.float64) - tf.constant(
            float(self.base_year), dtype=tf.float64
        )

        loss_growth, loss_depr, prior_loss_am = self.balance_sheet.capex_policy.loss(
            delta_nca,
            depr_aligned,
            sales_aligned,
            nca_prev,
            _one,
            _one,
        )
        loss_adv_ps, loss_adv_pp, loss_ar, loss_ap, loss_inv = (
            self.balance_sheet.working_capital.loss(
                data.sales,
                data.purchases,
                data.cogs,
                data.advance_payments_sales,
                data.advance_payments_purchases,
                data.accounts_receivable,
                data.accounts_payable,
                data.inventory,
                _one,
                _one,
                _one,
                _one,
                _one,
            )
        )
        loss_tl, loss_cash = self.cash_budget.liquidity_policy.loss(
            data.sales,
            data.cash,
            data.ims,
            time_indices,
            _one,
            _one,
        )
        loss_tax = self.tax_module.loss(data.tax, data.net_income, _one)
        loss_div = self.cash_budget.dividend_policy.loss(
            ni_prev,
            div_aligned,
            div_prev,
            _one,
        )
        loss_bb = self.cash_budget.buyback_policy.loss(
            data.stock_buyback,
            data.depreciation,
            _one,
        )
        loss_cost_ratio = self.balance_sheet.purchases_policy.loss(
            data.sales,
            data.cogs,
            data.inventory,
            time_indices,
            _one,
        )
        loss_eff_st_debt = self.cash_budget.debt_policy.loss_st_debt(
            data.effective_st_debt,
            data.sales,
            time_indices,
            _one,
        )
        loss_opex = self.opex_module.loss(
            data.opex,
            data.sales,
            cum_inf,
            _one,
        )

        return {
            "growth": loss_growth,
            "depr": loss_depr,
            "prior_am": prior_loss_am,
            "adv_ps": loss_adv_ps,
            "adv_pp": loss_adv_pp,
            "ar": loss_ar,
            "ap": loss_ap,
            "inv": loss_inv,
            "tl": loss_tl,
            "cash": loss_cash,
            "tax": loss_tax,
            "div": loss_div,
            "bb": loss_bb,
            "cost_ratio": loss_cost_ratio,
            "eff_st_debt": loss_eff_st_debt,
            "opex": loss_opex,
        }

    # ------------------------------------------------------------------
    # Summary printing
    # ------------------------------------------------------------------

    def print_policy_summary(self, n_years: int) -> None:
        """Print all policy parameter summaries."""
        self.balance_sheet.capex_policy.print_summary()
        self.balance_sheet.working_capital.print_summary()
        self.cash_budget.liquidity_policy.print_summary(n_years)
        self.tax_module.print_summary()
        self.cash_budget.dividend_policy.print_summary()
        self.cash_budget.buyback_policy.print_summary()
        self.cash_budget.debt_policy.print_policy_summary(n_years)
        self.balance_sheet.purchases_policy.print_summary(n_years)
        self.opex_module.print_summary()

    def print_structural_summary(self, n_years: int) -> None:
        """Print all structural parameter summaries."""
        self.income_statement.print_summary()
        self.cash_budget.debt_policy.print_structural_summary(n_years)

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

        d = self._historical_data
        for k in _REQUIRED_KEYS:
            d[k] = raw[k]

        # # Replace NaN with 0 in balance sheet fields that enter the
        # # recurrent state.  NaN (e.g. unreported current_lt_debt)
        # # would otherwise propagate through forecast_step and corrupt
        # # all structural parameter gradients.
        # _NAN_FILL_KEYS = (
        #     "nca", "advance_payments_purchases", "accounts_receivable",
        #     "inventory", "cash", "ims", "accounts_payable",
        #     "advance_payments_sales", "current_lt_debt",
        #     "current_liabilities", "non_current_liabilities", "equity",
        #     "net_income", "dividends",
        # )
        # for k in _NAN_FILL_KEYS:
        #     if k in d:
        #         d[k] = tf.where(tf.math.is_nan(d[k]), tf.zeros_like(d[k]), d[k])

        n_hist = len(d["sales"])
        if inflation is not None:
            if len(inflation) < n_hist:
                raise ValueError(
                    f"inflation has {len(inflation)} values but "
                    f"financial_statements has {n_hist} years"
                )
            # Align: take the last n_hist values (most recent years)
            d["inflation"] = inflation[-n_hist:]
        else:
            d["inflation"] = tf.zeros(n_hist, dtype=tf.float64)
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
            self._scaled_data[key] = d[key] / scale

        # Configure model
        self.base_year = int(raw["years"][0])
        self.amount_scale = scale

        # Initial state: the year just before the test window
        self._initial_state = self.build_state_from_index(-(test_years + 1))
        validate_recurrent_state(self._initial_state, "prepare")

        # Initialize parameters from historical averages
        self._init_parameters_from_data()

    def _init_parameters_from_data(self) -> None:
        """Initialize all policy parameters from historical averages.

        Delegates to each sub-module's ``init_from_data(s)`` method.
        For the simple model this provides the final parameter values;
        for the trainable model these serve as better starting points
        for gradient descent.
        """
        s = self._scaled_data
        self.balance_sheet.capex_policy.init_from_data(s)
        self.balance_sheet.working_capital.init_from_data(s)
        self.balance_sheet.purchases_policy.init_from_data(s)
        self.cash_budget.liquidity_policy.init_from_data(s)
        self.cash_budget.dividend_policy.init_from_data(s)
        self.cash_budget.buyback_policy.init_from_data(s)
        self.cash_budget.debt_policy.init_from_data(s)
        self.opex_module.init_from_data(s)
        self.income_statement.init_from_data(s)

    def build_state_from_index(self, index: int) -> RecurrentState:
        """Build a recurrent state dict from scaled historical data at *index*."""
        s = self._scaled_data
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
        state: RecurrentState,
        inputs: ForecastInputs,
        use_mean_opex: bool = True,
    ) -> Dict[str, tf.Tensor]:
        """Advance the financial state by one period (graph-compiled).

        Args:
            state: :class:`RecurrentState` dict at *t-1*.
            inputs: :class:`ForecastInputs` dict with ``sales_t``,
                ``year``, ``cum_inflation``.
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
        state: tf.Tensor,
        sales_t: tf.Tensor,
        year: tf.Tensor,
        cum_inflation: tf.Tensor,
        use_mean_opex: bool = True,
    ) -> tuple[tf.Tensor, tf.Tensor]:
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
