"""Abstract base class for all financial forecasting models.

Extracts the shared Pareja (2009) Cash Budget construction logic that is
common to :class:`SimpleFinancialModel` and :class:`TrainableFinancialModel`.
Concrete subclasses override ``_initialize_parameters`` to choose between
fixed ``tf.constant`` values, trainable ``tf.Variable`` values, or Bayesian
``TransformedVariable`` distributions.

Why TensorFlow instead of NumPy?
    Using TensorFlow primitives (``tf.Tensor``, ``tf.Variable``) enables:
    - **Automatic differentiation** via ``tf.GradientTape`` for training.
    - **GPU acceleration** for large-scale Monte Carlo simulations.
    - **Graph-mode optimization** via ``@tf.function`` for faster execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import tensorflow as tf

from financial_forecast.types import EconomicInputs, FinancialState


class BaseFinancialModel(tf.Module, ABC):
    """Abstract base class for Pareja (2009) Cash Budget financial models.

    Subclasses must implement ``_initialize_parameters`` to set model
    parameters as either constants, trainable variables, or distributions.

    The ``forecast_step`` method implements the full single-period forecast
    pipeline: asset evolution, income statement, liquidity/financing, and
    final state assembly with balance-sheet identity checks.
    """

    def __init__(self, name: str | None = None):
        """Initialize the base model.

        Args:
            name: Optional name for the ``tf.Module``.
        """
        super().__init__(name=name)
        self._initialize_parameters()

    @abstractmethod
    def _initialize_parameters(self) -> None:
        """Initialize model parameters.

        Subclasses set attributes like ``self.asset_growth``,
        ``self.depreciation_rate``, etc.  These may be ``tf.constant``
        (deterministic), ``tf.Variable`` (trainable), or
        ``tfp.util.TransformedVariable`` (Bayesian).
        """

    def forecast_step(
        self,
        state: Union[FinancialState, Dict[str, Any]],
        inputs: EconomicInputs,
    ) -> FinancialState:
        """Perform a single-period financial forecast.

        Implements the Pareja (2009) Cash Budget construction by:
        1. Evolving asset accounts based on policy parameters.
        2. Computing the income statement (COGS, OpEx, interest, tax).
        3. Determining financing needs via the liquidity budget.
        4. Assembling the final balance-sheet state with identity checks.

        Args:
            state: Previous period financial state (t-1).
            inputs: Economic drivers for the current period (t).

        Returns:
            ``FinancialState`` for the predicted period.
        """
        if isinstance(state, dict):
            state = FinancialState.from_dict(state)

        assets = self._evolve_assets(state, inputs)
        income = self._calculate_income_statement(state, inputs, assets)
        financing = self._manage_liquidity_and_financing(state, inputs, assets, income)
        return self._assemble_final_state(state, assets, income, financing, inputs)

    def _evolve_assets(
        self, state: FinancialState, inputs: EconomicInputs
    ) -> Dict[str, Any]:
        """Update asset accounts based on sales and growth policy.

        Args:
            state: Previous period financial state.
            inputs: Current period economic inputs.

        Returns:
            Dictionary of updated asset values and intermediate quantities.
        """
        depr = state.nca * self.depreciation_rate
        capex = depr + (inputs.sales_t * self.asset_growth)
        nca_curr = state.nca - depr + capex

        adv_pp_curr = inputs.purchases_t_plus_1 * self.advance_payments_purchases_pct
        ar_curr = inputs.sales_t * self.account_receivables_pct
        inv_curr = inputs.sales_t * self.inventory_pct

        total_liq_curr = inputs.sales_t * self.total_liquidity_pct
        cash_curr = total_liq_curr * self.cash_pct_of_liquidity
        ims_curr = total_liq_curr * (1 - self.cash_pct_of_liquidity)

        return {
            "nca": nca_curr,
            "depreciation": depr,
            "capex": capex,
            "advance_payments_purchases": adv_pp_curr,
            "accounts_receivable": ar_curr,
            "inventory": inv_curr,
            "cash": cash_curr,
            "investment_in_market_securities": ims_curr,
            "total_liquidity": total_liq_curr,
        }

    def _calculate_income_statement(
        self,
        state: FinancialState,
        inputs: EconomicInputs,
        assets: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute income statement components for the current period.

        Args:
            state: Previous period financial state.
            inputs: Current period economic inputs.
            assets: Updated asset values from ``_evolve_assets``.

        Returns:
            Dictionary of income statement items.
        """
        cogs = state.inventory + inputs.purchases_t - assets["inventory"]
        opex = (
            self.baseline_opex * inputs.cum_inflation
            + inputs.sales_t * self.variable_opex_pct
        )
        ebitda = inputs.sales_t - cogs - opex

        # Debt servicing based on PREVIOUS debt levels (avoids circularity)
        prin_lt_due = state.non_current_liabilities / (self.avg_maturity_years - 1)
        int_lt = (
            self.avg_long_term_interest_pct
            * state.non_current_liabilities
            / (1 - 1 / self.avg_maturity_years)
        )
        prin_st_due = state.current_liabilities - prin_lt_due
        int_st = self.avg_short_term_interest_pct * prin_st_due

        ms_return = (
            state.investment_in_market_securities * self.market_securities_return_pct
        )

        ebt = ebitda - assets["depreciation"] - (int_st + int_lt) + ms_return
        tax = ebt * self.income_tax_pct
        ni_curr = ebt - tax

        return {
            "cogs": cogs,
            "opex": opex,
            "ebitda": ebitda,
            "net_income": ni_curr,
            "tax": tax,
            "interest_total": int_st + int_lt,
            "principal_st": prin_st_due,
            "principal_lt": prin_lt_due,
            "ms_return": ms_return,
            "interest_st": int_st,
            "interest_lt": int_lt,
        }

    def _manage_liquidity_and_financing(
        self,
        state: FinancialState,
        inputs: EconomicInputs,
        assets: Dict[str, Any],
        income: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine cash budget and new financing requirements.

        Implements the five-module liquidity budget: operating, investing,
        financing, owner transactions, and discretionary.

        Args:
            state: Previous period financial state.
            inputs: Current period economic inputs.
            assets: Updated asset values.
            income: Income statement items.

        Returns:
            Dictionary of financing decisions.
        """
        adv_sales_curr = inputs.sales_t_plus_1 * self.advance_payments_sales_pct
        sales_cash_in = (
            inputs.sales_t * (1 - self.account_receivables_pct)
            - state.advance_payments_sales
            + state.accounts_receivable
            + adv_sales_curr
        )

        purch_cash_out = (
            inputs.purchases_t * (1 - self.account_payables_pct)
            - state.advance_payments_purchases
            + state.accounts_payable
            + assets["advance_payments_purchases"]
        )
        op_outflows = purch_cash_out + income["opex"] + income["tax"]
        op_nlb = sales_cash_in - op_outflows

        prev_total_liq = state.cash + state.investment_in_market_securities
        liq_deficit_st = (
            assets["total_liquidity"]
            - prev_total_liq
            - income["ms_return"]
            - op_nlb
            + income["principal_st"]
            + income["interest_st"]
        )
        new_st_loan = tf.maximum(0.0, liq_deficit_st)

        divs = state.net_income * self.dividend_payout_ratio_pct
        bb = assets["depreciation"] * self.stock_buyback_pct

        liq_deficit_lt = (
            liq_deficit_st
            - new_st_loan
            + assets["capex"]
            + income["principal_lt"]
            + income["interest_lt"]
            + divs
            + bb
        )

        lt_fin_needed = tf.maximum(0.0, liq_deficit_lt)
        equity_fin = lt_fin_needed * self.equity_financing_pct
        new_lt_loan = lt_fin_needed * (1 - self.equity_financing_pct)

        fin_nlb = (
            new_st_loan
            + new_lt_loan
            - income["principal_st"]
            - income["principal_lt"]
            - income["interest_total"]
        )
        owners_nlb = equity_fin - divs - bb

        total_nlb = (
            op_nlb - assets["capex"] + fin_nlb + income["ms_return"] + owners_nlb
        )

        return {
            "new_st_loan": new_st_loan,
            "new_lt_loan": new_lt_loan,
            "equity_financing": equity_fin,
            "dividends": divs,
            "buybacks": bb,
            "advance_payments_sales": adv_sales_curr,
            "total_nlb": total_nlb,
        }

    def _assemble_final_state(
        self,
        prev_state: FinancialState,
        assets: Dict[str, Any],
        income: Dict[str, Any],
        financing: Dict[str, Any],
        inputs: EconomicInputs,
    ) -> FinancialState:
        """Assemble the final balance-sheet state with identity checks.

        Args:
            prev_state: Previous period financial state.
            assets: Updated asset values.
            income: Income statement items.
            financing: Financing decisions.
            inputs: Current period economic inputs.

        Returns:
            New ``FinancialState`` with diagnostic check fields.
        """
        ap_curr = inputs.purchases_t * self.account_payables_pct

        total_long_term_liabilities = (
            financing["new_lt_loan"] + prev_state.non_current_liabilities
        )

        ncl_curr = total_long_term_liabilities * (1 - 1 / self.avg_maturity_years)
        cl_curr = (
            financing["new_st_loan"]
            + total_long_term_liabilities / self.avg_maturity_years
        )

        equity_curr = (
            prev_state.equity
            + financing["equity_financing"]
            + income["net_income"]
            - financing["dividends"]
            - financing["buybacks"]
        )

        # Balance sheet identity: Assets = Liabilities + Equity
        total_assets = (
            assets["nca"]
            + assets["advance_payments_purchases"]
            + assets["accounts_receivable"]
            + assets["inventory"]
            + assets["cash"]
            + assets["investment_in_market_securities"]
        )
        total_liab_eq = (
            ap_curr
            + financing["advance_payments_sales"]
            + cl_curr
            + ncl_curr
            + equity_curr
        )

        prev_total_liq = prev_state.cash + prev_state.investment_in_market_securities
        liq_check = prev_total_liq + financing["total_nlb"] - assets["total_liquidity"]

        return FinancialState(
            nca=assets["nca"],
            advance_payments_purchases=assets["advance_payments_purchases"],
            accounts_receivable=assets["accounts_receivable"],
            inventory=assets["inventory"],
            cash=assets["cash"],
            investment_in_market_securities=assets["investment_in_market_securities"],
            accounts_payable=ap_curr,
            advance_payments_sales=financing["advance_payments_sales"],
            current_liabilities=cl_curr,
            non_current_liabilities=ncl_curr,
            equity=equity_curr,
            net_income=income["net_income"],
            liquidity_check=liq_check,
            balance_sheet_check=total_assets - total_liab_eq,
            st_loan_issued=financing["new_st_loan"],
            lt_loan_issued=financing["new_lt_loan"],
            st_principal_paid=income["principal_st"],
            lt_principal_paid=income["principal_lt"],
        )
