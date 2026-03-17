"""Deterministic financial forecasting model with fixed parameters.

This module provides :class:`SimpleFinancialModel`, a concrete implementation
of :class:`BaseFinancialModel` that uses ``tf.constant`` values for all
model parameters.  Because the parameters are not trainable, the model
produces a fully deterministic forecast suitable for scenario analysis
and baseline projections.

The forecast logic (asset evolution, income statement, liquidity budget,
and balance-sheet assembly) is inherited from the base class; only the
parameter initialization is overridden here.
"""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

from financial_forecast.types import EconomicInputs, FinancialState
from financial_forecast.models.base import BaseFinancialModel


class SimpleFinancialModel(BaseFinancialModel):
    """Deterministic financial model with fixed policy and structural parameters.

    All parameters are set as ``tf.constant`` values, meaning they cannot be
    updated via gradient descent.  This makes the model appropriate for
    forward-looking scenario analysis once parameters have been calibrated
    externally (e.g., from historical averages).

    Example:
        >>> model = SimpleFinancialModel()
        >>> state = model.forecast_step(initial_state, economic_inputs)
    """

    def _initialize_parameters(self) -> None:
        """Initialize all model parameters as fixed ``tf.constant`` values.

        Parameters are grouped into policy parameters (asset growth,
        depreciation, working-capital ratios, OpEx structure, and
        shareholder-return policies) and structural parameters (interest
        rates, debt maturity, securities returns, and financing mix).
        """
        # --- Policy Parameters ---
        self.asset_growth = tf.constant(0.0076, dtype=tf.float64)
        self.depreciation_rate = tf.constant(0.055, dtype=tf.float64)
        self.advance_payments_sales_pct = tf.constant(
            0.020614523, dtype=tf.float64
        )
        self.advance_payments_purchases_pct = tf.constant(
            0.073525733, dtype=tf.float64
        )
        self.account_receivables_pct = tf.constant(
            0.159111366, dtype=tf.float64
        )
        self.account_payables_pct = tf.constant(0.35014191, dtype=tf.float64)
        self.inventory_pct = tf.constant(0.0165, dtype=tf.float64)
        self.total_liquidity_pct = tf.constant(0.16, dtype=tf.float64)
        self.cash_pct_of_liquidity = tf.constant(0.487, dtype=tf.float64)
        self.income_tax_pct = tf.constant(0.147, dtype=tf.float64)
        self.variable_opex_pct = tf.constant(0.222168147, dtype=tf.float64)
        self.baseline_opex = tf.constant(-30306718214.0, dtype=tf.float64)
        self.dividend_payout_ratio_pct = tf.constant(0.15, dtype=tf.float64)
        self.stock_buyback_pct = tf.constant(7.5, dtype=tf.float64)

        # --- Structural Parameters ---
        self.avg_short_term_interest_pct = tf.constant(0.6, dtype=tf.float64)
        self.avg_long_term_interest_pct = tf.constant(0.06, dtype=tf.float64)
        self.avg_maturity_years = tf.constant(3.0, dtype=tf.float64)
        self.market_securities_return_pct = tf.constant(0.05, dtype=tf.float64)
        self.equity_financing_pct = tf.constant(0.15, dtype=tf.float64)


def run_forecast() -> None:
    """Execute a deterministic forecast simulation using Apple base data.

    Creates a :class:`SimpleFinancialModel`, initializes a 2023 balance
    sheet state, and rolls forward three periods using analyst consensus
    sales and purchase forecasts.  Results are printed as a summary table.
    """
    model = SimpleFinancialModel()

    # Initial State (2023 Apple Balance Sheet)
    initial_data = {
        "nca": 2.1735e11,
        "advance_payments_purchases": 21223000000,
        "accounts_receivable": 60932000000,
        "inventory": 4946000000,
        "cash": 23646000000,
        "investment_in_market_securities": 24658000000,
        "accounts_payable": 64115000000,
        "advance_payments_sales": 7912000000,
        "current_liabilities": 81955000000,
        "non_current_liabilities": 1.48101e11,
        "equity": 50672000000,
        "net_income": 99803000000,
    }
    state = FinancialState.from_dict(initial_data)

    # Forecast Inputs
    sales_forecast = [3.94328e11, 3.83285e11, 3.91035e11, 4.16161e11]
    purch_forecast = [2.07694e11, 1.99862e11, 2.04003e11, 2.10808e11]
    inflation = [0.0] * 4

    print(
        f"\n{'Year':<5} | {'Assets':<15} | {'Liabilities':<15} "
        f"| {'Equity':<15} | {'Check':<15}"
    )
    print("-" * 75)

    # Initial print
    total_assets = (
        state.nca
        + state.advance_payments_purchases
        + state.accounts_receivable
        + state.inventory
        + state.cash
        + state.investment_in_market_securities
    )
    total_liabilities = (
        state.accounts_payable
        + state.advance_payments_sales
        + state.current_liabilities
        + state.non_current_liabilities
    )

    print(
        f"{0:<5} | {total_assets / 1e9:>14.2f}B | "
        f"{total_liabilities / 1e9:>14.2f}B | "
        f"{state.equity / 1e9:>14.2f}B | "
        f"{state.balance_sheet_check:>14.2f}"
    )

    # Forecast loop
    for t in range(len(sales_forecast) - 1):
        inputs = EconomicInputs(
            sales_t=tf.constant(sales_forecast[t], dtype=tf.float64),
            purchases_t=tf.constant(purch_forecast[t], dtype=tf.float64),
            sales_t_plus_1=tf.constant(sales_forecast[t + 1], dtype=tf.float64),
            purchases_t_plus_1=tf.constant(
                purch_forecast[t + 1], dtype=tf.float64
            ),
            cum_inflation=tf.constant(1.0, dtype=tf.float64),
            t=t + 1,
        )
        state = model.forecast_step(state, inputs)

        curr_assets = (
            state.nca
            + state.advance_payments_purchases
            + state.accounts_receivable
            + state.inventory
            + state.cash
            + state.investment_in_market_securities
        )
        curr_liabilities = (
            state.accounts_payable
            + state.advance_payments_sales
            + state.current_liabilities
            + state.non_current_liabilities
        )

        print(
            f"{t + 1:<5} | {curr_assets.numpy() / 1e9:>14.2f}B | "
            f"{curr_liabilities.numpy() / 1e9:>14.2f}B | "
            f"{state.equity.numpy() / 1e9:>14.2f}B | "
            f"{state.balance_sheet_check.numpy():>14.2f}"
        )


if __name__ == "__main__":
    run_forecast()
