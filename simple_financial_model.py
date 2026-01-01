"""
Simple Financial Model using TensorFlow.

This module provides a deterministic financial forecasting model based on the
Cash Budget construction logic (Pareja, 2009). It uses constant parameters
for forecasting future financial states.

Software Engineering Principles Applied:
- Modularity: Logic is divided into discrete, manageable steps.
- Readability: Use of descriptive variable names and PEP 257 docstrings.
- Type Safety: Integration of dataclasses and type hints.
"""

import tensorflow as tf
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Union, Any, Optional


@dataclass(frozen=True)
class FinancialState:
    """Represents the financial state of a company at a specific point in time."""

    nca: Any  # Non-current assets
    advance_payments_purchases: Any
    accounts_receivable: Any
    inventory: Any
    cash: Any
    investment_in_market_securities: Any
    accounts_payable: Any
    advance_payments_sales: Any
    current_liabilities: Any
    non_current_liabilities: Any
    equity: Any
    net_income: Any

    # Diagnostic fields
    liquidity_check: Any = 0.0
    check: Any = 0.0
    stloan: Any = 0.0
    ltloan: Any = 0.0
    st_principal_paid: Any = 0.0
    lt_principal_paid: Any = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinancialState":
        """Create a FinancialState from a dictionary, mapping legacy keys if needed."""
        mapping = {
            "adv_pp": "advance_payments_purchases",
            "adv_ps": "advance_payments_sales",
            "ar": "accounts_receivable",
            "inv": "inventory",
            "ap": "accounts_payable",
            "cl": "current_liabilities",
            "ncl": "non_current_liabilities",
            "ni": "net_income",
            "ims": "investment_in_market_securities",
        }
        mapped_data = {mapping.get(k, k): v for k, v in data.items()}
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in mapped_data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass(frozen=True)
class EconomicInputs:
    """Represents external economic drivers for a period."""

    sales_t: Any
    purchases_t: Any
    sales_t_plus_1: Any
    purchases_t_plus_1: Any
    inflation: Any
    t: int


class SimpleFinancialModel(tf.Module):
    """
    A deterministic financial forecasting model.

    Implements the Pareja (2009) Cash Budget logic using fixed policy
    and structural parameters.
    """

    def __init__(self):
        super().__init__()
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize fixed model parameters."""
        self.asset_growth = tf.constant(0.0076, dtype=tf.float64)
        self.depreciation_rate = tf.constant(0.055, dtype=tf.float64)
        self.advance_payments_sales_pct = tf.constant(0.020614523, dtype=tf.float64)
        self.advance_payments_purchases_pct = tf.constant(0.073525733, dtype=tf.float64)
        self.account_receivables_pct = tf.constant(0.159111366, dtype=tf.float64)
        self.account_payables_pct = tf.constant(0.35014191, dtype=tf.float64)
        self.inventory_pct = tf.constant(0.0165, dtype=tf.float64)
        self.total_liquidity_pct = tf.constant(0.16, dtype=tf.float64)
        self.cash_pct_of_liquidity = tf.constant(0.487, dtype=tf.float64)
        self.income_tax_pct = tf.constant(0.147, dtype=tf.float64)
        self.variable_opex_pct = tf.constant(0.222168147, dtype=tf.float64)
        self.baseline_opex = tf.constant(-30306718214.0, dtype=tf.float64)
        self.avg_short_term_interest_pct = tf.constant(0.6, dtype=tf.float64)
        self.avg_long_term_interest_pct = tf.constant(0.06, dtype=tf.float64)
        self.avg_maturity_years = tf.constant(3.0, dtype=tf.float64)
        self.market_securities_return_pct = tf.constant(0.05, dtype=tf.float64)
        self.equity_financing_pct = tf.constant(0.15, dtype=tf.float64)
        self.dividend_payout_ratio_pct = tf.constant(0.15, dtype=tf.float64)
        self.stock_buyback_pct = tf.constant(7.5, dtype=tf.float64)

    def forecast_step(
        self, state: Union[FinancialState, Dict[str, Any]], inputs: EconomicInputs
    ) -> FinancialState:
        """
        Calculate the next financial state.

        Args:
            state: The previous financial state (t-1).
            inputs: Economic drivers for the current period (t).

        Returns:
            The predicted financial state at time t.
        """
        if isinstance(state, dict):
            state = FinancialState.from_dict(state)

        # 1. Assets
        assets = self._evolve_assets(state, inputs)

        # 2. Income Statement
        income = self._calculate_income_statement(state, inputs, assets)

        # 3. Liquidity and Financing
        financing = self._manage_liquidity_and_financing(state, inputs, assets, income)

        # 4. Final state assembly
        return self._assemble_final_state(state, assets, income, financing, inputs)

    def _evolve_assets(
        self, state: FinancialState, inputs: EconomicInputs
    ) -> Dict[str, Any]:
        """Update asset accounts based on sales and growth policy."""
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
        self, state: FinancialState, inputs: EconomicInputs, assets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate income statement components."""
        cogs = state.inventory + inputs.purchases_t - assets["inventory"]
        opex = (
            self.baseline_opex * (1 + inputs.inflation) ** inputs.t
            + inputs.sales_t * self.variable_opex_pct
        )
        ebitda = inputs.sales_t - cogs - opex

        # Debt servicing (based on t-1 state)
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
            "net_income": ni_curr,
            "opex": opex,
            "ms_return": ms_return,
            "principal_st": prin_st_due,
            "principal_lt": prin_lt_due,
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
        """Calculate cash budget and financing requirements."""
        # Flows
        adv_sales_curr = inputs.sales_t_plus_1 * self.advance_payments_sales_pct
        sales_in = (
            inputs.sales_t * (1 - self.account_receivables_pct)
            - state.advance_payments_sales
            + state.accounts_receivable
            + adv_sales_curr
        )

        purch_out = (
            inputs.purchases_t * (1 - self.account_payables_pct)
            - state.advance_payments_purchases
            + state.accounts_payable
            + assets["advance_payments_purchases"]
        )
        op_out = (
            purch_out
            + income["opex"]
            + (
                income["net_income"] / (1 - self.income_tax_pct)
                - (
                    income["net_income"] / (1 - self.income_tax_pct)
                    - (
                        inputs.sales_t
                        - (state.inventory + inputs.purchases_t - assets["inventory"])
                        - income["opex"]
                    )
                    + assets["depreciation"]
                    + (income["interest_st"] + income["interest_lt"])
                    - income["ms_return"]
                )
            )
        )  # wait, original tax logic
        # Actually original tax was: tax = ebt * self.income_tax_pct
        # Let's just use the tax from income dict.

        # Recalculating operating flows based on original logic:
        tax = (income["net_income"] / (1 - self.income_tax_pct)) * self.income_tax_pct
        op_nlb = sales_in - (purch_out + income["opex"] + tax)

        # Deficits
        prev_total_liq = state.cash + state.investment_in_market_securities
        liq_def_st = (
            assets["total_liquidity"]
            - prev_total_liq
            - income["ms_return"]
            - op_nlb
            + income["principal_st"]
            + income["interest_st"]
        )
        new_st = tf.maximum(0.0, liq_def_st)

        divs = state.net_income * self.dividend_payout_ratio_pct
        bb = assets["depreciation"] * self.stock_buyback_pct

        liq_def_lt = (
            liq_def_st
            - new_st
            + assets["capex"]
            + income["principal_lt"]
            + income["interest_lt"]
            + divs
            + bb
        )
        lt_fin = tf.maximum(0.0, liq_def_lt)
        equity_fin = lt_fin * self.equity_financing_pct
        new_lt = lt_fin * (1 - self.equity_financing_pct)

        # Total NLB check
        fin_nlb = (
            new_st
            + new_lt
            - income["principal_st"]
            - income["principal_lt"]
            - (income["interest_st"] + income["interest_lt"])
        )
        owners_nlb = equity_fin - divs - bb
        total_nlb = (
            op_nlb - assets["capex"] + fin_nlb + income["ms_return"] + owners_nlb
        )

        return {
            "new_st": new_st,
            "new_lt": new_lt,
            "equity_financing": equity_fin,
            "dividends": divs,
            "buybacks": bb,
            "adv_sales": adv_sales_curr,
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
        """Assemble the final state and perform identity checks."""
        ap_curr = inputs.purchases_t * self.account_payables_pct
        ncl_curr = (financing["new_lt"] + prev_state.non_current_liabilities) * (
            (self.avg_maturity_years - 1) / self.avg_maturity_years
        )
        cl_curr = financing["new_st"] + ncl_curr / (self.avg_maturity_years - 1)
        equity_curr = (
            prev_state.equity
            + financing["equity_financing"]
            + income["net_income"]
            - financing["dividends"]
            - financing["buybacks"]
        )

        # Balance sheet identity
        total_assets = (
            assets["nca"]
            + assets["advance_payments_purchases"]
            + assets["accounts_receivable"]
            + assets["inventory"]
            + assets["cash"]
            + assets["investment_in_market_securities"]
        )
        total_liab_eq = (
            ap_curr + financing["adv_sales"] + cl_curr + ncl_curr + equity_curr
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
            advance_payments_sales=financing["adv_sales"],
            current_liabilities=cl_curr,
            non_current_liabilities=ncl_curr,
            equity=equity_curr,
            net_income=income["net_income"],
            liquidity_check=liq_check,
            check=total_assets - total_liab_eq,
            stloan=financing["new_st"],
            ltloan=financing["new_lt"],
            st_principal_paid=income["principal_st"],
            lt_principal_paid=income["principal_lt"],
        )


def run_forecast():
    """Execute the deterministic forecast simulation."""
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
        f"\n{'Year':<5} | {'Assets':<15} | {'Liabilities':<15} | {'Equity':<15} | {'Check':<15}"
    )
    print("-" * 75)

    # Initial Print
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
        f"{0:<5} | {total_assets/1e9:>14.2f}B | {total_liabilities/1e9:>14.2f}B | {state.equity/1e9:>14.2f}B | {state.check:>14.2f}"
    )

    # Forecast Loop
    for t in range(len(sales_forecast) - 1):
        inputs = EconomicInputs(
            sales_t=tf.constant(sales_forecast[t], dtype=tf.float64),
            purchases_t=tf.constant(purch_forecast[t], dtype=tf.float64),
            sales_t_plus_1=tf.constant(sales_forecast[t + 1], dtype=tf.float64),
            purchases_t_plus_1=tf.constant(purch_forecast[t + 1], dtype=tf.float64),
            inflation=tf.constant(inflation[t], dtype=tf.float64),
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
            f"{t+1:<5} | {curr_assets.numpy()/1e9:>14.2f}B | {curr_liabilities.numpy()/1e9:>14.2f}B | {state.equity.numpy()/1e9:>14.2f}B | {state.check.numpy():>14.2f}"
        )


if __name__ == "__main__":
    run_forecast()
