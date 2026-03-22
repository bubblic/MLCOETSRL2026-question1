"""Cash budget model — liquidity management and financing decisions.

Implements the five-module Pareja (2009) cash budget: operating,
investing, external, financing, and owner transactions.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from financial_forecast.inference.state_index import (
    R_AR, R_AP, R_ADV_PP, R_ADV_PS, R_CASH, R_IMS, R_NCL, R_EQUITY,
)

tfb = tfp.bijectors


class CashBudgetModel(tf.Module):
    """Computes the cash budget and financing decisions.

    Owns short-term debt policy, equity financing mix, and debt
    maturity parameters.
    """

    def __init__(self, name="cash_budget"):
        super().__init__(name=name)

        # --- Short-term debt policy (logit-linear trend) ---
        self.st_debt_alpha = tf.Variable(
            -1.59, dtype=tf.float64, name="st_debt_alpha",
        )
        self.st_debt_beta = tf.Variable(
            0.0, dtype=tf.float64, name="st_debt_beta",
        )

        # --- Equity financing mix (logit-linear trend) ---
        self.ef_alpha = tf.Variable(-1.73, dtype=tf.float64, name="ef_alpha")
        self.ef_beta = tf.Variable(0.0, dtype=tf.float64, name="ef_beta")

        # --- Debt maturity ---
        self.avg_maturity_years = tfp.util.TransformedVariable(
            initial_value=3.0,
            bijector=tfb.Chain(
                [tfb.Shift(tf.constant(1.001, dtype=tf.float64)), tfb.Softplus()]
            ),
            dtype=tf.float64, name="avg_maturity_years",
        )

    def manage_liquidity(self, state, assets, income, sales_t, time_index,
                          balance_sheet):
        """Compute cash budget and financing decisions.

        Args:
            state: ``[n_samples, 14]`` recurrent state tensor.
            assets: Dict from ``BalanceSheetModel.evolve_assets()``.
            income: Dict from ``IncomeStatementModel.calculate_income()``.
            sales_t: ``[n_samples]`` sales.
            time_index: Scalar ``year - base_year``.
            balance_sheet: ``BalanceSheetModel`` instance (for cross-cutting
                working capital ratios).

        Returns:
            Dict with financing decisions.
        """
        zero = tf.constant(0.0, dtype=tf.float64)
        ar_prev = state[:, R_AR]
        ap_prev = state[:, R_AP]
        adv_pp_prev = state[:, R_ADV_PP]
        adv_ps_prev = state[:, R_ADV_PS]
        cash_prev = state[:, R_CASH]
        ims_prev = state[:, R_IMS]

        # 1. Operating NLB
        sales_curr = (
            sales_t * (1 - balance_sheet.account_receivables_pct) - adv_ps_prev
        )
        adv_ps_curr = sales_t * balance_sheet.advance_payments_sales_pct
        inflows = sales_curr + ar_prev + adv_ps_curr

        purchases_curr = (
            assets["purchases_t"] * (1 - balance_sheet.account_payables_pct)
            - adv_pp_prev
        )
        outflows = (
            purchases_curr + ap_prev + assets["adv_pp_curr"]
            + income["opex"] + income["tax"]
        )
        operating_nlb = inflows - outflows

        # 2. Capital expenditure outflow
        capex_nlb = -assets["capex"]

        # 3. Return on market securities
        external_investment_nlb = income["ms_return"]

        # 4. Financing: ST debt is policy-driven
        st_debt_pct = tf.sigmoid(
            self.st_debt_alpha + self.st_debt_beta * time_index
        )
        eff_st_debt_curr = sales_t * st_debt_pct

        liquidity_deficit_st = (
            assets["total_liquidity_curr"]
            - (cash_prev + ims_prev)
            - operating_nlb
            + income["principal_st"]
            + income["interest_st"]
        )

        dividends_prev = assets["dividends_prev"]
        stock_buyback = assets["stock_buyback_base"]

        liquidity_deficit_lt = (
            liquidity_deficit_st
            - eff_st_debt_curr
            - external_investment_nlb
            - capex_nlb
            + income["principal_lt"]
            + income["interest_lt"]
            + dividends_prev
            + stock_buyback
        )
        long_term_financing = tf.maximum(zero, liquidity_deficit_lt)

        # Equity-financing mix
        ef_pct = tf.sigmoid(self.ef_alpha + self.ef_beta * time_index)
        new_lt_loan = long_term_financing * (1 - ef_pct)
        equity_financing = long_term_financing * ef_pct

        excess_cash_buyback = tf.maximum(zero, -liquidity_deficit_lt)
        stock_buyback = stock_buyback + excess_cash_buyback

        financing_nlb = (
            eff_st_debt_curr + new_lt_loan
            - income["principal_st"] - income["principal_lt"]
            - income["interest_st"] - income["interest_lt"]
        )
        # 5. Transaction with owners
        transaction_with_owners_nlb = (
            equity_financing - dividends_prev - stock_buyback
        )
        total_nlb = (
            operating_nlb + capex_nlb + financing_nlb
            + external_investment_nlb + transaction_with_owners_nlb
        )
        liquidity_check = (
            (cash_prev + ims_prev) + total_nlb - assets["total_liquidity_curr"]
        )

        return {
            "adv_ps_curr": adv_ps_curr,
            "eff_st_debt_curr": eff_st_debt_curr,
            "new_lt_loan": new_lt_loan,
            "equity_financing": equity_financing,
            "dividends_prev": dividends_prev,
            "stock_buyback": stock_buyback,
            "liquidity_deficit_st": liquidity_deficit_st,
            "liquidity_check": liquidity_check,
        }

    def assemble_state(self, state, assets, income, financing,
                        balance_sheet):
        """Evolve liabilities, check balance sheet, pack output tensors.

        Args:
            state: ``[n_samples, 14]`` recurrent state tensor.
            assets: Dict from ``BalanceSheetModel``.
            income: Dict from ``IncomeStatementModel``.
            financing: Dict from ``self.manage_liquidity()``.
            balance_sheet: ``BalanceSheetModel`` instance.

        Returns:
            Tuple ``(new_state, diagnostics)`` as ``[n_samples, 14]``
            and ``[n_samples, 27]``.
        """
        ncl_prev = state[:, R_NCL]
        equity_prev = state[:, R_EQUITY]

        ap_curr = assets["purchases_t"] * balance_sheet.account_payables_pct
        total_lt_liabilities = financing["new_lt_loan"] + ncl_prev
        ncl_curr = total_lt_liabilities * (1 - 1 / self.avg_maturity_years)
        cur_lt_debt_curr = total_lt_liabilities / self.avg_maturity_years

        equity_curr = (
            equity_prev + financing["equity_financing"] + income["ni_curr"]
            - financing["dividends_prev"] - financing["stock_buyback"]
        )

        total_assets = (
            assets["nca_curr"] + assets["adv_pp_curr"] + assets["ar_curr"]
            + assets["inv_curr"] + assets["cash_curr"] + assets["ims_curr"]
        )
        total_liab_equity = (
            ap_curr + financing["adv_ps_curr"] + financing["eff_st_debt_curr"]
            + cur_lt_debt_curr + ncl_curr + equity_curr
        )
        check = total_assets - total_liab_equity

        new_state = tf.stack([
            assets["nca_curr"],
            assets["adv_pp_curr"],
            assets["ar_curr"],
            assets["inv_curr"],
            assets["cash_curr"],
            assets["ims_curr"],
            ap_curr,
            financing["adv_ps_curr"],
            financing["eff_st_debt_curr"],
            cur_lt_debt_curr,
            ncl_curr,
            equity_curr,
            income["ni_curr"],
            financing["dividends_prev"],
        ], axis=1)

        diagnostics = tf.stack([
            total_assets,
            assets["nca_curr"],
            assets["adv_pp_curr"],
            assets["ar_curr"],
            assets["inv_curr"],
            assets["cash_curr"],
            assets["ims_curr"],
            ap_curr,
            financing["adv_ps_curr"],
            financing["eff_st_debt_curr"],
            cur_lt_debt_curr,
            ncl_curr,
            equity_curr,
            income["ni_curr"],
            assets["depreciation"],
            income["cogs"],
            income["opex"],
            income["tax"],
            income["ms_return"],
            income["interest_lt"] + income["interest_st"],
            financing["dividends_prev"],
            financing["stock_buyback"],
            financing["new_lt_loan"],
            financing["equity_financing"],
            financing["liquidity_deficit_st"],
            financing["liquidity_check"],
            check,
        ], axis=1)

        return new_state, diagnostics
