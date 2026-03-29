"""Cash budget model -- liquidity management and financing decisions.

Implements the five-module Pareja (2009) cash budget: operating,
investing, external, financing, and owner transactions.
Delegates debt financing decisions to a pluggable ``DebtPolicy``.
"""

from typing import Dict, Tuple

import tensorflow as tf

from financial_forecast.models.liquidity import LiquidityPolicy
from financial_forecast.models.debt import DebtPolicy
from financial_forecast.models.dividends import DividendPolicy
from financial_forecast.models.buyback import BuybackPolicy
from financial_forecast.inference.state_index import (
    R_AR,
    R_AP,
    R_ADV_PP,
    R_ADV_PS,
    R_CASH,
    R_IMS,
    R_NCL,
    R_EQUITY,
    R_NET_INCOME,
    R_DIVIDENDS,
)


class CashBudgetModel(tf.Module):
    """Computes the cash budget, financing, and owner transaction decisions.

    Delegates liquidity to a ``LiquidityPolicy``, ST/LT debt to a
    ``DebtPolicy``, dividends to a ``DividendPolicy``, and buybacks
    to a ``BuybackPolicy``.

    Args:
        liquidity_policy: ``SimpleLiquidityPolicy``, ``TrendLiquidityPolicy``,
            or ``CashTargetPolicy``. If ims_target == None, excess cash after
            all flows is invested in market securities instead of additional buybacks.
        debt_policy: ``SimpleDebtPolicy`` or ``TrendDebtPolicy``.
        dividend_policy: ``SimpleDividendPolicy`` or ``LintnerDividendPolicy``.
        buyback_policy: ``SimpleBuybackPolicy`` or ``BaselineBuybackPolicy``.
    """

    def __init__(
        self,
        liquidity_policy: LiquidityPolicy,
        debt_policy: DebtPolicy,
        dividend_policy: DividendPolicy,
        buyback_policy: BuybackPolicy,
        name: str = "cash_budget",
    ):
        super().__init__(name=name)
        self.liquidity_policy = liquidity_policy
        self.debt_policy = debt_policy
        self.dividend_policy = dividend_policy
        self.buyback_policy = buyback_policy

    def manage_liquidity(
        self,
        state: tf.Tensor,
        assets: Dict[str, tf.Tensor],
        income: Dict[str, tf.Tensor],
        sales_t: tf.Tensor,
        time_index: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Compute cash budget and financing decisions."""
        zero = tf.constant(0.0, dtype=tf.float64)
        ar_prev = state[:, R_AR]
        ap_prev = state[:, R_AP]
        adv_pp_prev = state[:, R_ADV_PP]
        adv_ps_prev = state[:, R_ADV_PS]
        cash_prev = state[:, R_CASH]
        ims_prev = state[:, R_IMS]

        # 1. Operating NLB
        sales_curr = sales_t - assets["ar_curr"] - adv_ps_prev
        adv_ps_curr = assets["adv_ps_curr"]
        inflows = sales_curr + ar_prev + adv_ps_curr

        purchases_curr = assets["purchases_t"] - assets["ap_curr"] - adv_pp_prev
        outflows = (
            purchases_curr
            + ap_prev
            + assets["adv_pp_curr"]
            + income["opex"]
            + income["tax"]
        )
        operating_nlb = inflows - outflows

        # 2. Capital expenditure outflow
        capex_nlb = -assets["capex"]

        # 3. Return on market securities
        external_investment_nlb = income["ms_return"]

        # Liquidity target
        total_liquidity_target, cash_target, ims_target = self.liquidity_policy.compute(
            sales_t,
            time_index,
        )

        # ST liquidity gap
        liquidity_deficit_st = (
            total_liquidity_target
            - (cash_prev + ims_prev)
            - operating_nlb
            + income["principal_st"]
            + income["interest_st"]
        )

        # 4. Debt policy determines ST debt
        eff_st_debt_curr = self.debt_policy.compute_st_debt(
            sales_t,
            time_index,
            liquidity_deficit_st,
        )

        # 5. Transaction with owners
        ni_prev = state[:, R_NET_INCOME]
        div_paid_lastyr = state[:, R_DIVIDENDS]
        dividends_paid_thisyr = self.dividend_policy.compute(ni_prev, div_paid_lastyr)
        stock_buyback = self.buyback_policy.compute(assets["depreciation"])

        liquidity_deficit_lt = (
            liquidity_deficit_st
            - eff_st_debt_curr
            - external_investment_nlb
            - capex_nlb
            + income["principal_lt"]
            + income["interest_lt"]
            + dividends_paid_thisyr
            + stock_buyback
        )
        long_term_financing = tf.maximum(zero, liquidity_deficit_lt)

        # Debt policy determines LT financing mix
        new_lt_loan, equity_financing = self.debt_policy.compute_financing_mix(
            long_term_financing,
            time_index,
        )

        if ims_target is not None:
            # Excess cash from over-borrowing → additional stock buybacks
            excess_cash_buyback = tf.maximum(zero, -liquidity_deficit_lt)
            stock_buyback = stock_buyback + excess_cash_buyback

        financing_nlb = (
            eff_st_debt_curr
            + new_lt_loan
            - income["principal_st"]
            - income["principal_lt"]
            - income["interest_st"]
            - income["interest_lt"]
        )
        transaction_with_owners_nlb = (
            equity_financing - dividends_paid_thisyr - stock_buyback
        )
        total_nlb = (
            operating_nlb
            + capex_nlb
            + financing_nlb
            + external_investment_nlb
            + transaction_with_owners_nlb
        )

        # Compute final cash and IMS
        cash_curr = cash_target
        # For liquidity policy that only defines cash target, the investment in market securities is from any excess cash beyond cash target.
        if ims_target is None:
            # IMS absorbs excess cash after meeting the cash target
            ending_liquidity = (cash_prev + ims_prev) + total_nlb
            ims_curr = tf.maximum(zero, ending_liquidity - cash_curr)
        else:
            ims_curr = ims_target

        liquidity_check = (cash_prev + ims_prev) + total_nlb - cash_curr - ims_curr

        return {
            "adv_ps_curr": adv_ps_curr,
            "eff_st_debt_curr": eff_st_debt_curr,
            "new_lt_loan": new_lt_loan,
            "equity_financing": equity_financing,
            "dividends_curr": dividends_paid_thisyr,
            "stock_buyback": stock_buyback,
            "liquidity_deficit_st": liquidity_deficit_st,
            "liquidity_check": liquidity_check,
            "cash_curr": cash_curr,
            "ims_curr": ims_curr,
        }

    def assemble_state(
        self,
        state: tf.Tensor,
        assets: Dict[str, tf.Tensor],
        income: Dict[str, tf.Tensor],
        financing: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Evolve liabilities, check balance sheet, pack output tensors."""
        ncl_prev = state[:, R_NCL]
        equity_prev = state[:, R_EQUITY]

        ap_curr = assets["ap_curr"]

        # Debt policy evolves LT liabilities
        ncl_curr, cur_lt_debt_curr = self.debt_policy.evolve_lt_liabilities(
            financing["new_lt_loan"],
            ncl_prev,
        )

        equity_curr = (
            equity_prev
            + financing["equity_financing"]
            + income["ni_curr"]
            - financing["dividends_curr"]
            - financing["stock_buyback"]
        )

        total_assets = (
            assets["nca_curr"]
            + assets["adv_pp_curr"]
            + assets["ar_curr"]
            + assets["inv_curr"]
            + financing["cash_curr"]
            + financing["ims_curr"]
        )
        total_liab_equity = (
            ap_curr
            + financing["adv_ps_curr"]
            + financing["eff_st_debt_curr"]
            + cur_lt_debt_curr
            + ncl_curr
            + equity_curr
        )
        check = total_assets - total_liab_equity

        new_state = tf.stack(
            [
                assets["nca_curr"],
                assets["adv_pp_curr"],
                assets["ar_curr"],
                assets["inv_curr"],
                financing["cash_curr"],
                financing["ims_curr"],
                ap_curr,
                financing["adv_ps_curr"],
                financing["eff_st_debt_curr"],
                cur_lt_debt_curr,
                ncl_curr,
                equity_curr,
                income["ni_curr"],
                financing["dividends_curr"],
            ],
            axis=1,
        )

        diagnostics = tf.stack(
            [
                total_assets,
                assets["nca_curr"],
                assets["adv_pp_curr"],
                assets["ar_curr"],
                assets["inv_curr"],
                financing["cash_curr"],
                financing["ims_curr"],
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
                financing["dividends_curr"],
                financing["stock_buyback"],
                financing["new_lt_loan"],
                financing["equity_financing"],
                financing["liquidity_deficit_st"],
                financing["liquidity_check"],
                check,
            ],
            axis=1,
        )

        return new_state, diagnostics
