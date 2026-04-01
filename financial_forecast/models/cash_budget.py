"""Cash budget model -- liquidity management and financing decisions.

Implements the five-module Pareja (2009) cash budget: operating,
investing, external, financing, and owner transactions.
Delegates debt financing decisions to a pluggable ``DebtPolicy``.
"""

from typing import Dict, Optional, Tuple

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
        """Compute cash budget and financing decisions (orchestrator)."""
        zero = tf.constant(0.0, dtype=tf.float64)
        cash_prev = state[:, R_CASH]
        ims_prev = state[:, R_IMS]

        operating_nlb = self._compute_operating_nlb(state, assets, income, sales_t)
        capex_nlb = -assets["capex"]
        external_investment_nlb = income["ms_return"]

        liquidity_deficit_st, cash_target, ims_target = self._compute_liquidity_gap(
            operating_nlb,
            cash_prev,
            ims_prev,
            income,
            sales_t,
            time_index,
        )

        eff_st_debt_curr = self.debt_policy.compute_st_debt(
            sales_t,
            time_index,
            liquidity_deficit_st,
        )
        dividends, stock_buyback = self._compute_owner_transactions(state, assets)

        liquidity_deficit_lt = (
            liquidity_deficit_st
            - eff_st_debt_curr
            - external_investment_nlb
            - capex_nlb
            + income["principal_lt"]
            + income["interest_lt"]
            + dividends
            + stock_buyback
        )

        new_lt_loan, equity_financing = self._compute_lt_financing(
            liquidity_deficit_lt,
            time_index,
        )

        if ims_target is not None:
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
        transaction_with_owners_nlb = equity_financing - dividends - stock_buyback
        total_nlb = (
            operating_nlb
            + capex_nlb
            + financing_nlb
            + external_investment_nlb
            + transaction_with_owners_nlb
        )

        cash_curr, ims_curr, liquidity_check = self._reconcile_cash_and_ims(
            cash_target,
            ims_target,
            cash_prev,
            ims_prev,
            total_nlb,
        )

        return {
            "adv_ps_curr": assets["adv_ps_curr"],
            "eff_st_debt_curr": eff_st_debt_curr,
            "new_lt_loan": new_lt_loan,
            "equity_financing": equity_financing,
            "dividends_curr": dividends,
            "stock_buyback": stock_buyback,
            "liquidity_deficit_st": liquidity_deficit_st,
            "liquidity_check": liquidity_check,
            "cash_curr": cash_curr,
            "ims_curr": ims_curr,
        }

    # ------------------------------------------------------------------
    # Cash budget sub-computations
    # ------------------------------------------------------------------

    def _compute_operating_nlb(
        self,
        state: tf.Tensor,
        assets: Dict[str, tf.Tensor],
        income: Dict[str, tf.Tensor],
        sales_t: tf.Tensor,
    ) -> tf.Tensor:
        """Net liquid balance from core operations: collections minus disbursements."""
        ar_prev = state[:, R_AR]
        ap_prev = state[:, R_AP]
        adv_pp_prev = state[:, R_ADV_PP]
        adv_ps_prev = state[:, R_ADV_PS]

        sales_collected = sales_t - assets["ar_curr"] - adv_ps_prev
        inflows = sales_collected + ar_prev + assets["adv_ps_curr"]

        purchases_paid = assets["purchases_t"] - assets["ap_curr"] - adv_pp_prev
        outflows = (
            purchases_paid
            + ap_prev
            + assets["adv_pp_curr"]
            + income["opex"]
            + income["tax"]
        )
        return inflows - outflows

    def _compute_liquidity_gap(
        self,
        operating_nlb: tf.Tensor,
        cash_prev: tf.Tensor,
        ims_prev: tf.Tensor,
        income: Dict[str, tf.Tensor],
        sales_t: tf.Tensor,
        time_index: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """Liquidity target minus available funds -- drives ST borrowing need.

        Returns:
            Tuple ``(liquidity_deficit_st, cash_target, ims_target)``.
        """
        total_liq_target, cash_target, ims_target = self.liquidity_policy.compute(
            sales_t,
            time_index,
        )
        liquidity_deficit_st = (
            total_liq_target
            - (cash_prev + ims_prev)
            - operating_nlb
            + income["principal_st"]
            + income["interest_st"]
        )
        return liquidity_deficit_st, cash_target, ims_target

    def _compute_owner_transactions(
        self,
        state: tf.Tensor,
        assets: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Dividends and stock buybacks based on prior-period income.

        Returns:
            Tuple ``(dividends, stock_buyback)``.
        """
        ni_prev = state[:, R_NET_INCOME]
        div_paid_lastyr = state[:, R_DIVIDENDS]
        dividends = self.dividend_policy.compute(ni_prev, div_paid_lastyr)
        stock_buyback = self.buyback_policy.compute(assets["depreciation"])
        return dividends, stock_buyback

    def _compute_lt_financing(
        self,
        liquidity_deficit_lt: tf.Tensor,
        time_index: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Debt/equity mix for long-term financing need.

        Returns:
            Tuple ``(new_lt_loan, equity_financing)``.
        """
        zero = tf.constant(0.0, dtype=tf.float64)
        long_term_financing = tf.maximum(zero, liquidity_deficit_lt)
        return self.debt_policy.compute_financing_mix(long_term_financing, time_index)

    def _reconcile_cash_and_ims(
        self,
        cash_target: tf.Tensor,
        ims_target: Optional[tf.Tensor],
        cash_prev: tf.Tensor,
        ims_prev: tf.Tensor,
        total_nlb: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Allocate ending cash and IMS; compute liquidity check.

        Returns:
            Tuple ``(cash_curr, ims_curr, liquidity_check)``.
        """
        zero = tf.constant(0.0, dtype=tf.float64)
        cash_curr = cash_target
        if ims_target is None:
            ending_liquidity = (cash_prev + ims_prev) + total_nlb
            ims_curr = tf.maximum(zero, ending_liquidity - cash_curr)
        else:
            ims_curr = ims_target
        liquidity_check = (cash_prev + ims_prev) + total_nlb - cash_curr - ims_curr
        return cash_curr, ims_curr, liquidity_check

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
