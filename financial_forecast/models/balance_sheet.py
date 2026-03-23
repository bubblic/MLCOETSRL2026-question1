"""Balance sheet model -- asset evolution and working capital.

Composes pluggable capex, working capital, liquidity, dividend, buyback,
and purchases modules.
"""

import tensorflow as tf

from financial_forecast.inference.state_index import (
    R_NCA, R_INV, R_NET_INCOME, R_DIVIDENDS,
)


class BalanceSheetModel(tf.Module):
    """Evolves balance sheet asset accounts.

    Delegates asset growth/depreciation to :class:`CapexPolicy`,
    working capital ratios to :class:`WorkingCapitalPolicy`, and
    liquidity, dividends, buybacks, and purchases to pluggable
    sub-modules.

    Args:
        capex_policy: Capital expenditure policy module.
        working_capital: Working capital ratio module.
        liquidity_policy: Liquidity allocation module.
        dividend_policy: Dividend policy module.
        buyback_policy: Buyback policy module.
        purchases_policy: Purchases/cost ratio module.
    """

    def __init__(self, capex_policy, working_capital, liquidity_policy,
                 dividend_policy, buyback_policy,
                 purchases_policy, name="balance_sheet"):
        super().__init__(name=name)

        self.capex_policy = capex_policy
        self.working_capital = working_capital
        self.liquidity_policy = liquidity_policy
        self.dividend_policy = dividend_policy
        self.buyback_policy = buyback_policy
        self.purchases_policy = purchases_policy

    def evolve_assets(self, state, sales_t, time_index):
        """Evolve asset accounts and compute payout decisions.

        Args:
            state: ``[n_samples, 14]`` recurrent state tensor.
            sales_t: ``[n_samples]`` sales for this period.
            time_index: Scalar ``year - base_year``.

        Returns:
            Dict with asset values and intermediate quantities.
        """
        nca_prev = state[:, R_NCA]
        inv_prev = state[:, R_INV]
        ni_prev = state[:, R_NET_INCOME]
        div_prev_actual = state[:, R_DIVIDENDS]

        depreciation, capex, nca_curr = self.capex_policy.compute(
            nca_prev, sales_t,
        )

        # Delegate to pluggable modules
        dividends_prev = self.dividend_policy.compute(ni_prev, div_prev_actual)
        stock_buyback = self.buyback_policy.compute(depreciation)

        wc = self.working_capital
        ar_curr = sales_t * wc.account_receivables_pct
        inv_curr = sales_t * wc.inventory_pct
        purchases_t = self.purchases_policy.compute(
            sales_t, inv_curr, inv_prev, time_index,
        )
        adv_pp_curr = purchases_t * wc.advance_payments_purchases_pct

        total_liquidity_curr, cash_curr, ims_curr = (
            self.liquidity_policy.compute(sales_t, time_index)
        )

        return {
            "depreciation": depreciation,
            "capex": capex,
            "nca_curr": nca_curr,
            "ar_curr": ar_curr,
            "inv_curr": inv_curr,
            "purchases_t": purchases_t,
            "adv_pp_curr": adv_pp_curr,
            "total_liquidity_curr": total_liquidity_curr,
            "cash_curr": cash_curr,
            "ims_curr": ims_curr,
            "dividends_prev": dividends_prev,
            "stock_buyback_base": stock_buyback,
        }
