"""Balance sheet model -- asset evolution and working capital.

Composes pluggable capex, working capital, liquidity, and purchases modules.
"""

import tensorflow as tf

from financial_forecast.inference.state_index import (
    R_NCA,
    R_INV,
)


class BalanceSheetModel(tf.Module):
    """Evolves balance sheet asset accounts.

    Delegates asset growth/depreciation to :class:`CapexPolicy`,
    working capital ratios to :class:`WorkingCapitalPolicy`, and
    liquidity and purchases to pluggable sub-modules.

    Args:
        capex_policy: Capital expenditure policy module.
        working_capital: Working capital ratio module.
        liquidity_policy: Liquidity allocation module.
        purchases_policy: Purchases/cost ratio module.
    """

    def __init__(
        self,
        capex_policy,
        working_capital,
        liquidity_policy,
        purchases_policy,
        name="balance_sheet",
    ):
        super().__init__(name=name)

        self.capex_policy = capex_policy
        self.working_capital = working_capital
        self.liquidity_policy = liquidity_policy
        self.purchases_policy = purchases_policy

    def evolve_assets(self, state, sales_t, time_index):
        """Evolve asset accounts.

        Args:
            state: ``[n_samples, 14]`` recurrent state tensor.
            sales_t: ``[n_samples]`` sales for this period.
            time_index: Scalar ``year - base_year``.

        Returns:
            Dict with asset values and intermediate quantities.
        """
        nca_prev = state[:, R_NCA]
        inv_prev = state[:, R_INV]

        depreciation, capex, nca_curr = self.capex_policy.compute(
            nca_prev,
            sales_t,
        )

        ar_curr, inv_curr, adv_ps_curr = self.working_capital.compute_sales_based(
            sales_t
        )
        purchases_t = self.purchases_policy.compute(
            sales_t,
            inv_curr,
            inv_prev,
            time_index,
        )
        ap_curr, adv_pp_curr = self.working_capital.compute_purchases_based(purchases_t)

        total_liquidity_curr, cash_curr, ims_curr = self.liquidity_policy.compute(
            sales_t, time_index
        )

        return {
            "depreciation": depreciation,
            "capex": capex,
            "nca_curr": nca_curr,
            "ar_curr": ar_curr,
            "inv_curr": inv_curr,
            "adv_ps_curr": adv_ps_curr,
            "purchases_t": purchases_t,
            "ap_curr": ap_curr,
            "adv_pp_curr": adv_pp_curr,
            "total_liquidity_curr": total_liquidity_curr,
            "cash_curr": cash_curr,
            "ims_curr": ims_curr,
        }
