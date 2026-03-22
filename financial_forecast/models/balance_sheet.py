"""Balance sheet model — asset evolution, working capital, and payout policy.

Owns all parameters related to asset accounts, liquidity allocation,
dividends, buybacks, and cost structure.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from financial_forecast.inference.state_index import (
    R_NCA, R_INV, R_NET_INCOME, R_DIVIDENDS,
)

tfb = tfp.bijectors


class BalanceSheetModel(tf.Module):
    """Evolves balance sheet asset accounts and payout decisions.

    Parameters cover asset growth, depreciation, working capital ratios,
    liquidity allocation (logit-linear trends), dividend smoothing
    (Lintner model), stock buyback policy, and cost ratio trends.
    """

    def __init__(self, name="balance_sheet"):
        super().__init__(name=name)

        # --- Asset growth & depreciation ---
        self.asset_growth = tfp.util.TransformedVariable(
            initial_value=0.0076, bijector=tfb.Softplus(),
            dtype=tf.float64, name="asset_growth",
        )
        self.asset_maintain = tfp.util.TransformedVariable(
            initial_value=0.99, bijector=tfb.Softplus(),
            dtype=tf.float64, name="asset_maintain",
        )
        self.depreciation_rate = tfp.util.TransformedVariable(
            initial_value=0.055, bijector=tfb.Softplus(),
            dtype=tf.float64, name="depr_rate",
        )

        # --- Working capital ratios ---
        self.advance_payments_sales_pct = tfp.util.TransformedVariable(
            initial_value=0.0206, bijector=tfb.Softplus(),
            dtype=tf.float64, name="adv_ps",
        )
        self.advance_payments_purchases_pct = tfp.util.TransformedVariable(
            initial_value=0.0735, bijector=tfb.Softplus(),
            dtype=tf.float64, name="adv_pp",
        )
        self.account_receivables_pct = tfp.util.TransformedVariable(
            initial_value=0.1591, bijector=tfb.Softplus(),
            dtype=tf.float64, name="ar_pct",
        )
        self.account_payables_pct = tfp.util.TransformedVariable(
            initial_value=0.3501, bijector=tfb.Softplus(),
            dtype=tf.float64, name="ap_pct",
        )
        self.inventory_pct = tfp.util.TransformedVariable(
            initial_value=0.0165, bijector=tfb.Softplus(),
            dtype=tf.float64, name="inv_pct",
        )

        # --- Total liquidity (logit-linear trend + baseline) ---
        self.tl_alpha = tf.Variable(-1.66, dtype=tf.float64, name="tl_alpha")
        self.tl_beta = tf.Variable(0.0, dtype=tf.float64, name="tl_beta")
        self.tl_baseline = tf.Variable(0.0, dtype=tf.float64, name="tl_baseline")

        # --- Cash % of liquidity (logit-linear trend) ---
        self.cash_alpha = tf.Variable(-0.05, dtype=tf.float64, name="cash_alpha")
        self.cash_beta = tf.Variable(0.0, dtype=tf.float64, name="cash_beta")

        # --- Dividend smoothing (Lintner model) ---
        self.dividend_payout_ratio_pct = tfp.util.TransformedVariable(
            initial_value=0.15, bijector=tfb.Sigmoid(),
            dtype=tf.float64, name="div_pct",
        )
        self.dividend_adjustment_speed = tfp.util.TransformedVariable(
            initial_value=0.01, bijector=tfb.Sigmoid(),
            dtype=tf.float64, name="div_adj_speed",
        )

        # --- Stock buyback policy ---
        self.sb_baseline = tf.Variable(0.0, dtype=tf.float64, name="sb_baseline")
        self.sb_ratio = tf.Variable(1.0, dtype=tf.float64, name="sb_ratio")

        # --- Cost ratio (logit-linear trend) ---
        self.cost_ratio_alpha = tf.Variable(
            0.35, dtype=tf.float64, name="cost_ratio_alpha",
        )
        self.cost_ratio_beta = tf.Variable(
            -0.05, dtype=tf.float64, name="cost_ratio_beta",
        )

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

        depreciation = nca_prev * self.depreciation_rate
        stock_buyback = self.sb_baseline + self.sb_ratio * depreciation

        dividend_target = ni_prev * self.dividend_payout_ratio_pct
        dividends_prev = (
            self.dividend_adjustment_speed * dividend_target
            + (1.0 - self.dividend_adjustment_speed) * div_prev_actual
        )

        cost_ratio_t = tf.sigmoid(
            self.cost_ratio_alpha + self.cost_ratio_beta * time_index
        )
        capex = self.asset_maintain * depreciation + sales_t * self.asset_growth
        nca_curr = nca_prev - depreciation + capex

        ar_curr = sales_t * self.account_receivables_pct
        inv_curr = sales_t * self.inventory_pct
        purchases_t = sales_t * cost_ratio_t + (inv_curr - inv_prev)
        adv_pp_curr = purchases_t * self.advance_payments_purchases_pct

        tl_pct = tf.sigmoid(self.tl_alpha + self.tl_beta * time_index)
        total_liquidity_curr = self.tl_baseline + sales_t * tl_pct
        cash_pct = tf.sigmoid(self.cash_alpha + self.cash_beta * time_index)
        cash_curr = total_liquidity_curr * cash_pct
        ims_curr = total_liquidity_curr - cash_curr

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
