"""Working capital policy module.

Owns the five balance-sheet ratio parameters that link working capital
accounts to sales, COGS, or purchases:

- Advance payments on sales (% of sales)
- Advance payments on purchases (% of purchases)
- Accounts receivable (% of sales, equivalent to DSO/365)
- Accounts payable (% of purchases, equivalent to DPO/365)
- Inventory (% of COGS, equivalent to DSI/365)
"""

from typing import Dict, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class WorkingCapitalPolicy(tf.Module):
    """Static working capital ratios."""

    def __init__(self, name="working_capital"):
        super().__init__(name=name)
        self.advance_payments_sales_pct = tfp.util.TransformedVariable(
            initial_value=0.0206,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="adv_ps",
        )
        self.advance_payments_purchases_pct = tfp.util.TransformedVariable(
            initial_value=0.0735,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="adv_pp",
        )
        self.account_receivables_pct = tfp.util.TransformedVariable(
            initial_value=0.1591,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="ar_pct",
        )
        self.account_payables_pct = tfp.util.TransformedVariable(
            initial_value=0.3501,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="ap_pct",
        )
        self.inventory_cogs_pct = tfp.util.TransformedVariable(
            initial_value=0.0165,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="inv_cogs_pct",
        )

    def compute_sales_based(
        self, sales_t: tf.Tensor, cogs_t: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute working capital accounts driven by sales and COGS.

        AR and advance payments scale with sales (DSO-based).
        Inventory scales with COGS (DSI-based) because inventory is
        carried at cost, not at selling price.

        Args:
            sales_t: ``[n_samples]`` current-period sales.
            cogs_t: ``[n_samples]`` current-period cost of goods sold.

        Returns:
            Tuple ``(ar_curr, inv_curr, adv_ps_curr)``.
        """
        return (
            sales_t * self.account_receivables_pct,
            cogs_t * self.inventory_cogs_pct,
            sales_t * self.advance_payments_sales_pct,
        )

    def compute_purchases_based(
        self, purchases_t: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute working capital accounts driven by purchases.

        Args:
            purchases_t: ``[n_samples]`` current-period purchases.

        Returns:
            Tuple ``(ap_curr, adv_pp_curr)``.
        """
        return (
            purchases_t * self.account_payables_pct,
            purchases_t * self.advance_payments_purchases_pct,
        )

    def loss(
        self,
        sales: tf.Tensor,
        purchases: tf.Tensor,
        cogs: tf.Tensor,
        adv_ps_actual: tf.Tensor,
        adv_pp_actual: tf.Tensor,
        ar_actual: tf.Tensor,
        ap_actual: tf.Tensor,
        inv_actual: tf.Tensor,
        scale_adv_ps: tf.Tensor,
        scale_adv_pp: tf.Tensor,
        scale_ar: tf.Tensor,
        scale_ap: tf.Tensor,
        scale_inv: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """MSE losses for all five working capital ratios.

        Returns:
            Tuple ``(loss_adv_ps, loss_adv_pp, loss_ar, loss_ap, loss_inv)``.
        """
        loss_adv_ps = tf.reduce_mean(
            tf.square(
                (adv_ps_actual - sales * self.advance_payments_sales_pct) / scale_adv_ps
            )
        )
        loss_adv_pp = tf.reduce_mean(
            tf.square(
                (adv_pp_actual - purchases * self.advance_payments_purchases_pct)
                / scale_adv_pp
            )
        )
        loss_ar = tf.reduce_mean(
            tf.square((ar_actual - sales * self.account_receivables_pct) / scale_ar)
        )
        loss_ap = tf.reduce_mean(
            tf.square((ap_actual - purchases * self.account_payables_pct) / scale_ap)
        )
        loss_inv = tf.reduce_mean(
            tf.square((inv_actual - cogs * self.inventory_cogs_pct) / scale_inv)
        )
        return loss_adv_ps, loss_adv_pp, loss_ar, loss_ap, loss_inv

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        """Initialize ratios from historical averages."""
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        _mr = lambda num, den: max(
            _EPS, float(tf.reduce_mean(num / tf.maximum(den, _EPS)))
        )
        self.advance_payments_sales_pct.assign(
            _f64(
                _mr(
                    s["advance_payments_sales"],
                    s["sales"],
                )
            )
        )
        self.advance_payments_purchases_pct.assign(
            _f64(
                _mr(
                    s["advance_payments_purchases"],
                    s["purchases"],
                )
            )
        )
        self.account_receivables_pct.assign(
            _f64(
                _mr(
                    s["accounts_receivable"],
                    s["sales"],
                )
            )
        )
        self.account_payables_pct.assign(
            _f64(
                _mr(
                    s["accounts_payable"],
                    s["purchases"],
                )
            )
        )
        self.inventory_cogs_pct.assign(_f64(_mr(s["inventory"], s["cogs"])))

    def print_summary(self) -> None:
        """Print learned parameters."""
        print(f"Final %AdvPS: {self.advance_payments_sales_pct.numpy():.5f}")
        print(f"Final %AdvPP: {self.advance_payments_purchases_pct.numpy():.5f}")
        print(f"Final %AR: {self.account_receivables_pct.numpy():.5f}")
        print(f"Final %AP: {self.account_payables_pct.numpy():.5f}")
        print(f"Final %Inv (COGS): {self.inventory_cogs_pct.numpy():.5f}")
