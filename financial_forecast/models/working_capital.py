"""Working capital policy module.

Owns the five balance-sheet ratio parameters that link working capital
accounts to sales or purchases:

- Advance payments on sales (% of sales)
- Advance payments on purchases (% of purchases)
- Accounts receivable (% of sales)
- Accounts payable (% of purchases)
- Inventory (% of sales)
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class WorkingCapitalPolicy(tf.Module):
    """Static working capital ratios."""

    def __init__(self, name="working_capital"):
        super().__init__(name=name)
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

    def loss(self, sales, purchases, adv_ps_actual, adv_pp_actual,
             ar_actual, ap_actual, inv_actual,
             scale_adv_ps, scale_adv_pp, scale_ar, scale_ap, scale_inv):
        """MSE losses for all five working capital ratios.

        Returns:
            Tuple ``(loss_adv_ps, loss_adv_pp, loss_ar, loss_ap, loss_inv)``.
        """
        loss_adv_ps = tf.reduce_mean(
            tf.square(
                (adv_ps_actual - sales * self.advance_payments_sales_pct)
                / scale_adv_ps
            )
        )
        loss_adv_pp = tf.reduce_mean(
            tf.square(
                (adv_pp_actual - purchases * self.advance_payments_purchases_pct)
                / scale_adv_pp
            )
        )
        loss_ar = tf.reduce_mean(
            tf.square(
                (ar_actual - sales * self.account_receivables_pct)
                / scale_ar
            )
        )
        loss_ap = tf.reduce_mean(
            tf.square(
                (ap_actual - purchases * self.account_payables_pct)
                / scale_ap
            )
        )
        loss_inv = tf.reduce_mean(
            tf.square(
                (inv_actual - sales * self.inventory_pct)
                / scale_inv
            )
        )
        return loss_adv_ps, loss_adv_pp, loss_ar, loss_ap, loss_inv

    def print_summary(self):
        """Print learned parameters."""
        print(f"Final %AdvPS: {self.advance_payments_sales_pct.numpy():.5f}")
        print(f"Final %AdvPP: {self.advance_payments_purchases_pct.numpy():.5f}")
        print(f"Final %AR: {self.account_receivables_pct.numpy():.5f}")
        print(f"Final %AP: {self.account_payables_pct.numpy():.5f}")
        print(f"Final %Inv: {self.inventory_pct.numpy():.5f}")
