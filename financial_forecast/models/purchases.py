"""Purchases/cost ratio policy modules.

- ``StaticCostRatioPolicy``: purchases = sales * cost_ratio + delta_inventory.
- ``TrendCostRatioPolicy``: purchases = sales * sigmoid(alpha + beta*t) + delta_inventory.
"""

from abc import abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class PurchasesPolicy(tf.Module):
    """Abstract base class for purchases/cost ratio policy."""

    @abstractmethod
    def compute(self, sales_t, inv_curr, inv_prev, time_index):
        """Compute purchases for this period.

        Args:
            sales_t: ``[n_samples]`` sales tensor.
            inv_curr: ``[n_samples]`` target inventory.
            inv_prev: ``[n_samples]`` previous inventory.
            time_index: Scalar ``year - base_year``.

        Returns:
            ``[n_samples]`` purchases tensor.
        """

    @abstractmethod
    def loss(self, sales, cogs, inventory, time_indices, scale):
        """Compute MSE loss for cost ratio."""

    @abstractmethod
    def print_summary(self, n_years):
        """Print learned parameters."""


class StaticCostRatioPolicy(PurchasesPolicy):
    """Static cost ratio: purchases = sales * cost_ratio + delta_inventory."""

    def __init__(self, name="static_cost_ratio"):
        super().__init__(name=name)
        self.cost_ratio = tfp.util.TransformedVariable(
            initial_value=0.55, bijector=tfb.Sigmoid(),
            dtype=tf.float64, name="cost_ratio",
        )

    def compute(self, sales_t, inv_curr, inv_prev, time_index):
        return sales_t * self.cost_ratio + (inv_curr - inv_prev)

    def loss(self, sales, cogs, inventory, time_indices, scale):
        # COGS = sales * cost_ratio by construction
        pred_cogs = sales * self.cost_ratio
        return tf.reduce_mean(tf.square((cogs - pred_cogs) / scale))

    def print_summary(self, n_years):
        print(f"Cost Ratio: {self.cost_ratio.numpy():.5f}")


class TrendCostRatioPolicy(PurchasesPolicy):
    """Logit-linear cost ratio: purchases = sales * sigmoid(alpha + beta*t) + delta_inventory."""

    def __init__(self, name="trend_cost_ratio"):
        super().__init__(name=name)
        self.cost_ratio_alpha = tf.Variable(
            0.35, dtype=tf.float64, name="cost_ratio_alpha",
        )
        self.cost_ratio_beta = tf.Variable(
            -0.05, dtype=tf.float64, name="cost_ratio_beta",
        )

    def compute(self, sales_t, inv_curr, inv_prev, time_index):
        cost_ratio_t = tf.sigmoid(
            self.cost_ratio_alpha + self.cost_ratio_beta * time_index
        )
        return sales_t * cost_ratio_t + (inv_curr - inv_prev)

    def loss(self, sales, cogs, inventory, time_indices, scale):
        logit_cr_hist = tf.math.log(
            (cogs / sales) / (1.0 - cogs / sales)
        )
        logit_cr_pred = self.cost_ratio_alpha + self.cost_ratio_beta * time_indices
        return tf.reduce_mean(tf.square((logit_cr_hist - logit_cr_pred) / scale))

    def print_summary(self, n_years):
        import tensorflow as tf
        print(
            f"Cost Ratio (logit-linear): "
            f"alpha={self.cost_ratio_alpha.numpy():.4f}, "
            f"beta={self.cost_ratio_beta.numpy():.6f}"
        )
        print(
            f"  => CR at t=0: "
            f"{tf.sigmoid(self.cost_ratio_alpha).numpy():.4f}, "
            f"CR at t={n_years-1}: "
            f"{tf.sigmoid(self.cost_ratio_alpha + self.cost_ratio_beta * (n_years-1)).numpy():.4f}"
        )
