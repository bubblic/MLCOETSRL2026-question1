"""Purchases/cost ratio policy modules.

- ``StaticCostRatioPolicy``: purchases = sales * cost_ratio + delta_inventory.
- ``TrendCostRatioPolicy``: purchases = sales * sigmoid(alpha + beta*t) + delta_inventory.
"""

from abc import abstractmethod
from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class PurchasesPolicy(tf.Module):
    """Abstract base class for purchases/cost ratio policy."""

    @abstractmethod
    def compute(
        self,
        sales_t: tf.Tensor,
        inv_curr: tf.Tensor,
        inv_prev: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
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
    def get_cost_ratio(self, time_index: tf.Tensor) -> tf.Tensor:
        """Return the cost ratio for the given time index."""

    @abstractmethod
    def loss(
        self,
        sales: tf.Tensor,
        cogs: tf.Tensor,
        inventory: tf.Tensor,
        time_indices: tf.Tensor,
        scale: tf.Tensor,
    ) -> tf.Tensor:
        """Compute MSE loss for cost ratio."""

    @abstractmethod
    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        """Initialize parameters from historical averages."""

    @abstractmethod
    def print_summary(self, n_years: int) -> None:
        """Print learned parameters."""


class StaticCostRatioPolicy(PurchasesPolicy):
    """Static cost ratio: purchases = sales * cost_ratio + delta_inventory."""

    def __init__(self, name="static_cost_ratio"):
        super().__init__(name=name)
        self.cost_ratio = tfp.util.TransformedVariable(
            initial_value=0.55,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="cost_ratio",
        )

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        ratio = float(tf.reduce_mean(s["cogs"] / tf.maximum(s["sales"], _EPS)))
        self.cost_ratio.assign(_f64(min(1 - _EPS, max(_EPS, ratio))))

    def get_cost_ratio(self, time_index: tf.Tensor) -> tf.Tensor:
        return self.cost_ratio + tf.zeros_like(time_index)

    def compute(
        self,
        sales_t: tf.Tensor,
        inv_curr: tf.Tensor,
        inv_prev: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        return sales_t * self.cost_ratio + (inv_curr - inv_prev)

    def loss(
        self,
        sales: tf.Tensor,
        cogs: tf.Tensor,
        inventory: tf.Tensor,
        time_indices: tf.Tensor,
        scale: tf.Tensor,
    ) -> tf.Tensor:
        # COGS = sales * cost_ratio by construction
        pred_cogs = sales * self.cost_ratio
        return tf.reduce_mean(tf.square((cogs - pred_cogs) / scale))

    def print_summary(self, n_years: int) -> None:
        print(f"Cost Ratio: {self.cost_ratio.numpy():.5f}")


class TrendCostRatioPolicy(PurchasesPolicy):
    """Logit-linear cost ratio: purchases = sales * sigmoid(alpha + beta*t) + delta_inventory."""

    def __init__(self, name="trend_cost_ratio"):
        super().__init__(name=name)
        self.cost_ratio_alpha = tf.Variable(
            0.35,
            dtype=tf.float64,
            name="cost_ratio_alpha",
        )
        self.cost_ratio_beta = tf.Variable(
            -0.05,
            dtype=tf.float64,
            name="cost_ratio_beta",
        )

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _EPS = 1e-12
        ratio = float(tf.reduce_mean(s["cogs"] / tf.maximum(s["sales"], _EPS)))
        ratio = min(1 - _EPS, max(_EPS, ratio))
        import math

        self.cost_ratio_alpha.assign(math.log(ratio / (1 - ratio)))
        self.cost_ratio_beta.assign(0.0)

    def get_cost_ratio(self, time_index: tf.Tensor) -> tf.Tensor:
        return tf.sigmoid(self.cost_ratio_alpha + self.cost_ratio_beta * time_index)

    def compute(
        self,
        sales_t: tf.Tensor,
        inv_curr: tf.Tensor,
        inv_prev: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        return sales_t * self.get_cost_ratio(time_index) + (inv_curr - inv_prev)

    def loss(
        self,
        sales: tf.Tensor,
        cogs: tf.Tensor,
        inventory: tf.Tensor,
        time_indices: tf.Tensor,
        scale: tf.Tensor,
    ) -> tf.Tensor:
        logit_cr_hist = tf.math.log((cogs / sales) / (1.0 - cogs / sales))
        logit_cr_pred = self.cost_ratio_alpha + self.cost_ratio_beta * time_indices
        return tf.reduce_mean(tf.square((logit_cr_hist - logit_cr_pred) / scale))

    def print_summary(self, n_years: int) -> None:
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
