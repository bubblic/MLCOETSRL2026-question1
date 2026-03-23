"""Stock buyback policy modules.

- ``SimpleBuybackPolicy``: buyback = depr * pct.
- ``BaselineBuybackPolicy``: buyback = baseline + ratio * depr.
"""

from abc import abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class BuybackPolicy(tf.Module):
    """Abstract base class for stock buyback policy."""

    @abstractmethod
    def compute(self, depreciation):
        """Compute base stock buyback amount.

        Args:
            depreciation: ``[n_samples]`` depreciation tensor.

        Returns:
            ``[n_samples]`` base buyback tensor (before excess cash adjustment).
        """

    @abstractmethod
    def loss(self, bb_actual, depr, scale_bb):
        """Compute MSE loss for buybacks."""

    @abstractmethod
    def print_summary(self):
        """Print learned parameters."""


class SimpleBuybackPolicy(BuybackPolicy):
    """Buyback = depr * pct."""

    def __init__(self, name="simple_buyback"):
        super().__init__(name=name)
        self.stock_buyback_pct = tfp.util.TransformedVariable(
            initial_value=7.5, bijector=tfb.Softplus(),
            dtype=tf.float64, name="stock_buyback_pct",
        )

    def compute(self, depreciation):
        return depreciation * self.stock_buyback_pct

    def loss(self, bb_actual, depr, scale_bb):
        pred = depr * self.stock_buyback_pct
        return tf.reduce_mean(tf.square((bb_actual - pred) / scale_bb))

    def print_summary(self):
        print(f"Stock Buyback %: {self.stock_buyback_pct.numpy():.5f}")


class BaselineBuybackPolicy(BuybackPolicy):
    """Buyback = baseline + ratio * depr."""

    def __init__(self, name="baseline_buyback"):
        super().__init__(name=name)
        self.sb_baseline = tf.Variable(0.0, dtype=tf.float64, name="sb_baseline")
        self.sb_ratio = tf.Variable(1.0, dtype=tf.float64, name="sb_ratio")

    def compute(self, depreciation):
        return self.sb_baseline + self.sb_ratio * depreciation

    def loss(self, bb_actual, depr, scale_bb):
        pred = self.sb_baseline + self.sb_ratio * depr
        return tf.reduce_mean(tf.square((bb_actual - pred) / scale_bb))

    def print_summary(self):
        print(
            f"Stock Buyback (baseline + ratio*depr): "
            f"baseline={self.sb_baseline.numpy():.4f}, "
            f"ratio={self.sb_ratio.numpy():.6f}"
        )
