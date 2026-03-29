"""Stock buyback policy modules.

- ``SimpleBuybackPolicy``: buyback = depr * pct.
- ``BaselineBuybackPolicy``: buyback = baseline + ratio * depr.
"""

from abc import abstractmethod
from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class BuybackPolicy(tf.Module):
    """Abstract base class for stock buyback policy."""

    @abstractmethod
    def compute(self, depreciation: tf.Tensor) -> tf.Tensor:
        """Compute base stock buyback amount.

        Args:
            depreciation: ``[n_samples]`` depreciation tensor.

        Returns:
            ``[n_samples]`` base buyback tensor (before excess cash adjustment).
        """

    @abstractmethod
    def loss(
        self,
        bb_actual: tf.Tensor,
        depr: tf.Tensor,
        scale_bb: tf.Tensor,
    ) -> tf.Tensor:
        """Compute MSE loss for buybacks."""

    @abstractmethod
    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        """Initialize parameters from historical averages."""

    @abstractmethod
    def print_summary(self) -> None:
        """Print learned parameters."""


class SimpleBuybackPolicy(BuybackPolicy):
    """Buyback = depr * pct."""

    def __init__(self, name="simple_buyback"):
        super().__init__(name=name)
        self.stock_buyback_pct = tfp.util.TransformedVariable(
            initial_value=7.5,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="stock_buyback_pct",
        )

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        self.stock_buyback_pct.assign(
            _f64(
                max(
                    _EPS,
                    float(
                        tf.reduce_mean(
                            s["stock_buyback"] / tf.maximum(s["depreciation"], _EPS)
                        )
                    ),
                )
            )
        )

    def compute(self, depreciation: tf.Tensor) -> tf.Tensor:
        return depreciation * self.stock_buyback_pct

    def loss(
        self,
        bb_actual: tf.Tensor,
        depr: tf.Tensor,
        scale_bb: tf.Tensor,
    ) -> tf.Tensor:
        pred = depr * self.stock_buyback_pct
        return tf.reduce_mean(tf.square((bb_actual - pred) / scale_bb))

    def print_summary(self) -> None:
        print(f"Stock Buyback %: {self.stock_buyback_pct.numpy():.5f}")


class BaselineBuybackPolicy(BuybackPolicy):
    """Buyback = baseline + ratio * depr."""

    def __init__(self, name="baseline_buyback"):
        super().__init__(name=name)
        self.sb_baseline = tf.Variable(0.0, dtype=tf.float64, name="sb_baseline")
        self.sb_ratio = tf.Variable(1.0, dtype=tf.float64, name="sb_ratio")

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _EPS = 1e-12
        self.sb_ratio.assign(
            float(
                tf.reduce_mean(s["stock_buyback"] / tf.maximum(s["depreciation"], _EPS))
            )
        )
        self.sb_baseline.assign(0.0)

    def compute(self, depreciation: tf.Tensor) -> tf.Tensor:
        return self.sb_baseline + self.sb_ratio * depreciation

    def loss(
        self,
        bb_actual: tf.Tensor,
        depr: tf.Tensor,
        scale_bb: tf.Tensor,
    ) -> tf.Tensor:
        pred = self.sb_baseline + self.sb_ratio * depr
        return tf.reduce_mean(tf.square((bb_actual - pred) / scale_bb))

    def print_summary(self) -> None:
        print(
            f"Stock Buyback (baseline + ratio*depr): "
            f"baseline={self.sb_baseline.numpy():.4f}, "
            f"ratio={self.sb_ratio.numpy():.6f}"
        )
