"""Dividend policy modules: simple payout and Lintner smoothing.

- ``SimpleDividendPolicy``: dividends = NI * pct.
- ``LintnerDividendPolicy``: Lintner smoothing with adjustment speed.
"""

from abc import abstractmethod
from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class DividendPolicy(tf.Module):
    """Abstract base class for dividend policy."""

    @abstractmethod
    def compute(self, ni_prev: tf.Tensor, div_prev_actual: tf.Tensor) -> tf.Tensor:
        """Compute dividends for this period.

        Args:
            ni_prev: ``[n_samples]`` previous-period net income.
            div_prev_actual: ``[n_samples]`` previous-period actual dividends.

        Returns:
            ``[n_samples]`` dividends tensor.
        """

    @abstractmethod
    def loss(
        self,
        ni_prev: tf.Tensor,
        div_actual: tf.Tensor,
        div_prev: tf.Tensor,
        scale_div: tf.Tensor,
    ) -> tf.Tensor:
        """Compute MSE loss for dividends."""

    @abstractmethod
    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        """Initialize parameters from historical averages."""

    @abstractmethod
    def print_summary(self) -> None:
        """Print learned parameters."""


class SimpleDividendPolicy(DividendPolicy):
    """Dividends = NI * payout_ratio (no smoothing)."""

    def __init__(self, name="simple_dividend"):
        super().__init__(name=name)
        self.dividend_payout_ratio_pct = tfp.util.TransformedVariable(
            initial_value=0.15,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="div_pct",
        )

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        ratio = float(
            tf.reduce_mean(s["dividends"] / tf.maximum(s["net_income"], _EPS))
        )
        self.dividend_payout_ratio_pct.assign(_f64(min(1 - _EPS, max(_EPS, ratio))))

    def compute(self, ni_prev: tf.Tensor, div_prev_actual: tf.Tensor) -> tf.Tensor:
        return ni_prev * self.dividend_payout_ratio_pct

    def loss(
        self,
        ni_prev: tf.Tensor,
        div_actual: tf.Tensor,
        div_prev: tf.Tensor,
        scale_div: tf.Tensor,
    ) -> tf.Tensor:
        pred = ni_prev * self.dividend_payout_ratio_pct
        return tf.reduce_mean(tf.square((div_actual - pred) / scale_div))

    def print_summary(self) -> None:
        print(f"Final %PR: {self.dividend_payout_ratio_pct.numpy():.5f}")


class LintnerDividendPolicy(DividendPolicy):
    """Lintner dividend smoothing.

    dividends = alpha * (PR * NI) + (1 - alpha) * D_prev
    """

    def __init__(self, name="lintner_dividend"):
        super().__init__(name=name)
        self.dividend_payout_ratio_pct = tfp.util.TransformedVariable(
            initial_value=0.15,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="div_pct",
        )
        self.dividend_adjustment_speed = tfp.util.TransformedVariable(
            initial_value=0.01,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="div_adj_speed",
        )

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        ratio = float(
            tf.reduce_mean(s["dividends"] / tf.maximum(s["net_income"], _EPS))
        )
        self.dividend_payout_ratio_pct.assign(_f64(min(1 - _EPS, max(_EPS, ratio))))

    def compute(self, ni_prev: tf.Tensor, div_prev_actual: tf.Tensor) -> tf.Tensor:
        dividend_target = ni_prev * self.dividend_payout_ratio_pct
        return (
            self.dividend_adjustment_speed * dividend_target
            + (1.0 - self.dividend_adjustment_speed) * div_prev_actual
        )

    def loss(
        self,
        ni_prev: tf.Tensor,
        div_actual: tf.Tensor,
        div_prev: tf.Tensor,
        scale_div: tf.Tensor,
    ) -> tf.Tensor:
        _one = tf.constant(1.0, dtype=tf.float64)
        div_target = ni_prev * self.dividend_payout_ratio_pct
        div_pred = (
            self.dividend_adjustment_speed * div_target
            + (_one - self.dividend_adjustment_speed) * div_prev
        )
        return tf.reduce_mean(tf.square((div_actual - div_pred) / scale_div))

    def print_summary(self) -> None:
        print(f"Final %PR: {self.dividend_payout_ratio_pct.numpy():.5f}")
        print(f"Final DivAdjSpeed: {self.dividend_adjustment_speed.numpy():.5f}")
