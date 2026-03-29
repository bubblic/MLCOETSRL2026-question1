"""Liquidity allocation modules: static and time-varying.

- ``SimpleLiquidityPolicy``: total_liq = sales * pct, cash = total_liq * cash_pct.
- ``TrendLiquidityPolicy``: logit-linear trends with baseline floor.
"""

from abc import abstractmethod
from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class LiquidityPolicy(tf.Module):
    """Abstract base class for liquidity allocation."""

    @abstractmethod
    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        """Initialize parameters from historical averages."""

    @abstractmethod
    def compute(self, sales_t: tf.Tensor, time_index: tf.Tensor) -> tuple:
        """Compute total liquidity, cash, and IMS.

        Args:
            sales_t: ``[n_samples]`` sales tensor.
            time_index: Scalar ``year - base_year``.

        Returns:
            Tuple ``(total_liquidity, cash, ims)``.
        """


class CashTargetPolicy(LiquidityPolicy):
    """Cash target only: cash = sales * pct. IMS is residual (not a target).

    compute() returns None for ims_target so that excess cash after all flows is
    invested in market securities instead of being used for additional stock buybacks.
    """

    def __init__(self, name="cash_target"):
        super().__init__(name=name)
        self.cash_target_pct = tfp.util.TransformedVariable(
            initial_value=0.08,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="cash_target_pct",
        )

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        self.cash_target_pct.assign(
            _f64(
                max(
                    _EPS,
                    float(tf.reduce_mean(s["cash"] / tf.maximum(s["sales"], _EPS))),
                )
            )
        )

    def compute(self, sales_t: tf.Tensor, time_index: tf.Tensor) -> tuple:
        cash_target = sales_t * self.cash_target_pct
        return cash_target, cash_target, None

    def loss(
        self,
        sales: tf.Tensor,
        cash: tf.Tensor,
        ims: tf.Tensor,
        time_indices: tf.Tensor,
        scale_tl: tf.Tensor,
        scale_cash: tf.Tensor,
    ) -> tuple:
        """MSE loss on cash target only; IMS has no target."""
        loss_cash = tf.reduce_mean(
            tf.square((cash - sales * self.cash_target_pct) / scale_cash)
        )
        return loss_cash, tf.constant(0.0, dtype=tf.float64)

    def print_summary(self, n_years: int) -> None:
        print(f"Cash Target %: {self.cash_target_pct.numpy():.5f}")


class SimpleLiquidityPolicy(LiquidityPolicy):
    """Static percentage liquidity: total_liq = sales * pct."""

    def __init__(self, name="simple_liquidity"):
        super().__init__(name=name)
        self.total_liquidity_pct = tfp.util.TransformedVariable(
            initial_value=0.16,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="total_liquidity_pct",
        )
        self.cash_pct_of_liquidity = tfp.util.TransformedVariable(
            initial_value=0.487,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="cash_pct_of_liquidity",
        )

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        total_liq = s["cash"] + s["ims"]
        self.total_liquidity_pct.assign(
            _f64(
                max(
                    _EPS,
                    float(tf.reduce_mean(total_liq / tf.maximum(s["sales"], _EPS))),
                )
            )
        )
        self.cash_pct_of_liquidity.assign(
            _f64(
                min(
                    1 - _EPS,
                    max(
                        _EPS,
                        float(tf.reduce_mean(s["cash"] / tf.maximum(total_liq, _EPS))),
                    ),
                )
            )
        )

    def compute(self, sales_t: tf.Tensor, time_index: tf.Tensor) -> tuple:
        total_liq_target = sales_t * self.total_liquidity_pct
        cash_target = total_liq_target * self.cash_pct_of_liquidity
        ims_target = total_liq_target - cash_target
        return total_liq_target, cash_target, ims_target

    def print_summary(self, n_years: int) -> None:
        """Print learned liquidity parameters."""
        print(f"Total Liquidity %: {self.total_liquidity_pct.numpy():.5f}")
        print(f"Cash % of Liquidity: {self.cash_pct_of_liquidity.numpy():.5f}")

    def loss(
        self,
        sales: tf.Tensor,
        cash: tf.Tensor,
        ims: tf.Tensor,
        time_indices: tf.Tensor,
        scale_tl: tf.Tensor,
        scale_cash: tf.Tensor,
    ) -> tuple:
        """MSE loss for total liquidity and cash split."""
        total_liq = cash + ims
        loss_tl = tf.reduce_mean(
            tf.square((total_liq - sales * self.total_liquidity_pct) / scale_tl)
        )
        loss_cash = tf.reduce_mean(
            tf.square((cash - total_liq * self.cash_pct_of_liquidity) / scale_cash)
        )
        return loss_tl, loss_cash


class TrendLiquidityPolicy(LiquidityPolicy):
    """Logit-linear trend liquidity with baseline floor.

    total_liq = baseline + sales * sigmoid(alpha + beta * t)
    cash = total_liq * sigmoid(cash_alpha + cash_beta * t)
    """

    def __init__(self, name="trend_liquidity"):
        super().__init__(name=name)
        self.tl_alpha = tf.Variable(-1.66, dtype=tf.float64, name="tl_alpha")
        self.tl_beta = tf.Variable(0.0, dtype=tf.float64, name="tl_beta")
        self.tl_baseline = tf.Variable(0.0, dtype=tf.float64, name="tl_baseline")
        self.cash_alpha = tf.Variable(-0.05, dtype=tf.float64, name="cash_alpha")
        self.cash_beta = tf.Variable(0.0, dtype=tf.float64, name="cash_beta")

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _EPS = 1e-12
        total_liq = s["cash"] + s["ims"]
        tl_ratio = float(tf.reduce_mean(total_liq / tf.maximum(s["sales"], _EPS)))
        tl_ratio = min(1 - _EPS, max(_EPS, tl_ratio))
        cash_ratio = float(tf.reduce_mean(s["cash"] / tf.maximum(total_liq, _EPS)))
        cash_ratio = min(1 - _EPS, max(_EPS, cash_ratio))
        # sigmoid(alpha) = ratio → alpha = logit(ratio)
        import math

        self.tl_alpha.assign(math.log(tl_ratio / (1 - tl_ratio)))
        self.tl_beta.assign(0.0)
        self.tl_baseline.assign(0.0)
        self.cash_alpha.assign(math.log(cash_ratio / (1 - cash_ratio)))
        self.cash_beta.assign(0.0)

    def compute(self, sales_t: tf.Tensor, time_index: tf.Tensor) -> tuple:
        tl_pct = tf.sigmoid(self.tl_alpha + self.tl_beta * time_index)
        total_liq_target = self.tl_baseline + sales_t * tl_pct
        cash_pct = tf.sigmoid(self.cash_alpha + self.cash_beta * time_index)
        cash_target = total_liq_target * cash_pct
        ims_target = total_liq_target - cash_target
        return total_liq_target, cash_target, ims_target

    def print_summary(self, n_years: int) -> None:
        """Print learned liquidity parameters."""
        import tensorflow as tf

        print(
            f"Total Liquidity (baseline + logit-linear): "
            f"baseline={self.tl_baseline.numpy():.4f}, "
            f"alpha={self.tl_alpha.numpy():.4f}, "
            f"beta={self.tl_beta.numpy():.6f}"
        )
        print(
            f"  => %TL at t=0: {tf.sigmoid(self.tl_alpha).numpy():.4f}, "
            f"%TL at t={n_years-1}: "
            f"{tf.sigmoid(self.tl_alpha + self.tl_beta * (n_years-1)).numpy():.4f}"
        )
        print(
            f"Cash % of Liquidity (logit-linear): "
            f"alpha={self.cash_alpha.numpy():.4f}, "
            f"beta={self.cash_beta.numpy():.6f}"
        )
        print(
            f"  => %Cash at t=0: {tf.sigmoid(self.cash_alpha).numpy():.4f}, "
            f"%Cash at t={n_years-1}: "
            f"{tf.sigmoid(self.cash_alpha + self.cash_beta * (n_years-1)).numpy():.4f}"
        )

    def loss(
        self,
        sales: tf.Tensor,
        cash: tf.Tensor,
        ims: tf.Tensor,
        time_indices: tf.Tensor,
        scale_tl: tf.Tensor,
        scale_cash: tf.Tensor,
    ) -> tuple:
        """MSE loss for logit-linear total liquidity and cash split."""
        total_liq = cash + ims
        tl_pct_t = tf.sigmoid(self.tl_alpha + self.tl_beta * time_indices)
        loss_tl = tf.reduce_mean(
            tf.square((total_liq - (self.tl_baseline + sales * tl_pct_t)) / scale_tl)
        )
        cash_pct_t = tf.sigmoid(self.cash_alpha + self.cash_beta * time_indices)
        loss_cash = tf.reduce_mean(
            tf.square((cash - total_liq * cash_pct_t) / scale_cash)
        )
        return loss_tl, loss_cash
