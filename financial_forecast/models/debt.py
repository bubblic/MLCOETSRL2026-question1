"""Debt financing policy modules.

- ``SimpleDebtPolicy``: deficit-driven ST loans, static equity financing %.
- ``TrendDebtPolicy``: policy-driven ST debt (logit-linear), time-varying equity mix.
"""

from abc import abstractmethod
from typing import Dict, List

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class DebtPolicy(tf.Module):
    """Abstract base class for debt financing policy."""

    @abstractmethod
    def compute_st_debt(
        self,
        sales_t: tf.Tensor,
        time_index: tf.Tensor,
        liquidity_deficit_st: tf.Tensor,
    ) -> tf.Tensor:
        """Compute effective short-term debt.

        Args:
            sales_t: ``[n_samples]`` sales tensor.
            time_index: Scalar ``year - base_year``.
            liquidity_deficit_st: ``[n_samples]`` ST liquidity gap.

        Returns:
            ``[n_samples]`` effective ST debt tensor.
        """

    @abstractmethod
    def compute_financing_mix(
        self,
        long_term_financing: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tuple:
        """Split long-term financing into debt and equity.

        Args:
            long_term_financing: ``[n_samples]`` total LT financing needed.
            time_index: Scalar ``year - base_year``.

        Returns:
            Tuple ``(new_lt_loan, equity_financing)``.
        """

    @abstractmethod
    def evolve_lt_liabilities(
        self,
        new_lt_loan: tf.Tensor,
        ncl_prev: tf.Tensor,
    ) -> tuple:
        """Evolve long-term debt balances.

        Returns:
            Tuple ``(ncl_curr, cur_lt_debt_curr)``.
        """

    @property
    @abstractmethod
    def policy_trainable_variables(self) -> List[tf.Variable]:
        """Variables optimized during policy training (ST debt trends)."""

    @property
    @abstractmethod
    def structural_trainable_variables(self) -> List[tf.Variable]:
        """Variables optimized during structural training (interest/debt phase)."""

    @abstractmethod
    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        """Initialize parameters from historical averages."""

    @abstractmethod
    def print_policy_summary(self, n_years: int) -> None:
        """Print policy-phase parameters."""

    @abstractmethod
    def print_structural_summary(self, n_years: int) -> None:
        """Print structural-phase parameters."""


class SimpleDebtPolicy(DebtPolicy):
    """Deficit-driven ST loans, static equity financing %."""

    def __init__(self, name="simple_debt"):
        super().__init__(name=name)
        self.equity_financing_pct = tfp.util.TransformedVariable(
            initial_value=0.15,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="equity_financing_pct",
        )
        self.avg_maturity_years = tfp.util.TransformedVariable(
            initial_value=3.0,
            bijector=tfb.Chain(
                [tfb.Shift(tf.constant(1.001, dtype=tf.float64)), tfb.Softplus()]
            ),
            dtype=tf.float64,
            name="avg_maturity_years",
        )

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        total_lt = s["non_current_liabilities"] + s["current_lt_debt"]
        avg_mat = float(
            tf.reduce_mean(total_lt / tf.maximum(s["current_lt_debt"], _EPS))
        )
        self.avg_maturity_years.assign(_f64(max(1.5, min(avg_mat, 30.0))))

    def compute_st_debt(
        self,
        sales_t: tf.Tensor,
        time_index: tf.Tensor,
        liquidity_deficit_st: tf.Tensor,
    ) -> tf.Tensor:
        return tf.maximum(
            tf.constant(0.0, dtype=tf.float64),
            liquidity_deficit_st,
        )

    def compute_financing_mix(
        self,
        long_term_financing: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tuple:
        new_lt_loan = long_term_financing * (1 - self.equity_financing_pct)
        equity_financing = long_term_financing * self.equity_financing_pct
        return new_lt_loan, equity_financing

    def evolve_lt_liabilities(
        self,
        new_lt_loan: tf.Tensor,
        ncl_prev: tf.Tensor,
    ) -> tuple:
        total = new_lt_loan + ncl_prev
        ncl_curr = total * (1 - 1 / self.avg_maturity_years)
        cur_lt_debt_curr = total / self.avg_maturity_years
        return ncl_curr, cur_lt_debt_curr

    @property
    def policy_trainable_variables(self) -> List[tf.Variable]:
        return []

    @property
    def structural_trainable_variables(self) -> List[tf.Variable]:
        return [
            self.avg_maturity_years.trainable_variables[0],
            self.equity_financing_pct.trainable_variables[0],
        ]

    def loss_st_debt(
        self,
        eff_st_debt: tf.Tensor,
        sales: tf.Tensor,
        time_indices: tf.Tensor,
        scale: tf.Tensor,
    ) -> tf.Tensor:
        """No ST debt loss for deficit-driven policy."""
        return tf.constant(0.0, dtype=tf.float64)

    def print_policy_summary(self, n_years: int) -> None:
        pass

    def print_structural_summary(self, n_years: int) -> None:
        print(f"Equity Financing %: {self.equity_financing_pct.numpy():.5f}")
        print(f"Avg Maturity Years: {self.avg_maturity_years.numpy():.4f}")


class TrendDebtPolicy(DebtPolicy):
    """Policy-driven ST debt (logit-linear), time-varying equity financing mix."""

    def __init__(self, name="trend_debt"):
        super().__init__(name=name)
        self.st_debt_alpha = tf.Variable(
            -1.59,
            dtype=tf.float64,
            name="st_debt_alpha",
        )
        self.st_debt_beta = tf.Variable(
            0.0,
            dtype=tf.float64,
            name="st_debt_beta",
        )
        self.ef_alpha = tf.Variable(-1.73, dtype=tf.float64, name="ef_alpha")
        self.ef_beta = tf.Variable(0.0, dtype=tf.float64, name="ef_beta")
        self.avg_maturity_years = tfp.util.TransformedVariable(
            initial_value=3.0,
            bijector=tfb.Chain(
                [tfb.Shift(tf.constant(1.001, dtype=tf.float64)), tfb.Softplus()]
            ),
            dtype=tf.float64,
            name="avg_maturity_years",
        )

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        import math

        # ST debt as % of sales → logit
        st_ratio = float(
            tf.reduce_mean(s["effective_st_debt"] / tf.maximum(s["sales"], _EPS))
        )
        st_ratio = min(1 - _EPS, max(_EPS, st_ratio))
        self.st_debt_alpha.assign(math.log(st_ratio / (1 - st_ratio)))
        self.st_debt_beta.assign(0.0)
        # avg maturity
        total_lt = s["non_current_liabilities"] + s["current_lt_debt"]
        avg_mat = float(
            tf.reduce_mean(total_lt / tf.maximum(s["current_lt_debt"], _EPS))
        )
        self.avg_maturity_years.assign(_f64(max(1.5, min(avg_mat, 30.0))))

    def compute_st_debt(
        self, sales_t: tf.Tensor, time_index: tf.Tensor, liquidity_deficit_st: tf.Tensor
    ) -> tf.Tensor:
        st_debt_pct = tf.sigmoid(self.st_debt_alpha + self.st_debt_beta * time_index)
        return sales_t * st_debt_pct

    def compute_financing_mix(
        self, long_term_financing: tf.Tensor, time_index: tf.Tensor
    ) -> tuple:
        ef_pct = tf.sigmoid(self.ef_alpha + self.ef_beta * time_index)
        new_lt_loan = long_term_financing * (1 - ef_pct)
        equity_financing = long_term_financing * ef_pct
        return new_lt_loan, equity_financing

    def evolve_lt_liabilities(
        self, new_lt_loan: tf.Tensor, ncl_prev: tf.Tensor
    ) -> tuple:
        total = new_lt_loan + ncl_prev
        ncl_curr = total * (1 - 1 / self.avg_maturity_years)
        cur_lt_debt_curr = total / self.avg_maturity_years
        return ncl_curr, cur_lt_debt_curr

    @property
    def policy_trainable_variables(self) -> List[tf.Variable]:
        return [self.st_debt_alpha, self.st_debt_beta]

    @property
    def structural_trainable_variables(self) -> List[tf.Variable]:
        return [
            self.avg_maturity_years.trainable_variables[0],
            self.ef_alpha,
            self.ef_beta,
        ]

    def loss_st_debt(
        self,
        eff_st_debt: tf.Tensor,
        sales: tf.Tensor,
        time_indices: tf.Tensor,
        scale: tf.Tensor,
    ) -> tf.Tensor:
        """MSE loss for policy-driven ST debt."""
        st_debt_pct_pred = tf.sigmoid(
            self.st_debt_alpha + self.st_debt_beta * time_indices
        )
        return tf.reduce_mean(
            tf.square((eff_st_debt - sales * st_debt_pct_pred) / scale)
        )

    def print_policy_summary(self, n_years: int) -> None:
        import tensorflow as tf

        print(
            f"Effective ST Debt % of Sales (logit-linear): "
            f"alpha={self.st_debt_alpha.numpy():.4f}, "
            f"beta={self.st_debt_beta.numpy():.6f}"
        )
        print(
            f"  => %EffSTDebt at t=0: "
            f"{tf.sigmoid(self.st_debt_alpha).numpy():.4f}, "
            f"%EffSTDebt at t={n_years-1}: "
            f"{tf.sigmoid(self.st_debt_alpha + self.st_debt_beta * (n_years-1)).numpy():.4f}"
        )

    def print_structural_summary(self, n_years: int) -> None:
        import tensorflow as tf

        print(
            f"Equity Financing (logit-linear): "
            f"alpha={self.ef_alpha.numpy():.4f}, "
            f"beta={self.ef_beta.numpy():.6f}"
        )
        print(f"Avg Maturity Years: {self.avg_maturity_years.numpy():.4f}")
