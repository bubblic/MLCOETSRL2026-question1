"""Income tax modules with pluggable anomaly support.

Provides two implementations:

- ``SimpleTax``: ``tax = ebt * rate`` -- no anomaly adjustment.
- ``TaxWithAnomalies``: extends ``SimpleTax``; ``compute()`` adds
  one-time payments when the forecast year has a known anomaly.

Usage in entry point::

    data = HistoricalDataLoader("aapl", include_tax_onetime=True)
    model = TrainableFinancialModel(
        opex_module=BayesianOpEx(),
        tax_module=TaxWithAnomalies(data.tax_onetime_payments),
    )
"""

from typing import Dict, Optional

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class SimpleTax(tf.Module):
    """Tax = ebt * income_tax_pct."""

    def __init__(
        self,
        initial_tax_pct: float = 0.147,
        name: str = "simple_tax",
    ):
        super().__init__(name=name)
        self.income_tax_pct = tfp.util.TransformedVariable(
            initial_value=initial_tax_pct,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="tax_pct",
        )

    def compute(self, ebt: tf.Tensor, year: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Compute income tax.

        Args:
            ebt: Earnings before tax tensor.
            year: Optional calendar year (int or scalar tensor).
                Ignored by SimpleTax.

        Returns:
            Tax tensor.
        """
        return ebt * self.income_tax_pct

    def prepare_for_training(
        self, amount_scale: float, years: Optional[tf.Tensor] = None
    ) -> None:
        """Called by the pipeline before training. No-op for SimpleTax."""

    def loss(
        self, observed_tax: tf.Tensor, net_income: tf.Tensor, loss_scale: tf.Tensor
    ) -> tf.Tensor:
        """Compute MSE loss for tax prediction.

        Derives predicted tax from net income:
        ``tax_pred = NI / (1/tax_pct - 1)``.
        """
        _one = tf.constant(1.0, dtype=tf.float64)
        tax_pred = net_income / (_one / self.income_tax_pct - _one)
        return tf.reduce_mean(tf.square((observed_tax - tax_pred) / loss_scale))

    def print_summary(self) -> None:
        """Print learned parameters."""
        print(f"Final %IT: {self.income_tax_pct.numpy():.5f}")


class TaxWithAnomalies(SimpleTax):
    """Extends SimpleTax with one-time tax anomaly support.

    ``compute(ebt, year)`` looks up the calendar year in the stored
    anomaly dict.  If a one-time payment exists for that year, it is
    added to the systematic ``ebt * tax_pct``.  For future years (or
    years without anomalies), only the systematic component is returned.

    ``loss()`` accounts for one-time payments during training so the
    tax rate parameter is not distorted by anomaly years.

    Args:
        tax_onetime_by_year: Dict mapping fiscal year (int) to one-time
            tax amount in USD.
    """

    def __init__(
        self,
        tax_onetime_by_year: Dict[int, float],
        initial_tax_pct: float = 0.147,
        name: str = "tax_with_anomalies",
    ):
        super().__init__(initial_tax_pct=initial_tax_pct, name=name)
        self._onetime_by_year = tax_onetime_by_year  # {year: usd_amount}
        self._onetime_scaled_by_year = None
        self._training_adjustments = None
        # Graph-mode lookup: dense tensor + base year offset
        self._lookup_tensor = None
        self._lookup_base_year = None

    def prepare_for_training(
        self, amount_scale: float, years: Optional[tf.Tensor] = None
    ) -> None:
        """Scale stored onetime data and build training adjustment tensor.

        Args:
            amount_scale: USD-to-scaled-units conversion factor.
            years: Optional 1-D tensor of training year labels.  If
                provided, builds a pre-computed adjustment tensor for
                use inside ``@tf.function`` compiled training.
        """
        self._onetime_scaled_by_year = {
            yr: amt / amount_scale for yr, amt in self._onetime_by_year.items()
        }
        # Build dense lookup tensor for graph-mode compute()
        all_years = sorted(self._onetime_by_year.keys())
        if years is not None:
            # Include both anomaly years and training years
            all_years = sorted(set(all_years) | {int(y) for y in years.numpy()})
        if all_years:
            self._lookup_base_year = min(all_years)
            span = max(all_years) - self._lookup_base_year + 1
            lookup = [0.0] * span
            for yr, amt in self._onetime_scaled_by_year.items():
                lookup[yr - self._lookup_base_year] = amt
            self._lookup_tensor = tf.constant(lookup, dtype=tf.float64)
        # Pre-compute training adjustment vector
        if years is not None:
            self._training_adjustments = tf.constant(
                [
                    self._onetime_scaled_by_year.get(int(yr), 0.0)
                    for yr in years.numpy()
                ],
                dtype=tf.float64,
            )
        else:
            self._training_adjustments = None

    def compute(self, ebt: tf.Tensor, year: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Compute income tax, adding one-time anomaly if year has one.

        Args:
            ebt: Earnings before tax tensor.
            year: Calendar year (int, float, or scalar tensor).
                If the year has a stored anomaly, it is added.
                If ``None`` or not in the anomaly dict, systematic only.

        Returns:
            Tax tensor.
        """
        tax = ebt * self.income_tax_pct
        if year is not None and self._lookup_tensor is not None:
            idx = tf.cast(year, tf.int32) - self._lookup_base_year
            n = tf.shape(self._lookup_tensor)[0]
            # Clamp index to valid range so tf.gather doesn't crash,
            # then zero out the result for out-of-range years.
            safe_idx = tf.clip_by_value(idx, 0, n - 1)
            in_range = tf.logical_and(idx >= 0, idx < n)
            adjustment = tf.where(
                in_range,
                tf.gather(self._lookup_tensor, safe_idx),
                tf.constant(0.0, dtype=tf.float64),
            )
            tax = tax + adjustment
        return tax

    def loss(
        self, observed_tax: tf.Tensor, net_income: tf.Tensor, loss_scale: tf.Tensor
    ) -> tf.Tensor:
        """Compute MSE loss including one-time payments.

        Uses the pre-computed adjustment tensor built by
        ``prepare_for_training``.

        Args:
            observed_tax: 1-D tensor of historical tax (scaled).
            net_income: 1-D tensor of historical net income (scaled).
            loss_scale: Normalization factor.

        Returns:
            Scalar MSE loss tensor.
        """
        _one = tf.constant(1.0, dtype=tf.float64)
        tax_pred = net_income / (_one / self.income_tax_pct - _one)
        if self._training_adjustments is not None:
            n = tf.shape(net_income)[0]
            tax_pred = tax_pred + self._training_adjustments[:n]
        return tf.reduce_mean(tf.square((observed_tax - tax_pred) / loss_scale))
