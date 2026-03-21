"""Income tax modules with pluggable anomaly support.

Provides two implementations:

- ``SimpleTax``: ``tax = ebt * rate`` -- no anomaly adjustment.
- ``TaxWithAnomalies``: extends ``SimpleTax`` with anomaly-aware
  ``loss()`` that accounts for one-time payments during training.

Usage in entry point::

    model = TrainableFinancialModel()

    # Without anomalies (default):
    # model.tax_module is already SimpleTax

    # With anomalies:
    model.tax_module = TaxWithAnomalies(get_tax_onetime_payments())
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class SimpleTax(tf.Module):
    """Tax = ebt * income_tax_pct."""

    def __init__(self, initial_tax_pct=0.147, name="simple_tax"):
        super().__init__(name=name)
        self.income_tax_pct = tfp.util.TransformedVariable(
            initial_value=initial_tax_pct,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="tax_pct",
        )

    def compute(self, ebt):
        """Compute income tax."""
        return ebt * self.income_tax_pct

    def prepare_for_training(self, amount_scale):
        """Called by the pipeline before training. No-op for SimpleTax."""

    def loss(self, observed_tax, net_income, loss_scale):
        """Compute MSE loss for tax prediction.

        Derives predicted tax from net income:
        ``tax_pred = NI / (1/tax_pct - 1)``.
        """
        _one = tf.constant(1.0, dtype=tf.float64)
        tax_pred = net_income / (_one / self.income_tax_pct - _one)
        return tf.reduce_mean(tf.square((observed_tax - tax_pred) / loss_scale))


class TaxWithAnomalies(SimpleTax):
    """Extends SimpleTax with anomaly-aware training loss.

    ``compute()`` is inherited -- produces systematic tax only.
    ``loss()`` additionally accounts for stored one-time payments so
    the tax rate is not distorted by anomaly years.

    Args:
        tax_onetime_usd: 1-D tensor of one-time tax amounts in USD,
            aligned to the full historical years.
    """

    def __init__(self, tax_onetime_usd, initial_tax_pct=0.147,
                 name="tax_with_anomalies"):
        super().__init__(initial_tax_pct=initial_tax_pct, name=name)
        self._onetime_usd = tax_onetime_usd
        self._onetime_scaled = None

    def prepare_for_training(self, amount_scale):
        """Scale stored onetime data for training.

        Called by the pipeline after ``amount_scale`` is computed.
        The full array is stored; ``loss()`` slices to match the
        training window dynamically.
        """
        self._onetime_scaled = self._onetime_usd / amount_scale

    def loss(self, observed_tax, net_income, loss_scale):
        """Compute MSE loss including one-time payments.

        Derives predicted tax from net income:
        ``tax_pred = NI / (1/tax_pct - 1) + onetime``.
        """
        _one = tf.constant(1.0, dtype=tf.float64)
        tax_pred = net_income / (_one / self.income_tax_pct - _one)
        if self._onetime_scaled is not None:
            n = tf.shape(net_income)[0]
            tax_pred = tax_pred + self._onetime_scaled[:n]
        return tf.reduce_mean(tf.square((observed_tax - tax_pred) / loss_scale))
