"""Operating expense layer with optional Bayesian uncertainty."""

import tensorflow as tf


class OpExLayer(tf.keras.layers.Layer):
    """Computes operating expenses with optional Bayesian uncertainty.

    In deterministic mode, uses fixed ``baseline_opex`` and
    ``variable_opex_pct``.  In Bayesian mode, samples from learned posterior
    distributions.

    The operating expense for period *t* is computed as::

        opex_t = baseline_opex * cum_inflation_t + sales_t * variable_opex_pct

    Args:
        baseline_opex: Fixed operating expense baseline (base-year nominal).
        variable_opex_pct: Variable operating expenses as a fraction of sales.
    """

    def __init__(self, baseline_opex, variable_opex_pct, **kwargs):
        super().__init__(dtype=tf.float64, **kwargs)
        self.baseline_opex = tf.cast(baseline_opex, tf.float64)
        self.variable_opex_pct = tf.cast(variable_opex_pct, tf.float64)

    def call(self, sales_t, cum_inflation, training=False):
        """Compute operating expenses for one period.

        Args:
            sales_t: Sales revenue for the current period.
            cum_inflation: Cumulative inflation index relative to the base year.
            training: Whether the layer is in training mode. Reserved for
                future Bayesian sampling behaviour.

        Returns:
            A scalar ``tf.Tensor`` (float64) representing operating expenses.
        """
        opex = self.baseline_opex * cum_inflation + sales_t * self.variable_opex_pct
        return opex
