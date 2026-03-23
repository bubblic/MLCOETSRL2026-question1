"""Capital expenditure policy module.

Owns asset growth, asset maintenance, and depreciation rate parameters.
Computes NCA evolution: ``NCA_t = NCA_{t-1} - depr + capex``.
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors


class CapexPolicy(tf.Module):
    """Asset growth and depreciation policy.

    Parameters:
        asset_growth: Fraction of sales invested in new capacity.
        asset_maintain: Multiplier on depreciation for maintenance capex
            (1.0 = exact replacement, >1 = net expansion).
        depreciation_rate: Fraction of NCA depreciated each period.
    """

    def __init__(self, prior_strength_asset_maintain=1.0, name="capex"):
        super().__init__(name=name)
        self.prior_strength_am = prior_strength_asset_maintain
        self.asset_growth = tfp.util.TransformedVariable(
            initial_value=0.0076, bijector=tfb.Softplus(),
            dtype=tf.float64, name="asset_growth",
        )
        self.asset_maintain = tfp.util.TransformedVariable(
            initial_value=0.99, bijector=tfb.Softplus(),
            dtype=tf.float64, name="asset_maintain",
        )
        self.depreciation_rate = tfp.util.TransformedVariable(
            initial_value=0.055, bijector=tfb.Softplus(),
            dtype=tf.float64, name="depr_rate",
        )

    def compute(self, nca_prev, sales_t):
        """Evolve non-current assets.

        Args:
            nca_prev: ``[n_samples]`` previous-period NCA.
            sales_t: ``[n_samples]`` current-period sales.

        Returns:
            Tuple ``(depreciation, capex, nca_curr)``.
        """
        depreciation = nca_prev * self.depreciation_rate
        capex = self.asset_maintain * depreciation + sales_t * self.asset_growth
        nca_curr = nca_prev - depreciation + capex
        return depreciation, capex, nca_curr

    def loss(self, delta_nca, depr_true, sales_aligned, nca_prev,
             scale_growth, scale_depr):
        """MSE losses for asset growth and depreciation, plus prior.

        Returns:
            Tuple ``(loss_growth, loss_depr, prior_loss_am)``.
        """
        _one = tf.constant(1.0, dtype=tf.float64)
        loss_growth = tf.reduce_mean(
            tf.square(
                (
                    delta_nca
                    - (
                        (self.asset_maintain - _one) * depr_true
                        + sales_aligned * self.asset_growth
                    )
                )
                / scale_growth
            )
        )
        loss_depr = tf.reduce_mean(
            tf.square(
                (depr_true - nca_prev * self.depreciation_rate)
                / scale_depr
            )
        )
        prior_loss_am = tf.constant(
            self.prior_strength_am, dtype=tf.float64,
        ) * tf.square(self.asset_maintain - _one)
        return loss_growth, loss_depr, prior_loss_am

    def print_summary(self):
        """Print learned parameters."""
        print(f"Final %AG: {self.asset_growth.numpy():.5f}")
        print(f"Final %AM: {self.asset_maintain.numpy():.5f}")
        print(f"Final %Depr: {self.depreciation_rate.numpy():.5f}")
