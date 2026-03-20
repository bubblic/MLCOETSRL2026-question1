"""Bayesian OpEx module with variational inference.

Encapsulates the six OpEx-related parameters, posterior sampling,
KL divergence, and the ELBO loss function.  Designed to be composed
into a financial model as ``self.opex_module = BayesianOpEx()``.
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

_ZERO = tf.constant(0.0, dtype=tf.float64)


class BayesianOpEx(tf.Module):
    """Operating expenses modeled with variational inference.

    OpEx = (base_opex * cum_inflation) + (var_opex * centered_sales) + noise

    Parameters are learned via a Normal variational posterior with wide
    priors.  The ELBO (Evidence Lower BOund)-based ``loss`` method computes the full training objective
    so the trainer does not need to know about KL divergence or sampling.
    """

    def __init__(self, name="bayesian_opex"):
        super().__init__(name=name)

        # Variable OpEx %
        self.q_var_opex_loc = tf.Variable(0.0, dtype=tf.float64, name="q_var_opex_loc")
        self.q_var_opex_scale = tfp.util.TransformedVariable(
            initial_value=1.0,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="q_var_opex_scale",
        )

        # Baseline OpEx
        self.q_base_opex_loc = tf.Variable(
            0.0, dtype=tf.float64, name="q_base_opex_loc"
        )
        self.q_base_opex_scale = tfp.util.TransformedVariable(
            initial_value=1.0,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="q_base_opex_scale",
        )

        # Aleatoric uncertainty (inherent noise in OpEx data)
        self.noise_sigma = tfp.util.TransformedVariable(
            initial_value=1.0,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="noise_sigma",
        )

        # Sales offset (for centering sales during training, not trainable)
        self.sales_offset = tf.Variable(
            0.0,
            dtype=tf.float64,
            name="sales_offset",
            trainable=False,
        )
        self.amount_scale = 1.0  # overridden by set_forecast_drivers

    def set_forecast_drivers(self, scaled_sales, amount_scale):
        """Set data-derived quantities needed for training and inference.

        Args:
            scaled_sales: 1-D tensor of historical sales (already scaled).
            amount_scale: USD-to-scaled-units conversion factor.
        """
        self.amount_scale = amount_scale
        self.sales_offset.assign(tf.reduce_mean(scaled_sales))

    def compute(self, sales_t, cum_inflation, var_opex, base_opex, noise):
        """Compute OpEx given pre-sampled parameters.

        Args:
            sales_t: ``[n_samples]`` sales for this period.
            cum_inflation: Scalar cumulative inflation factor.
            var_opex: ``[n_samples]`` sampled variable OpEx percentage.
            base_opex: ``[n_samples]`` sampled baseline OpEx.
            noise: ``[n_samples]`` sampled aleatoric noise.

        Returns:
            ``[n_samples]`` OpEx tensor.
        """
        sales_t_centered = sales_t - self.sales_offset
        return (base_opex * cum_inflation) + (sales_t_centered * var_opex) + noise

    def sample(self):
        """Sample OpEx parameters from the variational posterior.

        Returns:
            Tuple ``(var_opex_sample, base_opex_sample)`` of scalar tensors.
        """
        q_var = tfd.Normal(loc=self.q_var_opex_loc, scale=self.q_var_opex_scale)
        q_base = tfd.Normal(loc=self.q_base_opex_loc, scale=self.q_base_opex_scale)
        return q_var.sample(), q_base.sample()

    def kl_divergence(self):
        """Compute KL(posterior || prior) for both OpEx parameters.

        Returns:
            Scalar tensor with the summed KL divergence.
        """
        prior_var = tfd.Normal(loc=_ZERO, scale=1.0e10)
        prior_base = tfd.Normal(loc=_ZERO, scale=1.0e10)
        q_var = tfd.Normal(loc=self.q_var_opex_loc, scale=self.q_var_opex_scale)
        q_base = tfd.Normal(loc=self.q_base_opex_loc, scale=self.q_base_opex_scale)
        return tfd.kl_divergence(q_var, prior_var) + tfd.kl_divergence(
            q_base, prior_base
        )

    def loss(self, observed_opex, sales, cum_inflation, loss_scale):
        """Compute the ELBO loss for variational OpEx training.

        Encapsulates sales centering, sampling, negative log-likelihood,
        and KL divergence into a single scalar loss.

        Args:
            observed_opex: 1-D tensor of historical OpEx values (scaled).
            sales: 1-D tensor of historical sales (scaled, not centered).
            cum_inflation: 1-D tensor of cumulative inflation factors.
            scale: Normalization factor for residuals (e.g. std of OpEx).

        Returns:
            Scalar ELBO loss tensor.
        """
        var_opex_sample, base_opex_sample = self.sample()
        sales_centered = sales - self.sales_offset
        pred_opex = (base_opex_sample * cum_inflation) + (
            var_opex_sample * sales_centered
        )
        residuals = (observed_opex - pred_opex) / loss_scale
        likelihood = tfd.Normal(loc=_ZERO, scale=self.noise_sigma / loss_scale)
        nll = -tf.reduce_sum(likelihood.log_prob(residuals))
        kl = self.kl_divergence()
        n_obs = tf.cast(tf.shape(observed_opex)[0], tf.float64)
        return (nll + kl) / n_obs

    def print_summary(self) -> None:
        """Print learned OpEx parameter summary."""
        s = self.amount_scale
        print(
            f"Bayesian OpEx Variable %: "
            f"Mean={self.q_var_opex_loc.numpy():.4f}, "
            f"Std={self.q_var_opex_scale.numpy():.4f}"
        )
        print(
            f"Bayesian OpEx Baseline (USD):   "
            f"Mean={(self.q_base_opex_loc.numpy() * s):.2e}, "
            f"Std={(self.q_base_opex_scale.numpy() * s):.2e}"
        )
        print(
            f"OpEx aleatoric uncertainty (USD): "
            f"{(self.noise_sigma.numpy() * s):.2e}"
        )
