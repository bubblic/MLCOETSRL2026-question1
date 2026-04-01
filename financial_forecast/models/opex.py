"""OpEx modules: deterministic and Bayesian.

Provides an abstract base class and two implementations:

- ``OpExModule``: abstract interface all OpEx modules must satisfy.
- ``SimpleOpEx``: deterministic linear OpEx (no uncertainty).
- ``BayesianOpEx``: variational inference with aleatoric noise.

Designed to be composed into a financial model as
``self.opex_module = SimpleOpEx()`` or ``BayesianOpEx()``.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from financial_forecast.training.io_utils import get_training_results_path

tfd = tfp.distributions
tfb = tfp.bijectors

_ZERO = tf.constant(0.0, dtype=tf.float64)


class OpExModule(tf.Module):
    """Abstract base class for operating expense modules.

    Defines the interface that the model, trainers, and forecast
    engine depend on.  Subclasses choose the formula and whether
    the module is stochastic.
    """

    is_stochastic: bool = False

    @abstractmethod
    def predict(
        self,
        sales_t: tf.Tensor,
        cum_inflation: tf.Tensor,
        use_mean: bool = False,
    ) -> tf.Tensor:
        """Compute OpEx for a single forecast step.

        Args:
            sales_t: ``[n_samples]`` sales tensor.
            cum_inflation: Scalar cumulative inflation factor.
            use_mean: If ``True``, suppress any stochastic sampling.

        Returns:
            ``[n_samples]`` OpEx tensor.
        """

    @abstractmethod
    def prepare_mc(self, n_samples: int, n_years: int) -> None:
        """Pre-sample stochastic values for Monte Carlo forecast."""

    @abstractmethod
    def compute_mc_step(
        self,
        sales_t: tf.Tensor,
        cum_inflation: tf.Tensor,
        step: tf.Tensor,
    ) -> tf.Tensor:
        """Compute OpEx for one MC forecast step.

        Args:
            sales_t: ``[n_samples]`` sales tensor.
            cum_inflation: Scalar cumulative inflation factor.
            step: Integer year index into pre-sampled values.

        Returns:
            ``[n_samples]`` OpEx tensor.
        """

    @abstractmethod
    def loss(
        self,
        observed_opex: tf.Tensor,
        sales: tf.Tensor,
        cum_inflation: tf.Tensor,
        loss_scale: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the training loss for OpEx.

        Args:
            observed_opex: 1-D tensor of historical OpEx (scaled).
            sales: 1-D tensor of historical sales (scaled).
            cum_inflation: 1-D tensor of cumulative inflation factors.
            loss_scale: Normalization factor.

        Returns:
            Scalar loss tensor.
        """

    @abstractmethod
    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        """Initialize parameters from historical averages."""

    @abstractmethod
    def prepare_for_training(
        self,
        amount_scale: float,
        scaled_sales: tf.Tensor,
        scaled_opex: tf.Tensor,
        inflation: tf.Tensor,
    ) -> None:
        """Store data-derived quantities for training and plotting."""

    @abstractmethod
    def record_step(self, epoch: int, loss: float) -> None:
        """Record training state for diagnostics."""

    @abstractmethod
    def plot_diagnostics(self, show_plot: bool = False) -> None:
        """Plot training diagnostics (no-op if nothing to plot)."""

    @abstractmethod
    def print_summary(self) -> None:
        """Print learned parameter summary."""


class SimpleOpEx(OpExModule):
    """Deterministic linear OpEx model.

    OpEx = baseline_opex * cum_inflation + sales * variable_opex_pct

    No uncertainty, no sampling.  Training uses simple MSE loss.
    """

    is_stochastic = False

    def __init__(self, name="simple_opex"):
        super().__init__(name=name)
        self.variable_opex_pct = tf.Variable(
            0.0,
            dtype=tf.float64,
            name="variable_opex_pct",
        )
        self.baseline_opex = tf.Variable(
            0.0,
            dtype=tf.float64,
            name="baseline_opex",
        )
        # Non-trainable centering constant — decorrelates baseline from
        # variable_opex_pct so gradient descent converges reliably.
        self.sales_offset = tf.Variable(
            0.0,
            dtype=tf.float64,
            trainable=False,
            name="sales_offset",
        )
        self.amount_scale = 1.0
        self._historical_sales_scaled = None
        self._historical_opex_scaled = None
        self._historical_inflation = None
        self._training_history = {"epochs": [], "loss": []}

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        # Center sales to decorrelate the slope and intercept, matching
        # the BayesianOpEx pattern.  The LS fit on centered data gives
        # slope = cov(sales, opex) / var(sales), intercept = mean(opex).
        sales = tf.cast(s["sales"], tf.float64)
        opex = tf.cast(s["opex"], tf.float64)
        sales_mean = tf.reduce_mean(sales)
        opex_mean = tf.reduce_mean(opex)
        slope = tf.reduce_sum((sales - sales_mean) * (opex - opex_mean)) / (
            tf.reduce_sum(tf.square(sales - sales_mean)) + 1e-12
        )
        self.sales_offset.assign(float(sales_mean))
        self.variable_opex_pct.assign(float(slope))
        self.baseline_opex.assign(float(opex_mean))

    def prepare_for_training(
        self,
        amount_scale: float,
        scaled_sales: tf.Tensor,
        scaled_opex: tf.Tensor,
        inflation: tf.Tensor,
    ) -> None:
        """Store data-derived quantities."""
        self.amount_scale = amount_scale
        self._historical_sales_scaled = scaled_sales
        self._historical_opex_scaled = scaled_opex
        self._historical_inflation = inflation

    def predict(
        self,
        sales_t: tf.Tensor,
        cum_inflation: tf.Tensor,
        use_mean: bool = False,
    ) -> tf.Tensor:
        """Compute OpEx deterministically. ``use_mean`` is ignored."""
        sales_centered = sales_t - self.sales_offset
        return self.baseline_opex * cum_inflation + sales_centered * self.variable_opex_pct

    def prepare_mc(self, n_samples: int, n_years: int) -> None:
        """No-op — deterministic model has no stochastic values."""

    def compute_mc_step(
        self,
        sales_t: tf.Tensor,
        cum_inflation: tf.Tensor,
        step: tf.Tensor,
    ) -> tf.Tensor:
        """Compute OpEx deterministically (same for all samples)."""
        return self.predict(sales_t, cum_inflation)

    def loss(
        self,
        observed_opex: tf.Tensor,
        sales: tf.Tensor,
        cum_inflation: tf.Tensor,
        loss_scale: tf.Tensor,
    ) -> tf.Tensor:
        """MSE loss for deterministic OpEx."""
        sales_centered = sales - self.sales_offset
        pred = self.baseline_opex * cum_inflation + sales_centered * self.variable_opex_pct
        return tf.reduce_mean(tf.square((observed_opex - pred) / loss_scale))

    def record_step(self, epoch: int, loss: float) -> None:
        """Record training state for diagnostics."""
        self._training_history["epochs"].append(epoch)
        self._training_history["loss"].append(loss)

    def plot_diagnostics(self, show_plot: bool = False) -> None:
        """No-op for deterministic OpEx — nothing to plot."""

    def print_summary(self) -> None:
        """Print learned OpEx parameters."""
        s = self.amount_scale
        print(f"OpEx Variable %: {self.variable_opex_pct.numpy():.4f}")
        print(f"OpEx Baseline (USD): {(self.baseline_opex.numpy() * s):.2e}")


class BayesianOpEx(OpExModule):
    """Operating expenses modeled with variational inference.

    OpEx = (base_opex * cum_inflation) + (var_opex * centered_sales) + noise

    Parameters are learned via a Normal variational posterior with wide
    priors.  The ELBO-based ``loss`` method computes the full training
    objective so the trainer does not need to know about KL divergence
    or sampling.
    """

    is_stochastic = True

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

        # Populated by set_forecast_drivers
        self._historical_sales_scaled = None
        self._historical_opex_scaled = None
        self._historical_inflation = None

        # Populated by prepare_mc
        self._mc_var_opex = None
        self._mc_base_opex = None
        self._mc_noise = None

        # Training history (populated by record_step)
        self._training_history = {
            "epochs": [],
            "loss": [],
            "q_var_opex_loc": [],
            "q_var_opex_scale": [],
            "q_base_opex_loc": [],
            "q_base_opex_scale": [],
            "noise_sigma": [],
        }

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        _EPS = 1e-12
        self.q_var_opex_loc.assign(
            float(tf.reduce_mean(s["opex"] / tf.maximum(s["sales"], _EPS)))
        )

    def prepare_for_training(
        self,
        amount_scale: float,
        scaled_sales: tf.Tensor,
        scaled_opex: tf.Tensor,
        inflation: tf.Tensor,
    ) -> None:
        """Set data-derived quantities needed for training, inference, and plotting.

        Args:
            scaled_sales: 1-D tensor of historical sales (already scaled).
            scaled_opex: 1-D tensor of historical OpEx (already scaled).
            inflation: 1-D tensor of annual inflation rates (or zeros).
            amount_scale: USD-to-scaled-units conversion factor.
        """
        self.amount_scale = amount_scale
        self.sales_offset.assign(tf.reduce_mean(scaled_sales))
        self._historical_sales_scaled = scaled_sales
        self._historical_opex_scaled = scaled_opex
        self._historical_inflation = inflation

    def _compute(
        self,
        sales_t: tf.Tensor,
        cum_inflation: tf.Tensor,
        var_opex: tf.Tensor,
        base_opex: tf.Tensor,
        noise: tf.Tensor,
    ) -> tf.Tensor:
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

    def sample(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Sample OpEx parameters from the variational posterior.

        Returns:
            Tuple ``(var_opex_sample, base_opex_sample)`` of scalar tensors.
        """
        q_var = tfd.Normal(loc=self.q_var_opex_loc, scale=self.q_var_opex_scale)
        q_base = tfd.Normal(loc=self.q_base_opex_loc, scale=self.q_base_opex_scale)
        return q_var.sample(), q_base.sample()

    def predict(
        self, sales_t: tf.Tensor, cum_inflation: tf.Tensor, use_mean: bool = False
    ) -> tf.Tensor:
        """Sample parameters and compute OpEx in one call.

        Convenience method for the dict-based ``forecast_step`` wrapper.

        Args:
            sales_t: ``[n_samples]`` sales tensor.
            cum_inflation: Scalar cumulative inflation factor.
            use_mean: If ``True``, use posterior means with zero noise.

        Returns:
            ``[n_samples]`` OpEx tensor.
        """
        n = sales_t.shape[0] or 1
        var_opex, base_opex, noise = self._get_step_params(n, use_mean=use_mean)
        return self._compute(sales_t, cum_inflation, var_opex, base_opex, noise)

    def prepare_mc(self, n_samples: int, n_years: int, start_year: float) -> None:
        """Pre-sample all stochastic values for a Monte Carlo forecast.

        Stores the samples internally.  The loop body then calls
        :meth:`compute_mc_step` to get the OpEx for each year.

        Args:
            n_samples: Number of MC trajectories.
            n_years: Number of forecast years.
            start_year: Calendar year of the first forecast step.
        """
        self._mc_start_year = tf.constant(start_year, dtype=tf.float64)
        q_var = tfd.Normal(loc=self.q_var_opex_loc, scale=self.q_var_opex_scale)
        q_base = tfd.Normal(loc=self.q_base_opex_loc, scale=self.q_base_opex_scale)
        self._mc_var_opex = q_var.sample([n_samples])
        self._mc_base_opex = q_base.sample([n_samples])
        self._mc_noise = tfd.Normal(_ZERO, self.noise_sigma).sample(
            [n_years, n_samples]
        )

    def compute_mc_step(
        self, sales_t: tf.Tensor, cum_inflation: tf.Tensor, year: tf.Tensor
    ) -> tf.Tensor:
        """Compute OpEx for one MC forecast step using pre-sampled values.

        Args:
            sales_t: ``[n_samples]`` sales tensor.
            cum_inflation: Scalar cumulative inflation factor.
            year: Scalar float64 calendar year (used to index noise).

        Returns:
            ``[n_samples]`` OpEx tensor.
        """
        step = tf.cast(year - self._mc_start_year, tf.int32)
        return self._compute(
            sales_t,
            cum_inflation,
            self._mc_var_opex,
            self._mc_base_opex,
            self._mc_noise[step],
        )

    def _get_step_params(
        self, n_samples: int, use_mean: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return ``(var_opex, base_opex, noise)`` ready for a forecast step.

        Args:
            n_samples: Batch size (1 for dict-based forecast, N for MC).
            use_mean: If ``True``, return posterior means with zero noise.
                If ``False``, sample from the posterior and add noise.

        Returns:
            Tuple ``(var_opex, base_opex, noise)`` each of shape
            ``[n_samples]``.
        """
        if use_mean:
            var_opex = tf.fill([n_samples], tf.cast(self.q_var_opex_loc, tf.float64))
            base_opex = tf.fill([n_samples], tf.cast(self.q_base_opex_loc, tf.float64))
            noise = tf.zeros([n_samples], dtype=tf.float64)
        else:
            q_var = tfd.Normal(loc=self.q_var_opex_loc, scale=self.q_var_opex_scale)
            q_base = tfd.Normal(loc=self.q_base_opex_loc, scale=self.q_base_opex_scale)
            var_opex = q_var.sample([n_samples])
            base_opex = q_base.sample([n_samples])
            noise = tfd.Normal(_ZERO, self.noise_sigma).sample([n_samples])
        return var_opex, base_opex, noise

    def kl_divergence(self) -> tf.Tensor:
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

    def loss(
        self,
        observed_opex: tf.Tensor,
        sales: tf.Tensor,
        cum_inflation: tf.Tensor,
        loss_scale: tf.Tensor,
    ) -> tf.Tensor:
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

    def record_step(self, epoch: int, loss: float) -> None:
        """Record training state for VI diagnostics."""
        h = self._training_history
        h["epochs"].append(epoch)
        h["loss"].append(loss)
        h["q_var_opex_loc"].append(self.q_var_opex_loc.numpy())
        h["q_var_opex_scale"].append(self.q_var_opex_scale.numpy())
        h["q_base_opex_loc"].append(self.q_base_opex_loc.numpy())
        h["q_base_opex_scale"].append(self.q_base_opex_scale.numpy())
        h["noise_sigma"].append(self.noise_sigma.numpy())

    def plot_diagnostics(self, show_plot: bool = False) -> None:
        """Plot VI parameter convergence over training epochs."""
        from financial_forecast.training.diagnostics import plot_vi_diagnostics

        # Remap to the format expected by the existing plot function
        vi_history = {
            "epochs": self._training_history["epochs"],
            "loss_vi": self._training_history["loss"],
            "q_var_opex_loc": self._training_history["q_var_opex_loc"],
            "q_var_opex_scale": self._training_history["q_var_opex_scale"],
            "q_base_opex_loc": self._training_history["q_base_opex_loc"],
            "q_base_opex_scale": self._training_history["q_base_opex_scale"],
            "noise_sigma": self._training_history["noise_sigma"],
        }
        plot_vi_diagnostics(vi_history, self.amount_scale, show_plot)

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

    def plot_fit(
        self,
        n_samples: int = 2000,
        lower_q: float = 5.0,
        upper_q: float = 95.0,
        show_plot: bool = False,
        use_gaussian_ci: bool = False,
    ) -> None:
        """Plot historical OpEx against the model fit with predictive uncertainty.

        Uses historical data stored by :meth:`set_forecast_drivers`.
        Generates two figures: OpEx vs Year and OpEx vs Sales, each showing
        the posterior predictive mean and confidence/prediction intervals.

        Args:
            n_samples: Number of posterior predictive samples when
                *use_gaussian_ci* is ``False``.
            lower_q: Lower quantile for the prediction band (percent).
            upper_q: Upper quantile for the prediction band (percent).
            show_plot: Whether to call ``plt.show()`` after saving.
            use_gaussian_ci: If ``True``, compute analytical Gaussian intervals
                instead of Monte Carlo samples.
        """
        historical_sales_scaled = self._historical_sales_scaled
        historical_opex_scaled = self._historical_opex_scaled
        historical_inflation = self._historical_inflation
        historical_years = tf.cast(
            tf.range(1, len(historical_sales_scaled) + 1),
            dtype=tf.float64,
        )
        cum_inf = tf.math.cumprod(1 + historical_inflation)

        mean_var_opex = self.q_var_opex_loc.numpy()
        mean_base_opex = self.q_base_opex_loc.numpy()
        sigma_opex = self.noise_sigma.numpy()
        sales_offset = self.sales_offset.numpy()
        amount_scale = self.amount_scale

        # Center sales using the offset from training
        sales_centered = historical_sales_scaled - sales_offset
        mean_opex_bil = (mean_base_opex * cum_inf) + (mean_var_opex * sales_centered)

        if use_gaussian_ci:
            var_var = float(self.q_var_opex_scale.numpy()) ** 2
            var_base = float(self.q_base_opex_scale.numpy()) ** 2
            var_noise = float(sigma_opex) ** 2
            cum_inf_tf = tf.cast(cum_inf, dtype=tf.float64)
            sales_tf = tf.cast(sales_centered, dtype=tf.float64)
            std_opex_bil = tf.sqrt(
                (cum_inf_tf**2) * var_base + (sales_tf**2) * var_var + var_noise
            )
            z_low = float(tfd.Normal(0.0, 1.0).quantile(lower_q / 100.0))
            z_up = float(tfd.Normal(0.0, 1.0).quantile(upper_q / 100.0))
            lower_opex_bil = mean_opex_bil + z_low * std_opex_bil
            upper_opex_bil = mean_opex_bil + z_up * std_opex_bil
        else:
            q_var = tfd.Normal(loc=self.q_var_opex_loc, scale=self.q_var_opex_scale)
            q_base = tfd.Normal(loc=self.q_base_opex_loc, scale=self.q_base_opex_scale)
            var_samples = tf.reshape(q_var.sample(n_samples), (-1, 1))
            base_samples = tf.reshape(q_base.sample(n_samples), (-1, 1))
            sales = tf.reshape(
                tf.convert_to_tensor(sales_centered, dtype=tf.float64), (1, -1)
            )
            cum_inf_t = tf.reshape(
                tf.convert_to_tensor(cum_inf, dtype=tf.float64), (1, -1)
            )
            noise = tf.random.normal(
                shape=(n_samples, len(historical_sales_scaled)),
                mean=0.0,
                stddev=sigma_opex,
                dtype=tf.float64,
            )
            opex_samples = (base_samples * cum_inf_t) + (var_samples * sales) + noise
            lower_opex_bil = tfp.stats.percentile(opex_samples, lower_q, axis=0)
            upper_opex_bil = tfp.stats.percentile(opex_samples, upper_q, axis=0)

        mean_opex_usd = mean_opex_bil * amount_scale
        upper_opex_usd = upper_opex_bil * amount_scale
        lower_opex_usd = lower_opex_bil * amount_scale
        opex_hist_usd = historical_opex_scaled * amount_scale
        sales_hist_usd = historical_sales_scaled * amount_scale

        # --- Figure 1: OpEx vs Year ---
        plt.figure(figsize=(10, 5))
        plt.plot(
            historical_years,
            opex_hist_usd,
            "o-",
            label="Historical OpEx",
            color="black",
        )
        plt.plot(
            historical_years,
            mean_opex_usd,
            "o-",
            label="Mean OpEx (learned)",
            color="tab:blue",
        )
        plt.plot(
            historical_years,
            lower_opex_usd,
            "--",
            label=f"Posterior predictive {lower_q:.0f}%",
            color="tab:blue",
            alpha=0.8,
        )
        plt.plot(
            historical_years,
            upper_opex_usd,
            "--",
            label=f"Posterior predictive {upper_q:.0f}%",
            color="tab:blue",
            alpha=0.8,
        )
        plt.title("OpEx vs Year with Learned Probabilistic Linear Regression")
        plt.xlabel("Year")
        plt.ylabel("OpEx (USD)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "gaussian_ci" if use_gaussian_ci else "monte_carlo"
        plt.savefig(
            get_training_results_path(f"opex_probabilistic_fit_{timestamp}_{tag}.png"),
            dpi=150,
        )
        if show_plot:
            plt.show()
        else:
            plt.close()

        # --- Figure 2: OpEx vs Sales ---
        x_min = float(tf.reduce_min(sales_hist_usd))
        x_max = float(tf.reduce_max(sales_hist_usd))
        x_span = x_max - x_min if x_max > x_min else max(abs(x_max), 1.0)
        x_pad = 0.5 * x_span
        x_left, x_right = x_min - x_pad, x_max + x_pad

        sales_grid_usd = tf.linspace(
            tf.constant(x_left, dtype=tf.float64),
            tf.constant(x_right, dtype=tf.float64),
            200,
        )
        sales_grid_bil = sales_grid_usd / amount_scale
        sales_grid_centered = sales_grid_bil - sales_offset
        cum_inf_mean = float(tf.reduce_mean(cum_inf))
        mean_opex_grid_bil = (mean_base_opex * cum_inf_mean) + (
            mean_var_opex * sales_grid_centered
        )
        mean_opex_grid_usd = mean_opex_grid_bil * amount_scale

        if use_gaussian_ci:
            var_var = float(self.q_var_opex_scale.numpy()) ** 2
            var_base = float(self.q_base_opex_scale.numpy()) ** 2
            var_noise = float(sigma_opex) ** 2
            std_grid = tf.sqrt(
                (cum_inf_mean**2) * var_base
                + (sales_grid_centered**2) * var_var
                + var_noise
            )
            z_low = float(tfd.Normal(0.0, 1.0).quantile(lower_q / 100.0))
            z_up = float(tfd.Normal(0.0, 1.0).quantile(upper_q / 100.0))
            lower_grid_bil = mean_opex_grid_bil + z_low * std_grid
            upper_grid_bil = mean_opex_grid_bil + z_up * std_grid
        else:
            q_var = tfd.Normal(loc=self.q_var_opex_loc, scale=self.q_var_opex_scale)
            q_base = tfd.Normal(loc=self.q_base_opex_loc, scale=self.q_base_opex_scale)
            var_samples = tf.reshape(q_var.sample(n_samples), (-1, 1))
            base_samples = tf.reshape(q_base.sample(n_samples), (-1, 1))
            sales_grid_t = tf.reshape(
                tf.convert_to_tensor(sales_grid_centered, dtype=tf.float64), (1, -1)
            )
            cum_inf_grid_t = tf.reshape(
                tf.fill(
                    sales_grid_bil.shape, tf.constant(cum_inf_mean, dtype=tf.float64)
                ),
                (1, -1),
            )
            noise_grid = tf.random.normal(
                shape=(n_samples, len(sales_grid_bil)),
                mean=0.0,
                stddev=sigma_opex,
                dtype=tf.float64,
            )
            opex_grid_samples = (
                (base_samples * cum_inf_grid_t)
                + (var_samples * sales_grid_t)
                + noise_grid
            )
            lower_grid_bil = tfp.stats.percentile(opex_grid_samples, lower_q, axis=0)
            upper_grid_bil = tfp.stats.percentile(opex_grid_samples, upper_q, axis=0)

        lower_grid_usd = lower_grid_bil * amount_scale
        upper_grid_usd = upper_grid_bil * amount_scale

        plt.figure(figsize=(10, 5))
        plt.scatter(
            sales_hist_usd,
            opex_hist_usd,
            label="Historical OpEx",
            color="black",
            zorder=3,
        )
        plt.scatter(
            sales_hist_usd,
            mean_opex_usd,
            label="Mean OpEx per data point (learned)",
            color="tab:blue",
            marker="x",
            s=80,
            zorder=4,
        )
        plt.plot(
            sales_grid_usd,
            mean_opex_grid_usd,
            "-",
            label="Mean OpEx trend (avg. inflation)",
            color="tab:blue",
            alpha=0.5,
        )
        plt.plot(
            sales_grid_usd,
            lower_grid_usd,
            "--",
            label=f"Posterior predictive {lower_q:.0f}%",
            color="tab:blue",
            alpha=0.8,
        )
        plt.plot(
            sales_grid_usd,
            upper_grid_usd,
            "--",
            label=f"Posterior predictive {upper_q:.0f}%",
            color="tab:blue",
            alpha=0.8,
        )
        plt.xlim(x_left, x_right)
        plt.title("OpEx vs Sales with Learned Probabilistic Linear Regression")
        plt.xlabel("Sales (USD)")
        plt.ylabel("OpEx (USD)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "gaussian_ci" if use_gaussian_ci else "monte_carlo"
        plt.savefig(
            get_training_results_path(f"opex_vs_sales_fit_{timestamp}_{tag}.png"),
            dpi=150,
        )
        if show_plot:
            plt.show()
        else:
            plt.close()
