"""Ornstein-Uhlenbeck MLE calibration and Monte Carlo loan pricing.

The :class:`OrnsteinUhlenbeckCalibrator` fits an OU process to an
observed credit-spread series using maximum-likelihood estimation
implemented with ``tf.GradientTape`` and constrained parameters via
``tfp.util.TransformedVariable`` (matching the pattern in
``financial_forecast/models/capex.py``).

The :class:`MonteCarloLoanPricer` simulates joint OU (spread) and
Vasicek (rate) paths using fully vectorised TensorFlow operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from loan_pricing.exceptions import InsufficientDataError
from loan_pricing.logging_config import get_logger

logger = get_logger(__name__)

tfb = tfp.bijectors
tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OUParameters:
    """Calibrated Ornstein-Uhlenbeck parameters.

    The discrete-time OU model is:
        X_{t+1} = X_t + kappa * (theta - X_t) * dt + sigma * sqrt(dt) * Z_t

    Attributes:
        kappa: Mean-reversion speed.
        theta: Long-run mean level.
        sigma: Volatility.
        log_likelihood: Maximised log-likelihood.
        se_kappa: Standard error of kappa.
        se_theta: Standard error of theta.
        se_sigma: Standard error of sigma.
    """

    kappa: float
    theta: float
    sigma: float
    log_likelihood: float
    se_kappa: float
    se_theta: float
    se_sigma: float


@dataclass(frozen=True)
class VasicekParameters:
    """Parameters for a Vasicek short-rate model.

    Attributes:
        kappa: Mean-reversion speed.
        theta: Long-run mean level.
        sigma: Volatility.
    """

    kappa: float
    theta: float
    sigma: float


@dataclass(frozen=True)
class MonteCarloResult:
    """Output of :meth:`MonteCarloLoanPricer.simulate`.

    Attributes:
        simulated_spreads: Array of shape ``(n_simulations,)`` with
            terminal spread values.
        mean_spread: Mean of simulated spreads.
        std_spread: Standard deviation of simulated spreads.
        ci_lower_95: 2.5th percentile.
        ci_upper_95: 97.5th percentile.
        ci_lower_80: 10th percentile.
        ci_upper_80: 90th percentile.
    """

    simulated_spreads: np.ndarray
    mean_spread: float
    std_spread: float
    ci_lower_95: float
    ci_upper_95: float
    ci_lower_80: float
    ci_upper_80: float


# ---------------------------------------------------------------------------
# OU Calibrator
# ---------------------------------------------------------------------------

_DEFAULT_MLE_LR = 0.01
_DEFAULT_MLE_STEPS = 2000
_CONVERGENCE_TOLERANCE = 1e-8


class OrnsteinUhlenbeckCalibrator:
    """Fit an OU process via TensorFlow-based MLE.

    Parameters are stored as ``tfp.util.TransformedVariable`` with
    ``Softplus`` bijectors to enforce positivity on kappa and sigma,
    following the pattern in ``financial_forecast/models/capex.py``.

    Args:
        min_series_length: Minimum number of observations required.
        learning_rate: Adam learning rate for the MLE optimisation.
        max_steps: Maximum optimisation steps.
        dt: Time step between observations (default 1.0 for daily).
    """

    def __init__(
        self,
        min_series_length: int = 60,
        learning_rate: float = _DEFAULT_MLE_LR,
        max_steps: int = _DEFAULT_MLE_STEPS,
        dt: float = 1.0,
    ) -> None:
        self._min_length = min_series_length
        self._lr = learning_rate
        self._max_steps = max_steps
        self._dt = dt

    def fit(self, spread_series: pd.Series | np.ndarray) -> OUParameters:
        """Calibrate OU parameters to an observed spread time series.

        Args:
            spread_series: Ordered time series of spread observations.

        Returns:
            Frozen :class:`OUParameters` dataclass.

        Raises:
            InsufficientDataError: If the series has fewer than
                ``min_series_length`` observations.
        """
        data = np.asarray(spread_series, dtype=np.float64)
        if len(data) < self._min_length:
            raise InsufficientDataError(
                f"OU calibration requires at least {self._min_length} "
                f"observations, got {len(data)}"
            )

        x_prev = tf.constant(data[:-1], dtype=tf.float64)
        x_next = tf.constant(data[1:], dtype=tf.float64)
        dt = tf.constant(self._dt, dtype=tf.float64)

        # Initialise from simple method-of-moments estimates.
        init_theta = float(np.mean(data))
        init_sigma = float(np.std(np.diff(data)))
        autocorr = float(np.corrcoef(data[:-1], data[1:])[0, 1])
        init_kappa = max(-np.log(max(autocorr, 0.01)) / self._dt, 0.01)

        # Use raw tf.Variable in unconstrained space.  Apply softplus
        # manually to enforce positivity on kappa and sigma.
        softplus = tfb.Softplus()
        raw_kappa = tf.Variable(
            softplus.inverse(tf.constant(init_kappa, dtype=tf.float64)),
            dtype=tf.float64, name="raw_kappa",
        )
        theta_var = tf.Variable(init_theta, dtype=tf.float64, name="theta")
        raw_sigma = tf.Variable(
            softplus.inverse(tf.constant(init_sigma, dtype=tf.float64)),
            dtype=tf.float64, name="raw_sigma",
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self._lr)
        trainable = [raw_kappa, theta_var, raw_sigma]

        prev_loss = float("inf")
        for step in range(self._max_steps):
            with tf.GradientTape() as tape:
                kappa = softplus.forward(raw_kappa)
                theta = theta_var
                sigma = softplus.forward(raw_sigma)

                # Discrete-time conditional distribution:
                # X_{t+1} | X_t ~ N(mu_t, var_t)
                exp_neg_k_dt = tf.exp(-kappa * dt)
                mu = theta + (x_prev - theta) * exp_neg_k_dt
                var = (sigma ** 2) / (2.0 * kappa) * (1.0 - exp_neg_k_dt ** 2)
                var = tf.maximum(var, 1e-12)

                dist = tfd.Normal(loc=mu, scale=tf.sqrt(var))
                neg_log_lik = -tf.reduce_sum(dist.log_prob(x_next))

            grads = tape.gradient(neg_log_lik, trainable)
            optimizer.apply_gradients(zip(grads, trainable))

            loss = float(neg_log_lik)
            if abs(prev_loss - loss) < _CONVERGENCE_TOLERANCE:
                logger.info("OU MLE converged at step %d", step + 1)
                break
            prev_loss = loss

        # Extract fitted values (apply softplus to get constrained).
        kappa_fit = float(softplus.forward(raw_kappa))
        theta_fit = float(theta_var)
        sigma_fit = float(softplus.forward(raw_sigma))
        log_lik = -float(neg_log_lik)

        # Standard errors via numerical Hessian (finite differences).
        se_kappa, se_theta, se_sigma = self._compute_standard_errors(
            x_prev, x_next, dt, kappa_fit, theta_fit, sigma_fit,
        )

        logger.info(
            "OU MLE: kappa=%.4f theta=%.4f sigma=%.4f loglik=%.2f",
            kappa_fit, theta_fit, sigma_fit, log_lik,
        )

        return OUParameters(
            kappa=kappa_fit,
            theta=theta_fit,
            sigma=sigma_fit,
            log_likelihood=log_lik,
            se_kappa=se_kappa,
            se_theta=se_theta,
            se_sigma=se_sigma,
        )

    @staticmethod
    def _compute_standard_errors(
        x_prev: tf.Tensor,
        x_next: tf.Tensor,
        dt: tf.Tensor,
        kappa: float,
        theta: float,
        sigma: float,
    ) -> tuple[float, float, float]:
        """Estimate standard errors via the observed information matrix.

        Uses second-order finite differences of the negative
        log-likelihood at the MLE.

        Args:
            x_prev: Lagged observations.
            x_next: Lead observations.
            dt: Time step.
            kappa: Fitted kappa.
            theta: Fitted theta.
            sigma: Fitted sigma.

        Returns:
            Tuple ``(se_kappa, se_theta, se_sigma)``.
        """
        params = np.array([kappa, theta, sigma])
        n_params = len(params)
        eps = 1e-5

        def neg_ll(p: np.ndarray) -> float:
            k, th, s = float(p[0]), float(p[1]), float(p[2])
            k = max(k, 1e-8)
            s = max(s, 1e-8)
            exp_neg = np.exp(-k * float(dt))
            mu = th + (x_prev.numpy() - th) * exp_neg
            var = (s ** 2) / (2.0 * k) * (1.0 - exp_neg ** 2)
            var = np.maximum(var, 1e-12)
            ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (x_next.numpy() - mu) ** 2 / var)
            return -ll

        # Numerical Hessian.
        hess = np.zeros((n_params, n_params))
        f0 = neg_ll(params)
        for i in range(n_params):
            for j in range(i, n_params):
                p_pp = params.copy()
                p_pp[i] += eps
                p_pp[j] += eps
                p_pm = params.copy()
                p_pm[i] += eps
                p_pm[j] -= eps
                p_mp = params.copy()
                p_mp[i] -= eps
                p_mp[j] += eps
                p_mm = params.copy()
                p_mm[i] -= eps
                p_mm[j] -= eps

                hess[i, j] = (neg_ll(p_pp) - neg_ll(p_pm) - neg_ll(p_mp) + neg_ll(p_mm)) / (4 * eps ** 2)
                hess[j, i] = hess[i, j]

        try:
            inv_hess = np.linalg.inv(hess)
            se = np.sqrt(np.maximum(np.diag(inv_hess), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(n_params, np.nan)

        return float(se[0]), float(se[1]), float(se[2])


# ---------------------------------------------------------------------------
# Monte Carlo Loan Pricer
# ---------------------------------------------------------------------------


class MonteCarloLoanPricer:
    """Simulate joint OU spread / Vasicek rate paths using vectorised TF ops.

    Args:
        ou_params: Calibrated spread-process parameters.
        vasicek_params: Calibrated short-rate parameters.
        n_simulations: Number of Monte Carlo paths.
        horizon_years: Simulation horizon in years.
        random_seed: Seed for reproducible draws.
        steps_per_year: Euler-Maruyama discretisation resolution.
    """

    def __init__(
        self,
        ou_params: OUParameters,
        vasicek_params: VasicekParameters,
        n_simulations: int = 10_000,
        horizon_years: float = 5.0,
        random_seed: int = 42,
        steps_per_year: int = 252,
    ) -> None:
        self._ou = ou_params
        self._vasicek = vasicek_params
        self._n_sim = n_simulations
        self._horizon = horizon_years
        self._seed = random_seed
        self._steps_per_year = steps_per_year

    def simulate(
        self,
        current_spread_bps: float,
        current_treasury_yield_pct: float,
    ) -> MonteCarloResult:
        """Run vectorised Monte Carlo simulation.

        Args:
            current_spread_bps: Starting credit spread (bps).
            current_treasury_yield_pct: Starting risk-free rate (%).

        Returns:
            A :class:`MonteCarloResult` with terminal spread statistics.
        """
        n_steps = int(self._horizon * self._steps_per_year)
        dt = tf.constant(1.0 / self._steps_per_year, dtype=tf.float64)
        sqrt_dt = tf.sqrt(dt)

        # Draw all random increments upfront: shape (n_steps, n_sim, 2)
        # Column 0 = spread noise, column 1 = rate noise.
        seed_tensor = tf.constant([self._seed, self._seed + 1], dtype=tf.int32)
        all_noise = tf.random.stateless_normal(
            shape=(n_steps, self._n_sim, 2),
            seed=seed_tensor,
            dtype=tf.float64,
        )

        # OU spread path.
        ou_k = tf.constant(self._ou.kappa, dtype=tf.float64)
        ou_th = tf.constant(self._ou.theta, dtype=tf.float64)
        ou_s = tf.constant(self._ou.sigma, dtype=tf.float64)

        # Vasicek rate path.
        v_k = tf.constant(self._vasicek.kappa, dtype=tf.float64)
        v_th = tf.constant(self._vasicek.theta, dtype=tf.float64)
        v_s = tf.constant(self._vasicek.sigma, dtype=tf.float64)

        # Initialise paths.
        spread = tf.fill([self._n_sim], tf.constant(current_spread_bps, dtype=tf.float64))
        rate = tf.fill([self._n_sim], tf.constant(current_treasury_yield_pct, dtype=tf.float64))

        # Euler-Maruyama loop — vectorised over simulations.
        for t in range(n_steps):
            dW_spread = all_noise[t, :, 0] * sqrt_dt
            dW_rate = all_noise[t, :, 1] * sqrt_dt

            spread = spread + ou_k * (ou_th - spread) * dt + ou_s * dW_spread
            rate = rate + v_k * (v_th - rate) * dt + v_s * dW_rate

        terminal_spreads = spread.numpy()

        percentile_2_5 = 2.5
        percentile_10 = 10.0
        percentile_90 = 90.0
        percentile_97_5 = 97.5

        return MonteCarloResult(
            simulated_spreads=terminal_spreads,
            mean_spread=float(np.mean(terminal_spreads)),
            std_spread=float(np.std(terminal_spreads)),
            ci_lower_95=float(np.percentile(terminal_spreads, percentile_2_5)),
            ci_upper_95=float(np.percentile(terminal_spreads, percentile_97_5)),
            ci_lower_80=float(np.percentile(terminal_spreads, percentile_10)),
            ci_upper_80=float(np.percentile(terminal_spreads, percentile_90)),
        )
