"""Tests for OU calibration and Monte Carlo loan pricing."""

from __future__ import annotations

import numpy as np
import pytest

from loan_pricing.exceptions import InsufficientDataError
from loan_pricing.models.ou_calibration import (
    MonteCarloLoanPricer,
    MonteCarloResult,
    OrnsteinUhlenbeckCalibrator,
    OUParameters,
    VasicekParameters,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _generate_ou_path(
    kappa: float,
    theta: float,
    sigma: float,
    n: int,
    dt: float = 1.0,
    x0: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate a discrete OU path with known parameters."""
    rng = np.random.default_rng(seed)
    x0 = x0 if x0 is not None else theta
    path = np.empty(n)
    path[0] = x0
    for t in range(1, n):
        exp_neg = np.exp(-kappa * dt)
        mu = theta + (path[t - 1] - theta) * exp_neg
        var = (sigma ** 2) / (2.0 * kappa) * (1.0 - exp_neg ** 2)
        path[t] = rng.normal(mu, np.sqrt(var))
    return path


# ------------------------------------------------------------------
# Known parameters for recovery test
# ------------------------------------------------------------------

TRUE_KAPPA = 0.5
TRUE_THETA = 200.0
TRUE_SIGMA = 15.0


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_ou_path() -> np.ndarray:
    """Long OU path (500 obs) for parameter recovery."""
    return _generate_ou_path(
        kappa=TRUE_KAPPA,
        theta=TRUE_THETA,
        sigma=TRUE_SIGMA,
        n=500,
        dt=1.0,
        seed=42,
    )


@pytest.fixture(scope="module")
def fitted_ou_params(synthetic_ou_path: np.ndarray) -> OUParameters:
    """OU parameters fitted on the synthetic path."""
    cal = OrnsteinUhlenbeckCalibrator(
        min_series_length=60,
        learning_rate=0.02,
        max_steps=3000,
        dt=1.0,
    )
    return cal.fit(synthetic_ou_path)


# ------------------------------------------------------------------
# OrnsteinUhlenbeckCalibrator
# ------------------------------------------------------------------


class TestOUCalibrator:
    def test_kappa_recovery(self, fitted_ou_params: OUParameters) -> None:
        """MLE recovers kappa within 2 standard errors."""
        p = fitted_ou_params
        assert abs(p.kappa - TRUE_KAPPA) < 2 * p.se_kappa, (
            f"kappa={p.kappa:.4f}, se={p.se_kappa:.4f}, true={TRUE_KAPPA}"
        )

    def test_theta_recovery(self, fitted_ou_params: OUParameters) -> None:
        """MLE recovers theta within 2 standard errors."""
        p = fitted_ou_params
        assert abs(p.theta - TRUE_THETA) < 2 * p.se_theta, (
            f"theta={p.theta:.4f}, se={p.se_theta:.4f}, true={TRUE_THETA}"
        )

    def test_sigma_recovery(self, fitted_ou_params: OUParameters) -> None:
        """MLE recovers sigma within 2 standard errors."""
        p = fitted_ou_params
        assert abs(p.sigma - TRUE_SIGMA) < 2 * p.se_sigma, (
            f"sigma={p.sigma:.4f}, se={p.se_sigma:.4f}, true={TRUE_SIGMA}"
        )

    def test_positive_kappa(self, fitted_ou_params: OUParameters) -> None:
        assert fitted_ou_params.kappa > 0

    def test_positive_sigma(self, fitted_ou_params: OUParameters) -> None:
        assert fitted_ou_params.sigma > 0

    def test_log_likelihood_finite(self, fitted_ou_params: OUParameters) -> None:
        assert np.isfinite(fitted_ou_params.log_likelihood)


class TestOUCalibratorErrors:
    def test_short_series_raises(self) -> None:
        cal = OrnsteinUhlenbeckCalibrator(min_series_length=60)
        short = np.random.default_rng(0).normal(200, 10, 30)
        with pytest.raises(InsufficientDataError, match="60"):
            cal.fit(short)

    def test_accepts_pandas_series(self, synthetic_ou_path: np.ndarray) -> None:
        import pandas as pd

        series = pd.Series(synthetic_ou_path)
        cal = OrnsteinUhlenbeckCalibrator(
            min_series_length=60,
            learning_rate=0.02,
            max_steps=500,
        )
        result = cal.fit(series)
        assert isinstance(result, OUParameters)


# ------------------------------------------------------------------
# MonteCarloLoanPricer
# ------------------------------------------------------------------


class TestMonteCarloLoanPricer:
    def test_result_shape(self, fitted_ou_params: OUParameters) -> None:
        n_sim = 1000
        mc = MonteCarloLoanPricer(
            ou_params=fitted_ou_params,
            vasicek_params=VasicekParameters(kappa=0.3, theta=3.5, sigma=0.5),
            n_simulations=n_sim,
            horizon_years=1.0,
            random_seed=42,
            steps_per_year=52,
        )
        result = mc.simulate(
            current_spread_bps=200.0,
            current_treasury_yield_pct=4.0,
        )
        assert isinstance(result, MonteCarloResult)
        assert result.simulated_spreads.shape == (n_sim,)

    def test_mean_near_theta(self, fitted_ou_params: OUParameters) -> None:
        """With long horizon the mean should revert toward theta."""
        mc = MonteCarloLoanPricer(
            ou_params=fitted_ou_params,
            vasicek_params=VasicekParameters(kappa=0.3, theta=3.5, sigma=0.5),
            n_simulations=5000,
            horizon_years=10.0,
            random_seed=42,
            steps_per_year=52,
        )
        result = mc.simulate(
            current_spread_bps=300.0,
            current_treasury_yield_pct=4.0,
        )
        # After 10 years with kappa≈0.5, spread should be close to theta.
        assert abs(result.mean_spread - fitted_ou_params.theta) < 30

    def test_ci_ordering(self, fitted_ou_params: OUParameters) -> None:
        mc = MonteCarloLoanPricer(
            ou_params=fitted_ou_params,
            vasicek_params=VasicekParameters(kappa=0.3, theta=3.5, sigma=0.5),
            n_simulations=2000,
            horizon_years=2.0,
            random_seed=42,
            steps_per_year=52,
        )
        result = mc.simulate(
            current_spread_bps=200.0,
            current_treasury_yield_pct=4.0,
        )
        assert result.ci_lower_95 <= result.ci_lower_80
        assert result.ci_lower_80 <= result.mean_spread
        assert result.mean_spread <= result.ci_upper_80
        assert result.ci_upper_80 <= result.ci_upper_95

    def test_reproducibility(self, fitted_ou_params: OUParameters) -> None:
        """Same seed produces identical results."""
        kwargs = dict(
            ou_params=fitted_ou_params,
            vasicek_params=VasicekParameters(kappa=0.3, theta=3.5, sigma=0.5),
            n_simulations=500,
            horizon_years=1.0,
            random_seed=99,
            steps_per_year=52,
        )
        r1 = MonteCarloLoanPricer(**kwargs).simulate(200.0, 4.0)
        r2 = MonteCarloLoanPricer(**kwargs).simulate(200.0, 4.0)
        np.testing.assert_array_equal(r1.simulated_spreads, r2.simulated_spreads)

    def test_std_positive(self, fitted_ou_params: OUParameters) -> None:
        mc = MonteCarloLoanPricer(
            ou_params=fitted_ou_params,
            vasicek_params=VasicekParameters(kappa=0.3, theta=3.5, sigma=0.5),
            n_simulations=1000,
            horizon_years=1.0,
            random_seed=42,
            steps_per_year=52,
        )
        result = mc.simulate(200.0, 4.0)
        assert result.std_spread > 0
