"""Tests for quantile regression and conformal prediction intervals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loan_pricing.models.quantile_model import (
    ConformalPredictionInterval,
    QuantileModelConfig,
    TFQuantileRegressor,
    pinball_loss,
)


# ------------------------------------------------------------------
# Small config for fast tests
# ------------------------------------------------------------------

_Q_CFG = QuantileModelConfig(
    hidden_sizes=(32, 16),
    dropout_rates=(0.1, 0.0),
    learning_rate=5e-3,
    epochs=60,
    batch_size=32,
    early_stopping_patience=15,
    random_seed=42,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """300-row dataset with known linear relationship + noise."""
    rng = np.random.default_rng(99)
    n, d = 300, 5
    X = rng.normal(0, 1, (n, d))
    y = 200.0 + 40.0 * X[:, 0] - 20.0 * X[:, 1] + rng.normal(0, 30, n)
    return X, y


@pytest.fixture(scope="module")
def fitted_lower(
    synthetic_regression_data: tuple[np.ndarray, np.ndarray],
) -> TFQuantileRegressor:
    X, y = synthetic_regression_data
    model = TFQuantileRegressor(quantile_alpha=0.025, config=_Q_CFG)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def fitted_upper(
    synthetic_regression_data: tuple[np.ndarray, np.ndarray],
) -> TFQuantileRegressor:
    X, y = synthetic_regression_data
    model = TFQuantileRegressor(quantile_alpha=0.975, config=_Q_CFG)
    model.fit(X, y)
    return model


# ------------------------------------------------------------------
# Pinball loss
# ------------------------------------------------------------------


class TestPinballLoss:
    def test_zero_residual_gives_zero_loss(self) -> None:
        import tensorflow as tf

        y = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
        loss = pinball_loss(y, y, alpha=0.5)
        assert float(loss) == pytest.approx(0.0, abs=1e-10)

    def test_asymmetric_for_nonzero_alpha(self) -> None:
        import tensorflow as tf

        y_true = tf.constant([10.0], dtype=tf.float64)
        y_under = tf.constant([8.0], dtype=tf.float64)  # predict too low
        y_over = tf.constant([12.0], dtype=tf.float64)  # predict too high

        # For alpha=0.9, under-predicting is penalised more heavily.
        loss_under = float(pinball_loss(y_true, y_under, alpha=0.9))
        loss_over = float(pinball_loss(y_true, y_over, alpha=0.9))
        assert loss_under > loss_over


# ------------------------------------------------------------------
# TFQuantileRegressor
# ------------------------------------------------------------------


class TestTFQuantileRegressor:
    def test_output_shape(
        self,
        fitted_lower: TFQuantileRegressor,
        synthetic_regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, _ = synthetic_regression_data
        preds = fitted_lower.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_lower_below_upper(
        self,
        fitted_lower: TFQuantileRegressor,
        fitted_upper: TFQuantileRegressor,
        synthetic_regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, _ = synthetic_regression_data
        lo = fitted_lower.predict(X)
        hi = fitted_upper.predict(X)
        # Most predictions should have lower < upper.
        assert np.mean(lo < hi) > 0.8

    def test_predict_before_fit_raises(self) -> None:
        model = TFQuantileRegressor(quantile_alpha=0.5, config=_Q_CFG)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.zeros((5, 5)))

    def test_coverage_test_method(
        self,
        fitted_upper: TFQuantileRegressor,
        synthetic_regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = synthetic_regression_data
        coverage = fitted_upper.coverage_test(X, y)
        assert 0.0 <= coverage <= 1.0


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------


class TestQuantileSaveLoad:
    def test_round_trip(
        self,
        fitted_lower: TFQuantileRegressor,
        synthetic_regression_data: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        X, _ = synthetic_regression_data
        original = fitted_lower.predict(X)

        path = tmp_path / "quantile_lower.npz"
        fitted_lower.save(path)

        loaded = TFQuantileRegressor.load(path)
        restored = loaded.predict(X)
        np.testing.assert_array_almost_equal(original, restored)

    def test_alpha_preserved(
        self,
        fitted_lower: TFQuantileRegressor,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "quantile_lower.npz"
        fitted_lower.save(path)
        loaded = TFQuantileRegressor.load(path)
        assert loaded.quantile_alpha == pytest.approx(0.025)


# ------------------------------------------------------------------
# ConformalPredictionInterval
# ------------------------------------------------------------------


class TestConformalPredictionInterval:
    def test_calibration_sets_q_hat(
        self,
        fitted_lower: TFQuantileRegressor,
        fitted_upper: TFQuantileRegressor,
        synthetic_regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = synthetic_regression_data
        cpi = ConformalPredictionInterval(fitted_lower, fitted_upper, coverage_target=0.95)
        assert not cpi.is_calibrated
        cpi.calibrate(X, y)
        assert cpi.is_calibrated
        assert isinstance(cpi.q_hat, float)

    def test_predict_interval_before_calibrate_raises(
        self,
        fitted_lower: TFQuantileRegressor,
        fitted_upper: TFQuantileRegressor,
    ) -> None:
        cpi = ConformalPredictionInterval(fitted_lower, fitted_upper)
        with pytest.raises(RuntimeError, match="calibrate"):
            cpi.predict_interval(np.zeros((5, 5)))

    def test_empirical_coverage_above_threshold(
        self,
        fitted_lower: TFQuantileRegressor,
        fitted_upper: TFQuantileRegressor,
        synthetic_regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Conformal intervals must achieve >= 0.93 coverage on held-out data."""
        X, y = synthetic_regression_data
        rng = np.random.default_rng(7)
        idx = rng.permutation(len(X))
        cal_idx, test_idx = idx[:200], idx[200:]

        cpi = ConformalPredictionInterval(
            fitted_lower, fitted_upper, coverage_target=0.95,
        )
        cpi.calibrate(X[cal_idx], y[cal_idx])
        coverage = cpi.empirical_coverage(X[test_idx], y[test_idx])
        assert coverage >= 0.93

    def test_interval_lower_leq_upper(
        self,
        fitted_lower: TFQuantileRegressor,
        fitted_upper: TFQuantileRegressor,
        synthetic_regression_data: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = synthetic_regression_data
        cpi = ConformalPredictionInterval(fitted_lower, fitted_upper)
        cpi.calibrate(X, y)
        lower, upper = cpi.predict_interval(X)
        assert np.all(lower <= upper)
