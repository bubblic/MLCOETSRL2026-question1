"""Credit-spread regression model and loan-pricing public API.

The :class:`TFSpreadRegressor` composes a ``tf.keras.Model`` for
spread prediction and injects PD estimates from a fitted
:class:`~loan_pricing.models.pd_model.BaseClassifier`.

The :class:`LoanPricer` is the top-level public API that combines the
PD model, spread model, quantile models, and feature engineer into a
single ``price()`` call.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tensorflow as tf

from loan_pricing.logging_config import get_logger
from loan_pricing.models.pd_model import BaseClassifier

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpreadEvaluationResult:
    """Evaluation metrics for a credit-spread regression model.

    Attributes:
        rmse_bps: Root mean squared error in basis points.
        mae_bps: Mean absolute error in basis points.
        r_squared: Coefficient of determination.
        mape: Mean absolute percentage error.
        median_absolute_error_bps: Median absolute error in bps.
    """

    rmse_bps: float
    mae_bps: float
    r_squared: float
    mape: float
    median_absolute_error_bps: float


@dataclass(frozen=True)
class LoanPricingInput:
    """Input to :meth:`LoanPricer.price`.

    Attributes:
        financial_ratios: Borrower financial ratios keyed by name.
        loan_maturity_years: Loan term in years.
        loan_size_mm: Loan notional in millions of USD.
        is_secured: Whether the loan is collateralised.
        industry_naics: NAICS industry code of the borrower.
        treasury_yield_pct: Risk-free rate at the loan maturity.
    """

    financial_ratios: dict[str, float]
    loan_maturity_years: float
    loan_size_mm: float
    is_secured: bool
    industry_naics: str
    treasury_yield_pct: float


@dataclass(frozen=True)
class LoanPricingOutput:
    """Result of :meth:`LoanPricer.price`.

    Attributes:
        estimated_pd: Estimated probability of default.
        credit_spread_bps: Point estimate of the credit spread (bps).
        all_in_rate_pct: ``treasury_yield_pct + spread / 100``.
        spread_ci_lower_bps: Lower bound of the confidence interval.
        spread_ci_upper_bps: Upper bound of the confidence interval.
        internal_rating: Mapped internal credit rating string.
    """

    estimated_pd: float
    credit_spread_bps: float
    all_in_rate_pct: float
    spread_ci_lower_bps: float
    spread_ci_upper_bps: float
    internal_rating: str


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpreadModelConfig:
    """Hyperparameters for :class:`TFSpreadRegressor`.

    Attributes:
        hidden_sizes: Widths of the hidden dense layers.
        dropout_rates: Dropout rate after each hidden layer.
        learning_rate: Adam optimiser learning rate.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        early_stopping_patience: Epochs without val-loss improvement
            before stopping.
        random_seed: Seed for weight initialisation and shuffling.
    """

    hidden_sizes: tuple[int, ...] = (128, 64)
    dropout_rates: tuple[float, ...] = (0.3, 0.0)
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseRegressor(abc.ABC):
    """Interface that every spread regressor must satisfy."""

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseRegressor:
        """Train the regressor.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)`` (spread in bps).

        Returns:
            ``self`` for method chaining.
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted credit spreads.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of spread predictions in basis points.
        """


# ---------------------------------------------------------------------------
# Keras network builder
# ---------------------------------------------------------------------------


def _build_regression_network(
    n_features: int,
    config: SpreadModelConfig,
) -> tf.keras.Model:
    """Construct a dense regression network.

    Args:
        n_features: Number of input features.
        config: Model hyperparameters.

    Returns:
        An uncompiled ``tf.keras.Model``.
    """
    inputs = tf.keras.Input(shape=(n_features,), dtype=tf.float64)
    x = inputs
    for size, drop in zip(config.hidden_sizes, config.dropout_rates):
        x = tf.keras.layers.Dense(
            size, activation="relu", dtype=tf.float64,
        )(x)
        if drop > 0:
            x = tf.keras.layers.Dropout(drop, dtype=tf.float64)(x)
    outputs = tf.keras.layers.Dense(1, dtype=tf.float64)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _train_regression_epoch(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    X: tf.Tensor,
    y: tf.Tensor,
    batch_size: int,
) -> float:
    """Run one training epoch with MSE loss and return the mean loss.

    Args:
        model: Keras regression network.
        optimizer: TF optimiser instance.
        X: Training features as a float64 tensor.
        y: Training targets as a float64 tensor.
        batch_size: Mini-batch size.

    Returns:
        Scalar mean MSE loss for the epoch.
    """
    loss_fn = tf.keras.losses.MeanSquaredError()
    n = X.shape[0]
    indices = tf.random.shuffle(tf.range(n))
    epoch_loss = tf.constant(0.0, dtype=tf.float64)
    n_batches = 0

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        x_batch = tf.gather(X, batch_idx)
        y_batch = tf.gather(y, batch_idx)

        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)
            loss = loss_fn(y_batch, preds)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += tf.cast(loss, tf.float64)
        n_batches += 1

    return float(epoch_loss / n_batches)


def _evaluate_regression_loss(
    model: tf.keras.Model,
    X: tf.Tensor,
    y: tf.Tensor,
) -> float:
    """Compute MSE loss without gradient tracking.

    Args:
        model: Keras regression network.
        X: Feature tensor.
        y: Target tensor.

    Returns:
        Scalar loss value.
    """
    loss_fn = tf.keras.losses.MeanSquaredError()
    preds = model(X, training=False)
    return float(loss_fn(y, preds))


# ---------------------------------------------------------------------------
# Concrete regressor
# ---------------------------------------------------------------------------


class TFSpreadRegressor(BaseRegressor):
    """TensorFlow-based credit-spread regressor.

    Composes a ``tf.keras.Model`` internally and prepends PD estimates
    from an injected :class:`BaseClassifier` as an additional feature.

    Args:
        pd_model: A fitted PD classifier whose
            :meth:`~BaseClassifier.predict_proba` output is appended
            to the feature matrix.
        config: Model hyperparameters.
    """

    def __init__(
        self,
        pd_model: BaseClassifier,
        config: SpreadModelConfig | None = None,
    ) -> None:
        self._pd_model = pd_model
        self._config = config or SpreadModelConfig()
        self._model: tf.keras.Model | None = None
        self._n_features: int | None = None

    def _augment_with_pd(self, X: np.ndarray) -> np.ndarray:
        """Append PD predictions as the last column of *X*.

        Args:
            X: Original feature matrix.

        Returns:
            Augmented matrix with one extra column.
        """
        pd_probs = self._pd_model.predict_proba(X).reshape(-1, 1)
        return np.hstack([X, pd_probs])

    # -- public API ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> TFSpreadRegressor:
        """Train the spread model with PD-augmented features.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.
            y: Target spread in bps ``(n_samples,)``.

        Returns:
            ``self``.
        """
        cfg = self._config
        tf.random.set_seed(cfg.random_seed)

        X_aug = self._augment_with_pd(X)
        self._n_features = X_aug.shape[1]

        self._model = _build_regression_network(self._n_features, cfg)
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

        # Simple train/val split (last 15%) for early stopping.
        n = len(X_aug)
        split = int(n * 0.85)
        X_train = tf.constant(X_aug[:split], dtype=tf.float64)
        y_train = tf.constant(y[:split].reshape(-1, 1), dtype=tf.float64)
        X_val = tf.constant(X_aug[split:], dtype=tf.float64)
        y_val = tf.constant(y[split:].reshape(-1, 1), dtype=tf.float64)

        best_val_loss = float("inf")
        best_weights: list[np.ndarray] | None = None
        patience_counter = 0

        for epoch in range(cfg.epochs):
            _train_regression_epoch(
                self._model, optimizer, X_train, y_train, cfg.batch_size,
            )
            val_loss = _evaluate_regression_loss(self._model, X_val, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = list(self._model.get_weights())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (val_loss=%.4f)",
                    epoch + 1,
                    best_val_loss,
                )
                break

        if best_weights is not None:
            self._model.set_weights(best_weights)

        logger.info("Spread model training complete — best val loss %.4f", best_val_loss)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted credit spreads (bps).

        Args:
            X: Feature matrix ``(n_samples, n_features)``.

        Returns:
            1-D array of spread predictions.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._model is None:
            msg = "Call fit() before predict()"
            raise RuntimeError(msg)
        X_aug = self._augment_with_pd(X)
        X_t = tf.constant(X_aug, dtype=tf.float64)
        preds = self._model(X_t, training=False)
        return preds.numpy().ravel()

    # -- serialisation ------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise model weights and config to an ``.npz`` file.

        Args:
            path: Destination file.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if self._model is None:
            msg = "Cannot save an unfitted model"
            raise RuntimeError(msg)

        params: dict[str, object] = {"n_features": self._n_features}
        for i, w in enumerate(self._model.get_weights()):
            params[f"weight_{i}"] = w

        params["hidden_sizes"] = np.array(self._config.hidden_sizes)
        params["dropout_rates"] = np.array(self._config.dropout_rates)
        params["learning_rate"] = self._config.learning_rate

        tmp = path.parent / (path.stem + "_tmp.npz")
        np.savez(str(tmp), **params)
        if not tmp.exists() and tmp.with_suffix(".npz").exists():
            tmp = tmp.with_suffix(".npz")
        tmp.replace(path)
        logger.info("Spread model saved to %s", path)

    @classmethod
    def load(
        cls,
        path: Path,
        pd_model: BaseClassifier,
    ) -> TFSpreadRegressor:
        """Load a saved model from an ``.npz`` file.

        Args:
            path: Source file written by :meth:`save`.
            pd_model: A fitted PD classifier (needed for inference).

        Returns:
            A fitted :class:`TFSpreadRegressor`.
        """
        data = np.load(str(path), allow_pickle=True)
        n_features = int(data["n_features"])
        hidden_sizes = tuple(int(x) for x in data["hidden_sizes"])
        dropout_rates = tuple(float(x) for x in data["dropout_rates"])
        learning_rate = float(data["learning_rate"])

        config = SpreadModelConfig(
            hidden_sizes=hidden_sizes,
            dropout_rates=dropout_rates,
            learning_rate=learning_rate,
        )

        instance = cls(pd_model=pd_model, config=config)
        instance._n_features = n_features
        instance._model = _build_regression_network(n_features, config)

        dummy = tf.zeros((1, n_features), dtype=tf.float64)
        instance._model(dummy, training=False)

        weights = []
        i = 0
        while f"weight_{i}" in data:
            weights.append(data[f"weight_{i}"])
            i += 1
        instance._model.set_weights(weights)

        logger.info("Spread model loaded from %s", path)
        return instance


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_spread_model(
    model: BaseRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> SpreadEvaluationResult:
    """Compute regression metrics for a spread model.

    Args:
        model: A fitted regressor satisfying :class:`BaseRegressor`.
        X_test: Test features.
        y_test: Test target spreads in bps.

    Returns:
        A :class:`SpreadEvaluationResult`.
    """
    y_pred = model.predict(X_test)

    residuals = y_test - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    median_ae = float(np.median(np.abs(residuals)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Avoid division by zero in MAPE.
    nonzero_mask = np.abs(y_test) > 1e-8
    if nonzero_mask.any():
        mape = float(np.mean(np.abs(residuals[nonzero_mask] / y_test[nonzero_mask])))
    else:
        mape = float("inf")

    return SpreadEvaluationResult(
        rmse_bps=rmse,
        mae_bps=mae,
        r_squared=r2,
        mape=mape,
        median_absolute_error_bps=median_ae,
    )


# ---------------------------------------------------------------------------
# Internal rating mapper
# ---------------------------------------------------------------------------

_PD_RATING_THRESHOLDS: list[tuple[float, str]] = [
    (0.001, "AAA"),
    (0.005, "AA"),
    (0.01, "A"),
    (0.02, "BBB"),
    (0.05, "BB"),
    (0.10, "B"),
    (0.20, "CCC"),
]


def _pd_to_rating(pd_value: float) -> str:
    """Map a probability of default to an internal rating string.

    Args:
        pd_value: Estimated PD in ``[0, 1]``.

    Returns:
        Rating string from ``AAA`` to ``D``.
    """
    for threshold, rating in _PD_RATING_THRESHOLDS:
        if pd_value <= threshold:
            return rating
    return "D"


# ---------------------------------------------------------------------------
# Public API — LoanPricer
# ---------------------------------------------------------------------------

# Forward reference for quantile models to avoid circular imports.
# The actual type is loan_pricing.models.quantile_model.ConformalPredictionInterval
# but we accept Any-like duck typing here.


class LoanPricer:
    """Top-level API that combines all model stages into one ``price()`` call.

    When a fitted :class:`~loan_pricing.features.engineering.FeatureEngineer`
    is provided, ``price()`` routes the raw loan input through the same
    normalisation pipeline that was used during training.  This ensures
    that inference-time feature values are on the same scale as the
    training data, preventing scale-dependent prediction errors.

    Args:
        pd_model: Fitted PD classifier.
        spread_model: Fitted spread regressor.
        quantile_models: A pair ``(lower, upper)`` of fitted quantile
            regressors, or ``None`` if confidence intervals are not
            needed.
        feature_engineer: Fitted feature engineer.  When provided,
            ``price()`` normalises the input through it.  When
            ``None``, raw feature values are zero-padded to the
            model's expected width (useful only for testing).
    """

    def __init__(
        self,
        pd_model: BaseClassifier,
        spread_model: BaseRegressor,
        quantile_models: tuple[BaseRegressor, BaseRegressor] | None = None,
        feature_engineer: object | None = None,
    ) -> None:
        self._pd_model = pd_model
        self._spread_model = spread_model
        self._quantile_lower = quantile_models[0] if quantile_models else None
        self._quantile_upper = quantile_models[1] if quantile_models else None
        self._feature_engineer = feature_engineer

    def _build_feature_vector(self, loan: LoanPricingInput) -> np.ndarray:
        """Convert a :class:`LoanPricingInput` into a normalised feature row.

        When a feature engineer is available the input is routed through
        its ``transform_features`` method, which applies z-score
        normalisation using the statistics learned during training.
        Otherwise a raw zero-padded vector is returned (test-only path).

        Args:
            loan: Loan and borrower characteristics.

        Returns:
            Feature matrix of shape ``(1, n_features)``.
        """
        import pandas as pd

        # Assemble a single-row DataFrame with the same column names
        # the FeatureEngineer expects.
        row: dict[str, float] = {**loan.financial_ratios}
        row["maturity_years"] = loan.loan_maturity_years
        row["loan_size_mm"] = loan.loan_size_mm
        row["is_secured"] = float(loan.is_secured)
        row["treasury_yield"] = loan.treasury_yield_pct

        if self._feature_engineer is not None:
            df = pd.DataFrame([row])
            # Use the inference-time method that normalises without
            # requiring a target column.
            X = self._feature_engineer.transform_features(df)  # type: ignore[union-attr]
            return X

        # Fallback: raw values, zero-padded to model width (test path).
        raw = np.array([[row.get(k, 0.0) for k in sorted(row)]], dtype=np.float64)
        pd_n = self._pd_model._n_features  # type: ignore[attr-defined]
        if raw.shape[1] < pd_n:
            pad = np.zeros((1, pd_n - raw.shape[1]), dtype=np.float64)
            return np.hstack([raw, pad])
        return raw[:, :pd_n]

    def price(self, loan: LoanPricingInput) -> LoanPricingOutput:
        """Price a single loan.

        When a feature engineer was provided at construction time, the
        raw loan characteristics are normalised through it before being
        fed to the models.  This guarantees that inference operates on
        the same scale as the training data.

        Args:
            loan: Loan and borrower characteristics.

        Returns:
            Full pricing output including spread, all-in rate, CI,
            and internal rating.
        """
        X = self._build_feature_vector(loan)

        pd_est = float(self._pd_model.predict_proba(X)[0])
        spread = float(self._spread_model.predict(X)[0])

        bps_to_pct = 100.0
        all_in = loan.treasury_yield_pct + spread / bps_to_pct

        # Confidence interval from quantile models.
        if self._quantile_lower is not None and self._quantile_upper is not None:
            ci_lower = float(self._quantile_lower.predict(X)[0])
            ci_upper = float(self._quantile_upper.predict(X)[0])
        else:
            # Fallback: symmetric ±20% band.
            fallback_half_width = 0.20
            ci_lower = spread * (1.0 - fallback_half_width)
            ci_upper = spread * (1.0 + fallback_half_width)

        rating = _pd_to_rating(pd_est)

        return LoanPricingOutput(
            estimated_pd=pd_est,
            credit_spread_bps=spread,
            all_in_rate_pct=all_in,
            spread_ci_lower_bps=ci_lower,
            spread_ci_upper_bps=ci_upper,
            internal_rating=rating,
        )
