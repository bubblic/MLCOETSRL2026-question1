"""Quantile regression and split-conformal prediction intervals.

:class:`TFQuantileRegressor` trains a ``tf.keras.Model`` with pinball
loss at a specified quantile level.  :class:`ConformalPredictionInterval`
wraps a lower/upper pair and calibrates them on a held-out set to
provide guaranteed marginal coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from loan_pricing.logging_config import get_logger
from loan_pricing.models.spread_model import BaseRegressor

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantileModelConfig:
    """Hyperparameters for :class:`TFQuantileRegressor`.

    Attributes:
        hidden_sizes: Widths of the hidden dense layers.
        dropout_rates: Dropout rate after each hidden layer.
        learning_rate: Adam optimiser learning rate.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        early_stopping_patience: Epochs without val-loss improvement
            before stopping.
        random_seed: Seed for reproducibility.
    """

    hidden_sizes: tuple[int, ...] = (128, 64)
    dropout_rates: tuple[float, ...] = (0.3, 0.0)
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Pinball (quantile) loss
# ---------------------------------------------------------------------------


def pinball_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float,
) -> tf.Tensor:
    """Compute the pinball (quantile) loss.

    Args:
        y_true: Ground-truth targets.
        y_pred: Predicted values.
        alpha: Quantile level in ``(0, 1)``.

    Returns:
        Scalar mean pinball loss.
    """
    error = y_true - y_pred
    return tf.reduce_mean(
        tf.maximum(alpha * error, (alpha - 1.0) * error)
    )


# ---------------------------------------------------------------------------
# Network builder
# ---------------------------------------------------------------------------


def _build_quantile_network(
    n_features: int,
    config: QuantileModelConfig,
) -> tf.keras.Model:
    """Construct a dense network for quantile regression.

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


def _train_quantile_epoch(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    X: tf.Tensor,
    y: tf.Tensor,
    alpha: float,
    batch_size: int,
) -> float:
    """Run one training epoch with pinball loss.

    Args:
        model: Keras quantile network.
        optimizer: TF optimiser.
        X: Feature tensor.
        y: Target tensor.
        alpha: Quantile level.
        batch_size: Mini-batch size.

    Returns:
        Mean pinball loss for the epoch.
    """
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
            loss = pinball_loss(y_batch, preds, alpha)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += tf.cast(loss, tf.float64)
        n_batches += 1

    return float(epoch_loss / n_batches)


def _evaluate_quantile_loss(
    model: tf.keras.Model,
    X: tf.Tensor,
    y: tf.Tensor,
    alpha: float,
) -> float:
    """Compute pinball loss without gradient tracking.

    Args:
        model: Keras quantile network.
        X: Feature tensor.
        y: Target tensor.
        alpha: Quantile level.

    Returns:
        Scalar loss value.
    """
    preds = model(X, training=False)
    return float(pinball_loss(y, preds, alpha))


# ---------------------------------------------------------------------------
# Concrete quantile regressor
# ---------------------------------------------------------------------------


class TFQuantileRegressor(BaseRegressor):
    """TensorFlow quantile regressor trained with pinball loss.

    Args:
        quantile_alpha: Quantile level in ``(0, 1)``.
        config: Model hyperparameters.
    """

    def __init__(
        self,
        quantile_alpha: float,
        config: QuantileModelConfig | None = None,
    ) -> None:
        self._alpha = quantile_alpha
        self._config = config or QuantileModelConfig()
        self._model: tf.keras.Model | None = None
        self._n_features: int | None = None

    @property
    def quantile_alpha(self) -> float:
        """The quantile level this model targets."""
        return self._alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> TFQuantileRegressor:
        """Train the quantile regressor.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.
            y: Target vector ``(n_samples,)``.

        Returns:
            ``self``.
        """
        cfg = self._config
        tf.random.set_seed(cfg.random_seed)
        self._n_features = X.shape[1]

        self._model = _build_quantile_network(self._n_features, cfg)
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

        n = len(X)
        split = int(n * 0.85)
        X_train = tf.constant(X[:split], dtype=tf.float64)
        y_train = tf.constant(y[:split].reshape(-1, 1), dtype=tf.float64)
        X_val = tf.constant(X[split:], dtype=tf.float64)
        y_val = tf.constant(y[split:].reshape(-1, 1), dtype=tf.float64)

        best_val_loss = float("inf")
        best_weights: list[np.ndarray] | None = None
        patience_counter = 0

        for epoch in range(cfg.epochs):
            _train_quantile_epoch(
                self._model, optimizer, X_train, y_train,
                self._alpha, cfg.batch_size,
            )
            val_loss = _evaluate_quantile_loss(
                self._model, X_val, y_val, self._alpha,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = list(self._model.get_weights())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg.early_stopping_patience:
                logger.info(
                    "Quantile(%.2f) early stop epoch %d (val_loss=%.4f)",
                    self._alpha, epoch + 1, best_val_loss,
                )
                break

        if best_weights is not None:
            self._model.set_weights(best_weights)

        logger.info(
            "Quantile(%.2f) training complete — best val loss %.4f",
            self._alpha, best_val_loss,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return quantile predictions.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.

        Returns:
            1-D array of quantile predictions.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._model is None:
            msg = "Call fit() before predict()"
            raise RuntimeError(msg)
        X_t = tf.constant(X, dtype=tf.float64)
        return self._model(X_t, training=False).numpy().ravel()

    def coverage_test(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> float:
        """Compute empirical coverage on a calibration set.

        For a lower quantile (``alpha < 0.5``), coverage is the
        fraction of samples where ``y >= predicted``.  For an upper
        quantile (``alpha >= 0.5``), coverage is the fraction where
        ``y <= predicted``.

        Args:
            X_cal: Calibration features.
            y_cal: Calibration targets.

        Returns:
            Empirical coverage in ``[0, 1]``.
        """
        preds = self.predict(X_cal)
        if self._alpha < 0.5:
            covered = np.sum(y_cal >= preds)
        else:
            covered = np.sum(y_cal <= preds)
        return float(covered / len(y_cal))

    # -- serialisation ------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise to ``.npz``.

        Args:
            path: Destination file.
        """
        if self._model is None:
            msg = "Cannot save an unfitted model"
            raise RuntimeError(msg)

        params: dict[str, object] = {
            "n_features": self._n_features,
            "alpha": self._alpha,
            "hidden_sizes": np.array(self._config.hidden_sizes),
            "dropout_rates": np.array(self._config.dropout_rates),
            "learning_rate": self._config.learning_rate,
        }
        for i, w in enumerate(self._model.get_weights()):
            params[f"weight_{i}"] = w

        tmp = path.parent / (path.stem + "_tmp.npz")
        np.savez(str(tmp), **params)
        if not tmp.exists() and tmp.with_suffix(".npz").exists():
            tmp = tmp.with_suffix(".npz")
        tmp.replace(path)
        logger.info("Quantile model (alpha=%.2f) saved to %s", self._alpha, path)

    @classmethod
    def load(cls, path: Path) -> TFQuantileRegressor:
        """Load from ``.npz``.

        Args:
            path: Source file written by :meth:`save`.

        Returns:
            A fitted :class:`TFQuantileRegressor`.
        """
        data = np.load(str(path), allow_pickle=True)
        n_features = int(data["n_features"])
        alpha = float(data["alpha"])
        config = QuantileModelConfig(
            hidden_sizes=tuple(int(x) for x in data["hidden_sizes"]),
            dropout_rates=tuple(float(x) for x in data["dropout_rates"]),
            learning_rate=float(data["learning_rate"]),
        )

        instance = cls(quantile_alpha=alpha, config=config)
        instance._n_features = n_features
        instance._model = _build_quantile_network(n_features, config)

        dummy = tf.zeros((1, n_features), dtype=tf.float64)
        instance._model(dummy, training=False)

        weights = []
        i = 0
        while f"weight_{i}" in data:
            weights.append(data[f"weight_{i}"])
            i += 1
        instance._model.set_weights(weights)
        return instance


# ---------------------------------------------------------------------------
# Split-conformal prediction interval
# ---------------------------------------------------------------------------


class ConformalPredictionInterval:
    """Calibrated prediction interval with guaranteed marginal coverage.

    Uses split-conformal prediction: the residuals on a calibration set
    determine a correction factor ``q_hat`` that inflates or deflates
    the raw quantile predictions to achieve the desired coverage.

    Args:
        lower_model: Fitted quantile regressor targeting the lower
            quantile (e.g. ``alpha=0.025``).
        upper_model: Fitted quantile regressor targeting the upper
            quantile (e.g. ``alpha=0.975``).
        coverage_target: Desired marginal coverage (e.g. ``0.95``).
    """

    def __init__(
        self,
        lower_model: TFQuantileRegressor,
        upper_model: TFQuantileRegressor,
        coverage_target: float = 0.95,
    ) -> None:
        self._lower = lower_model
        self._upper = upper_model
        self._coverage_target = coverage_target
        self._q_hat: float | None = None

    @property
    def is_calibrated(self) -> bool:
        """Whether :meth:`calibrate` has been called."""
        return self._q_hat is not None

    @property
    def q_hat(self) -> float:
        """The conformal correction factor.

        Raises:
            RuntimeError: If not yet calibrated.
        """
        if self._q_hat is None:
            msg = "Call calibrate() first"
            raise RuntimeError(msg)
        return self._q_hat

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Compute the conformal correction ``q_hat`` from a calibration set.

        Args:
            X_cal: Calibration features.
            y_cal: Calibration targets.
        """
        lower_preds = self._lower.predict(X_cal)
        upper_preds = self._upper.predict(X_cal)

        # Non-conformity scores: how far each observation falls outside
        # the raw interval.
        scores = np.maximum(lower_preds - y_cal, y_cal - upper_preds)

        # Quantile at level ceil((n+1) * coverage) / n.
        n = len(y_cal)
        quantile_level = min(
            np.ceil((n + 1) * self._coverage_target) / n, 1.0,
        )
        self._q_hat = float(np.quantile(scores, quantile_level))

        logger.info(
            "Conformal calibration: q_hat=%.4f (n_cal=%d, target=%.2f)",
            self._q_hat, n, self._coverage_target,
        )

    def predict_interval(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated lower and upper bounds.

        Args:
            X: Feature matrix.

        Returns:
            Tuple ``(lower_bounds, upper_bounds)`` as 1-D arrays.

        Raises:
            RuntimeError: If not yet calibrated.
        """
        if self._q_hat is None:
            msg = "Call calibrate() before predict_interval()"
            raise RuntimeError(msg)

        lower = self._lower.predict(X) - self._q_hat
        upper = self._upper.predict(X) + self._q_hat
        return lower, upper

    def empirical_coverage(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute empirical coverage of the calibrated intervals.

        Args:
            X: Features.
            y: Targets.

        Returns:
            Fraction of samples where ``lower <= y <= upper``.
        """
        lower, upper = self.predict_interval(X)
        covered = np.sum((y >= lower) & (y <= upper))
        return float(covered / len(y))
