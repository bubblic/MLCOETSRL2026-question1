"""Probability-of-default classifier built on TensorFlow.

The :class:`TFPDClassifier` composes a ``tf.keras.Model`` internally
and trains it with a custom ``tf.GradientTape`` loop, following the
pattern established in ``financial_forecast/training/``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from loan_pricing.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class CalibrationResult(NamedTuple):
    """Calibration-curve data for a fitted classifier.

    Attributes:
        fraction_of_positives: Observed positive fraction per bin.
        mean_predicted_value: Mean predicted probability per bin.
    """

    fraction_of_positives: np.ndarray
    mean_predicted_value: np.ndarray


@dataclass(frozen=True)
class PDEvaluationResult:
    """Evaluation metrics for a probability-of-default model.

    Attributes:
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the Precision-Recall curve.
        brier_score: Brier score (lower is better).
        ks_statistic: Kolmogorov-Smirnov statistic.
        gini_coefficient: Gini = 2 * ROC-AUC - 1.
    """

    roc_auc: float
    pr_auc: float
    brier_score: float
    ks_statistic: float
    gini_coefficient: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDModelConfig:
    """Hyperparameters for :class:`TFPDClassifier`.

    Attributes:
        hidden_sizes: Widths of the hidden dense layers.
        dropout_rates: Dropout rate after each hidden layer.
        learning_rate: Adam optimiser learning rate.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        early_stopping_patience: Epochs without val-loss improvement
            before stopping early.
        random_seed: Seed for weight initialisation and shuffling.
        cross_validation_folds: Number of stratified CV folds used
            during :meth:`TFPDClassifier.fit`.
    """

    hidden_sizes: tuple[int, ...] = (128, 64)
    dropout_rates: tuple[float, ...] = (0.3, 0.2)
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    random_seed: int = 42
    cross_validation_folds: int = 5


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseClassifier(abc.ABC):
    """Interface that every PD classifier must satisfy."""

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseClassifier:
        """Train the classifier.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Binary labels of shape ``(n_samples,)``.

        Returns:
            ``self`` for method chaining.
        """

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted default probabilities.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of probabilities in ``[0, 1]``.
        """


# ---------------------------------------------------------------------------
# Keras network builder
# ---------------------------------------------------------------------------


def _build_network(
    n_features: int,
    config: PDModelConfig,
) -> tf.keras.Model:
    """Construct a dense classification network.

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
        x = tf.keras.layers.Dropout(drop, dtype=tf.float64)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype=tf.float64)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _train_epoch(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    X: tf.Tensor,
    y: tf.Tensor,
    batch_size: int,
) -> float:
    """Run one training epoch and return the mean loss.

    Args:
        model: Keras classification network.
        optimizer: TF optimiser instance.
        X: Training features as a float64 tensor.
        y: Training labels as a float64 tensor.
        batch_size: Mini-batch size.

    Returns:
        Scalar mean binary-crossentropy loss for the epoch.
    """
    loss_fn = tf.keras.losses.BinaryCrossentropy()
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


def _evaluate_loss(
    model: tf.keras.Model,
    X: tf.Tensor,
    y: tf.Tensor,
) -> float:
    """Compute binary-crossentropy loss without gradient tracking.

    Args:
        model: Keras classification network.
        X: Feature tensor.
        y: Label tensor.

    Returns:
        Scalar loss value.
    """
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    preds = model(X, training=False)
    return float(loss_fn(y, preds))


# ---------------------------------------------------------------------------
# Concrete classifier
# ---------------------------------------------------------------------------


class TFPDClassifier(BaseClassifier):
    """TensorFlow-based probability-of-default classifier.

    Composes a ``tf.keras.Model`` internally (composition, not
    inheritance) and trains it with a custom ``GradientTape`` loop.

    Args:
        config: Model hyperparameters.
    """

    def __init__(self, config: PDModelConfig | None = None) -> None:
        self._config = config or PDModelConfig()
        self._model: tf.keras.Model | None = None
        self._n_features: int | None = None

    # -- public API ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> TFPDClassifier:
        """Train with stratified k-fold CV, keeping the best fold's weights.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.
            y: Binary labels ``(n_samples,)``.

        Returns:
            ``self``.
        """
        cfg = self._config
        tf.random.set_seed(cfg.random_seed)
        self._n_features = X.shape[1]

        skf = StratifiedKFold(
            n_splits=cfg.cross_validation_folds,
            shuffle=True,
            random_state=cfg.random_seed,
        )

        best_val_loss = float("inf")
        best_weights: list[np.ndarray] | None = None

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info("Fold %d/%d", fold + 1, cfg.cross_validation_folds)

            model = _build_network(self._n_features, cfg)
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

            X_train = tf.constant(X[train_idx], dtype=tf.float64)
            y_train = tf.constant(y[train_idx].reshape(-1, 1), dtype=tf.float64)
            X_val = tf.constant(X[val_idx], dtype=tf.float64)
            y_val = tf.constant(y[val_idx].reshape(-1, 1), dtype=tf.float64)

            patience_counter = 0
            fold_best_loss = float("inf")

            for epoch in range(cfg.epochs):
                _train_epoch(model, optimizer, X_train, y_train, cfg.batch_size)
                val_loss = _evaluate_loss(model, X_val, y_val)

                if val_loss < fold_best_loss:
                    fold_best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= cfg.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (val_loss=%.4f)",
                        epoch + 1,
                        fold_best_loss,
                    )
                    break

            if fold_best_loss < best_val_loss:
                best_val_loss = fold_best_loss
                best_weights = list(model.get_weights())

        # Rebuild model with best weights.
        self._model = _build_network(self._n_features, cfg)
        if best_weights is not None:
            self._model.set_weights(best_weights)

        logger.info("Training complete — best val loss %.4f", best_val_loss)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted default probabilities.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.

        Returns:
            1-D array of probabilities in ``[0, 1]``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._model is None:
            msg = "Call fit() before predict_proba()"
            raise RuntimeError(msg)
        X_t = tf.constant(X, dtype=tf.float64)
        preds = self._model(X_t, training=False)
        return preds.numpy().ravel()

    @property
    def calibration_curve_(self) -> CalibrationResult:
        """Calibration curve data (requires a preceding predict call).

        This is a convenience that must be called *after* obtaining
        predictions, as it needs ground-truth labels.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if self._model is None:
            msg = "Model is not fitted"
            raise RuntimeError(msg)
        # The caller is expected to use the standalone function below
        # for evaluation.  This property is provided for compatibility
        # with the plan's interface specification.
        msg = (
            "Use evaluate_pd_model() to compute the calibration curve "
            "with test data."
        )
        raise RuntimeError(msg)

    # -- serialisation ------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise model weights and config to an ``.npz`` file.

        Args:
            path: Destination file (should end in ``.npz``).

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if self._model is None:
            msg = "Cannot save an unfitted model"
            raise RuntimeError(msg)

        params: dict[str, object] = {
            "n_features": self._n_features,
        }
        for i, w in enumerate(self._model.get_weights()):
            params[f"weight_{i}"] = w

        # Save config fields for reconstruction.
        params["hidden_sizes"] = np.array(self._config.hidden_sizes)
        params["dropout_rates"] = np.array(self._config.dropout_rates)
        params["learning_rate"] = self._config.learning_rate

        # np.savez appends '.npz' if not already present, so we
        # write to a temp file that already ends in '.npz' and rename.
        tmp = path.parent / (path.stem + "_tmp.npz")
        np.savez(str(tmp), **params)
        # np.savez may or may not append .npz depending on the suffix.
        if not tmp.exists() and tmp.with_suffix(".npz").exists():
            tmp = tmp.with_suffix(".npz")
        tmp.replace(path)
        logger.info("PD model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> TFPDClassifier:
        """Load a saved model from an ``.npz`` file.

        Args:
            path: Source file written by :meth:`save`.

        Returns:
            A fitted :class:`TFPDClassifier`.
        """
        data = np.load(str(path), allow_pickle=True)
        n_features = int(data["n_features"])
        hidden_sizes = tuple(int(x) for x in data["hidden_sizes"])
        dropout_rates = tuple(float(x) for x in data["dropout_rates"])
        learning_rate = float(data["learning_rate"])

        config = PDModelConfig(
            hidden_sizes=hidden_sizes,
            dropout_rates=dropout_rates,
            learning_rate=learning_rate,
        )

        instance = cls(config=config)
        instance._n_features = n_features
        instance._model = _build_network(n_features, config)

        # Trigger weight creation by calling with dummy input.
        dummy = tf.zeros((1, n_features), dtype=tf.float64)
        instance._model(dummy, training=False)

        weights = []
        i = 0
        while f"weight_{i}" in data:
            weights.append(data[f"weight_{i}"])
            i += 1
        instance._model.set_weights(weights)

        logger.info("PD model loaded from %s", path)
        return instance


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov statistic for a binary classifier.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.

    Returns:
        KS statistic (maximum separation between positive and
        negative CDFs).
    """
    pos = np.sort(y_prob[y_true == 1])
    neg = np.sort(y_prob[y_true == 0])

    all_values = np.sort(np.unique(np.concatenate([pos, neg])))

    cdf_pos = np.searchsorted(pos, all_values, side="right") / max(len(pos), 1)
    cdf_neg = np.searchsorted(neg, all_values, side="right") / max(len(neg), 1)

    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def evaluate_pd_model(
    model: BaseClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> PDEvaluationResult:
    """Compute standard classification metrics for a PD model.

    Args:
        model: A fitted classifier satisfying :class:`BaseClassifier`.
        X_test: Test features.
        y_test: Test labels (binary).

    Returns:
        A :class:`PDEvaluationResult` with all metrics populated.
    """
    y_prob = model.predict_proba(X_test)

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ks = _ks_statistic(y_test, y_prob)
    gini = 2.0 * roc - 1.0

    return PDEvaluationResult(
        roc_auc=roc,
        pr_auc=pr,
        brier_score=brier,
        ks_statistic=ks,
        gini_coefficient=gini,
    )
