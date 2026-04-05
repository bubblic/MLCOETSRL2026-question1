"""Ordinal logistic regression (proportional-odds model).

Provides a baseline that respects the ordered nature of credit ratings.
Uses ``mord.LogisticAT`` (All-Thresholds variant) when available,
falling back to a TensorFlow/Keras cumulative-logit model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from credit_rating.config.settings import CreditRatingSettings, RatingClass
from credit_rating.domain.features import FinancialRatios
from credit_rating.domain.rating import RatingPrediction
from credit_rating.domain.report import AnnualReport
from credit_rating.features.normalizer import RatioNormalizer
from credit_rating.features.ratio_calculator import FinancialRatioCalculator

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics produced during ordinal logistic training.

    Args:
        accuracy: Overall classification accuracy.
        mae_notches: Mean absolute error in rating notches.
        confusion_matrix: Confusion matrix as a 2-D list.
    """

    accuracy: float = 0.0
    mae_notches: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)


class OrdinalLogisticRatingModel:
    """Proportional-odds ordinal logistic regression for credit ratings.

    Args:
        settings: System configuration.
        normalizer: Pre-fitted ratio normalizer. If ``None``, a new
            one will be created and fitted during training.
    """

    def __init__(
        self,
        settings: Optional[CreditRatingSettings] = None,
        normalizer: Optional[RatioNormalizer] = None,
    ) -> None:
        self._settings = settings or CreditRatingSettings()
        self._normalizer = normalizer
        self._model = _create_ordinal_model()
        self._is_trained = False
        self._calculator = FinancialRatioCalculator()

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> TrainingMetrics:
        """Train the ordinal logistic model.

        Args:
            features: 2-D array of shape ``(n_samples, 25)``.
            labels: 1-D array of integer rating class labels.

        Returns:
            Training metrics including accuracy and MAE.
        """
        self._model.fit(features, labels)
        self._is_trained = True

        predictions = self._model.predict(features)
        accuracy = float(np.mean(predictions == labels))
        mae = float(np.mean(np.abs(predictions - labels)))

        logger.info(
            "Ordinal logistic trained: accuracy=%.3f, MAE=%.2f notches",
            accuracy,
            mae,
        )
        return TrainingMetrics(accuracy=accuracy, mae_notches=mae)

    def predict(self, report: AnnualReport) -> RatingPrediction:
        """Predict a credit rating from an annual report.

        Args:
            report: A fully populated annual report.

        Returns:
            A :class:`RatingPrediction` with rating and probabilities.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError("Model has not been trained. Call .fit() first.")
        ratios = self._calculator.calculate(report.financial_statements)
        return self.predict_from_ratios(ratios)

    def predict_from_ratios(
        self,
        ratios: FinancialRatios,
    ) -> RatingPrediction:
        """Predict from a pre-computed ratio vector.

        Args:
            ratios: The 25-dimensional financial ratio vector.

        Returns:
            A :class:`RatingPrediction`.
        """
        features = np.array([ratios.to_flat_list()], dtype=np.float64)
        if self._normalizer is not None and self._normalizer.is_fitted:
            features = self._normalizer.transform(
                [ratios],
            )
        return self._predict_array(features[0])

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        """Batch predict from a 2-D feature array.

        Args:
            features: Array of shape ``(n_samples, 25)``.

        Returns:
            1-D array of integer rating class predictions.
        """
        return self._model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities for a 2-D feature array.

        Args:
            features: Array of shape ``(n_samples, 25)``.

        Returns:
            2-D array of shape ``(n_samples, 7)`` with probabilities.
        """
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(features)
        return _one_hot_predictions(
            self._model.predict(features),
            self._settings.num_rating_classes,
        )

    def _predict_array(self, features_1d: np.ndarray) -> RatingPrediction:
        """Predict a single sample and build a RatingPrediction."""
        features_2d = features_1d.reshape(1, -1)
        predicted_class = int(self._model.predict(features_2d)[0])
        proba = self.predict_proba(features_2d)[0]

        probabilities: Dict[RatingClass, float] = {
            RatingClass(i): float(proba[i])
            for i in range(self._settings.num_rating_classes)
            if i < len(proba)
        }

        return RatingPrediction(
            rating=RatingClass(predicted_class),
            probabilities=probabilities,
        )


class _KerasOrdinalLogistic:
    """Cumulative-logit (proportional-odds) model built on Keras.

    Each of the ``K-1`` thresholds shares a single weight vector, and
    the model learns monotonically increasing cutpoints so that
    ``P(Y <= k) = sigmoid(cutpoint_k - X @ w)``.  This mirrors the
    ``mord.LogisticAT`` formulation.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._model: Optional[tf.keras.Model] = None
        self._num_classes = 0

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
    ) -> None:
        self._num_classes = int(labels.max()) + 1
        n_thresholds = self._num_classes - 1
        n_features = features.shape[1]

        inputs = tf.keras.Input(shape=(n_features,))
        # Shared linear projection (no bias — thresholds act as biases)
        logits = tf.keras.layers.Dense(
            1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self._alpha),
            name="shared_weights",
        )(inputs)
        # Learnable cutpoints, initialised uniformly across the range
        cutpoints = tf.Variable(
            tf.linspace(-2.0, 2.0, n_thresholds),
            name="cutpoints",
        )
        # Cumulative probabilities P(Y <= k) for k in 0..K-2
        # Wrapped in Lambda because tf.sigmoid can't operate on
        # Keras symbolic tensors directly in TF 2.20+ / Keras 3.
        cum_probs = tf.keras.layers.Lambda(
            lambda x: tf.sigmoid(cutpoints - x),
        )(logits)

        self._model = tf.keras.Model(inputs=inputs, outputs=cum_probs)
        self._model.cutpoints = cutpoints

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
            loss=self._ordinal_loss,
        )

        # Encode labels as cumulative binary targets: Y_k = 1 if label <= k
        targets = np.zeros(
            (len(labels), n_thresholds), dtype=np.float32
        )
        for k in range(n_thresholds):
            targets[:, k] = (labels <= k).astype(np.float32)

        self._model.fit(
            features.astype(np.float32),
            targets,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

    @staticmethod
    def _ordinal_loss(
        y_true: tf.Tensor, y_pred: tf.Tensor
    ) -> tf.Tensor:
        return tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y_true, y_pred)
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        cum = self._model.predict(
            features.astype(np.float32), verbose=0
        )
        proba = np.zeros(
            (len(features), self._num_classes), dtype=np.float64
        )
        # P(Y = 0) = cum_0;  P(Y = k) = cum_k - cum_{k-1};  P(Y = K-1) = 1 - cum_{K-2}
        proba[:, 0] = cum[:, 0]
        for k in range(1, self._num_classes - 1):
            proba[:, k] = cum[:, k] - cum[:, k - 1]
        proba[:, -1] = 1.0 - cum[:, -1]
        proba = np.clip(proba, 0.0, 1.0)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)


def _create_ordinal_model() -> object:
    """Create an ordinal logistic model, with Keras fallback."""
    try:
        from mord import LogisticAT

        return LogisticAT(alpha=1.0)
    except ImportError:
        logger.info(
            "mord not installed; falling back to Keras ordinal logistic"
        )
        return _KerasOrdinalLogistic(alpha=1.0)


def _one_hot_predictions(
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Convert hard predictions to a pseudo-probability matrix."""
    n = len(predictions)
    proba = np.zeros((n, num_classes), dtype=np.float64)
    for i, pred in enumerate(predictions):
        pred_int = int(pred)
        if 0 <= pred_int < num_classes:
            proba[i, pred_int] = 1.0
    return proba
