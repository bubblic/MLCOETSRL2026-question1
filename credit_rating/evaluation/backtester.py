"""Backtesting harness for credit rating models.

Runs a trained :class:`CreditRatingModel` on held-out test data
and produces a :class:`BacktestReport` containing classification
metrics, a rating migration matrix, and an investment-grade vs.
high-yield ROC-AUC score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from credit_rating.config.settings import CreditRatingSettings, RatingClass
from credit_rating.models.protocols import CreditRatingModel
from credit_rating.training.metrics import TrainingMetrics, compute_metrics

logger = logging.getLogger(__name__)

# Boundary between investment-grade (<=2) and high-yield (>2)
_IG_BOUNDARY = 2


# ------------------------------------------------------------------
# Report containers
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PerCompanyResult:
    """Prediction result for a single test sample.

    Args:
        index: Row index in the test set.
        true_label: Ground-truth rating class.
        predicted_label: Model-predicted rating class.
        probabilities: Full probability vector over rating classes.
    """

    index: int
    true_label: int
    predicted_label: int
    probabilities: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestReport:
    """Complete evaluation report from a backtest run.

    Args:
        metrics: Standard classification metrics (accuracy, F1,
            MAE, Spearman rho, IG accuracy, confusion matrix).
        migration_matrix: Rating migration matrix as a nested list
            of shape ``(num_classes, num_classes)`` where entry
            ``[i][j]`` counts transitions from true class *i* to
            predicted class *j*.
        roc_auc_ig_hy: ROC-AUC for the binary investment-grade vs.
            high-yield classification.
        per_company: Per-sample prediction results.
    """

    metrics: TrainingMetrics
    migration_matrix: List[List[int]]
    roc_auc_ig_hy: float
    per_company: List[PerCompanyResult] = field(default_factory=list)


# ------------------------------------------------------------------
# Backtester
# ------------------------------------------------------------------


class CreditRatingBacktester:
    """Evaluate a credit rating model on held-out test data.

    Args:
        model: Any object satisfying the :class:`CreditRatingModel`
            protocol (must support batch-level ``predict`` or the
            backtester uses the raw probability arrays directly).
        settings: Project-wide configuration.
    """

    def __init__(
        self,
        model: CreditRatingModel,
        settings: Optional[CreditRatingSettings] = None,
    ) -> None:
        self._model = model
        self._settings = settings or CreditRatingSettings()

    def run(
        self,
        test_features: np.ndarray,
        test_labels: np.ndarray,
    ) -> BacktestReport:
        """Execute the backtest and return a full report.

        The method delegates to the model for predictions, then
        computes all evaluation metrics, the migration matrix, and
        the IG/HY ROC-AUC.

        Args:
            test_features: Feature matrix of shape ``(n, d)`` where
                *d* is the feature dimension.
            test_labels: Ground-truth integer labels of shape ``(n,)``.

        Returns:
            A frozen :class:`BacktestReport`.
        """
        y_pred, y_prob = self._generate_predictions(test_features)
        y_true = test_labels.astype(int)

        metrics = compute_metrics(y_true, y_pred, y_prob)
        migration = self._compute_migration_matrix(y_true, y_pred)
        roc_auc = self._compute_ig_hy_roc_auc(y_true, y_prob)
        per_company = self._build_per_company(y_true, y_pred, y_prob)

        report = BacktestReport(
            metrics=metrics,
            migration_matrix=migration,
            roc_auc_ig_hy=roc_auc,
            per_company=per_company,
        )
        logger.info(
            "Backtest complete: accuracy=%.4f, macro_f1=%.4f, "
            "IG/HY ROC-AUC=%.4f",
            metrics.accuracy,
            metrics.macro_f1,
            roc_auc,
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_predictions(
        self,
        features: np.ndarray,
    ) -> tuple:
        """Run the model and return predicted labels and probabilities.

        Supports two model interfaces: TensorFlow-style models that
        accept a raw feature tensor and return logits, and protocol-
        compliant models that operate on :class:`AnnualReport` objects.
        Falls back to logit-based prediction via
        ``predict_from_structured`` when available.

        Args:
            features: Feature matrix of shape ``(n, d)``.

        Returns:
            Tuple of ``(y_pred, y_prob)`` where *y_pred* has shape
            ``(n,)`` and *y_prob* has shape ``(n, num_classes)``.
        """
        import tensorflow as tf

        x = tf.constant(features, dtype=tf.float32)

        if hasattr(self._model, "predict_from_structured"):
            logits = self._model.predict_from_structured(x)  # type: ignore[union-attr]
        elif hasattr(self._model, "__call__"):
            logits = self._model(x)  # type: ignore[operator]
        else:
            raise TypeError(
                f"Model {type(self._model).__name__} does not support "
                f"batch prediction from structured features."
            )

        logits_np: np.ndarray = logits.numpy() if hasattr(logits, "numpy") else np.asarray(logits)
        probs = _softmax(logits_np)
        preds = np.argmax(probs, axis=-1)
        return preds, probs

    def _compute_migration_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> List[List[int]]:
        """Build the rating migration (confusion) matrix.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.

        Returns:
            A ``(num_classes, num_classes)`` nested list where entry
            ``[i][j]`` is the count of samples with true label *i*
            predicted as *j*.
        """
        from sklearn.metrics import confusion_matrix

        num_classes = self._settings.num_rating_classes
        labels = list(range(num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return cm.tolist()

    def _compute_ig_hy_roc_auc(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> float:
        """Compute ROC-AUC for the IG vs. HY binary split.

        Investment-grade labels are ``<= _IG_BOUNDARY`` (AAA/AA, A,
        BBB), high-yield are ``> _IG_BOUNDARY`` (BB, B, CCC/CC, D).

        Args:
            y_true: Ground-truth integer labels.
            y_prob: Probability matrix of shape ``(n, num_classes)``.

        Returns:
            The binary ROC-AUC score.  Returns ``0.0`` if only one
            class is present in the test set.
        """
        from sklearn.metrics import roc_auc_score

        binary_true = (y_true <= _IG_BOUNDARY).astype(int)

        # Probability of being investment-grade = sum of IG class probs
        ig_prob = y_prob[:, : _IG_BOUNDARY + 1].sum(axis=1)

        if len(np.unique(binary_true)) < 2:
            logger.warning(
                "Only one class (IG or HY) present in test set; "
                "ROC-AUC is undefined, returning 0.0."
            )
            return 0.0

        return float(roc_auc_score(binary_true, ig_prob))

    @staticmethod
    def _build_per_company(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> List[PerCompanyResult]:
        """Build per-sample result records.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            y_prob: Probability matrix.

        Returns:
            List of :class:`PerCompanyResult` for every sample.
        """
        results: List[PerCompanyResult] = []
        for i in range(len(y_true)):
            results.append(
                PerCompanyResult(
                    index=i,
                    true_label=int(y_true[i]),
                    predicted_label=int(y_pred[i]),
                    probabilities=y_prob[i].tolist(),
                )
            )
        return results


# ------------------------------------------------------------------
# Numerical helpers
# ------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Args:
        logits: Array of shape ``(n, c)`` or ``(c,)``.

    Returns:
        Probability array of the same shape.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
