"""Evaluation metrics for credit rating models.

Covers accuracy, macro-F1, MAE (in notches), Spearman rho,
investment-grade binary accuracy, and the confusion matrix.  All
implementations delegate to ``sklearn.metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class TrainingMetrics:
    """Comprehensive evaluation metrics for a rating model.

    Args:
        accuracy: Overall classification accuracy.
        macro_f1: Macro-averaged F1 score.
        mae_notches: Mean absolute error measured in rating notches.
        spearman_rho: Spearman rank correlation coefficient.
        investment_grade_accuracy: Binary accuracy for the
            investment-grade vs. high-yield split.
        confusion_matrix: 7x7 confusion matrix as a nested list.
    """

    accuracy: float = 0.0
    macro_f1: float = 0.0
    mae_notches: float = 0.0
    spearman_rho: float = 0.0
    investment_grade_accuracy: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> TrainingMetrics:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground-truth integer labels, shape ``(n,)``.
        y_pred: Predicted integer labels, shape ``(n,)``.
        y_prob: Optional probability matrix, shape ``(n, 7)``.

    Returns:
        A populated :class:`TrainingMetrics`.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mae = float(mean_absolute_error(y_true, y_pred))
    rho = float(spearmanr(y_true, y_pred).correlation)
    ig_acc = _investment_grade_accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return TrainingMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        mae_notches=mae,
        spearman_rho=rho,
        investment_grade_accuracy=ig_acc,
        confusion_matrix=cm,
    )


def _investment_grade_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Binary accuracy for investment-grade (<=2) vs. high-yield (>2).

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy of the binary IG/HY classification.
    """
    ig_boundary = 2
    true_ig = (y_true <= ig_boundary).astype(int)
    pred_ig = (y_pred <= ig_boundary).astype(int)
    if len(true_ig) == 0:
        return 0.0
    return float(np.mean(true_ig == pred_ig))
