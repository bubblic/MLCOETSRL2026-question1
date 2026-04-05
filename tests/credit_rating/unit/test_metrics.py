"""Tests for training metrics computation."""

from __future__ import annotations

import numpy as np
import pytest

from credit_rating.training.metrics import compute_metrics


class TestComputeMetrics:
    """Verify all metric computations."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 3, 4, 5, 6])
        y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == 1.0
        assert m.mae_notches == 0.0
        assert m.spearman_rho == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([6, 6, 6, 6])
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == 0.0
        assert m.mae_notches == 6.0

    def test_investment_grade_accuracy(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        m = compute_metrics(y_true, y_pred)
        assert m.investment_grade_accuracy == 1.0

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        m = compute_metrics(y_true, y_pred)
        assert len(m.confusion_matrix) == 3
        assert len(m.confusion_matrix[0]) == 3

    def test_macro_f1_range(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 1, 3])
        m = compute_metrics(y_true, y_pred)
        assert 0 <= m.macro_f1 <= 1.0
