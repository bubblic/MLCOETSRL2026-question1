"""Tests for the TensorFlow PD classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loan_pricing.models.pd_model import (
    PDEvaluationResult,
    PDModelConfig,
    TFPDClassifier,
    evaluate_pd_model,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

# Use small config so tests run fast.
FAST_CONFIG = PDModelConfig(
    hidden_sizes=(16, 8),
    dropout_rates=(0.1, 0.1),
    learning_rate=1e-2,
    epochs=20,
    batch_size=16,
    early_stopping_patience=5,
    random_seed=42,
    cross_validation_folds=2,
)


@pytest.fixture()
def synthetic_X_y() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic dataset with separable classes."""
    rng = np.random.default_rng(42)
    n = 200
    n_features = 10

    # Create two somewhat-separable clusters.
    X_pos = rng.normal(loc=1.0, scale=1.0, size=(n // 2, n_features))
    X_neg = rng.normal(loc=-1.0, scale=1.0, size=(n // 2, n_features))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.float64)

    # Shuffle.
    idx = rng.permutation(n)
    return X[idx], y[idx]


@pytest.fixture()
def fitted_classifier(
    synthetic_X_y: tuple[np.ndarray, np.ndarray],
) -> TFPDClassifier:
    """A TFPDClassifier trained on the synthetic dataset."""
    X, y = synthetic_X_y
    clf = TFPDClassifier(config=FAST_CONFIG)
    clf.fit(X, y)
    return clf


# ------------------------------------------------------------------
# predict_proba
# ------------------------------------------------------------------


class TestPredictProba:
    def test_output_in_unit_interval(
        self,
        fitted_classifier: TFPDClassifier,
        synthetic_X_y: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, _ = synthetic_X_y
        probs = fitted_classifier.predict_proba(X)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_output_shape(
        self,
        fitted_classifier: TFPDClassifier,
        synthetic_X_y: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, _ = synthetic_X_y
        probs = fitted_classifier.predict_proba(X)
        assert probs.shape == (X.shape[0],)

    def test_predict_before_fit_raises(self) -> None:
        clf = TFPDClassifier(config=FAST_CONFIG)
        X = np.zeros((5, 10))
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_proba(X)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------


class TestEvaluation:
    def test_roc_auc_above_chance(
        self,
        fitted_classifier: TFPDClassifier,
        synthetic_X_y: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = synthetic_X_y
        result = evaluate_pd_model(fitted_classifier, X, y)
        assert result.roc_auc > 0.5

    def test_returns_frozen_dataclass(
        self,
        fitted_classifier: TFPDClassifier,
        synthetic_X_y: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = synthetic_X_y
        result = evaluate_pd_model(fitted_classifier, X, y)
        assert isinstance(result, PDEvaluationResult)

    def test_gini_equals_two_auc_minus_one(
        self,
        fitted_classifier: TFPDClassifier,
        synthetic_X_y: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = synthetic_X_y
        result = evaluate_pd_model(fitted_classifier, X, y)
        assert result.gini_coefficient == pytest.approx(
            2.0 * result.roc_auc - 1.0
        )

    def test_brier_score_below_random(
        self,
        fitted_classifier: TFPDClassifier,
        synthetic_X_y: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = synthetic_X_y
        result = evaluate_pd_model(fitted_classifier, X, y)
        # Random baseline Brier score for balanced classes = 0.25.
        assert result.brier_score < 0.25

    def test_ks_statistic_positive(
        self,
        fitted_classifier: TFPDClassifier,
        synthetic_X_y: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = synthetic_X_y
        result = evaluate_pd_model(fitted_classifier, X, y)
        assert result.ks_statistic > 0.0


# ------------------------------------------------------------------
# Save / load round-trip
# ------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip_predictions_match(
        self,
        fitted_classifier: TFPDClassifier,
        synthetic_X_y: tuple[np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        X, _ = synthetic_X_y
        original_probs = fitted_classifier.predict_proba(X)

        save_path = tmp_path / "pd_model.npz"
        fitted_classifier.save(save_path)
        assert save_path.exists()

        loaded = TFPDClassifier.load(save_path)
        loaded_probs = loaded.predict_proba(X)

        np.testing.assert_array_almost_equal(original_probs, loaded_probs)

    def test_save_unfitted_raises(self, tmp_path: Path) -> None:
        clf = TFPDClassifier(config=FAST_CONFIG)
        with pytest.raises(RuntimeError, match="unfitted"):
            clf.save(tmp_path / "bad.npz")

    def test_save_is_atomic(
        self,
        fitted_classifier: TFPDClassifier,
        tmp_path: Path,
    ) -> None:
        """The .tmp file should not remain after a successful save."""
        save_path = tmp_path / "pd_model.npz"
        fitted_classifier.save(save_path)
        assert not save_path.with_suffix(".tmp").exists()
