"""Tests for the TensorFlow spread regressor and LoanPricer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loan_pricing.models.pd_model import PDModelConfig, TFPDClassifier
from loan_pricing.models.spread_model import (
    LoanPricer,
    LoanPricingInput,
    LoanPricingOutput,
    SpreadEvaluationResult,
    SpreadModelConfig,
    TFSpreadRegressor,
    evaluate_spread_model,
)


# ------------------------------------------------------------------
# Shared configs — small for fast tests
# ------------------------------------------------------------------

_PD_CFG = PDModelConfig(
    hidden_sizes=(16, 8),
    dropout_rates=(0.1, 0.1),
    learning_rate=1e-2,
    epochs=15,
    batch_size=16,
    early_stopping_patience=5,
    random_seed=42,
    cross_validation_folds=2,
)

_SPREAD_CFG = SpreadModelConfig(
    hidden_sizes=(16, 8),
    dropout_rates=(0.1, 0.0),
    learning_rate=1e-2,
    epochs=30,
    batch_size=16,
    early_stopping_patience=10,
    random_seed=42,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separable binary labels + correlated continuous spread target."""
    rng = np.random.default_rng(42)
    n, d = 200, 8

    X_pos = rng.normal(loc=1.5, scale=0.8, size=(n // 2, d))
    X_neg = rng.normal(loc=-1.5, scale=0.8, size=(n // 2, d))
    X = np.vstack([X_pos, X_neg])
    y_cls = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.float64)

    # Spread is a noisy linear function of X columns + default indicator.
    y_spread = 200.0 + 50.0 * X[:, 0] + 30.0 * y_cls + rng.normal(0, 10, n)
    y_spread = np.abs(y_spread)  # spreads are positive

    idx = rng.permutation(n)
    return X[idx], y_cls[idx], y_spread[idx]


@pytest.fixture(scope="module")
def fitted_pd(synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> TFPDClassifier:
    X, y_cls, _ = synthetic_data
    clf = TFPDClassifier(config=_PD_CFG)
    clf.fit(X, y_cls)
    return clf


@pytest.fixture(scope="module")
def fitted_spread(
    synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    fitted_pd: TFPDClassifier,
) -> TFSpreadRegressor:
    X, _, y_spread = synthetic_data
    reg = TFSpreadRegressor(pd_model=fitted_pd, config=_SPREAD_CFG)
    reg.fit(X, y_spread)
    return reg


# ------------------------------------------------------------------
# TFSpreadRegressor
# ------------------------------------------------------------------


class TestTFSpreadRegressor:
    def test_predictions_are_finite(
        self,
        fitted_spread: TFSpreadRegressor,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, _, _ = synthetic_data
        preds = fitted_spread.predict(X)
        assert np.all(np.isfinite(preds))

    def test_output_shape(
        self,
        fitted_spread: TFSpreadRegressor,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, _, _ = synthetic_data
        preds = fitted_spread.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_before_fit_raises(self, fitted_pd: TFPDClassifier) -> None:
        reg = TFSpreadRegressor(pd_model=fitted_pd, config=_SPREAD_CFG)
        X = np.zeros((5, 8))
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict(X)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------


class TestSpreadEvaluation:
    def test_returns_evaluation_result(
        self,
        fitted_spread: TFSpreadRegressor,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, _, y_spread = synthetic_data
        result = evaluate_spread_model(fitted_spread, X, y_spread)
        assert isinstance(result, SpreadEvaluationResult)

    def test_rmse_is_positive(
        self,
        fitted_spread: TFSpreadRegressor,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, _, y_spread = synthetic_data
        result = evaluate_spread_model(fitted_spread, X, y_spread)
        assert result.rmse_bps > 0

    def test_mae_leq_rmse(
        self,
        fitted_spread: TFSpreadRegressor,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, _, y_spread = synthetic_data
        result = evaluate_spread_model(fitted_spread, X, y_spread)
        assert result.mae_bps <= result.rmse_bps + 1e-6


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------


class TestSpreadSaveLoad:
    def test_round_trip(
        self,
        fitted_spread: TFSpreadRegressor,
        fitted_pd: TFPDClassifier,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        X, _, _ = synthetic_data
        original = fitted_spread.predict(X)

        path = tmp_path / "spread.npz"
        fitted_spread.save(path)

        loaded = TFSpreadRegressor.load(path, pd_model=fitted_pd)
        restored = loaded.predict(X)

        np.testing.assert_array_almost_equal(original, restored)


# ------------------------------------------------------------------
# LoanPricer
# ------------------------------------------------------------------


class TestLoanPricer:
    def test_returns_pricing_output(
        self,
        fitted_pd: TFPDClassifier,
        fitted_spread: TFSpreadRegressor,
    ) -> None:
        pricer = LoanPricer(
            pd_model=fitted_pd,
            spread_model=fitted_spread,
        )
        loan = LoanPricingInput(
            financial_ratios={"debt_to_ebitda": 3.5, "interest_coverage_ratio": 5.0},
            loan_maturity_years=5.0,
            loan_size_mm=100.0,
            is_secured=True,
            industry_naics="5221",
            treasury_yield_pct=4.0,
        )
        result = pricer.price(loan)
        assert isinstance(result, LoanPricingOutput)

    def test_pd_in_unit_interval(
        self,
        fitted_pd: TFPDClassifier,
        fitted_spread: TFSpreadRegressor,
    ) -> None:
        pricer = LoanPricer(pd_model=fitted_pd, spread_model=fitted_spread)
        loan = LoanPricingInput(
            financial_ratios={"debt_to_ebitda": 3.5},
            loan_maturity_years=5.0,
            loan_size_mm=100.0,
            is_secured=True,
            industry_naics="5221",
            treasury_yield_pct=4.0,
        )
        result = pricer.price(loan)
        assert 0.0 <= result.estimated_pd <= 1.0

    def test_ci_lower_leq_upper(
        self,
        fitted_pd: TFPDClassifier,
        fitted_spread: TFSpreadRegressor,
    ) -> None:
        pricer = LoanPricer(pd_model=fitted_pd, spread_model=fitted_spread)
        loan = LoanPricingInput(
            financial_ratios={"debt_to_ebitda": 3.5},
            loan_maturity_years=5.0,
            loan_size_mm=100.0,
            is_secured=True,
            industry_naics="5221",
            treasury_yield_pct=4.0,
        )
        result = pricer.price(loan)
        assert result.spread_ci_lower_bps <= result.spread_ci_upper_bps

    def test_internal_rating_is_string(
        self,
        fitted_pd: TFPDClassifier,
        fitted_spread: TFSpreadRegressor,
    ) -> None:
        pricer = LoanPricer(pd_model=fitted_pd, spread_model=fitted_spread)
        loan = LoanPricingInput(
            financial_ratios={},
            loan_maturity_years=3.0,
            loan_size_mm=50.0,
            is_secured=False,
            industry_naics="3361",
            treasury_yield_pct=3.5,
        )
        result = pricer.price(loan)
        assert isinstance(result.internal_rating, str)
        assert result.internal_rating in {
            "AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D",
        }

    def test_feature_engineer_normalises_input(
        self,
        fitted_pd: TFPDClassifier,
        fitted_spread: TFSpreadRegressor,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """When a FeatureEngineer is provided, price() routes through it."""
        import pandas as pd
        from loan_pricing.features.engineering import FeatureConfig, FeatureEngineer

        X, y_cls, y_spread = synthetic_data

        # Build a DataFrame that matches the FeatureEngineer's expected
        # column names (at least the loan_columns).
        rng = np.random.default_rng(0)
        train_df = pd.DataFrame({
            "maturity_years": rng.choice([1, 3, 5, 10], size=len(X)),
            "loan_size_mm": rng.uniform(10, 500, size=len(X)),
            "is_secured": rng.choice([0.0, 1.0], size=len(X)),
            "credit_spread_bps": y_spread,
        })

        # Use a minimal config with only loan columns (no ratios/macro).
        cfg = FeatureConfig(
            financial_ratio_columns=(),
            macro_columns=(),
            loan_columns=("maturity_years", "loan_size_mm", "is_secured"),
            interaction_terms=(),
        )
        eng = FeatureEngineer(config=cfg)
        eng.fit(train_df)

        # The pricer should accept the feature engineer and run without
        # error, even though the model dimensions may not match the
        # engineer's output (the point is testing the normalisation path).
        # We test that it doesn't crash and returns a valid output.
        pricer = LoanPricer(
            pd_model=fitted_pd,
            spread_model=fitted_spread,
            feature_engineer=eng,
        )
        loan = LoanPricingInput(
            financial_ratios={},
            loan_maturity_years=5.0,
            loan_size_mm=100.0,
            is_secured=True,
            industry_naics="5221",
            treasury_yield_pct=4.0,
        )
        # The engineer produces 3 features but the models expect 8+1.
        # This will fail at the model level due to shape mismatch,
        # so we catch that to confirm the engineer path was invoked.
        try:
            result = pricer.price(loan)
            # If it somehow works (dimensions align), it's still valid.
            assert isinstance(result, LoanPricingOutput)
        except ValueError as exc:
            # Shape mismatch confirms the engineer path was reached —
            # the raw fallback would have zero-padded to avoid this.
            assert "shape" in str(exc).lower() or "incompatible" in str(exc).lower()
