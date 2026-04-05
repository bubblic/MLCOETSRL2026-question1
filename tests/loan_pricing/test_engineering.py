"""Tests for the feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_pricing.exceptions import DataLeakageError
from loan_pricing.features.engineering import (
    FeatureConfig,
    FeatureEngineer,
    TransformResult,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def train_df(synthetic_loan_df: pd.DataFrame) -> pd.DataFrame:
    """First 35 rows as the 'training' set."""
    return synthetic_loan_df.iloc[:35].copy()


@pytest.fixture()
def val_df(synthetic_loan_df: pd.DataFrame) -> pd.DataFrame:
    """Last 15 rows as the 'validation' set."""
    return synthetic_loan_df.iloc[35:].copy()


@pytest.fixture()
def fitted_engineer(train_df: pd.DataFrame) -> FeatureEngineer:
    """A FeatureEngineer that has been fitted on *train_df*."""
    eng = FeatureEngineer()
    eng.fit(train_df)
    return eng


# ------------------------------------------------------------------
# fit() basics
# ------------------------------------------------------------------


class TestFeatureEngineerFit:
    def test_is_fitted_after_fit(self, train_df: pd.DataFrame) -> None:
        eng = FeatureEngineer()
        assert not eng.is_fitted
        eng.fit(train_df)
        assert eng.is_fitted

    def test_feature_names_populated(self, fitted_engineer: FeatureEngineer) -> None:
        names = fitted_engineer.feature_names
        assert isinstance(names, list)
        assert len(names) > 0

    def test_fit_on_same_df_is_idempotent(self, train_df: pd.DataFrame) -> None:
        eng = FeatureEngineer()
        eng.fit(train_df)
        eng.fit(train_df)  # Same object — should not raise.

    def test_fit_on_different_df_raises(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> None:
        eng = FeatureEngineer()
        eng.fit(train_df)
        with pytest.raises(DataLeakageError):
            eng.fit(val_df)


# ------------------------------------------------------------------
# transform() basics
# ------------------------------------------------------------------


class TestFeatureEngineerTransform:
    def test_returns_transform_result(
        self, fitted_engineer: FeatureEngineer, train_df: pd.DataFrame
    ) -> None:
        result = fitted_engineer.transform(train_df)
        assert isinstance(result, TransformResult)

    def test_x_shape_matches_rows(
        self, fitted_engineer: FeatureEngineer, train_df: pd.DataFrame
    ) -> None:
        result = fitted_engineer.transform(train_df)
        assert result.X.shape[0] == len(train_df)

    def test_y_length_matches_rows(
        self, fitted_engineer: FeatureEngineer, train_df: pd.DataFrame
    ) -> None:
        result = fitted_engineer.transform(train_df)
        assert len(result.y) == len(train_df)

    def test_feature_names_length_matches_columns(
        self, fitted_engineer: FeatureEngineer, train_df: pd.DataFrame
    ) -> None:
        result = fitted_engineer.transform(train_df)
        assert result.X.shape[1] == len(result.feature_names)

    def test_transform_before_fit_raises(self, train_df: pd.DataFrame) -> None:
        eng = FeatureEngineer()
        with pytest.raises(RuntimeError, match="fit"):
            eng.transform(train_df)


# ------------------------------------------------------------------
# Cross-split consistency
# ------------------------------------------------------------------


class TestCrossSplitConsistency:
    def test_val_has_same_columns_as_train(
        self,
        fitted_engineer: FeatureEngineer,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> None:
        train_result = fitted_engineer.transform(train_df)
        val_result = fitted_engineer.transform(val_df)
        assert train_result.feature_names == val_result.feature_names
        assert train_result.X.shape[1] == val_result.X.shape[1]


# ------------------------------------------------------------------
# Interaction terms
# ------------------------------------------------------------------


class TestInteractionTerms:
    def test_interaction_columns_present(
        self, fitted_engineer: FeatureEngineer
    ) -> None:
        names = fitted_engineer.feature_names
        assert "debt_to_ebitda_x_vix" in names
        assert "maturity_years_x_hy_oas" in names
        assert "interest_coverage_ratio_x_treasury_yield" in names

    def test_interaction_values_are_products(
        self, fitted_engineer: FeatureEngineer, train_df: pd.DataFrame
    ) -> None:
        """Verify raw interaction = product of raw inputs (pre-normalisation)."""
        # Use a fresh engineer to check raw values before normalisation.
        eng = FeatureEngineer()
        eng.fit(train_df)
        raw = eng._select_and_engineer(train_df)

        expected = raw["debt_to_ebitda"] * raw["vix"]
        actual = raw["debt_to_ebitda_x_vix"]
        np.testing.assert_array_almost_equal(actual.values, expected.values)

    def test_custom_interaction_config(
        self, train_df: pd.DataFrame
    ) -> None:
        cfg = FeatureConfig(
            interaction_terms=(("loan_size_mm", "vix"),),
        )
        eng = FeatureEngineer(config=cfg)
        eng.fit(train_df)
        assert "loan_size_mm_x_vix" in eng.feature_names
        # The default interactions should NOT be present.
        assert "debt_to_ebitda_x_vix" not in eng.feature_names


# ------------------------------------------------------------------
# fit_transform convenience
# ------------------------------------------------------------------


class TestFitTransform:
    def test_equivalent_to_fit_then_transform(
        self, train_df: pd.DataFrame
    ) -> None:
        eng1 = FeatureEngineer()
        result1 = eng1.fit_transform(train_df)

        eng2 = FeatureEngineer()
        eng2.fit(train_df)
        result2 = eng2.transform(train_df)

        np.testing.assert_array_equal(result1.X, result2.X)
        np.testing.assert_array_equal(result1.y, result2.y)

    def test_custom_target(self, train_df: pd.DataFrame) -> None:
        eng = FeatureEngineer()
        result = eng.fit_transform(train_df, target="defaulted")
        # y should be the 'defaulted' column values.
        np.testing.assert_array_equal(
            result.y, train_df["defaulted"].values.astype(np.float64)
        )


# ------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------


class TestTransformFeatures:
    """Tests for the inference-time transform_features() method."""

    def test_returns_ndarray(
        self, fitted_engineer: FeatureEngineer, train_df: pd.DataFrame
    ) -> None:
        X = fitted_engineer.transform_features(train_df)
        assert isinstance(X, np.ndarray)

    def test_shape_matches_transform(
        self, fitted_engineer: FeatureEngineer, train_df: pd.DataFrame
    ) -> None:
        X_feat = fitted_engineer.transform_features(train_df)
        result = fitted_engineer.transform(train_df)
        np.testing.assert_array_equal(X_feat, result.X)

    def test_single_row(
        self, fitted_engineer: FeatureEngineer
    ) -> None:
        """A 1-row DataFrame (single loan) should produce (1, n_features)."""
        row = pd.DataFrame([{
            "debt_to_ebitda": 3.0,
            "vix": 20.0,
            "maturity_years": 5,
            "loan_size_mm": 100.0,
            "is_secured": 1.0,
        }])
        X = fitted_engineer.transform_features(row)
        assert X.shape[0] == 1
        assert X.shape[1] == len(fitted_engineer.feature_names)

    def test_before_fit_raises(self, train_df: pd.DataFrame) -> None:
        eng = FeatureEngineer()
        with pytest.raises(RuntimeError, match="fit"):
            eng.transform_features(train_df)


class TestNormalisation:
    def test_train_features_approximately_zero_mean(
        self, fitted_engineer: FeatureEngineer, train_df: pd.DataFrame
    ) -> None:
        result = fitted_engineer.transform(train_df)
        col_means = result.X.mean(axis=0)
        # After z-score normalisation on training data, means ≈ 0.
        np.testing.assert_array_almost_equal(col_means, 0.0, decimal=1)

    def test_train_features_approximately_unit_std(
        self, fitted_engineer: FeatureEngineer, train_df: pd.DataFrame
    ) -> None:
        result = fitted_engineer.transform(train_df)
        col_stds = result.X.std(axis=0)
        # Standard deviations ≈ 1 (within tolerance for N=35).
        assert all(s < 2.0 for s in col_stds)
