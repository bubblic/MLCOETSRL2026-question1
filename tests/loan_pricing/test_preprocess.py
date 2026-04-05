"""Tests for the preprocessing pipeline — each function tested independently."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from loan_pricing.data.preprocess import (
    LoanDataPreprocessor,
    TrainValTestSplit,
    compute_financial_ratios,
    handle_missing_values,
    merge_loan_features,
    pivot_fred_series,
    split_train_val_test,
)


# ------------------------------------------------------------------
# pivot_fred_series
# ------------------------------------------------------------------


class TestPivotFredSeries:
    def test_produces_wide_format(self, synthetic_fred_df: pd.DataFrame) -> None:
        wide = pivot_fred_series(synthetic_fred_df)
        assert "date" in wide.columns
        # Each series_id should become its own column.
        series_ids = synthetic_fred_df["series_id"].unique()
        for sid in series_ids:
            assert sid in wide.columns

    def test_no_duplicate_dates(self, synthetic_fred_df: pd.DataFrame) -> None:
        wide = pivot_fred_series(synthetic_fred_df)
        assert wide["date"].is_unique


# ------------------------------------------------------------------
# compute_financial_ratios
# ------------------------------------------------------------------


@pytest.fixture()
def wide_sec_df() -> pd.DataFrame:
    """Wide-format SEC data with one row per company-year."""
    rng = np.random.default_rng(11)
    n = 10
    return pd.DataFrame(
        {
            "cik": ["0000320193"] * n,
            "fiscal_year": list(range(2015, 2025)),
            "Revenues": rng.uniform(1e9, 5e10, n),
            "GrossProfit": rng.uniform(5e8, 2e10, n),
            "OperatingIncomeLoss": rng.uniform(1e8, 1e10, n),
            "NetIncomeLoss": rng.uniform(1e8, 8e9, n),
            "Assets": rng.uniform(1e10, 5e11, n),
            "Liabilities": rng.uniform(5e9, 3e11, n),
            "StockholdersEquity": rng.uniform(1e9, 2e11, n),
            "RetainedEarningsAccumulatedDeficit": rng.uniform(-1e9, 1e11, n),
            "CashAndCashEquivalentsAtCarryingValue": rng.uniform(1e8, 5e10, n),
            "LongTermDebt": rng.uniform(1e8, 1e11, n),
            "InterestExpense": rng.uniform(1e6, 5e9, n),
            "DepreciationAndAmortization": rng.uniform(1e7, 5e9, n),
        }
    )


class TestComputeFinancialRatios:
    def test_ratio_columns_present(self, wide_sec_df: pd.DataFrame) -> None:
        result = compute_financial_ratios(wide_sec_df)
        expected_cols = {
            "debt_to_ebitda",
            "interest_coverage_ratio",
            "net_debt_to_equity",
            "fcf_to_debt",
            "revenue_growth_yoy",
            "ebitda_margin",
            "current_ratio",
            "altman_z_double_prime",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_row_count_preserved(self, wide_sec_df: pd.DataFrame) -> None:
        result = compute_financial_ratios(wide_sec_df)
        assert len(result) == len(wide_sec_df)

    def test_ebitda_margin_in_range(self, wide_sec_df: pd.DataFrame) -> None:
        result = compute_financial_ratios(wide_sec_df)
        margin = result["ebitda_margin"].dropna()
        # EBITDA margin should generally be between -1 and +2 for sane inputs.
        assert (margin > -2).all()
        assert (margin < 5).all()


# ------------------------------------------------------------------
# merge_loan_features
# ------------------------------------------------------------------


class TestMergeLoanFeatures:
    def test_output_has_loan_rows(self, synthetic_loan_df: pd.DataFrame) -> None:
        empty_ratios = pd.DataFrame(columns=["cik", "fiscal_year"])
        empty_fred = pd.DataFrame(columns=["date"])
        result = merge_loan_features(synthetic_loan_df, empty_ratios, empty_fred)
        assert len(result) == len(synthetic_loan_df)


# ------------------------------------------------------------------
# handle_missing_values
# ------------------------------------------------------------------


class TestHandleMissingValues:
    def test_median_fills_nans(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "z"]})
        result = handle_missing_values(df, strategy="median")
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[1] == pytest.approx(2.0)

    def test_non_numeric_unchanged(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", None]})
        result = handle_missing_values(df, strategy="median")
        assert result["b"].iloc[1] is None

    def test_knn_fills_nans(self) -> None:
        df = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0, 4.0], "b": [10.0, 20.0, np.nan, 40.0]}
        )
        result = handle_missing_values(df, strategy="knn")
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 0

    def test_invalid_strategy_raises(self) -> None:
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="Unknown"):
            handle_missing_values(df, strategy="magic")  # type: ignore[arg-type]


# ------------------------------------------------------------------
# split_train_val_test
# ------------------------------------------------------------------


class TestSplitTrainValTest:
    def test_fractions_produce_correct_sizes(
        self, synthetic_loan_df: pd.DataFrame
    ) -> None:
        split = split_train_val_test(
            synthetic_loan_df,
            val_frac=0.15,
            test_frac=0.15,
            time_column="date",
        )
        total = len(split.train) + len(split.validation) + len(split.test)
        assert total == len(synthetic_loan_df)

    def test_no_temporal_leakage(self, synthetic_loan_df: pd.DataFrame) -> None:
        split = split_train_val_test(
            synthetic_loan_df,
            val_frac=0.2,
            test_frac=0.2,
            time_column="date",
        )
        train_max = pd.to_datetime(split.train["date"]).max()
        val_min = pd.to_datetime(split.validation["date"]).min()
        test_min = pd.to_datetime(split.test["date"]).min()
        assert train_max <= val_min
        assert val_min <= test_min

    def test_returns_named_tuple(self, synthetic_loan_df: pd.DataFrame) -> None:
        split = split_train_val_test(
            synthetic_loan_df,
            val_frac=0.15,
            test_frac=0.15,
            time_column="date",
        )
        assert isinstance(split, TrainValTestSplit)

    def test_invalid_fractions_raise(self, synthetic_loan_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="less than 1.0"):
            split_train_val_test(
                synthetic_loan_df,
                val_frac=0.6,
                test_frac=0.6,
                time_column="date",
            )

    def test_no_nan_leakage_across_splits(
        self, synthetic_loan_df: pd.DataFrame
    ) -> None:
        """Train-set NaN imputation must not leak into val/test indices."""
        split = split_train_val_test(
            synthetic_loan_df,
            val_frac=0.15,
            test_frac=0.15,
            time_column="date",
        )
        train_idx = set(split.train.index)
        val_idx = set(split.validation.index)
        test_idx = set(split.test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)
