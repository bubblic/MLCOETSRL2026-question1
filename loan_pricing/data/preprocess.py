"""Preprocessing pipeline for loan-pricing data.

Each public function in this module performs a single, independently
testable transformation step.  The :class:`LoanDataPreprocessor`
orchestrates them into a coherent pipeline that converts raw FRED and
SEC DataFrames into a single modelling-ready table.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from loan_pricing.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_MATURITY_BUCKETS: dict[str, str] = {
    "1": "DGS1",
    "2": "DGS2",
    "3": "DGS5",
    "5": "DGS5",
    "7": "DGS10",
    "10": "DGS10",
    "30": "DGS30",
}
"""Map from loan maturity (years, as string) to the nearest Treasury series."""


# ---------------------------------------------------------------------------
# Named result for the train / val / test split
# ---------------------------------------------------------------------------


class TrainValTestSplit(NamedTuple):
    """Result container for :func:`split_train_val_test`."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


# ---------------------------------------------------------------------------
# Individual transformation steps
# ---------------------------------------------------------------------------


def pivot_fred_series(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-format FRED DataFrame to wide format.

    Args:
        df: Tidy DataFrame with columns ``[date, series_id, value]``.

    Returns:
        Wide DataFrame indexed by ``date`` with one column per series.
    """
    wide = df.pivot_table(
        index="date",
        columns="series_id",
        values="value",
        aggfunc="last",
    )
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Derive credit-relevant financial ratios from raw SEC data.

    The input is a *wide* DataFrame with one row per company-year and
    raw accounting items as columns (e.g. ``Revenues``,
    ``OperatingIncomeLoss``, ``Assets``, …).

    Args:
        df: Wide-format SEC data.  Must contain the columns listed in
            :data:`~loan_pricing.data.fetch_sec.REQUIRED_XBRL_TAGS`.

    Returns:
        A copy of *df* with the following ratio columns appended:
        ``debt_to_ebitda``, ``interest_coverage_ratio``,
        ``net_debt_to_equity``, ``fcf_to_debt``,
        ``revenue_growth_yoy``, ``ebitda_margin``, ``current_ratio``,
        ``altman_z_double_prime``.
    """
    out = df.copy()

    ebitda = (
        out.get("OperatingIncomeLoss", 0)
        + out.get("DepreciationAndAmortization", 0)
    )
    out["ebitda"] = ebitda

    long_term_debt = out.get("LongTermDebt", pd.Series(np.nan, index=out.index))
    cash = out.get(
        "CashAndCashEquivalentsAtCarryingValue",
        pd.Series(np.nan, index=out.index),
    )
    equity = out.get(
        "StockholdersEquity",
        pd.Series(np.nan, index=out.index),
    )
    assets = out.get("Assets", pd.Series(np.nan, index=out.index))
    liabilities = out.get("Liabilities", pd.Series(np.nan, index=out.index))
    revenues = out.get("Revenues", pd.Series(np.nan, index=out.index))
    interest = out.get("InterestExpense", pd.Series(np.nan, index=out.index))
    net_income = out.get("NetIncomeLoss", pd.Series(np.nan, index=out.index))
    retained = out.get(
        "RetainedEarningsAccumulatedDeficit",
        pd.Series(np.nan, index=out.index),
    )

    # Ratios -----------------------------------------------------------------
    out["debt_to_ebitda"] = long_term_debt / ebitda.replace(0, np.nan)
    out["interest_coverage_ratio"] = ebitda / interest.replace(0, np.nan)

    net_debt = long_term_debt - cash
    out["net_debt_to_equity"] = net_debt / equity.replace(0, np.nan)

    # Simplified FCF = EBITDA - CapEx (not available) ≈ OperatingIncomeLoss
    operating = out.get(
        "OperatingIncomeLoss", pd.Series(np.nan, index=out.index)
    )
    out["fcf_to_debt"] = operating / long_term_debt.replace(0, np.nan)

    # Year-over-year revenue growth (requires groupby company)
    if "cik" in out.columns:
        out["revenue_growth_yoy"] = out.groupby("cik")["Revenues"].pct_change()
    else:
        out["revenue_growth_yoy"] = revenues.pct_change()

    out["ebitda_margin"] = ebitda / revenues.replace(0, np.nan)

    current_assets = assets  # rough proxy when current assets unavailable
    current_liabilities = liabilities
    out["current_ratio"] = current_assets / current_liabilities.replace(0, np.nan)

    # Altman Z''-score for private firms (no market-cap term):
    # Z'' = 6.56 * (WC/TA) + 3.26 * (RE/TA) + 6.72 * (EBIT/TA) + 1.05 * (BV_E/TL)
    working_capital = current_assets - current_liabilities
    wc_ta = working_capital / assets.replace(0, np.nan)
    re_ta = retained / assets.replace(0, np.nan)
    ebit_ta = operating / assets.replace(0, np.nan)
    bve_tl = equity / liabilities.replace(0, np.nan)

    z_pp_constant = 6.56
    z_pp_re_coeff = 3.26
    z_pp_ebit_coeff = 6.72
    z_pp_bve_coeff = 1.05

    out["altman_z_double_prime"] = (
        z_pp_constant * wc_ta
        + z_pp_re_coeff * re_ta
        + z_pp_ebit_coeff * ebit_ta
        + z_pp_bve_coeff * bve_tl
    )

    return out


def _nearest_maturity_series(maturity_years: int) -> str:
    """Map a loan maturity to the nearest Treasury yield series.

    Args:
        maturity_years: Loan maturity in whole years.

    Returns:
        FRED series identifier (e.g. ``"DGS10"``).
    """
    key = str(maturity_years)
    if key in _MATURITY_BUCKETS:
        return _MATURITY_BUCKETS[key]
    # Fall back to nearest available bucket.
    available = sorted(int(k) for k in _MATURITY_BUCKETS)
    nearest = min(available, key=lambda x: abs(x - maturity_years))
    return _MATURITY_BUCKETS[str(nearest)]


def merge_loan_features(
    loans_df: pd.DataFrame,
    ratios_df: pd.DataFrame,
    fred_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join borrower ratios and macro data onto the loan table.

    Args:
        loans_df: Loan-level table with at least ``borrower_id``,
            ``date``, and ``maturity_years`` columns.
        ratios_df: Financial ratios keyed by ``cik`` and
            ``fiscal_year``.
        fred_df: *Wide-format* FRED data keyed by ``date``.

    Returns:
        Merged DataFrame with borrower ratios, macro variables, and
        the matched Treasury yield for each loan's maturity.
    """
    merged = loans_df.copy()

    # Merge financial ratios by borrower and year.
    if "cik" in merged.columns and "cik" in ratios_df.columns:
        merged["fiscal_year"] = pd.to_datetime(merged["date"]).dt.year
        ratio_cols = [
            c
            for c in ratios_df.columns
            if c not in ("cik", "fiscal_year", "ticker")
        ]
        merged = merged.merge(
            ratios_df[["cik", "fiscal_year"] + ratio_cols],
            on=["cik", "fiscal_year"],
            how="left",
        )

    # Merge macro variables by date.
    if not fred_df.empty:
        merged["date"] = pd.to_datetime(merged["date"]).dt.date
        fred_copy = fred_df.copy()
        fred_copy["date"] = pd.to_datetime(fred_copy["date"]).dt.date
        merged = merged.merge(fred_copy, on="date", how="left")

    # Match Treasury yield to loan maturity.
    if "maturity_years" in merged.columns:
        treasury_yields = []
        for _, row in merged.iterrows():
            series = _nearest_maturity_series(int(row["maturity_years"]))
            treasury_yields.append(row.get(series, np.nan))
        merged["treasury_yield_matched"] = treasury_yields

    return merged


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Literal["median", "knn"] = "median",
) -> pd.DataFrame:
    """Impute missing numeric values in *df*.

    Args:
        df: DataFrame that may contain ``NaN`` values.
        strategy: Imputation method — ``"median"`` fills with column
            medians; ``"knn"`` uses k-nearest-neighbours imputation.

    Returns:
        A new DataFrame with missing numerics imputed.  Non-numeric
        columns are passed through unchanged.
    """
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns.tolist()

    if strategy == "median":
        medians = out[numeric_cols].median()
        out[numeric_cols] = out[numeric_cols].fillna(medians)
    elif strategy == "knn":
        from sklearn.impute import KNNImputer

        imputer = KNNImputer()
        out[numeric_cols] = imputer.fit_transform(out[numeric_cols])
    else:
        msg = f"Unknown imputation strategy: {strategy!r}"
        raise ValueError(msg)

    return out


def split_train_val_test(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    time_column: str,
) -> TrainValTestSplit:
    """Time-aware train / validation / test split.

    Rows are sorted by *time_column* and then sliced so that training
    data always precedes validation data, which always precedes test
    data.  This prevents future information from leaking into earlier
    sets.

    Args:
        df: Full dataset.
        val_frac: Fraction of rows for the validation set.
        test_frac: Fraction of rows for the test set.
        time_column: Column name containing the temporal ordering key.

    Returns:
        A :class:`TrainValTestSplit` named tuple.

    Raises:
        ValueError: If the fractions are invalid.
    """
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        msg = (
            f"val_frac={val_frac} and test_frac={test_frac} must be "
            f"non-negative and sum to less than 1.0"
        )
        raise ValueError(msg)

    sorted_df = df.sort_values(time_column).reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * (1.0 - val_frac - test_frac))
    val_end = int(n * (1.0 - test_frac))

    return TrainValTestSplit(
        train=sorted_df.iloc[:train_end].copy(),
        validation=sorted_df.iloc[train_end:val_end].copy(),
        test=sorted_df.iloc[val_end:].copy(),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class LoanDataPreprocessor:
    """End-to-end preprocessor that chains the individual transform steps.

    Args:
        missing_strategy: Imputation strategy passed to
            :func:`handle_missing_values`.
        val_frac: Validation-set fraction.
        test_frac: Test-set fraction.
        time_column: Column used for temporal ordering in the split.
    """

    def __init__(
        self,
        missing_strategy: Literal["median", "knn"] = "median",
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        time_column: str = "date",
    ) -> None:
        self._missing_strategy = missing_strategy
        self._val_frac = val_frac
        self._test_frac = test_frac
        self._time_column = time_column

    def run(
        self,
        loans_df: pd.DataFrame,
        sec_wide_df: pd.DataFrame,
        fred_wide_df: pd.DataFrame,
    ) -> TrainValTestSplit:
        """Execute the full preprocessing pipeline.

        Args:
            loans_df: Raw loan-level table.
            sec_wide_df: Wide-format SEC financials (one row per
                company-year, columns are XBRL tag values).
            fred_wide_df: Wide-format FRED macro data (one row per
                date).

        Returns:
            A :class:`TrainValTestSplit` with imputed, merged data.
        """
        logger.info("Computing financial ratios")
        ratios = compute_financial_ratios(sec_wide_df)

        logger.info("Merging loan features")
        merged = merge_loan_features(loans_df, ratios, fred_wide_df)

        logger.info("Handling missing values (strategy=%s)", self._missing_strategy)
        clean = handle_missing_values(merged, strategy=self._missing_strategy)

        logger.info("Splitting into train / val / test")
        return split_train_val_test(
            clean,
            val_frac=self._val_frac,
            test_frac=self._test_frac,
            time_column=self._time_column,
        )
