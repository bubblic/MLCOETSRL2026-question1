"""Shared fixtures for loan-pricing tests."""

from __future__ import annotations

# TensorFlow must be imported before numpy/pandas on Windows to avoid a
# DLL initialisation race in the native runtime.  Importing it here in
# conftest ensures it is loaded first during pytest collection.
try:
    import tensorflow as tf  # noqa: F401
except ImportError:
    pass

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loan_pricing.config import ProjectConfig


# ---------------------------------------------------------------------------
# Synthetic FRED data (long format)
# ---------------------------------------------------------------------------

FRED_SERIES_IDS = [
    "DGS1",
    "DGS2",
    "DGS5",
    "DGS10",
    "DGS30",
    "BAMLC0A0CM",
    "BAMLH0A0HYM2",
    "VIXCLS",
    "T10Y2Y",
    "UNRATE",
    "A191RL1Q225SBEA",
]


@pytest.fixture()
def synthetic_fred_df() -> pd.DataFrame:
    """Tidy long-format DataFrame mimicking FRED fetch output.

    Columns: ``[date, series_id, value]``.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(
        start="2015-01-02",
        periods=50,
        freq="B",
    )
    rows: list[dict[str, object]] = []
    for series_id in FRED_SERIES_IDS:
        for date in dates:
            rows.append(
                {
                    "date": date.date(),
                    "series_id": series_id,
                    "value": float(rng.uniform(0.5, 10.0)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic SEC data (long format)
# ---------------------------------------------------------------------------

SEC_TAGS = [
    "Revenues",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "RetainedEarningsAccumulatedDeficit",
    "CashAndCashEquivalentsAtCarryingValue",
    "LongTermDebt",
    "InterestExpense",
    "DepreciationAndAmortization",
]


@pytest.fixture()
def synthetic_sec_df() -> pd.DataFrame:
    """Tidy long-format DataFrame mimicking SEC EDGAR fetch output.

    Columns: ``[cik, ticker, fiscal_year, tag, value]``.
    """
    rng = np.random.default_rng(99)
    ciks = [("0000320193", "AAPL"), ("0000789019", "MSFT")]
    years = range(2015, 2025)
    rows: list[dict[str, object]] = []
    for cik, ticker in ciks:
        for year in years:
            for tag in SEC_TAGS:
                rows.append(
                    {
                        "cik": cik,
                        "ticker": ticker,
                        "fiscal_year": year,
                        "tag": tag,
                        "value": float(rng.uniform(1e6, 1e11)),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic loan-level DataFrame (wide, model-ready shape)
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_loan_df() -> pd.DataFrame:
    """50-row wide-format DataFrame with realistic loan feature columns."""
    rng = np.random.default_rng(7)
    n_rows = 50
    return pd.DataFrame(
        {
            "borrower_id": [f"B{i:04d}" for i in range(n_rows)],
            "date": pd.date_range("2018-01-01", periods=n_rows, freq="QS"),
            "maturity_years": rng.choice([1, 2, 3, 5, 7, 10], size=n_rows),
            "loan_size_mm": rng.uniform(10, 500, size=n_rows).round(1),
            "is_secured": rng.choice([True, False], size=n_rows),
            "debt_to_ebitda": rng.uniform(1, 8, size=n_rows).round(2),
            "interest_coverage_ratio": rng.uniform(1, 15, size=n_rows).round(2),
            "net_debt_to_equity": rng.uniform(-0.5, 5, size=n_rows).round(2),
            "fcf_to_debt": rng.uniform(-0.2, 0.6, size=n_rows).round(3),
            "revenue_growth_yoy": rng.uniform(-0.3, 0.5, size=n_rows).round(3),
            "ebitda_margin": rng.uniform(0.05, 0.5, size=n_rows).round(3),
            "current_ratio": rng.uniform(0.5, 4, size=n_rows).round(2),
            "altman_z_double_prime": rng.uniform(-1, 8, size=n_rows).round(2),
            "treasury_yield": rng.uniform(1, 5, size=n_rows).round(3),
            "ig_oas": rng.uniform(80, 250, size=n_rows).round(1),
            "hy_oas": rng.uniform(300, 800, size=n_rows).round(1),
            "vix": rng.uniform(10, 40, size=n_rows).round(1),
            "yield_curve_slope": rng.uniform(-0.5, 2.5, size=n_rows).round(3),
            "unemployment_rate": rng.uniform(3, 10, size=n_rows).round(1),
            "gdp_growth": rng.uniform(-2, 6, size=n_rows).round(2),
            "credit_spread_bps": rng.uniform(50, 600, size=n_rows).round(1),
            "defaulted": rng.choice([0, 1], size=n_rows, p=[0.92, 0.08]),
        }
    )


# ---------------------------------------------------------------------------
# Temporary artefact directory and config fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_artifacts_dir(tmp_path: Path) -> Path:
    """Create a temporary directory tree mirroring ``loan_pricing/artifacts/``."""
    for sub in (
        "raw/fred",
        "raw/sec",
        "processed/figures",
        "processed/tables",
        "processed/models",
    ):
        (tmp_path / sub).mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def project_config(tmp_artifacts_dir: Path) -> ProjectConfig:
    """A ``ProjectConfig`` whose paths all point at a temp directory."""
    return ProjectConfig(
        artifacts_dir=tmp_artifacts_dir,
        raw_fred_dir=tmp_artifacts_dir / "raw" / "fred",
        raw_sec_dir=tmp_artifacts_dir / "raw" / "sec",
        figures_dir=tmp_artifacts_dir / "processed" / "figures",
        tables_dir=tmp_artifacts_dir / "processed" / "tables",
        models_dir=tmp_artifacts_dir / "processed" / "models",
    )
