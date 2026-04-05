"""Tests for the Kaggle Corporate Credit Ratings CSV loader."""

from __future__ import annotations

import math
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from credit_rating.config.settings import RatingClass
from credit_rating.ingestion.kaggle_loader import KaggleRatingsLoader

# Matches the actual Kaggle CSV column headers
_SAMPLE_CSV = """\
Rating Agency,Corporation,Rating,Rating Date,CIK,Binary Rating,SIC Code,Sector,Ticker,Current Ratio,Long-term Debt / Capital,Debt/Equity Ratio,Gross Margin,Operating Margin,EBIT Margin,EBITDA Margin,Pre-Tax Profit Margin,Net Profit Margin,Asset Turnover,ROE - Return On Equity,Return On Tangible Equity,ROA - Return On Assets,ROI - Return On Investment,Operating Cash Flow Per Share,Free Cash Flow Per Share
S&P,Apple Inc.,AA+,2023-01-15,320193,1,3571,Technology,AAPL,0.988,0.611,1.786,0.438,0.302,0.335,0.362,0.281,0.253,1.087,1.561,1.788,0.275,0.311,6.42,5.87
S&P,Ford Motor Co.,BBB-,2023-02-01,37996,1,3711,Automotive,F,1.203,0.732,3.420,0.155,0.048,0.049,0.101,0.032,0.018,0.623,0.112,0.143,0.012,0.028,2.15,0.89
S&P,Default Corp.,D,2023-03-01,99999,0,1234,Other,DFLT,0.312,0.950,15.200,-0.120,-0.450,-0.440,-0.350,-0.510,-0.580,0.210,-2.100,-3.500,-0.450,-0.380,-1.20,-2.50
S&P,Bad Data Inc.,ZZZZZ,2023-04-01,11111,0,9999,Other,BAD,1.0,0.5,1.0,0.5,0.3,0.3,0.3,0.3,0.2,0.8,0.2,0.3,0.1,0.1,1.0,0.5
"""


@pytest.fixture
def csv_path(tmp_path) -> Path:
    """Write sample CSV to a temp file."""
    path = tmp_path / "corporate_ratings.csv"
    path.write_text(_SAMPLE_CSV)
    return path


@pytest.fixture
def loader(csv_path) -> KaggleRatingsLoader:
    return KaggleRatingsLoader(csv_path)


class TestKaggleLoader:
    """Tests for KaggleRatingsLoader with actual CSV column format."""

    def test_loads_valid_rows(self, loader):
        results = list(loader.load())
        assert len(results) == 3  # 4 rows minus 1 bad rating

    def test_rating_mapping(self, loader):
        results = list(loader.load())
        ratings = [r[1] for r in results]
        assert RatingClass.AAA_AA in ratings  # AA+
        assert RatingClass.BBB in ratings      # BBB-
        assert RatingClass.D in ratings         # D

    def test_skips_unmappable_rating(self, loader):
        results = list(loader.load())
        tickers_loaded = set()
        for ratios, rating in results:
            # BAD with rating "ZZZZZ" should be skipped
            pass
        assert len(results) == 3

    def test_current_ratio_parsed(self, loader):
        results = list(loader.load())
        apple_ratios = results[0][0]
        assert apple_ratios.liquidity.current_ratio == pytest.approx(0.988)

    def test_debt_to_equity_parsed(self, loader):
        results = list(loader.load())
        apple_ratios = results[0][0]
        assert apple_ratios.leverage.debt_to_equity == pytest.approx(1.786)

    def test_debt_to_capital_parsed(self, loader):
        results = list(loader.load())
        apple_ratios = results[0][0]
        assert apple_ratios.leverage.debt_to_capital == pytest.approx(0.611)

    def test_profitability_ratios_parsed(self, loader):
        results = list(loader.load())
        apple_ratios = results[0][0]
        assert apple_ratios.profitability.gross_margin == pytest.approx(0.438)
        assert apple_ratios.profitability.operating_margin == pytest.approx(0.302)
        assert apple_ratios.profitability.net_margin == pytest.approx(0.253)
        assert apple_ratios.profitability.return_on_assets == pytest.approx(0.275)
        assert apple_ratios.profitability.return_on_equity == pytest.approx(1.561)

    def test_efficiency_ratios_parsed(self, loader):
        results = list(loader.load())
        apple_ratios = results[0][0]
        assert apple_ratios.efficiency.asset_turnover == pytest.approx(1.087)
        assert apple_ratios.efficiency.cost_to_income == pytest.approx(1.0 - 0.302)

    def test_feature_count_is_25(self, loader):
        results = list(loader.load())
        assert len(results[0][0]) == 25

    def test_no_nan_in_features(self, loader):
        for ratios, _ in loader.load():
            for value in ratios:
                assert not math.isnan(value), f"NaN in features: {ratios.to_dict()}"

    def test_negative_margins_handled(self, loader):
        """Distressed company with negative margins should load fine."""
        results = list(loader.load())
        default_ratios = results[2][0]  # D-rated company
        assert default_ratios.profitability.gross_margin == pytest.approx(-0.120)
        assert default_ratios.profitability.operating_margin == pytest.approx(-0.450)

    def test_load_dataframe(self, loader):
        df = loader.load_dataframe()
        assert len(df) == 4
        assert "Rating" in df.columns
        assert "Current Ratio" in df.columns
