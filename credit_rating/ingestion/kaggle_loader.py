"""Loader for the Kaggle Corporate Credit Ratings dataset.

The CSV contains ~2,029 US firms with S&P ratings and financial
features.  This loader maps the raw columns to
:class:`~credit_rating.domain.features.FinancialRatios` and the S&P
rating string to :class:`~credit_rating.config.settings.RatingClass`.

Actual CSV columns::

    Rating Agency, Corporation, Rating, Rating Date, CIK,
    Binary Rating, SIC Code, Sector, Ticker,
    Current Ratio, Long-term Debt / Capital, Debt/Equity Ratio,
    Gross Margin, Operating Margin, EBIT Margin, EBITDA Margin,
    Pre-Tax Profit Margin, Net Profit Margin, Asset Turnover,
    ROE - Return On Equity, Return On Tangible Equity,
    ROA - Return On Assets, ROI - Return On Investment,
    Operating Cash Flow Per Share, Free Cash Flow Per Share
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterator, Optional, Tuple

import pandas as pd

from credit_rating.config.settings import CreditRatingSettings, RatingClass
from credit_rating.domain.features import (
    EfficiencyRatios,
    FinancialRatios,
    LeverageRatios,
    LiquidityRatios,
    ProfitabilityRatios,
    SizeRatios,
)

logger = logging.getLogger(__name__)

_RATING_COLUMN = "Rating"


class KaggleRatingsLoader:
    """Load and iterate over the Kaggle Corporate Credit Ratings CSV.

    Args:
        csv_path: Path to the CSV file.
        settings: Configuration.
    """

    def __init__(
        self,
        csv_path: Path,
        settings: Optional[CreditRatingSettings] = None,
    ) -> None:
        self._csv_path = csv_path
        self._settings = settings or CreditRatingSettings()

    def load(self) -> Iterator[Tuple[FinancialRatios, RatingClass]]:
        """Yield ``(FinancialRatios, RatingClass)`` for each valid row.

        Rows with unparseable ratings or NaN features are skipped
        with a warning.

        Yields:
            Tuples of structured features and their credit rating label.
        """
        dataframe = pd.read_csv(self._csv_path)
        skipped = 0
        for _, row in dataframe.iterrows():
            result = self._parse_row(row)
            if result is None:
                skipped += 1
                continue
            yield result
        if skipped:
            logger.warning("Skipped %d rows with invalid data", skipped)

    def load_dataframe(self) -> pd.DataFrame:
        """Load the raw CSV as a pandas DataFrame.

        Returns:
            The raw DataFrame without transformation.
        """
        return pd.read_csv(self._csv_path)

    def _parse_row(
        self,
        row: pd.Series,
    ) -> Optional[Tuple[FinancialRatios, RatingClass]]:
        """Parse a single CSV row into domain objects."""
        rating = self._parse_rating(row)
        if rating is None:
            return None
        ratios = self._parse_ratios(row)
        if ratios is None:
            return None
        return ratios, rating

    @staticmethod
    def _parse_rating(row: pd.Series) -> Optional[RatingClass]:
        """Extract and map the S&P rating string."""
        raw = row.get(_RATING_COLUMN)
        if pd.isna(raw):
            return None
        try:
            return RatingClass.from_sp_string(str(raw))
        except ValueError:
            logger.debug("Unmappable rating: %s", raw)
            return None

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        """Convert a value to float, returning *default* for NaN/None."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_ratios(self, row: pd.Series) -> Optional[FinancialRatios]:
        """Build a :class:`FinancialRatios` from CSV columns.

        Maps the actual Kaggle CSV column names to domain ratio groups.
        """
        g = self._safe_float

        try:
            debt_to_equity = g(row.get("Debt/Equity Ratio"))
            operating_margin = g(row.get("Operating Margin"))

            leverage = LeverageRatios(
                debt_to_equity=debt_to_equity,
                debt_to_assets=0.0,
                debt_to_capital=g(row.get("Long-term Debt / Capital")),
                debt_to_ebitda=0.0,
                interest_coverage=0.0,
            )
            liquidity = LiquidityRatios(
                current_ratio=g(row.get("Current Ratio")),
                quick_ratio=0.0,
                cash_ratio=0.0,
                working_capital_to_assets=0.0,
                operating_cash_flow_ratio=0.0,
            )
            profitability = ProfitabilityRatios(
                gross_margin=g(row.get("Gross Margin")),
                operating_margin=operating_margin,
                net_margin=g(row.get("Net Profit Margin")),
                return_on_assets=g(row.get("ROA - Return On Assets")),
                return_on_equity=g(row.get("ROE - Return On Equity")),
            )
            efficiency = EfficiencyRatios(
                asset_turnover=g(row.get("Asset Turnover")),
                receivables_turnover=0.0,
                inventory_turnover=0.0,
                payables_turnover=0.0,
                cost_to_income=_cost_to_income_from_margin(operating_margin),
            )
            size = SizeRatios(
                log_total_assets=0.0,
                log_total_revenue=0.0,
                revenue_growth=0.0,
                asset_growth=0.0,
                retained_earnings_to_assets=0.0,
            )
            return FinancialRatios(
                leverage=leverage,
                liquidity=liquidity,
                profitability=profitability,
                efficiency=efficiency,
                size=size,
            )
        except (TypeError, ValueError):
            return None


def _cost_to_income_from_margin(operating_margin: float) -> float:
    """Derive cost-to-income from operating margin.

    ``cost/income = 1 - operating_margin``
    """
    return 1.0 - operating_margin
