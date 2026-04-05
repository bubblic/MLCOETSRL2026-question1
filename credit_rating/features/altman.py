"""Altman Z-Score and Z''-Score calculators.

The original Z-Score (1968) is designed for manufacturing firms.
The Z''-Score variant works for non-manufacturing and emerging-market
firms by dropping the sales/assets ratio.

All coefficients are sourced from :class:`CreditRatingSettings`.
"""

from __future__ import annotations

from credit_rating.config.settings import AltmanZone, CreditRatingSettings
from credit_rating.domain.features import FinancialRatios
from credit_rating.domain.financial_statements import FinancialStatements


class AltmanZScoreCalculator:
    """Calculate Altman Z-Score and Z''-Score.

    Args:
        settings: Configuration supplying all coefficients and
            zone thresholds.
    """

    def __init__(self, settings: CreditRatingSettings | None = None) -> None:
        self._s = settings or CreditRatingSettings()

    def calculate_z(self, statements: FinancialStatements) -> float:
        """Compute the original Altman Z-Score (manufacturing firms).

        ``Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5``

        where:
            X1 = Working Capital / Total Assets
            X2 = Retained Earnings / Total Assets
            X3 = EBIT / Total Assets
            X4 = Book Equity / Total Liabilities
            X5 = Sales / Total Assets

        Args:
            statements: Financial statements for one period.

        Returns:
            The Z-Score as a float.
        """
        bs = statements.balance_sheet
        inc = statements.income_statement
        ta = bs.total_assets
        if abs(ta) < 1e-12:
            return 0.0

        x1 = bs.computed_working_capital() / ta
        x2 = bs.retained_earnings / ta
        x3 = inc.computed_ebit() / ta
        x4 = _safe_divide(bs.total_equity, bs.total_liabilities)
        x5 = inc.total_revenue / ta

        return (
            self._s.altman_x1_coefficient * x1
            + self._s.altman_x2_coefficient * x2
            + self._s.altman_x3_coefficient * x3
            + self._s.altman_x4_coefficient * x4
            + self._s.altman_x5_coefficient * x5
        )

    def calculate_z_prime_prime(
        self,
        statements: FinancialStatements,
    ) -> float:
        """Compute the Altman Z''-Score (non-manufacturing firms).

        ``Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4 + 3.25``

        Same X1..X4 as original Z but without X5 (sales/assets),
        plus a constant term.

        Args:
            statements: Financial statements for one period.

        Returns:
            The Z''-Score as a float.
        """
        bs = statements.balance_sheet
        inc = statements.income_statement
        ta = bs.total_assets
        if abs(ta) < 1e-12:
            return 0.0

        x1 = bs.computed_working_capital() / ta
        x2 = bs.retained_earnings / ta
        x3 = inc.computed_ebit() / ta
        x4 = _safe_divide(bs.total_equity, bs.total_liabilities)

        return (
            self._s.altman_zpp_x1_coefficient * x1
            + self._s.altman_zpp_x2_coefficient * x2
            + self._s.altman_zpp_x3_coefficient * x3
            + self._s.altman_zpp_x4_coefficient * x4
            + self._s.altman_zpp_constant
        )

    def classify_z(self, z_score: float) -> AltmanZone:
        """Classify a Z-Score into Safe / Grey / Distress.

        Args:
            z_score: A computed Z or Z'' score.

        Returns:
            The :class:`AltmanZone` classification.
        """
        if z_score > self._s.altman_safe_threshold:
            return AltmanZone.SAFE
        if z_score < self._s.altman_distress_threshold:
            return AltmanZone.DISTRESS
        return AltmanZone.GREY

    def classify_z_prime_prime(self, z_score: float) -> AltmanZone:
        """Classify a Z''-Score into Safe / Grey / Distress.

        Args:
            z_score: A computed Z'' score.

        Returns:
            The :class:`AltmanZone` classification.
        """
        if z_score > self._s.altman_zpp_safe_threshold:
            return AltmanZone.SAFE
        if z_score < self._s.altman_zpp_distress_threshold:
            return AltmanZone.DISTRESS
        return AltmanZone.GREY


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide with zero-protection."""
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator
