"""Beneish M-Score earnings manipulation detector.

Computes the eight-variable Beneish model to flag likely earnings
manipulators.  Each index variable has a dedicated method with a
docstring explaining its economic interpretation.  All coefficients
are sourced from :class:`CreditRatingSettings`.
"""

from __future__ import annotations

import logging
from typing import Optional

from credit_rating.config.settings import CreditRatingSettings
from credit_rating.domain.financial_statements import FinancialStatements
from credit_rating.domain.shenanigans import BeneishScore

logger = logging.getLogger(__name__)


class BeneishMScoreDetector:
    """Compute the Beneish M-Score from two consecutive periods.

    Args:
        settings: Configuration supplying all eight coefficients
            and the manipulation threshold.
    """

    def __init__(
        self,
        settings: Optional[CreditRatingSettings] = None,
    ) -> None:
        self._s = settings or CreditRatingSettings()

    def calculate(
        self,
        current: FinancialStatements,
        prior: FinancialStatements,
    ) -> BeneishScore:
        """Compute all eight Beneish variables and the M-Score.

        Args:
            current: Current period financial statements.
            prior: Prior period financial statements.

        Returns:
            A frozen :class:`BeneishScore` with all variables and
            the manipulation flag.
        """
        dsri = self._days_sales_receivables_index(current, prior)
        gmi = self._gross_margin_index(current, prior)
        aqi = self._asset_quality_index(current, prior)
        sgi = self._sales_growth_index(current, prior)
        depi = self._depreciation_index(current, prior)
        sgai = self._sga_index(current, prior)
        lvgi = self._leverage_index(current, prior)
        tata = self._total_accruals_to_total_assets(current)

        m_score = (
            self._s.beneish_constant
            + self._s.beneish_dsri_coeff * dsri
            + self._s.beneish_gmi_coeff * gmi
            + self._s.beneish_aqi_coeff * aqi
            + self._s.beneish_sgi_coeff * sgi
            + self._s.beneish_depi_coeff * depi
            + self._s.beneish_sgai_coeff * sgai
            + self._s.beneish_lvgi_coeff * lvgi
            + self._s.beneish_tata_coeff * tata
        )

        return BeneishScore(
            dsri=dsri,
            gmi=gmi,
            aqi=aqi,
            sgi=sgi,
            depi=depi,
            sgai=sgai,
            lvgi=lvgi,
            tata=tata,
            m_score=m_score,
            is_likely_manipulator=m_score > self._s.beneish_manipulation_threshold,
        )

    @staticmethod
    def _days_sales_receivables_index(
        current: FinancialStatements,
        prior: FinancialStatements,
    ) -> float:
        """DSRI: measures whether receivables grew faster than revenue.

        A large increase suggests revenue may be recognised prematurely.
        """
        cur_ratio = _safe_divide(
            current.balance_sheet.net_receivables,
            current.income_statement.total_revenue,
        )
        pri_ratio = _safe_divide(
            prior.balance_sheet.net_receivables,
            prior.income_statement.total_revenue,
        )
        return _safe_divide(cur_ratio, pri_ratio)

    @staticmethod
    def _gross_margin_index(
        current: FinancialStatements,
        prior: FinancialStatements,
    ) -> float:
        """GMI: measures deterioration in gross margins.

        A ratio > 1 signals worsening margins, increasing
        incentive to manipulate earnings.
        """
        cur_margin = _gross_margin(current)
        pri_margin = _gross_margin(prior)
        return _safe_divide(pri_margin, cur_margin)

    @staticmethod
    def _asset_quality_index(
        current: FinancialStatements,
        prior: FinancialStatements,
    ) -> float:
        """AQI: measures the proportion of non-hard assets.

        A rising ratio suggests increasing capitalisation of costs
        that should be expensed.
        """
        cur_aq = _asset_quality(current)
        pri_aq = _asset_quality(prior)
        return _safe_divide(cur_aq, pri_aq)

    @staticmethod
    def _sales_growth_index(
        current: FinancialStatements,
        prior: FinancialStatements,
    ) -> float:
        """SGI: measures year-over-year revenue growth.

        High-growth firms face more pressure to sustain growth
        and are more prone to manipulation.
        """
        return _safe_divide(
            current.income_statement.total_revenue,
            prior.income_statement.total_revenue,
        )

    @staticmethod
    def _depreciation_index(
        current: FinancialStatements,
        prior: FinancialStatements,
    ) -> float:
        """DEPI: measures the rate of depreciation.

        A ratio > 1 means the firm is depreciating assets more
        slowly, boosting reported earnings.
        """
        cur_rate = _depreciation_rate(current)
        pri_rate = _depreciation_rate(prior)
        return _safe_divide(pri_rate, cur_rate)

    @staticmethod
    def _sga_index(
        current: FinancialStatements,
        prior: FinancialStatements,
    ) -> float:
        """SGAI: measures SG&A expense relative to revenue.

        A disproportionate rise in SG&A may signal declining
        operational efficiency.
        """
        cur_ratio = _safe_divide(
            current.income_statement.selling_general_admin,
            current.income_statement.total_revenue,
        )
        pri_ratio = _safe_divide(
            prior.income_statement.selling_general_admin,
            prior.income_statement.total_revenue,
        )
        return _safe_divide(cur_ratio, pri_ratio)

    @staticmethod
    def _leverage_index(
        current: FinancialStatements,
        prior: FinancialStatements,
    ) -> float:
        """LVGI: measures the change in leverage.

        Increasing leverage (debt/assets) raises default risk and
        the incentive to manipulate earnings.
        """
        cur_leverage = _safe_divide(
            current.balance_sheet.total_liabilities,
            current.balance_sheet.total_assets,
        )
        pri_leverage = _safe_divide(
            prior.balance_sheet.total_liabilities,
            prior.balance_sheet.total_assets,
        )
        return _safe_divide(cur_leverage, pri_leverage)

    @staticmethod
    def _total_accruals_to_total_assets(
        current: FinancialStatements,
    ) -> float:
        """TATA: measures the accrual component of earnings.

        High accruals relative to cash flow signal that earnings
        quality is poor.
        """
        accruals = (
            current.income_statement.net_income
            - current.cash_flow.cash_from_operations
        )
        return _safe_divide(accruals, current.balance_sheet.total_assets)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide with zero-protection, returning 0.0 on zero denominator."""
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _gross_margin(stmts: FinancialStatements) -> float:
    """Compute gross margin from financial statements."""
    gp = stmts.income_statement.computed_gross_profit()
    return _safe_divide(gp, stmts.income_statement.total_revenue)


def _asset_quality(stmts: FinancialStatements) -> float:
    """Compute asset quality (1 - hard assets / total assets)."""
    bs = stmts.balance_sheet
    hard_assets = bs.total_current_assets + bs.net_property_plant_equipment
    return 1.0 - _safe_divide(hard_assets, bs.total_assets)


def _depreciation_rate(stmts: FinancialStatements) -> float:
    """Compute depreciation / (depreciation + PPE)."""
    dep = stmts.income_statement.depreciation_expense
    ppe = stmts.balance_sheet.net_property_plant_equipment
    return _safe_divide(dep, dep + ppe)
