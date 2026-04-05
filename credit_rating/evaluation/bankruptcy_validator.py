"""Bankruptcy prediction validator for credit rating models.

Validates model performance by examining rating trajectories and
shenanigans signals for companies in the years leading up to their
actual bankruptcy filing dates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from credit_rating.config.settings import CreditRatingSettings, RatingClass, RiskLevel
from credit_rating.domain.financial_statements import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from credit_rating.domain.rating import RatingPrediction
from credit_rating.domain.report import AnnualReport
from credit_rating.domain.shenanigans import ShenanigansReport
from credit_rating.features.altman import AltmanZScoreCalculator
from credit_rating.features.ratio_calculator import FinancialRatioCalculator
from credit_rating.models.protocols import CreditRatingModel
from credit_rating.shenanigans.beneish import BeneishMScoreDetector
from credit_rating.shenanigans.cash_flow import CashFlowShenanigansDetector
from credit_rating.shenanigans.earnings_manipulation import EarningsManipulationDetector
from credit_rating.shenanigans.report_builder import ShenanigansReportBuilder

logger = logging.getLogger(__name__)

# Number of fiscal years prior to bankruptcy to examine.
_LOOKBACK_YEARS = 3


# ------------------------------------------------------------------
# Report containers
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AnnualSignalSnapshot:
    """Model and shenanigans output for a single fiscal year.

    Args:
        fiscal_year: The fiscal year these signals cover.
        rating_prediction: Model's predicted rating (``None`` if
            prediction failed).
        shenanigans_report: Shenanigans analysis (``None`` if
            analysis could not be run).
        altman_z_score: Altman Z'' score for the year.
    """

    fiscal_year: int
    rating_prediction: Optional[RatingPrediction] = None
    shenanigans_report: Optional[ShenanigansReport] = None
    altman_z_score: Optional[float] = None


@dataclass(frozen=True)
class CompanyBankruptcyResult:
    """Validation result for a single bankrupt company.

    Args:
        ticker: Stock ticker symbol.
        bankruptcy_date: Date the company filed for bankruptcy.
        signal_timeline: Ordered list of annual snapshots from
            earliest to latest (up to ``_LOOKBACK_YEARS`` years
            prior to bankruptcy).
        final_rating: The model's last predicted rating before
            bankruptcy (``None`` if no predictions succeeded).
        flags_escalated: Whether the total flags raised increased
            over the lookback window.
    """

    ticker: str
    bankruptcy_date: date
    signal_timeline: List[AnnualSignalSnapshot] = field(default_factory=list)
    final_rating: Optional[RatingClass] = None
    flags_escalated: bool = False


@dataclass(frozen=True)
class BankruptcyValidationReport:
    """Aggregate validation report across all bankrupt companies.

    Args:
        company_results: Per-company validation results.
        companies_with_distress_signal: Number of companies where
            the model predicted CCC/CC or D in the final year.
        companies_with_escalating_flags: Number of companies where
            shenanigans flags increased over time.
        total_companies: Total number of companies validated.
    """

    company_results: List[CompanyBankruptcyResult] = field(
        default_factory=list,
    )
    companies_with_distress_signal: int = 0
    companies_with_escalating_flags: int = 0
    total_companies: int = 0


# ------------------------------------------------------------------
# Validator
# ------------------------------------------------------------------


class BankruptcyValidator:
    """Validate a credit rating model against known bankruptcies.

    For each company, retrieves financial data for the three years
    prior to the bankruptcy date, runs both the rating model and the
    shenanigans detection suite, and assembles a timeline of signals.

    Args:
        bankruptcy_cases: List of ``(ticker, bankruptcy_date)``
            pairs to validate.
        settings: Project-wide configuration.
    """

    def __init__(
        self,
        bankruptcy_cases: List[Tuple[str, date]],
        settings: Optional[CreditRatingSettings] = None,
    ) -> None:
        self._cases = bankruptcy_cases
        self._settings = settings or CreditRatingSettings()
        self._ratio_calculator = FinancialRatioCalculator()
        self._altman = AltmanZScoreCalculator(self._settings)
        self._beneish = BeneishMScoreDetector(self._settings)

    def validate(
        self,
        model: CreditRatingModel,
        shenanigans_runner: Optional[Any] = None,
    ) -> BankruptcyValidationReport:
        """Run the full validation across all bankruptcy cases.

        For each company the method:

        1. Loads financial statements for the three fiscal years
           preceding bankruptcy (via ``shenanigans_runner`` if
           provided, otherwise uses placeholder data).
        2. Runs the credit rating model on each year's report.
        3. Runs the shenanigans detection suite on each year.
        4. Assembles a chronological signal timeline.

        Args:
            model: A model satisfying the :class:`CreditRatingModel`
                protocol.
            shenanigans_runner: Optional callable that, given a
                ticker and fiscal year, returns a
                :class:`ShenanigansReport`.  When ``None`` the
                validator builds reports from the detectors directly.

        Returns:
            A :class:`BankruptcyValidationReport` with per-company
            timelines and aggregate statistics.
        """
        company_results: List[CompanyBankruptcyResult] = []
        distress_count = 0
        escalation_count = 0

        for ticker, bankruptcy_date in self._cases:
            result = self._validate_company(
                ticker, bankruptcy_date, model, shenanigans_runner,
            )
            company_results.append(result)

            if result.final_rating is not None and result.final_rating.value >= RatingClass.CCC_CC:
                distress_count += 1
            if result.flags_escalated:
                escalation_count += 1

        report = BankruptcyValidationReport(
            company_results=company_results,
            companies_with_distress_signal=distress_count,
            companies_with_escalating_flags=escalation_count,
            total_companies=len(self._cases),
        )
        logger.info(
            "Bankruptcy validation complete: %d/%d flagged distress, "
            "%d/%d showed escalating flags.",
            distress_count,
            len(self._cases),
            escalation_count,
            len(self._cases),
        )
        return report

    # ------------------------------------------------------------------
    # Per-company validation
    # ------------------------------------------------------------------

    def _validate_company(
        self,
        ticker: str,
        bankruptcy_date: date,
        model: CreditRatingModel,
        shenanigans_runner: Optional[Any],
    ) -> CompanyBankruptcyResult:
        """Build a signal timeline for a single company.

        Args:
            ticker: Stock ticker.
            bankruptcy_date: Known bankruptcy filing date.
            model: Credit rating model.
            shenanigans_runner: Optional shenanigans callable.

        Returns:
            A :class:`CompanyBankruptcyResult`.
        """
        bankruptcy_year = bankruptcy_date.year
        fiscal_years = list(
            range(bankruptcy_year - _LOOKBACK_YEARS, bankruptcy_year)
        )

        snapshots: List[AnnualSignalSnapshot] = []
        prior_statements: Optional[FinancialStatements] = None

        for fy in fiscal_years:
            statements = self._load_statements(ticker, fy)
            report = AnnualReport(
                ticker=ticker,
                fiscal_year=fy,
                financial_statements=statements,
            )

            # Rating prediction
            prediction: Optional[RatingPrediction] = None
            try:
                prediction = model.predict(report)
            except Exception:
                logger.warning(
                    "Model prediction failed for %s FY%d", ticker, fy,
                )

            # Altman Z''-Score
            z_score: Optional[float] = None
            try:
                z_score = self._altman.calculate_z_prime_prime(statements)
            except Exception:
                logger.warning(
                    "Altman Z'' failed for %s FY%d", ticker, fy,
                )

            # Shenanigans report
            shenanigans: Optional[ShenanigansReport] = None
            if shenanigans_runner is not None:
                try:
                    shenanigans = shenanigans_runner(ticker, fy)
                except Exception:
                    logger.warning(
                        "Shenanigans runner failed for %s FY%d",
                        ticker, fy,
                    )
            else:
                shenanigans = self._run_shenanigans(
                    statements, prior_statements,
                )

            snapshots.append(
                AnnualSignalSnapshot(
                    fiscal_year=fy,
                    rating_prediction=prediction,
                    shenanigans_report=shenanigans,
                    altman_z_score=z_score,
                )
            )
            prior_statements = statements

        # Determine final rating
        final_rating: Optional[RatingClass] = None
        for snapshot in reversed(snapshots):
            if snapshot.rating_prediction is not None:
                final_rating = snapshot.rating_prediction.rating
                break

        # Determine if flags escalated
        flags_escalated = self._check_flag_escalation(snapshots)

        return CompanyBankruptcyResult(
            ticker=ticker,
            bankruptcy_date=bankruptcy_date,
            signal_timeline=snapshots,
            final_rating=final_rating,
            flags_escalated=flags_escalated,
        )

    # ------------------------------------------------------------------
    # Shenanigans detection
    # ------------------------------------------------------------------

    def _run_shenanigans(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements],
    ) -> ShenanigansReport:
        """Run the full shenanigans detection suite.

        Args:
            current: Current period financial statements.
            prior: Prior period statements (may be ``None``).

        Returns:
            A :class:`ShenanigansReport`.
        """
        builder = ShenanigansReportBuilder()

        # Beneish M-Score (requires prior period)
        if prior is not None:
            beneish_score = self._beneish.calculate(current, prior)
            builder.with_beneish(beneish_score)

        # Earnings manipulation signals
        ems_detector = EarningsManipulationDetector(current, prior)
        builder.with_earnings_signals(ems_detector.detect_all())

        # Cash flow shenanigans
        cfs_detector = CashFlowShenanigansDetector(current, prior)
        builder.with_cash_flow_signals(cfs_detector.detect_all())

        return builder.build()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_statements(ticker: str, fiscal_year: int) -> FinancialStatements:
        """Load or construct financial statements for a given year.

        In a production system this would fetch from a database or
        SEC EDGAR.  The current implementation returns a placeholder
        set of zero-valued statements so the pipeline can be
        exercised end-to-end without external data.

        Args:
            ticker: Stock ticker.
            fiscal_year: Fiscal year to load.

        Returns:
            A :class:`FinancialStatements` instance.
        """
        logger.debug(
            "Loading statements for %s FY%d (placeholder)", ticker, fiscal_year,
        )
        income = IncomeStatement(
            total_revenue=0.0,
            cost_of_goods_sold=0.0,
            total_operating_expenses=0.0,
            selling_general_admin=0.0,
            depreciation_expense=0.0,
            interest_expense=0.0,
            income_tax_expense=0.0,
            net_income=0.0,
        )
        balance = BalanceSheet(
            cash_and_equivalents=0.0,
            short_term_investments=0.0,
            net_receivables=0.0,
            inventory=0.0,
            total_current_assets=0.0,
            net_property_plant_equipment=0.0,
            total_assets=0.0,
            accounts_payable=0.0,
            short_term_debt=0.0,
            total_current_liabilities=0.0,
            long_term_debt=0.0,
            total_liabilities=0.0,
            total_equity=0.0,
            retained_earnings=0.0,
        )
        cash_flow = CashFlowStatement(
            depreciation_and_amortization=0.0,
            cash_from_operations=0.0,
            capital_expenditures=0.0,
            cash_from_investing=0.0,
            cash_from_financing=0.0,
            net_change_in_cash=0.0,
        )
        return FinancialStatements(
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cash_flow,
            fiscal_year=fiscal_year,
        )

    @staticmethod
    def _check_flag_escalation(
        snapshots: List[AnnualSignalSnapshot],
    ) -> bool:
        """Check whether shenanigans flags increased over time.

        Args:
            snapshots: Chronologically ordered annual snapshots.

        Returns:
            ``True`` if the total flag count in the final year
            exceeds that in the first year.
        """
        flag_counts: List[int] = []
        for snapshot in snapshots:
            if snapshot.shenanigans_report is not None:
                flag_counts.append(
                    snapshot.shenanigans_report.total_flags_raised,
                )

        if len(flag_counts) < 2:
            return False
        return flag_counts[-1] > flag_counts[0]
