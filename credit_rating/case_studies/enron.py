"""Enron 2000 case study.

Constructs Enron's approximate FY1999--2000 financial data from the
10-K filing, runs ratio calculation, Altman Z-Score, the hybrid
rating model, and the full shenanigans detection suite.  The
expected result is an M-Score above -2.22, triggering the
manipulation flag.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from credit_rating.config.settings import (
    AltmanZone,
    CreditRatingSettings,
    RatingClass,
    RiskLevel,
)
from credit_rating.domain.financial_statements import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from credit_rating.domain.report import AnnualReport
from credit_rating.features.altman import AltmanZScoreCalculator
from credit_rating.features.ratio_calculator import FinancialRatioCalculator
from credit_rating.shenanigans.beneish import BeneishMScoreDetector
from credit_rating.shenanigans.cash_flow import CashFlowShenanigansDetector
from credit_rating.shenanigans.earnings_manipulation import EarningsManipulationDetector
from credit_rating.shenanigans.report_builder import ShenanigansReportBuilder

logger = logging.getLogger(__name__)


class EnronAnalysis:
    """End-to-end case study of Enron Corp (2001 bankruptcy).

    Demonstrates the credit rating system's ability to detect
    earnings manipulation signals using Enron's FY2000 10-K data.
    The Beneish M-Score is expected to exceed the -2.22 threshold,
    correctly flagging Enron as a likely earnings manipulator.

    Args:
        settings: Project-wide configuration.  When ``None`` a
            default :class:`CreditRatingSettings` is used.
    """

    def __init__(
        self,
        settings: Optional[CreditRatingSettings] = None,
    ) -> None:
        self._settings = settings or CreditRatingSettings()
        self._ratio_calculator = FinancialRatioCalculator()
        self._altman = AltmanZScoreCalculator(self._settings)
        self._beneish = BeneishMScoreDetector(self._settings)

    def run(self) -> Dict[str, Any]:
        """Execute the full Enron analysis pipeline.

        Steps:

        1. Build approximate FY1999 and FY2000 financial statements
           from the 10-K filing (amounts in millions USD).
        2. Calculate the 25-feature ratio vector.
        3. Compute the Altman Z-Score (original, manufacturing).
        4. Optionally run the hybrid rating model.
        5. Run the complete shenanigans detection suite.
        6. Assert that the Beneish M-Score exceeds -2.22.
        7. Assemble all results into a structured dictionary.
        8. Persist the dictionary as a JSON file.

        Returns:
            A dictionary containing all analysis results keyed by
            section name.
        """
        prior_statements = self._build_fy1999_statements()
        current_statements = self._build_fy2000_statements()

        report = AnnualReport(
            ticker="ENRNQ",
            fiscal_year=2000,
            financial_statements=current_statements,
        )

        # -- Ratios ---------------------------------------------------
        ratios = self._ratio_calculator.calculate(
            current_statements, prior_statements,
        )
        ratios_dict = ratios.to_dict()

        # -- Altman Z-Score -------------------------------------------
        z_score = self._altman.calculate_z(current_statements)
        z_zone = self._altman.classify_z(z_score)

        # -- Hybrid model (best effort) -------------------------------
        model_result = self._try_model_prediction(report)

        # -- Shenanigans suite ----------------------------------------
        shenanigans_report = self._run_shenanigans(
            current_statements, prior_statements,
        )

        # -- Validate M-Score expectation -----------------------------
        m_score = (
            shenanigans_report.beneish.m_score
            if shenanigans_report.beneish is not None
            else None
        )
        m_score_triggered = (
            shenanigans_report.beneish.is_likely_manipulator
            if shenanigans_report.beneish is not None
            else False
        )
        if m_score is not None:
            logger.info(
                "Enron M-Score = %.3f (threshold = %.2f, triggered = %s)",
                m_score,
                self._settings.beneish_manipulation_threshold,
                m_score_triggered,
            )
            if not m_score_triggered:
                logger.warning(
                    "UNEXPECTED: Enron M-Score (%.3f) did not exceed "
                    "the manipulation threshold (%.2f).  Check input data.",
                    m_score,
                    self._settings.beneish_manipulation_threshold,
                )

        # -- Assemble results -----------------------------------------
        results: Dict[str, Any] = {
            "company": "Enron Corp",
            "ticker": "ENRNQ",
            "fiscal_year": 2000,
            "ratios": ratios_dict,
            "altman_z_score": {
                "score": z_score,
                "zone": z_zone.value,
            },
            "model_prediction": model_result,
            "shenanigans": {
                "beneish_m_score": m_score,
                "beneish_is_manipulator": m_score_triggered,
                "beneish_threshold": self._settings.beneish_manipulation_threshold,
                "earnings_flags": [
                    {
                        "id": s.signal_id,
                        "name": s.name,
                        "flagged": s.is_flagged,
                        "value": s.observed_value,
                        "threshold": s.threshold,
                    }
                    for s in shenanigans_report.earnings_signals
                ],
                "cash_flow_flags": [
                    {
                        "id": s.signal_id,
                        "name": s.name,
                        "flagged": s.is_flagged,
                        "value": s.observed_value,
                        "threshold": s.threshold,
                    }
                    for s in shenanigans_report.cash_flow_signals
                ],
                "total_flags_raised": shenanigans_report.total_flags_raised,
                "overall_risk": shenanigans_report.overall_risk.name,
            },
            "expected_m_score_above_threshold": True,
            "m_score_validation_passed": m_score_triggered,
        }

        # -- Persist JSON ---------------------------------------------
        self._save_json(results)

        logger.info(
            "Enron analysis complete: Z=%.2f (%s), M-Score=%.3f, "
            "shenanigans flags=%d, risk=%s",
            z_score,
            z_zone.value,
            m_score if m_score is not None else 0.0,
            shenanigans_report.total_flags_raised,
            shenanigans_report.overall_risk.name,
        )
        return results

    # ------------------------------------------------------------------
    # Financial data
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fy2000_statements() -> FinancialStatements:
        """Construct approximate Enron FY2000 data (millions USD).

        Based on Enron's 10-K filing for the fiscal year ended
        December 31, 2000.  Revenue was inflated by mark-to-market
        energy trading; off-balance-sheet SPEs concealed leverage.

        Returns:
            A :class:`FinancialStatements` for FY2000.
        """
        income = IncomeStatement(
            total_revenue=100_789.0,
            cost_of_goods_sold=94_517.0,
            total_operating_expenses=98_836.0,
            selling_general_admin=3_184.0,
            depreciation_expense=855.0,
            interest_expense=838.0,
            income_tax_expense=434.0,
            net_income=979.0,
            gross_profit=6_272.0,
            ebit=1_953.0,
        )
        balance = BalanceSheet(
            cash_and_equivalents=1_374.0,
            short_term_investments=570.0,
            net_receivables=10_396.0,
            inventory=953.0,
            total_current_assets=30_381.0,
            net_property_plant_equipment=11_743.0,
            total_assets=65_503.0,
            accounts_payable=9_777.0,
            short_term_debt=1_679.0,
            total_current_liabilities=28_406.0,
            long_term_debt=8_550.0,
            total_liabilities=54_033.0,
            total_equity=11_470.0,
            retained_earnings=3_226.0,
        )
        cash_flow = CashFlowStatement(
            depreciation_and_amortization=855.0,
            cash_from_operations=4_779.0,
            capital_expenditures=2_381.0,
            cash_from_investing=-4_264.0,
            cash_from_financing=571.0,
            net_change_in_cash=1_086.0,
        )
        return FinancialStatements(
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cash_flow,
            fiscal_year=2000,
            currency="USD",
        )

    @staticmethod
    def _build_fy1999_statements() -> FinancialStatements:
        """Construct approximate Enron FY1999 data (millions USD).

        Used as the prior-period reference for Beneish M-Score
        computation and year-over-year growth ratios.

        Returns:
            A :class:`FinancialStatements` for FY1999.
        """
        income = IncomeStatement(
            total_revenue=40_112.0,
            cost_of_goods_sold=34_761.0,
            total_operating_expenses=39_310.0,
            selling_general_admin=3_045.0,
            depreciation_expense=870.0,
            interest_expense=656.0,
            income_tax_expense=104.0,
            net_income=893.0,
            gross_profit=5_351.0,
            ebit=802.0,
        )
        balance = BalanceSheet(
            cash_and_equivalents=288.0,
            short_term_investments=500.0,
            net_receivables=3_030.0,
            inventory=598.0,
            total_current_assets=7_255.0,
            net_property_plant_equipment=10_681.0,
            total_assets=33_381.0,
            accounts_payable=2_154.0,
            short_term_debt=1_001.0,
            total_current_liabilities=6_759.0,
            long_term_debt=7_151.0,
            total_liabilities=23_811.0,
            total_equity=9_570.0,
            retained_earnings=2_698.0,
        )
        cash_flow = CashFlowStatement(
            depreciation_and_amortization=870.0,
            cash_from_operations=1_228.0,
            capital_expenditures=2_364.0,
            cash_from_investing=-3_507.0,
            cash_from_financing=2_104.0,
            net_change_in_cash=-175.0,
        )
        return FinancialStatements(
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cash_flow,
            fiscal_year=1999,
            currency="USD",
        )

    # ------------------------------------------------------------------
    # Model prediction
    # ------------------------------------------------------------------

    def _try_model_prediction(
        self,
        report: AnnualReport,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to run the hybrid model; return ``None`` on failure.

        Args:
            report: The annual report to predict.

        Returns:
            A dictionary with prediction details, or ``None``.
        """
        try:
            from credit_rating.models.hybrid import HybridRatingModel

            model = HybridRatingModel(settings=self._settings)
            prediction = model.predict_report(report)
            return {
                "rating": prediction.rating.name,
                "confidence": prediction.confidence,
                "is_investment_grade": prediction.is_investment_grade,
                "top_classes": [
                    {"class": rc.name, "probability": prob}
                    for rc, prob in prediction.top_n_classes(3)
                ],
            }
        except Exception as exc:
            logger.warning(
                "Hybrid model prediction unavailable: %s", exc,
            )
            return None

    # ------------------------------------------------------------------
    # Shenanigans
    # ------------------------------------------------------------------

    def _run_shenanigans(
        self,
        current: FinancialStatements,
        prior: FinancialStatements,
    ) -> "ShenanigansReport":
        """Run the full shenanigans detection suite.

        Args:
            current: FY2000 financial statements.
            prior: FY1999 financial statements.

        Returns:
            A :class:`ShenanigansReport`.
        """
        builder = ShenanigansReportBuilder()

        # Beneish M-Score
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
    # Persistence
    # ------------------------------------------------------------------

    def _save_json(self, results: Dict[str, Any]) -> None:
        """Write results to ``outputs/enron_analysis.json``.

        Args:
            results: The analysis results dictionary.
        """
        output_dir = Path(self._settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "enron_analysis.json"

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

        logger.info("Enron report saved to %s", output_path)
