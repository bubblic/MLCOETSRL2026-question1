"""Evergrande 2022 case study.

Constructs Evergrande's approximate FY2021--2022 financial data,
runs ratio calculation, Altman Z''-Score, the hybrid rating model,
and the full shenanigans detection suite.  Results are returned as
a structured dictionary and persisted as a JSON report.
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


class EvergrandeAnalysis:
    """End-to-end case study of China Evergrande Group (2022 default).

    Demonstrates the credit rating system's ability to flag deep
    financial distress and accounting red flags using publicly
    available financial data.

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
        """Execute the full Evergrande analysis pipeline.

        Steps:

        1. Build approximate FY2021 and FY2022 financial statements
           from publicly reported figures (amounts in millions CNY).
        2. Calculate the 25-feature ratio vector.
        3. Compute the Altman Z''-Score.
        4. Optionally run the hybrid rating model (if a checkpoint
           is available).
        5. Run the complete shenanigans detection suite.
        6. Assemble all results into a structured dictionary.
        7. Persist the dictionary as a JSON file under
           ``settings.output_dir``.

        Returns:
            A dictionary containing all analysis results keyed by
            section name.
        """
        prior_statements = self._build_fy2021_statements()
        current_statements = self._build_fy2022_statements()

        report = AnnualReport(
            ticker="3333.HK",
            fiscal_year=2022,
            financial_statements=current_statements,
        )

        # -- Ratios ---------------------------------------------------
        ratios = self._ratio_calculator.calculate(
            current_statements, prior_statements,
        )
        ratios_dict = ratios.to_dict()

        # -- Altman Z''-Score -----------------------------------------
        z_pp = self._altman.calculate_z_prime_prime(current_statements)
        z_zone = self._altman.classify_z_prime_prime(z_pp)

        # -- Hybrid model (best effort) -------------------------------
        model_result = self._try_model_prediction(report)

        # -- Shenanigans suite ----------------------------------------
        shenanigans_report = self._run_shenanigans(
            current_statements, prior_statements,
        )

        # -- Assemble results -----------------------------------------
        results: Dict[str, Any] = {
            "company": "China Evergrande Group",
            "ticker": "3333.HK",
            "fiscal_year": 2022,
            "ratios": ratios_dict,
            "altman_z_prime_prime": {
                "score": z_pp,
                "zone": z_zone.value,
            },
            "model_prediction": model_result,
            "shenanigans": {
                "beneish_m_score": (
                    shenanigans_report.beneish.m_score
                    if shenanigans_report.beneish is not None
                    else None
                ),
                "beneish_is_manipulator": (
                    shenanigans_report.beneish.is_likely_manipulator
                    if shenanigans_report.beneish is not None
                    else None
                ),
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
        }

        # -- Persist JSON ---------------------------------------------
        self._save_json(results)

        logger.info(
            "Evergrande analysis complete: Z''=%.2f (%s), "
            "shenanigans flags=%d, risk=%s",
            z_pp,
            z_zone.value,
            shenanigans_report.total_flags_raised,
            shenanigans_report.overall_risk.name,
        )
        return results

    # ------------------------------------------------------------------
    # Financial data
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fy2022_statements() -> FinancialStatements:
        """Construct approximate Evergrande FY2022 data (millions CNY).

        Based on publicly available interim and estimated full-year
        figures as reported in 2022/2023 filings.  Evergrande had
        negative equity, massive losses, and negligible cash flow.

        Returns:
            A :class:`FinancialStatements` for FY2022.
        """
        income = IncomeStatement(
            total_revenue=230_000.0,
            cost_of_goods_sold=210_000.0,
            total_operating_expenses=250_000.0,
            selling_general_admin=15_000.0,
            depreciation_expense=3_500.0,
            interest_expense=57_000.0,
            income_tax_expense=1_000.0,
            net_income=-106_000.0,
            gross_profit=20_000.0,
            ebit=-20_000.0,
        )
        balance = BalanceSheet(
            cash_and_equivalents=14_300.0,
            short_term_investments=0.0,
            net_receivables=90_000.0,
            inventory=900_000.0,
            total_current_assets=1_500_000.0,
            net_property_plant_equipment=58_000.0,
            total_assets=1_740_000.0,
            accounts_payable=660_000.0,
            short_term_debt=240_000.0,
            total_current_liabilities=1_480_000.0,
            long_term_debt=350_000.0,
            total_liabilities=2_580_000.0,
            total_equity=-840_000.0,
            retained_earnings=-900_000.0,
        )
        cash_flow = CashFlowStatement(
            depreciation_and_amortization=5_000.0,
            cash_from_operations=-40_000.0,
            capital_expenditures=2_000.0,
            cash_from_investing=-15_000.0,
            cash_from_financing=-10_000.0,
            net_change_in_cash=-65_000.0,
        )
        return FinancialStatements(
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cash_flow,
            fiscal_year=2022,
            currency="CNY",
        )

    @staticmethod
    def _build_fy2021_statements() -> FinancialStatements:
        """Construct approximate Evergrande FY2021 data (millions CNY).

        Used as the prior-period reference for Beneish M-Score and
        year-over-year growth ratios.

        Returns:
            A :class:`FinancialStatements` for FY2021.
        """
        income = IncomeStatement(
            total_revenue=250_000.0,
            cost_of_goods_sold=220_000.0,
            total_operating_expenses=260_000.0,
            selling_general_admin=18_000.0,
            depreciation_expense=4_000.0,
            interest_expense=45_000.0,
            income_tax_expense=2_000.0,
            net_income=-476_000.0,
            gross_profit=30_000.0,
            ebit=-10_000.0,
        )
        balance = BalanceSheet(
            cash_and_equivalents=36_000.0,
            short_term_investments=0.0,
            net_receivables=110_000.0,
            inventory=1_200_000.0,
            total_current_assets=1_700_000.0,
            net_property_plant_equipment=70_000.0,
            total_assets=2_300_000.0,
            accounts_payable=700_000.0,
            short_term_debt=280_000.0,
            total_current_liabilities=1_650_000.0,
            long_term_debt=400_000.0,
            total_liabilities=2_440_000.0,
            total_equity=-140_000.0,
            retained_earnings=-200_000.0,
        )
        cash_flow = CashFlowStatement(
            depreciation_and_amortization=6_000.0,
            cash_from_operations=-100_000.0,
            capital_expenditures=5_000.0,
            cash_from_investing=-20_000.0,
            cash_from_financing=50_000.0,
            net_change_in_cash=-70_000.0,
        )
        return FinancialStatements(
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cash_flow,
            fiscal_year=2021,
            currency="CNY",
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
            current: FY2022 financial statements.
            prior: FY2021 financial statements.

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
        """Write results to ``outputs/evergrande_analysis.json``.

        Args:
            results: The analysis results dictionary.
        """
        output_dir = Path(self._settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "evergrande_analysis.json"

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

        logger.info("Evergrande report saved to %s", output_path)
