"""Integration test: statements -> all detectors -> ShenanigansReport."""

from __future__ import annotations

import pytest

from credit_rating.shenanigans.beneish import BeneishMScoreDetector
from credit_rating.shenanigans.cash_flow import CashFlowShenanigansDetector
from credit_rating.shenanigans.earnings_manipulation import (
    EarningsManipulationDetector,
)
from credit_rating.shenanigans.report_builder import ShenanigansReportBuilder
from credit_rating.shenanigans.text_signals import TextSignalDetector


class TestShenanigansFullPipeline:
    """Full shenanigans analysis from statements to report."""

    def test_full_pipeline_with_prior(
        self, sample_statements, sample_prior_statements,
    ):
        beneish = BeneishMScoreDetector().calculate(
            sample_statements, sample_prior_statements,
        )
        ems = EarningsManipulationDetector(
            sample_statements, sample_prior_statements,
        ).detect_all()
        cfs = CashFlowShenanigansDetector(
            sample_statements, sample_prior_statements,
        ).detect_all()
        text_signals = TextSignalDetector().detect(
            risk_factors_text="The company faces risk.",
        )

        report = (
            ShenanigansReportBuilder()
            .with_beneish(beneish)
            .with_earnings_signals(ems)
            .with_cash_flow_signals(cfs)
            .with_text_signals(text_signals)
            .build()
        )

        assert report.beneish is not None
        assert len(report.earnings_signals) == 8
        assert len(report.cash_flow_signals) == 4
        assert report.text_signals is not None
        assert report.total_flags_raised >= 0

    def test_pipeline_without_prior(self, sample_statements):
        ems = EarningsManipulationDetector(sample_statements).detect_all()
        cfs = CashFlowShenanigansDetector(sample_statements).detect_all()

        report = (
            ShenanigansReportBuilder()
            .with_earnings_signals(ems)
            .with_cash_flow_signals(cfs)
            .build()
        )

        assert len(report.earnings_signals) == 8
        assert len(report.cash_flow_signals) == 4
        assert report.beneish is None
