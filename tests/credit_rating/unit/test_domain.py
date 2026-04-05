"""Tests for domain dataclasses."""

from __future__ import annotations

import pytest

from credit_rating.config.settings import RatingClass, RiskLevel
from credit_rating.domain.features import FEATURE_NAMES, NUM_FEATURES, FinancialRatios
from credit_rating.domain.financial_statements import FinancialStatements
from credit_rating.domain.rating import RatingPrediction
from credit_rating.domain.shenanigans import (
    BeneishScore,
    EarningsManipulationSignal,
    ShenanigansReport,
)


class TestFinancialStatements:
    """Tests for financial statement dataclasses."""

    def test_frozen(self, sample_statements):
        with pytest.raises(AttributeError):
            sample_statements.fiscal_year = 2024

    def test_computed_ebit(self, sample_income_statement):
        ebit = sample_income_statement.computed_ebit()
        expected = 96_995_000_000 + 3_933_000_000 + 16_741_000_000
        assert ebit == expected

    def test_computed_gross_profit(self, sample_income_statement):
        gp = sample_income_statement.computed_gross_profit()
        expected = 383_285_000_000 - 214_137_000_000
        assert gp == expected

    def test_total_debt(self, sample_balance_sheet):
        debt = sample_balance_sheet.total_debt()
        assert debt == 15_807_000_000 + 95_281_000_000

    def test_working_capital(self, sample_balance_sheet):
        wc = sample_balance_sheet.computed_working_capital()
        assert wc == 143_566_000_000 - 145_308_000_000

    def test_free_cash_flow(self, sample_cash_flow):
        fcf = sample_cash_flow.free_cash_flow()
        assert fcf == 110_543_000_000 - 10_959_000_000


class TestFinancialRatios:
    """Tests for the FinancialRatios composite."""

    def test_num_features(self):
        assert NUM_FEATURES == 25

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 25

    def test_len(self, sample_ratios):
        assert len(sample_ratios) == 25

    def test_iter(self, sample_ratios):
        values = list(sample_ratios)
        assert len(values) == 25
        assert all(isinstance(v, float) for v in values)

    def test_getitem(self, sample_ratios):
        assert sample_ratios[0] == sample_ratios.leverage.debt_to_equity

    def test_to_dict(self, sample_ratios):
        d = sample_ratios.to_dict()
        assert len(d) == 25
        assert d["debt_to_equity"] == 1.79

    def test_to_tensor(self, sample_ratios):
        import tensorflow as tf

        tensor = sample_ratios.to_tensor()
        assert tensor.shape == (25,)
        assert tensor.dtype == tf.float32

    def test_frozen(self, sample_ratios):
        with pytest.raises(AttributeError):
            sample_ratios.leverage = None


class TestRatingPrediction:
    """Tests for RatingPrediction."""

    def test_is_investment_grade(self):
        pred = RatingPrediction(
            rating=RatingClass.BBB,
            probabilities={RatingClass.BBB: 0.8},
        )
        assert pred.is_investment_grade is True

    def test_confidence(self):
        pred = RatingPrediction(
            rating=RatingClass.BB,
            probabilities={RatingClass.BB: 0.65},
        )
        assert pred.confidence == 0.65

    def test_top_n_classes(self):
        probs = {
            RatingClass.A: 0.1,
            RatingClass.BBB: 0.7,
            RatingClass.BB: 0.2,
        }
        pred = RatingPrediction(rating=RatingClass.BBB, probabilities=probs)
        top2 = pred.top_n_classes(2)
        assert top2[0][0] == RatingClass.BBB
        assert top2[1][0] == RatingClass.BB


class TestShenanigansReport:
    """Tests for ShenanigansReport."""

    def test_total_flags(self):
        ems = EarningsManipulationSignal(
            signal_id="EMS-1",
            name="Test",
            is_flagged=True,
            observed_value=0.2,
            threshold=0.1,
            explanation="test",
        )
        report = ShenanigansReport(
            beneish=BeneishScore(
                dsri=1.0, gmi=1.0, aqi=1.0, sgi=1.0,
                depi=1.0, sgai=1.0, lvgi=1.0, tata=0.1,
                m_score=-1.5, is_likely_manipulator=True,
            ),
            earnings_signals=[ems],
        )
        assert report.total_flags_raised == 2
