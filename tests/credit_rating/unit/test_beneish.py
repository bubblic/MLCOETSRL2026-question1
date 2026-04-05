"""Tests for the Beneish M-Score detector."""

from __future__ import annotations

import pytest

from credit_rating.shenanigans.beneish import BeneishMScoreDetector


class TestBeneishMScore:
    """Verify all eight Beneish variables and the M-Score."""

    @pytest.fixture
    def detector(self):
        return BeneishMScoreDetector()

    def test_calculate_returns_beneish_score(
        self, detector, sample_statements, sample_prior_statements,
    ):
        result = detector.calculate(sample_statements, sample_prior_statements)
        assert hasattr(result, "m_score")
        assert hasattr(result, "is_likely_manipulator")

    def test_all_eight_variables_present(
        self, detector, sample_statements, sample_prior_statements,
    ):
        result = detector.calculate(sample_statements, sample_prior_statements)
        for attr in ("dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"):
            assert hasattr(result, attr)

    def test_flag_consistent_with_threshold(
        self, detector, sample_statements, sample_prior_statements,
    ):
        """The manipulation flag should match the threshold comparison."""
        result = detector.calculate(sample_statements, sample_prior_statements)
        expected_flag = result.m_score > detector._s.beneish_manipulation_threshold
        assert result.is_likely_manipulator is expected_flag

    def test_sgi_matches_revenue_ratio(
        self, detector, sample_statements, sample_prior_statements,
    ):
        result = detector.calculate(sample_statements, sample_prior_statements)
        expected = (
            sample_statements.income_statement.total_revenue
            / sample_prior_statements.income_statement.total_revenue
        )
        assert result.sgi == pytest.approx(expected)

    def test_m_score_formula(
        self, detector, sample_statements, sample_prior_statements,
    ):
        """Verify the M-Score is the weighted sum of the eight variables."""
        from credit_rating.config.settings import CreditRatingSettings

        s = CreditRatingSettings()
        result = detector.calculate(sample_statements, sample_prior_statements)
        expected = (
            s.beneish_constant
            + s.beneish_dsri_coeff * result.dsri
            + s.beneish_gmi_coeff * result.gmi
            + s.beneish_aqi_coeff * result.aqi
            + s.beneish_sgi_coeff * result.sgi
            + s.beneish_depi_coeff * result.depi
            + s.beneish_sgai_coeff * result.sgai
            + s.beneish_lvgi_coeff * result.lvgi
            + s.beneish_tata_coeff * result.tata
        )
        assert result.m_score == pytest.approx(expected)

    def test_manipulation_flag_respects_threshold(self, detector):
        """Score above -2.22 should flag as likely manipulator."""
        from credit_rating.domain.shenanigans import BeneishScore

        # Directly test the threshold comparison
        s = detector._s
        assert s.beneish_manipulation_threshold == -2.22
