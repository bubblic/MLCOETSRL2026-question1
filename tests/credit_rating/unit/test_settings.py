"""Tests for CreditRatingSettings and RatingClass."""

from __future__ import annotations

import pytest

from credit_rating.config.settings import (
    AltmanZone,
    CreditRatingSettings,
    RatingClass,
    RiskLevel,
)


class TestRatingClass:
    """Tests for the RatingClass enum."""

    def test_ordinal_values(self):
        assert RatingClass.AAA_AA == 0
        assert RatingClass.D == 6

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("AAA", RatingClass.AAA_AA),
            ("AA+", RatingClass.AAA_AA),
            ("A", RatingClass.A),
            ("BBB-", RatingClass.BBB),
            ("BB+", RatingClass.BB),
            ("B", RatingClass.B),
            ("CCC", RatingClass.CCC_CC),
            ("D", RatingClass.D),
            ("SD", RatingClass.D),
        ],
    )
    def test_from_sp_string(self, label, expected):
        assert RatingClass.from_sp_string(label) == expected

    def test_from_sp_string_case_insensitive(self):
        assert RatingClass.from_sp_string("bbb+") == RatingClass.BBB

    def test_from_sp_string_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown S&P rating"):
            RatingClass.from_sp_string("XYZ")

    def test_is_investment_grade(self):
        assert RatingClass.AAA_AA.is_investment_grade is True
        assert RatingClass.BBB.is_investment_grade is True
        assert RatingClass.BB.is_investment_grade is False
        assert RatingClass.D.is_investment_grade is False


class TestCreditRatingSettings:
    """Tests for the settings class."""

    def test_defaults_load(self, settings):
        assert settings.num_rating_classes == 7
        assert settings.random_seed == 42
        assert settings.lambda_ordinal == 0.1

    def test_altman_coefficients(self, settings):
        assert settings.altman_x1_coefficient == 1.2
        assert settings.altman_x3_coefficient == 3.3

    def test_beneish_threshold(self, settings):
        assert settings.beneish_manipulation_threshold == -2.22

    def test_fog_cutoff(self, settings):
        assert settings.fog_index_cutoff == 18.0
