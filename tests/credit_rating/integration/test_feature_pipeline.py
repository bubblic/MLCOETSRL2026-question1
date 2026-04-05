"""Integration test: statements -> ratios -> normalized."""

from __future__ import annotations

import math

import numpy as np
import pytest

from credit_rating.features.altman import AltmanZScoreCalculator
from credit_rating.features.normalizer import RatioNormalizer
from credit_rating.features.ratio_calculator import FinancialRatioCalculator


class TestFeaturePipeline:
    """End-to-end feature engineering pipeline."""

    def test_statements_to_ratios_to_normalized(
        self, sample_statements, sample_prior_statements,
    ):
        calculator = FinancialRatioCalculator()
        ratios = calculator.calculate(sample_statements, sample_prior_statements)

        assert len(ratios) == 25
        assert all(not math.isnan(v) for v in ratios)

        normalizer = RatioNormalizer()
        normalizer.fit([ratios, ratios])
        scaled = normalizer.transform([ratios])

        assert scaled.shape == (1, 25)
        assert not np.any(np.isnan(scaled))

    def test_altman_z_score_from_statements(self, sample_statements):
        altman = AltmanZScoreCalculator()
        z = altman.calculate_z(sample_statements)
        zpp = altman.calculate_z_prime_prime(sample_statements)

        assert isinstance(z, float)
        assert isinstance(zpp, float)
        assert not math.isnan(z)
        assert not math.isnan(zpp)

    def test_normalizer_roundtrip(self, sample_ratios):
        normalizer = RatioNormalizer()
        original = [sample_ratios, sample_ratios]
        scaled = normalizer.fit_transform(original)
        unscaled = normalizer.inverse_transform(scaled)

        original_flat = np.array(
            [sample_ratios.to_flat_list(), sample_ratios.to_flat_list()],
        )
        np.testing.assert_allclose(unscaled, original_flat, atol=1e-10)

    def test_normalizer_save_load(self, sample_ratios, tmp_path):
        normalizer = RatioNormalizer()
        normalizer.fit([sample_ratios])

        save_path = tmp_path / "scaler.pkl"
        normalizer.save(save_path)

        loaded = RatioNormalizer(scaler_path=save_path)
        assert loaded.is_fitted
        result = loaded.transform([sample_ratios])
        assert result.shape == (1, 25)
