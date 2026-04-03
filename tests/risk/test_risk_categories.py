"""Tests for risk_categories module."""

from risk.risk_categories import (
    ALL_CATEGORIES,
    DEFAULT_CHUNK_SIZE,
    RiskCategory,
)


class TestRiskCategory:
    def test_category_count(self):
        assert len(RiskCategory) == 9

    def test_all_categories_have_unique_values(self):
        values = [cat.value for cat in RiskCategory]
        assert len(values) == len(set(values))

    def test_categories_are_str_enum(self):
        for cat in RiskCategory:
            assert isinstance(cat.value, str)
            # str enum values can be used as dict keys directly
            d = {cat: True}
            assert d[cat] is True

    def test_all_categories_list_matches_enum(self):
        assert ALL_CATEGORIES == [cat.value for cat in RiskCategory]

    def test_default_chunk_size_is_positive(self):
        assert DEFAULT_CHUNK_SIZE > 0
