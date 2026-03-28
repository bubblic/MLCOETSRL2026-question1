"""Tests for the extraction pipeline classes.

These tests validate class construction, argument handling, and control
flow without triggering real LLM calls or network I/O.

Run:
    python -m pytest -q tests/test_statement_normalization.py
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from financial_forecast.extraction.financial_statement_extractor import (
    FinancialStatementExtractor,
)
from financial_forecast.extraction.statement_config import StatementType
from financial_forecast.extraction.statement_normalizer import (
    StatementNormalizer,
)
from financial_forecast.extraction.statement_ratios import (
    RatioCalculator,
    MedianRatioCalculator,
)
from financial_forecast.extraction.utils import (
    safe_divide,
    sum_if_all_present,
)


# ---------------------------------------------------------------------------
# FinancialStatementExtractor
# ---------------------------------------------------------------------------


def test_extractor_stores_config():
    """Constructor should store queries, client, and parameters."""
    client = Mock()
    extractor = FinancialStatementExtractor(
        queries=[StatementType.BALANCE_SHEET],
        llm_client=client,
        batch_size=50,
        parameters={"temperature": 0.5},
        max_workers=2,
    )
    assert extractor.queries == [StatementType.BALANCE_SHEET]
    assert extractor.llm_client is client
    assert extractor.batch_size == 50
    assert extractor.parameters == {"temperature": 0.5}
    assert extractor.max_workers == 2


def test_extractor_default_parameters():
    """Default parameters should be set when not provided."""
    extractor = FinancialStatementExtractor(
        queries=[],
        llm_client=Mock(),
    )
    assert extractor.parameters == {"temperature": 0, "top_k": 1}
    assert extractor.batch_size == 100
    assert extractor.max_workers == 9


def test_extractor_run_raises_on_missing_path():
    """run() should raise FileNotFoundError for non-existent path."""
    extractor = FinancialStatementExtractor(
        queries=[],
        llm_client=Mock(),
    )
    with pytest.raises(FileNotFoundError):
        extractor.run("does_not_exist_at_all")


def test_extractor_run_raises_on_empty_dir(tmp_path):
    """run() should raise FileNotFoundError for directory with no PDFs."""
    extractor = FinancialStatementExtractor(
        queries=[],
        llm_client=Mock(),
    )
    with pytest.raises(FileNotFoundError, match="No PDF files found"):
        extractor.run(str(tmp_path))


# ---------------------------------------------------------------------------
# StatementNormalizer
# ---------------------------------------------------------------------------


def test_normalizer_stores_config():
    """Constructor should store paths and parameters."""
    normalizer = StatementNormalizer(
        input_dir="in",
        output_dir="out",
        temperature=0.5,
        max_tokens=4000,
        top_k=3,
        max_workers=2,
    )
    assert normalizer.input_dir == Path("in")
    assert normalizer.output_dir == Path("out")
    assert normalizer.parameters == {
        "temperature": 0.5,
        "max_tokens": 4000,
        "top_k": 3,
    }
    assert normalizer.max_workers == 2


def test_normalizer_run_raises_on_empty_dir(tmp_path):
    """run() should raise ValueError when no *.llm.json files exist."""
    normalizer = StatementNormalizer(
        input_dir=str(tmp_path),
        output_dir=str(tmp_path / "out"),
    )
    with pytest.raises(ValueError, match="No statement files found"):
        normalizer.run()


# ---------------------------------------------------------------------------
# RatioCalculator
# ---------------------------------------------------------------------------


def test_ratio_calculator_stores_config():
    """Constructor should store paths."""
    calc = RatioCalculator(
        input_dir="normalized",
        output_file="ratios.json",
    )
    assert calc.input_dir == Path("normalized")


def test_ratio_calculator_run_raises_on_empty_dir(tmp_path):
    """run() should raise ValueError when no normalized files exist."""
    calc = RatioCalculator(
        input_dir=str(tmp_path),
        output_file="ratios.json",
    )
    with pytest.raises(ValueError, match="No normalized files found"):
        calc.run()


def test_median_ratio_calculator_raises_without_run_dirs(tmp_path):
    """run() should raise ValueError when no run directories exist."""
    calc = MedianRatioCalculator(
        runs_output_dir=str(tmp_path),
    )
    with pytest.raises(ValueError, match="No run directories found"):
        calc.run()


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


def test_safe_divide_normal():
    assert safe_divide(10.0, 2.0) == 5.0


def test_safe_divide_none():
    assert safe_divide(None, 2.0) is None
    assert safe_divide(10.0, None) is None


def test_safe_divide_zero():
    assert safe_divide(10.0, 0.0) is None


def test_sum_if_all_present_complete():
    assert sum_if_all_present(1.0, 2.0, 3.0) == 6.0


def test_sum_if_all_present_missing():
    assert sum_if_all_present(1.0, None, 3.0) is None


def test_calculate_ratios_all_none():
    """All-None input should produce all-None ratios."""
    derived, ratios = RatioCalculator._calculate_ratios({})
    assert all(v is None for v in ratios.values())
    assert all(v is None for v in derived.values())


def test_calculate_ratios_basic():
    """Basic ratio computation with complete inputs."""
    values = {
        "total_revenue": 100.0,
        "total_operating_cost": 60.0,
        "cash_and_cash_equivalents": 10.0,
        "short_term_market_securities": 5.0,
        "net_accounts_receivable": 15.0,
        "total_current_liabilities": 20.0,
        "total_debt_short_term_and_long_term": 50.0,
        "total_equity": 100.0,
        "total_assets": 200.0,
        "net_income": 25.0,
        "income_tax_expense": 8.0,
        "interest_expenses": 7.0,
        "depreciation_and_amortization": 10.0,
    }
    derived, ratios = RatioCalculator._calculate_ratios(values)
    assert derived["ebit"] == pytest.approx(40.0)
    assert derived["ebitda"] == pytest.approx(50.0)
    assert ratios["cost_to_income_ratio"] == pytest.approx(0.6)
    assert ratios["debt_to_equity_ratio"] == pytest.approx(0.5)
