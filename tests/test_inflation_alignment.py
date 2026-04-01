"""Tests for inflation/financial-data length alignment in prepare().

Catches the bug where an 8-element inflation tensor was silently stored
alongside 4-element financial data, causing misaligned cum_inflation in
OpEx training.
"""

import pytest
import tensorflow as tf

from financial_forecast.models.base import BaseFinancialModel
from financial_forecast.models.opex import SimpleOpEx
from financial_forecast.models.liquidity import CashTargetPolicy
from financial_forecast.models.dividends import SimpleDividendPolicy
from financial_forecast.models.buyback import SimpleBuybackPolicy
from financial_forecast.models.purchases import StaticCostRatioPolicy
from financial_forecast.models.debt import SimpleDebtPolicy
from financial_forecast.models.capex import CapexPolicy
from financial_forecast.models.working_capital import WorkingCapitalPolicy
from financial_forecast.models.tax import SimpleTax
from financial_forecast.inference.trajectory_simulator import DeterministicSimulator


def _make_model():
    return BaseFinancialModel(
        opex_module=SimpleOpEx(),
        trajectory_simulator=DeterministicSimulator(),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=CashTargetPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
        purchases_policy=StaticCostRatioPolicy(),
        debt_policy=SimpleDebtPolicy(),
        tax_module=SimpleTax(),
    )


def _make_financial_statements(n_years):
    """Minimal valid financial statements for n_years."""
    ones = tf.ones(n_years, dtype=tf.float64)
    return {
        "years": tf.cast(tf.range(2020, 2020 + n_years), tf.float64),
        "sales": ones * 100.0,
        "purchases": ones * 60.0,
        "cogs": ones * 55.0,
        "nca": ones * 200.0,
        "depreciation": ones * 10.0,
        "advance_payments_purchases": ones * 2.0,
        "accounts_receivable": ones * 15.0,
        "accounts_payable": ones * 12.0,
        "advance_payments_sales": ones * 3.0,
        "cash": ones * 20.0,
        "ims": ones * 10.0,
        "inventory": ones * 8.0,
        "current_liabilities": ones * 30.0,
        "non_current_liabilities": ones * 50.0,
        "equity": ones * 160.0,
        "net_income": ones * 12.0,
        "dividends": ones * 2.0,
        "stock_buyback": ones * 1.0,
        "opex": ones * 10.0,
        "tax": ones * 3.0,
        "current_lt_debt": ones * 5.0,
        "interest_payment": ones * 2.0,
        "ms_return": ones * 0.5,
    }


class TestInflationAlignment:

    def test_longer_inflation_is_tail_aligned(self):
        """Inflation with more years than data should use last n_hist values."""
        model = _make_model()
        fs = _make_financial_statements(4)
        inflation_8yr = tf.constant(
            [0.024, 0.018, 0.012, 0.047, 0.08, 0.041, 0.029, 0.027],
            dtype=tf.float64,
        )
        model.prepare(fs, inflation=inflation_8yr)

        stored = model.historical_data["inflation"]
        assert len(stored) == 4
        # Should be the LAST 4 values: [0.08, 0.041, 0.029, 0.027]
        assert float(stored[0]) == pytest.approx(0.08)
        assert float(stored[1]) == pytest.approx(0.041)
        assert float(stored[2]) == pytest.approx(0.029)
        assert float(stored[3]) == pytest.approx(0.027)

    def test_exact_length_inflation_passes(self):
        """Inflation with exactly n_hist values should be stored as-is."""
        model = _make_model()
        fs = _make_financial_statements(4)
        inflation_4yr = tf.constant([0.08, 0.041, 0.029, 0.027], dtype=tf.float64)
        model.prepare(fs, inflation=inflation_4yr)

        stored = model.historical_data["inflation"]
        assert len(stored) == 4
        assert float(stored[0]) == pytest.approx(0.08)

    def test_shorter_inflation_raises(self):
        """Inflation with fewer years than data should raise ValueError."""
        model = _make_model()
        fs = _make_financial_statements(4)
        inflation_2yr = tf.constant([0.029, 0.027], dtype=tf.float64)

        with pytest.raises(ValueError, match="inflation has 2 values"):
            model.prepare(fs, inflation=inflation_2yr)

    def test_no_inflation_gives_zeros(self):
        """No inflation should default to zeros matching n_hist."""
        model = _make_model()
        fs = _make_financial_statements(4)
        model.prepare(fs, inflation=None)

        stored = model.historical_data["inflation"]
        assert len(stored) == 4
        assert float(stored[0]) == pytest.approx(0.0)
