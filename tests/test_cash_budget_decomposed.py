"""Unit tests for decomposed CashBudgetModel sub-methods.

Tests use concrete float64 tensors and assert specific numerical outputs.
"""

import pytest
import tensorflow as tf

from financial_forecast.models.cash_budget import CashBudgetModel
from financial_forecast.models.liquidity import SimpleLiquidityPolicy
from financial_forecast.models.debt import SimpleDebtPolicy
from financial_forecast.models.dividends import SimpleDividendPolicy
from financial_forecast.models.buyback import SimpleBuybackPolicy
from financial_forecast.inference.state_index import (
    R_AR,
    R_AP,
    R_ADV_PP,
    R_ADV_PS,
    R_CASH,
    R_IMS,
    R_NET_INCOME,
    R_DIVIDENDS,
    N_RECURRENT,
)


def _f64(v):
    return tf.constant(v, dtype=tf.float64)


def _make_cash_budget():
    """Build a CashBudgetModel with simple policies for unit testing."""
    return CashBudgetModel(
        liquidity_policy=SimpleLiquidityPolicy(),
        debt_policy=SimpleDebtPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
    )


def _make_state_tensor(
    ar=100.0,
    ap=80.0,
    adv_pp=10.0,
    adv_ps=15.0,
    cash=200.0,
    ims=50.0,
    net_income=120.0,
    dividends=30.0,
):
    """Build a [1, 14] state tensor with specified values, zeros elsewhere."""
    state = tf.zeros([1, N_RECURRENT], dtype=tf.float64)
    indices = [
        (R_AR, ar),
        (R_AP, ap),
        (R_ADV_PP, adv_pp),
        (R_ADV_PS, adv_ps),
        (R_CASH, cash),
        (R_IMS, ims),
        (R_NET_INCOME, net_income),
        (R_DIVIDENDS, dividends),
    ]
    values = state.numpy()
    for idx, val in indices:
        values[0, idx] = val
    return tf.constant(values, dtype=tf.float64)


class TestComputeOperatingNlb:
    """Tests for CashBudgetModel._compute_operating_nlb()."""

    def test_basic_operating_nlb(self):
        """Verify operating NLB = inflows - outflows with known values.

        Scenario (all values for n_samples=1):
            Previous state: AR=100, AP=80, adv_pp=10, adv_ps=15
            Current period: sales=500, ar_curr=120, adv_ps_curr=20,
                           purchases=300, ap_curr=90, adv_pp_curr=12,
                           opex=60, tax=25

        Expected calculation:
            sales_collected = 500 - 120 - 15 = 365
            inflows = 365 + 100 + 20 = 485

            purchases_paid = 300 - 90 - 10 = 200
            outflows = 200 + 80 + 12 + 60 + 25 = 377

            operating_nlb = 485 - 377 = 108
        """
        cb = _make_cash_budget()
        state = _make_state_tensor(ar=100.0, ap=80.0, adv_pp=10.0, adv_ps=15.0)

        assets = {
            "ar_curr": _f64([[120.0]]),
            "adv_ps_curr": _f64([[20.0]]),
            "purchases_t": _f64([[300.0]]),
            "ap_curr": _f64([[90.0]]),
            "adv_pp_curr": _f64([[12.0]]),
        }
        income = {
            "opex": _f64([[60.0]]),
            "tax": _f64([[25.0]]),
        }
        sales_t = _f64([[500.0]])

        result = cb._compute_operating_nlb(state, assets, income, sales_t)

        assert float(result[0]) == pytest.approx(108.0)

    def test_negative_operating_nlb(self):
        """Operating NLB is negative when outflows exceed inflows.

        Scenario:
            Previous: AR=50, AP=200, adv_pp=5, adv_ps=10
            Current: sales=300, ar_curr=80, adv_ps_curr=8,
                    purchases=400, ap_curr=60, adv_pp_curr=15,
                    opex=100, tax=40

        Expected:
            sales_collected = 300 - 80 - 10 = 210
            inflows = 210 + 50 + 8 = 268

            purchases_paid = 400 - 60 - 5 = 335
            outflows = 335 + 200 + 15 + 100 + 40 = 690

            operating_nlb = 268 - 690 = -422
        """
        cb = _make_cash_budget()
        state = _make_state_tensor(ar=50.0, ap=200.0, adv_pp=5.0, adv_ps=10.0)

        assets = {
            "ar_curr": _f64([[80.0]]),
            "adv_ps_curr": _f64([[8.0]]),
            "purchases_t": _f64([[400.0]]),
            "ap_curr": _f64([[60.0]]),
            "adv_pp_curr": _f64([[15.0]]),
        }
        income = {
            "opex": _f64([[100.0]]),
            "tax": _f64([[40.0]]),
        }
        sales_t = _f64([[300.0]])

        result = cb._compute_operating_nlb(state, assets, income, sales_t)

        assert float(result[0]) == pytest.approx(-422.0)

    def test_zero_operating_nlb(self):
        """Operating NLB is exactly zero when inflows equal outflows.

        Scenario designed so inflows = outflows = 100:
            Previous: AR=0, AP=0, adv_pp=0, adv_ps=0
            Current: sales=100, ar_curr=0, adv_ps_curr=0,
                    purchases=0, ap_curr=0, adv_pp_curr=0,
                    opex=100, tax=0

            inflows = (100 - 0 - 0) + 0 + 0 = 100
            outflows = (0 - 0 - 0) + 0 + 0 + 100 + 0 = 100
            nlb = 0
        """
        cb = _make_cash_budget()
        state = _make_state_tensor(ar=0.0, ap=0.0, adv_pp=0.0, adv_ps=0.0)

        assets = {
            "ar_curr": _f64([[0.0]]),
            "adv_ps_curr": _f64([[0.0]]),
            "purchases_t": _f64([[0.0]]),
            "ap_curr": _f64([[0.0]]),
            "adv_pp_curr": _f64([[0.0]]),
        }
        income = {
            "opex": _f64([[100.0]]),
            "tax": _f64([[0.0]]),
        }
        sales_t = _f64([[100.0]])

        result = cb._compute_operating_nlb(state, assets, income, sales_t)

        assert float(result[0]) == pytest.approx(0.0)

    def test_batched_operating_nlb(self):
        """Operating NLB works with n_samples > 1 (batched Monte Carlo).

        Two samples with different AR values. All other values identical.
            Sample 0: AR_prev=100 → inflows differ
            Sample 1: AR_prev=200 → 100 more inflow

            nlb_0 = (500 - 120 - 15 + 100 + 20) - (300 - 90 - 10 + 80 + 12 + 60 + 25)
                   = 485 - 377 = 108
            nlb_1 = (500 - 120 - 15 + 200 + 20) - same outflows
                   = 585 - 377 = 208
        """
        cb = _make_cash_budget()

        values = tf.zeros([2, N_RECURRENT], dtype=tf.float64).numpy()
        values[0, R_AR] = 100.0
        values[1, R_AR] = 200.0
        values[:, R_AP] = 80.0
        values[:, R_ADV_PP] = 10.0
        values[:, R_ADV_PS] = 15.0
        state = tf.constant(values, dtype=tf.float64)

        assets = {
            "ar_curr": _f64([120.0, 120.0]),
            "adv_ps_curr": _f64([20.0, 20.0]),
            "purchases_t": _f64([300.0, 300.0]),
            "ap_curr": _f64([90.0, 90.0]),
            "adv_pp_curr": _f64([12.0, 12.0]),
        }
        income = {
            "opex": _f64([60.0, 60.0]),
            "tax": _f64([25.0, 25.0]),
        }
        sales_t = _f64([500.0, 500.0])

        result = cb._compute_operating_nlb(state, assets, income, sales_t)

        assert result[0].numpy() == pytest.approx(108.0)
        assert result[1].numpy() == pytest.approx(208.0)
