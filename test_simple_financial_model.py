"""Tests for simple_financial_model.py."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

import simple_financial_model as module


@pytest.fixture
def model():
    return module.SimpleFinancialModel()


@pytest.fixture
def state_dict():
    # Balanced opening state: assets 1.45 = liabilities 0.76 + equity 0.69
    return {
        "nca": tf.constant(1.00, dtype=tf.float64),
        "adv_pp": tf.constant(0.05, dtype=tf.float64),  # legacy key mapping
        "ar": tf.constant(0.15, dtype=tf.float64),  # legacy key mapping
        "inv": tf.constant(0.03, dtype=tf.float64),  # legacy key mapping
        "cash": tf.constant(0.12, dtype=tf.float64),
        "ims": tf.constant(0.10, dtype=tf.float64),  # legacy key mapping
        "ap": tf.constant(0.24, dtype=tf.float64),  # legacy key mapping
        "adv_ps": tf.constant(0.02, dtype=tf.float64),  # legacy key mapping
        "cl": tf.constant(0.18, dtype=tf.float64),  # legacy key mapping
        "ncl": tf.constant(0.32, dtype=tf.float64),  # legacy key mapping
        "equity": tf.constant(0.69, dtype=tf.float64),
        "ni": tf.constant(0.08, dtype=tf.float64),  # legacy key mapping
    }


@pytest.fixture
def econ_inputs():
    return module.EconomicInputs(
        sales_t=tf.constant(1.2, dtype=tf.float64),
        purchases_t=tf.constant(0.7, dtype=tf.float64),
        sales_t_plus_1=tf.constant(1.24, dtype=tf.float64),
        purchases_t_plus_1=tf.constant(0.72, dtype=tf.float64),
        inflation=tf.constant(0.02, dtype=tf.float64),
        t=1,
    )


def test_financial_state_from_dict_maps_legacy_fields():
    state = module.FinancialState.from_dict(
        {
            "nca": 1.0,
            "adv_pp": 0.1,
            "ar": 0.2,
            "inv": 0.3,
            "cash": 0.4,
            "ims": 0.5,
            "ap": 0.6,
            "adv_ps": 0.7,
            "cl": 0.8,
            "ncl": 0.9,
            "equity": 1.0,
            "ni": 0.05,
        }
    )
    assert state.advance_payments_purchases == 0.1
    assert state.accounts_receivable == 0.2
    assert state.net_income == 0.05


def test_forecast_step_returns_finite_outputs(model, state_dict, econ_inputs):
    out = model.forecast_step(state_dict, econ_inputs)
    assert isinstance(out, module.FinancialState)
    assert np.isfinite(float(out.net_income.numpy()))
    assert np.isfinite(float(out.liquidity_check.numpy()))
    check_value = float(out.check.numpy())
    assert np.isfinite(check_value)

    # Validate internal accounting consistency: `check` should equal
    # Assets - (Liabilities + Equity) using the returned state values.
    total_assets = float(
        (
            out.nca
            + out.advance_payments_purchases
            + out.accounts_receivable
            + out.inventory
            + out.cash
            + out.investment_in_market_securities
        ).numpy()
    )
    total_liab_equity = float(
        (
            out.accounts_payable
            + out.advance_payments_sales
            + out.current_liabilities
            + out.non_current_liabilities
            + out.equity
        ).numpy()
    )
    recomputed_check = total_assets - total_liab_equity
    assert check_value == pytest.approx(recomputed_check, rel=0.0, abs=1e-9)


def test_parameters_are_expected_fixed_constants(model):
    assert float(model.asset_growth.numpy()) >= 0.0
    assert 0.0 <= float(model.cash_pct_of_liquidity.numpy()) <= 1.0
    assert 0.0 <= float(model.income_tax_pct.numpy()) <= 1.0
    assert float(model.avg_maturity_years.numpy()) > 1.0
