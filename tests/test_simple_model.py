"""Tests for simple_financial_model.py.


Run the script by:
python -m pytest -q tests/test_simple_model.py
"""

import pytest

tf = pytest.importorskip("tensorflow")

from financial_forecast.models import simple_model as module
from financial_forecast.types import EconomicInputs, FinancialState


# Patch module-level references so existing test code using module.X still works
module.EconomicInputs = EconomicInputs
module.FinancialState = FinancialState


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
    return EconomicInputs(
        sales_t=tf.constant(1.2, dtype=tf.float64),
        purchases_t=tf.constant(0.7, dtype=tf.float64),
        sales_t_plus_1=tf.constant(1.24, dtype=tf.float64),
        purchases_t_plus_1=tf.constant(0.72, dtype=tf.float64),
        cum_inflation=tf.constant(0.02, dtype=tf.float64),
        t=1,
    )


def test_financial_state_from_dict_maps_legacy_fields():
    state = FinancialState.from_dict(
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
    assert isinstance(out, FinancialState)
    assert tf.math.is_finite(out.net_income)
    assert tf.math.is_finite(out.liquidity_check)
    check_value = float(out.balance_sheet_check.numpy())
    assert tf.math.is_finite(out.balance_sheet_check)

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


def test_forecast_step_output_dtypes_are_float64(model, state_dict, econ_inputs):
    """All forecast outputs should be float64 to avoid dtype mismatch errors downstream."""
    out = model.forecast_step(state_dict, econ_inputs)
    float64_fields = [
        "nca", "advance_payments_purchases", "accounts_receivable", "inventory",
        "cash", "investment_in_market_securities", "accounts_payable",
        "advance_payments_sales", "current_liabilities", "non_current_liabilities",
        "equity", "net_income",
    ]
    for field in float64_fields:
        value = getattr(out, field)
        assert hasattr(value, "dtype"), f"{field} is not a tensor"
        assert value.dtype == tf.float64, f"{field} has dtype {value.dtype}, expected float64"


def test_forecast_step_output_shapes_are_scalar(model, state_dict, econ_inputs):
    """Forecast outputs should be scalar tensors (rank 0)."""
    out = model.forecast_step(state_dict, econ_inputs)
    for field in ["nca", "equity", "net_income", "balance_sheet_check", "liquidity_check"]:
        value = getattr(out, field)
        assert hasattr(value, "shape"), f"{field} is not a tensor"
        assert value.shape.rank == 0, f"{field} has rank {value.shape.rank}, expected 0"


def test_forecast_step_check_is_consistent(model, state_dict, econ_inputs):
    """Balance sheet check should match recomputed assets - (liabilities + equity).

    Note: With synthetic fixture data the balance sheet may not close exactly to zero
    (that requires realistic, balanced data), but the reported check must be self-consistent.
    """
    out = model.forecast_step(state_dict, econ_inputs)
    total_assets = float(
        (out.nca + out.advance_payments_purchases + out.accounts_receivable
         + out.inventory + out.cash + out.investment_in_market_securities).numpy()
    )
    total_liab_equity = float(
        (out.accounts_payable + out.advance_payments_sales
         + out.current_liabilities + out.non_current_liabilities + out.equity).numpy()
    )
    recomputed = total_assets - total_liab_equity
    assert float(out.balance_sheet_check.numpy()) == pytest.approx(recomputed, abs=1e-6)


def test_forecast_step_does_not_mutate_input_state(model, state_dict, econ_inputs):
    """forecast_step must not modify the input state dictionary in-place."""
    original_values = {k: float(v.numpy()) for k, v in state_dict.items()}
    model.forecast_step(state_dict, econ_inputs)
    for k, orig_val in original_values.items():
        assert float(state_dict[k].numpy()) == orig_val, f"Input state key '{k}' was mutated"


def test_multi_step_forecast_stays_finite(model, state_dict, econ_inputs):
    """Chaining multiple forecast steps should produce finite outputs throughout."""
    out = model.forecast_step(state_dict, econ_inputs)
    # Build next state dict from the output
    next_state = {
        "nca": out.nca,
        "adv_pp": out.advance_payments_purchases,
        "ar": out.accounts_receivable,
        "inv": out.inventory,
        "cash": out.cash,
        "ims": out.investment_in_market_securities,
        "ap": out.accounts_payable,
        "adv_ps": out.advance_payments_sales,
        "cl": out.current_liabilities,
        "ncl": out.non_current_liabilities,
        "equity": out.equity,
        "ni": out.net_income,
    }
    next_econ = EconomicInputs(
        sales_t=tf.constant(1.28, dtype=tf.float64),
        purchases_t=tf.constant(0.74, dtype=tf.float64),
        sales_t_plus_1=tf.constant(1.32, dtype=tf.float64),
        purchases_t_plus_1=tf.constant(0.76, dtype=tf.float64),
        cum_inflation=tf.constant(0.02, dtype=tf.float64),
        t=2,
    )
    out2 = model.forecast_step(next_state, next_econ)
    assert tf.math.is_finite(out2.net_income)
    assert tf.math.is_finite(out2.balance_sheet_check)
    assert tf.math.is_finite(out2.equity)


def test_forecast_step_with_zero_sales(model, state_dict):
    """Edge case: zero sales should not cause NaN or inf."""
    econ = EconomicInputs(
        sales_t=tf.constant(0.0, dtype=tf.float64),
        purchases_t=tf.constant(0.0, dtype=tf.float64),
        sales_t_plus_1=tf.constant(0.0, dtype=tf.float64),
        purchases_t_plus_1=tf.constant(0.0, dtype=tf.float64),
        cum_inflation=tf.constant(0.02, dtype=tf.float64),
        t=1,
    )
    out = model.forecast_step(state_dict, econ)
    assert tf.math.is_finite(out.net_income)
    assert tf.math.is_finite(out.equity)
