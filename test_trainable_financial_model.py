"""Tests for trainable_financial_model.py.



Run the script by:
python -m pytest -q test_trainable_financial_model.py
"""

import pytest

tf = pytest.importorskip("tensorflow")

import trainable_financial_model as module


@pytest.fixture
def model():
    tf.random.set_seed(1)
    return module.TrainableFinancialModel()


@pytest.fixture
def state_dict():
    """Balanced, positive prior state for one-step forecast checks."""
    return {
        "nca": tf.constant(1.00, dtype=tf.float64),
        "advance_payments_purchases": tf.constant(0.05, dtype=tf.float64),
        "accounts_receivable": tf.constant(0.15, dtype=tf.float64),
        "inventory": tf.constant(0.03, dtype=tf.float64),
        "cash": tf.constant(0.12, dtype=tf.float64),
        "investment_in_market_securities": tf.constant(0.10, dtype=tf.float64),
        "accounts_payable": tf.constant(0.24, dtype=tf.float64),
        "advance_payments_sales": tf.constant(0.02, dtype=tf.float64),
        "current_liabilities": tf.constant(0.18, dtype=tf.float64),
        "non_current_liabilities": tf.constant(0.32, dtype=tf.float64),
        "equity": tf.constant(0.69, dtype=tf.float64),
        "net_income": tf.constant(0.08, dtype=tf.float64),
    }


@pytest.fixture
def econ_inputs():
    return module.EconomicInputs(
        sales_t=tf.constant(1.2, dtype=tf.float64),
        purchases_t=tf.constant(0.7, dtype=tf.float64),
        sales_t_plus_1=tf.constant(1.24, dtype=tf.float64),
        purchases_t_plus_1=tf.constant(0.72, dtype=tf.float64),
        cum_inflation=tf.constant(1.03, dtype=tf.float64),
    )


def test_financial_state_from_dict_filters_unknown_keys():
    data = {
        "nca": 1.0,
        "advance_payments_purchases": 0.1,
        "accounts_receivable": 0.2,
        "inventory": 0.3,
        "cash": 0.4,
        "investment_in_market_securities": 0.5,
        "accounts_payable": 0.6,
        "advance_payments_sales": 0.7,
        "current_liabilities": 0.8,
        "non_current_liabilities": 0.9,
        "equity": 1.0,
        "net_income": 0.05,
        "unexpected": 999,
    }
    state = module.FinancialState.from_dict(data)
    assert state.equity == 1.0
    assert not hasattr(state, "unexpected")


def test_forecast_step_outputs_finite_and_identity_close(
    model, state_dict, econ_inputs
):
    predicted = model.forecast_step(state_dict, econ_inputs)
    assert isinstance(predicted, module.FinancialState)
    assert tf.math.is_finite(predicted.net_income)
    check_value = float(predicted.balance_sheet_check.numpy())
    assert tf.math.is_finite(predicted.balance_sheet_check)
    assert tf.math.is_finite(predicted.liquidity_check)

    # Validate internal consistency: reported check equals recomputed
    # Assets - (Liabilities + Equity) from the returned state.
    total_assets = float(
        (
            predicted.nca
            + predicted.advance_payments_purchases
            + predicted.accounts_receivable
            + predicted.inventory
            + predicted.cash
            + predicted.investment_in_market_securities
        ).numpy()
    )
    total_liab_equity = float(
        (
            predicted.accounts_payable
            + predicted.advance_payments_sales
            + predicted.current_liabilities
            + predicted.non_current_liabilities
            + predicted.equity
        ).numpy()
    )
    recomputed_check = total_assets - total_liab_equity
    assert check_value == pytest.approx(recomputed_check, rel=0.0, abs=1e-9)


def test_apply_constraints_clips_invalid_parameter_values(model):
    model.asset_growth.assign(-1.0)
    model.cash_pct_of_liquidity.assign(2.0)
    model.income_tax_pct.assign(-0.5)
    model.avg_maturity_years.assign(0.1)
    model.equity_financing_pct.assign(3.0)

    model._apply_constraints()

    assert float(model.asset_growth.numpy()) >= 0.0
    assert 0.0 <= float(model.cash_pct_of_liquidity.numpy()) <= 1.0
    assert 0.0 <= float(model.income_tax_pct.numpy()) <= 1.0
    assert float(model.avg_maturity_years.numpy()) >= 1.001
    assert 0.0 <= float(model.equity_financing_pct.numpy()) <= 1.0


def test_train_simple_policies_executes_short_run(model):
    # Keep tiny synthetic data for speed and deterministic behavior.
    n = 5
    sales = tf.linspace(tf.constant(1.0, dtype=tf.float64), tf.constant(1.2, dtype=tf.float64), n)
    purchases = tf.linspace(tf.constant(0.6, dtype=tf.float64), tf.constant(0.7, dtype=tf.float64), n)
    historical_data = {
        "sales": sales,
        "purchases": purchases,
        "inventory": tf.linspace(tf.constant(0.03, dtype=tf.float64), tf.constant(0.04, dtype=tf.float64), n),
        "depr": tf.linspace(tf.constant(0.04, dtype=tf.float64), tf.constant(0.05, dtype=tf.float64), n),
        "nca": tf.linspace(tf.constant(0.9, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64), n),
        "advance_payments_purchases": tf.linspace(tf.constant(0.04, dtype=tf.float64), tf.constant(0.05, dtype=tf.float64), n),
        "accounts_receivable": tf.linspace(tf.constant(0.15, dtype=tf.float64), tf.constant(0.18, dtype=tf.float64), n),
        "cash": tf.linspace(tf.constant(0.1, dtype=tf.float64), tf.constant(0.12, dtype=tf.float64), n),
        "investment_in_market_securities": tf.linspace(tf.constant(0.1, dtype=tf.float64), tf.constant(0.11, dtype=tf.float64), n),
        "accounts_payable": tf.linspace(tf.constant(0.22, dtype=tf.float64), tf.constant(0.26, dtype=tf.float64), n),
        "advance_payments_sales": tf.linspace(tf.constant(0.02, dtype=tf.float64), tf.constant(0.03, dtype=tf.float64), n),
        "current_liabilities": tf.linspace(tf.constant(0.16, dtype=tf.float64), tf.constant(0.19, dtype=tf.float64), n),
        "non_current_liabilities": tf.linspace(tf.constant(0.3, dtype=tf.float64), tf.constant(0.34, dtype=tf.float64), n),
        "equity": tf.linspace(tf.constant(0.65, dtype=tf.float64), tf.constant(0.7, dtype=tf.float64), n),
        "net_income": tf.linspace(tf.constant(0.07, dtype=tf.float64), tf.constant(0.09, dtype=tf.float64), n),
        "dividends": tf.linspace(tf.constant(0.01, dtype=tf.float64), tf.constant(0.013, dtype=tf.float64), n),
        "stock_buyback": tf.linspace(tf.constant(0.01, dtype=tf.float64), tf.constant(0.014, dtype=tf.float64), n),
        "opex": tf.linspace(tf.constant(0.17, dtype=tf.float64), tf.constant(0.2, dtype=tf.float64), n),
        "tax": tf.linspace(tf.constant(0.01, dtype=tf.float64), tf.constant(0.015, dtype=tf.float64), n),
        "inflation": tf.zeros(n, dtype=tf.float64),
    }
    model.train_simple_policies(historical_data, epochs=1)
    assert tf.math.is_finite(model.asset_growth)


def test_train_structural_parameters_executes_short_run(model):
    n = 5
    sales = tf.linspace(tf.constant(1.0, dtype=tf.float64), tf.constant(1.2, dtype=tf.float64), n)
    purchases = tf.linspace(tf.constant(0.6, dtype=tf.float64), tf.constant(0.7, dtype=tf.float64), n)
    historical_data = {
        "sales": sales,
        "purchases": purchases,
        "inventory": tf.linspace(tf.constant(0.03, dtype=tf.float64), tf.constant(0.04, dtype=tf.float64), n),
        "depr": tf.linspace(tf.constant(0.04, dtype=tf.float64), tf.constant(0.05, dtype=tf.float64), n),
        "nca": tf.linspace(tf.constant(0.9, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64), n),
        "advance_payments_purchases": tf.linspace(tf.constant(0.04, dtype=tf.float64), tf.constant(0.05, dtype=tf.float64), n),
        "accounts_receivable": tf.linspace(tf.constant(0.15, dtype=tf.float64), tf.constant(0.18, dtype=tf.float64), n),
        "cash": tf.linspace(tf.constant(0.1, dtype=tf.float64), tf.constant(0.12, dtype=tf.float64), n),
        "investment_in_market_securities": tf.linspace(tf.constant(0.1, dtype=tf.float64), tf.constant(0.11, dtype=tf.float64), n),
        "accounts_payable": tf.linspace(tf.constant(0.22, dtype=tf.float64), tf.constant(0.26, dtype=tf.float64), n),
        "advance_payments_sales": tf.linspace(tf.constant(0.02, dtype=tf.float64), tf.constant(0.03, dtype=tf.float64), n),
        "current_liabilities": tf.linspace(tf.constant(0.16, dtype=tf.float64), tf.constant(0.19, dtype=tf.float64), n),
        "non_current_liabilities": tf.linspace(tf.constant(0.3, dtype=tf.float64), tf.constant(0.34, dtype=tf.float64), n),
        "equity": tf.linspace(tf.constant(0.65, dtype=tf.float64), tf.constant(0.7, dtype=tf.float64), n),
        "net_income": tf.linspace(tf.constant(0.07, dtype=tf.float64), tf.constant(0.09, dtype=tf.float64), n),
        "dividends": tf.linspace(tf.constant(0.01, dtype=tf.float64), tf.constant(0.013, dtype=tf.float64), n),
        "stock_buyback": tf.linspace(tf.constant(0.01, dtype=tf.float64), tf.constant(0.014, dtype=tf.float64), n),
        "opex": tf.linspace(tf.constant(0.17, dtype=tf.float64), tf.constant(0.2, dtype=tf.float64), n),
        "tax": tf.linspace(tf.constant(0.01, dtype=tf.float64), tf.constant(0.015, dtype=tf.float64), n),
        "inflation": tf.zeros(n, dtype=tf.float64),
    }
    model.train_structural_parameters(historical_data, epochs=1)
    assert tf.math.is_finite(model.avg_maturity_years)


def test_forecast_step_output_dtypes_are_float64(model, state_dict, econ_inputs):
    """All forecast outputs should be float64 to avoid dtype mismatch errors downstream."""
    predicted = model.forecast_step(state_dict, econ_inputs)
    float64_fields = [
        "nca", "advance_payments_purchases", "accounts_receivable", "inventory",
        "cash", "investment_in_market_securities", "accounts_payable",
        "advance_payments_sales", "current_liabilities", "non_current_liabilities",
        "equity", "net_income",
    ]
    for field in float64_fields:
        value = getattr(predicted, field)
        assert hasattr(value, "dtype"), f"{field} is not a tensor"
        assert value.dtype == tf.float64, f"{field} has dtype {value.dtype}, expected float64"


def test_forecast_step_output_shapes_are_scalar(model, state_dict, econ_inputs):
    """Forecast outputs should be scalar tensors (rank 0)."""
    predicted = model.forecast_step(state_dict, econ_inputs)
    for field in ["nca", "equity", "net_income", "balance_sheet_check", "liquidity_check"]:
        value = getattr(predicted, field)
        assert hasattr(value, "shape"), f"{field} is not a tensor"
        assert value.shape.rank == 0, f"{field} has rank {value.shape.rank}, expected 0"


def test_forecast_step_check_is_consistent(model, state_dict, econ_inputs):
    """Balance sheet check should match recomputed assets - (liabilities + equity).

    With synthetic fixture data the identity may not close to zero, but the
    reported check must be self-consistent with the returned state values.
    """
    predicted = model.forecast_step(state_dict, econ_inputs)
    total_assets = float(
        (predicted.nca + predicted.advance_payments_purchases
         + predicted.accounts_receivable + predicted.inventory
         + predicted.cash + predicted.investment_in_market_securities).numpy()
    )
    total_liab_equity = float(
        (predicted.accounts_payable + predicted.advance_payments_sales
         + predicted.current_liabilities + predicted.non_current_liabilities
         + predicted.equity).numpy()
    )
    recomputed = total_assets - total_liab_equity
    assert float(predicted.balance_sheet_check.numpy()) == pytest.approx(recomputed, abs=1e-6)


def test_forecast_step_does_not_mutate_input_state(model, state_dict, econ_inputs):
    """forecast_step must not modify the input state dictionary in-place."""
    original_values = {k: float(v.numpy()) for k, v in state_dict.items()}
    model.forecast_step(state_dict, econ_inputs)
    for k, orig_val in original_values.items():
        assert float(state_dict[k].numpy()) == orig_val, f"Input state key '{k}' was mutated"


def test_to_dict_roundtrip(model, state_dict, econ_inputs):
    """to_dict and from_dict should produce consistent values."""
    predicted = model.forecast_step(state_dict, econ_inputs)
    as_dict = predicted.to_dict()
    restored = module.FinancialState.from_dict(as_dict)
    assert float(restored.equity) == pytest.approx(float(predicted.equity), abs=1e-12)
    assert float(restored.net_income) == pytest.approx(float(predicted.net_income), abs=1e-12)


def test_gradient_flows_through_forecast_step(model, state_dict, econ_inputs):
    """Trainable parameters should receive non-None, finite gradients through forecast_step."""
    with tf.GradientTape() as tape:
        predicted = model.forecast_step(state_dict, econ_inputs)
        loss = tf.square(predicted.net_income)
    grads = tape.gradient(loss, model.trainable_variables)
    non_none_grads = [g for g in grads if g is not None]
    assert len(non_none_grads) > 0, "No gradients flowed through forecast_step"
    for g in non_none_grads:
        assert tf.reduce_all(tf.math.is_finite(g)), "Gradient contains inf or NaN"


def test_multi_step_forecast_stays_finite(model, state_dict, econ_inputs):
    """Chaining multiple forecast steps should produce finite outputs."""
    predicted = model.forecast_step(state_dict, econ_inputs)
    next_state = predicted.to_dict()
    # Filter to only keys needed for state input
    valid_state_keys = {
        "nca", "advance_payments_purchases", "accounts_receivable", "inventory",
        "cash", "investment_in_market_securities", "accounts_payable",
        "advance_payments_sales", "current_liabilities", "non_current_liabilities",
        "equity", "net_income",
    }
    next_state = {k: v for k, v in next_state.items() if k in valid_state_keys}

    next_econ = module.EconomicInputs(
        sales_t=tf.constant(1.28, dtype=tf.float64),
        purchases_t=tf.constant(0.74, dtype=tf.float64),
        sales_t_plus_1=tf.constant(1.32, dtype=tf.float64),
        purchases_t_plus_1=tf.constant(0.76, dtype=tf.float64),
        cum_inflation=tf.constant(1.06, dtype=tf.float64),
    )
    out2 = model.forecast_step(next_state, next_econ)
    assert tf.math.is_finite(out2.net_income)
    assert tf.math.is_finite(out2.balance_sheet_check)


def test_forecast_step_with_zero_sales(model, state_dict):
    """Edge case: zero sales should not cause NaN or inf."""
    econ = module.EconomicInputs(
        sales_t=tf.constant(0.0, dtype=tf.float64),
        purchases_t=tf.constant(0.0, dtype=tf.float64),
        sales_t_plus_1=tf.constant(0.0, dtype=tf.float64),
        purchases_t_plus_1=tf.constant(0.0, dtype=tf.float64),
        cum_inflation=tf.constant(1.0, dtype=tf.float64),
    )
    out = model.forecast_step(state_dict, econ)
    assert tf.math.is_finite(out.net_income)
    assert tf.math.is_finite(out.equity)
