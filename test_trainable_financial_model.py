"""Tests for trainable_financial_model.py."""

import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")

import trainable_financial_model as module


@pytest.fixture
def model():
    np.random.seed(1)
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


def test_forecast_step_outputs_finite_and_identity_close(model, state_dict, econ_inputs):
    predicted = model.forecast_step(state_dict, econ_inputs)
    assert isinstance(predicted, module.FinancialState)
    assert np.isfinite(float(predicted.net_income.numpy()))
    check_value = float(predicted.balance_sheet_check.numpy())
    assert np.isfinite(check_value)
    assert np.isfinite(float(predicted.liquidity_check.numpy()))

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
    sales = np.linspace(1.0, 1.2, n, dtype=np.float64)
    purchases = np.linspace(0.6, 0.7, n, dtype=np.float64)
    historical_data = {
        "sales": sales,
        "purchases": purchases,
        "inventory": np.linspace(0.03, 0.04, n, dtype=np.float64),
        "depr": np.linspace(0.04, 0.05, n, dtype=np.float64),
        "nca": np.linspace(0.9, 1.0, n, dtype=np.float64),
        "advance_payments_purchases": np.linspace(0.04, 0.05, n, dtype=np.float64),
        "accounts_receivable": np.linspace(0.15, 0.18, n, dtype=np.float64),
        "cash": np.linspace(0.1, 0.12, n, dtype=np.float64),
        "investment_in_market_securities": np.linspace(0.1, 0.11, n, dtype=np.float64),
        "accounts_payable": np.linspace(0.22, 0.26, n, dtype=np.float64),
        "advance_payments_sales": np.linspace(0.02, 0.03, n, dtype=np.float64),
        "current_liabilities": np.linspace(0.16, 0.19, n, dtype=np.float64),
        "non_current_liabilities": np.linspace(0.3, 0.34, n, dtype=np.float64),
        "equity": np.linspace(0.65, 0.7, n, dtype=np.float64),
        "net_income": np.linspace(0.07, 0.09, n, dtype=np.float64),
        "dividends": np.linspace(0.01, 0.013, n, dtype=np.float64),
        "stock_buyback": np.linspace(0.01, 0.014, n, dtype=np.float64),
        "opex": np.linspace(0.17, 0.2, n, dtype=np.float64),
        "tax": np.linspace(0.01, 0.015, n, dtype=np.float64),
        "inflation": np.zeros(n, dtype=np.float64),
    }
    model.train_simple_policies(historical_data, epochs=1)
    assert np.isfinite(float(model.asset_growth.numpy()))


def test_train_structural_parameters_executes_short_run(model):
    n = 5
    sales = np.linspace(1.0, 1.2, n, dtype=np.float64)
    purchases = np.linspace(0.6, 0.7, n, dtype=np.float64)
    historical_data = {
        "sales": sales,
        "purchases": purchases,
        "inventory": np.linspace(0.03, 0.04, n, dtype=np.float64),
        "depr": np.linspace(0.04, 0.05, n, dtype=np.float64),
        "nca": np.linspace(0.9, 1.0, n, dtype=np.float64),
        "advance_payments_purchases": np.linspace(0.04, 0.05, n, dtype=np.float64),
        "accounts_receivable": np.linspace(0.15, 0.18, n, dtype=np.float64),
        "cash": np.linspace(0.1, 0.12, n, dtype=np.float64),
        "investment_in_market_securities": np.linspace(0.1, 0.11, n, dtype=np.float64),
        "accounts_payable": np.linspace(0.22, 0.26, n, dtype=np.float64),
        "advance_payments_sales": np.linspace(0.02, 0.03, n, dtype=np.float64),
        "current_liabilities": np.linspace(0.16, 0.19, n, dtype=np.float64),
        "non_current_liabilities": np.linspace(0.3, 0.34, n, dtype=np.float64),
        "equity": np.linspace(0.65, 0.7, n, dtype=np.float64),
        "net_income": np.linspace(0.07, 0.09, n, dtype=np.float64),
        "dividends": np.linspace(0.01, 0.013, n, dtype=np.float64),
        "stock_buyback": np.linspace(0.01, 0.014, n, dtype=np.float64),
        "opex": np.linspace(0.17, 0.2, n, dtype=np.float64),
        "tax": np.linspace(0.01, 0.015, n, dtype=np.float64),
        "inflation": np.zeros(n, dtype=np.float64),
    }
    model.train_structural_parameters(historical_data, epochs=1)
    assert np.isfinite(float(model.avg_maturity_years.numpy()))
