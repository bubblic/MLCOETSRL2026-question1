"""Tests for TrainableFinancialModel with SimpleOpEx.

Verifies that the model works correctly with deterministic OpEx:
- Forecast step produces finite outputs with balance sheet identity.
- Training executes without errors.
- Monte Carlo forecast auto-reduces to 1 sample.
- Save/load roundtrip preserves SimpleOpEx parameters.
- OpEx loss is finite and non-negative.

Run:
    python -m pytest -q tests/test_trainable_financial_model_simpleopex.py
"""

import pytest
import tensorflow as tf

from financial_forecast.models.trainable_financial_model import TrainableFinancialModel
from financial_forecast.models.opex import SimpleOpEx
from financial_forecast.models.liquidity import SimpleLiquidityPolicy
from financial_forecast.models.dividends import SimpleDividendPolicy
from financial_forecast.models.buyback import SimpleBuybackPolicy
from financial_forecast.models.purchases import StaticCostRatioPolicy
from financial_forecast.models.debt import SimpleDebtPolicy
from financial_forecast.models.capex import CapexPolicy
from financial_forecast.models.working_capital import WorkingCapitalPolicy
from financial_forecast.models.tax import SimpleTax
from financial_forecast.inference.trajectory_simulator import DeterministicSimulator
from financial_forecast.training.policy_trainer import PolicyTrainer
from financial_forecast.training.structural_trainer import StructuralTrainer


@pytest.fixture
def model():
    tf.random.set_seed(7)
    m = TrainableFinancialModel(
        opex_module=SimpleOpEx(),
        trajectory_simulator=DeterministicSimulator(),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=SimpleLiquidityPolicy(),
        dividend_policy=SimpleDividendPolicy(),
        buyback_policy=SimpleBuybackPolicy(),
        purchases_policy=StaticCostRatioPolicy(),
        debt_policy=SimpleDebtPolicy(),
        tax_module=SimpleTax(),
    )
    m.base_year = 2018
    m.amount_scale = 1.0
    return m


@pytest.fixture
def mock_state():
    return {
        "nca": tf.constant(1.00, dtype=tf.float64),
        "advance_payments_purchases": tf.constant(0.05, dtype=tf.float64),
        "accounts_receivable": tf.constant(0.15, dtype=tf.float64),
        "inventory": tf.constant(0.03, dtype=tf.float64),
        "cash": tf.constant(0.12, dtype=tf.float64),
        "investment_in_market_securities": tf.constant(0.10, dtype=tf.float64),
        "accounts_payable": tf.constant(0.24, dtype=tf.float64),
        "advance_payments_sales": tf.constant(0.02, dtype=tf.float64),
        "effective_st_debt": tf.constant(0.12, dtype=tf.float64),
        "current_lt_debt": tf.constant(0.10, dtype=tf.float64),
        "non_current_liabilities": tf.constant(0.28, dtype=tf.float64),
        "equity": tf.constant(0.69, dtype=tf.float64),
        "net_income": tf.constant(0.08, dtype=tf.float64),
        "dividends": tf.constant(0.013, dtype=tf.float64),
    }


@pytest.fixture
def mock_inputs():
    return {
        "sales_t": tf.constant(1.20, dtype=tf.float64),
        "year": tf.constant(2023.0, dtype=tf.float64),
        "cum_inflation": tf.constant(1.04, dtype=tf.float64),
    }


@pytest.fixture
def mock_historical():
    sales = tf.constant([1.00, 1.05, 1.10, 1.15, 1.20], dtype=tf.float64)
    cogs = tf.constant([0.58, 0.60, 0.62, 0.64, 0.66], dtype=tf.float64)
    inventory = tf.constant([0.03, 0.032, 0.034, 0.036, 0.038], dtype=tf.float64)
    purchases_first = tf.expand_dims(cogs[0], axis=0)
    purchases_rest = cogs[1:] + (inventory[1:] - inventory[:-1])
    purchases = tf.concat([purchases_first, purchases_rest], axis=0)
    return {
        "sales": sales,
        "purchases": purchases,
        "cogs": cogs,
        "nca": tf.constant([0.90, 0.92, 0.95, 0.98, 1.01], dtype=tf.float64),
        "depreciation": tf.constant(
            [0.045, 0.046, 0.047, 0.048, 0.049], dtype=tf.float64
        ),
        "adv_pay_sales": tf.constant(
            [0.020, 0.021, 0.022, 0.023, 0.024], dtype=tf.float64
        ),
        "adv_pay_purch": tf.constant(
            [0.040, 0.041, 0.042, 0.043, 0.044], dtype=tf.float64
        ),
        "ar": tf.constant([0.16, 0.165, 0.170, 0.175, 0.180], dtype=tf.float64),
        "ap": tf.constant([0.25, 0.255, 0.260, 0.265, 0.270], dtype=tf.float64),
        "inventory": inventory,
        "cash": tf.constant([0.11, 0.112, 0.114, 0.116, 0.118], dtype=tf.float64),
        "ims": tf.constant([0.12, 0.123, 0.126, 0.129, 0.132], dtype=tf.float64),
        "net_income": tf.constant(
            [0.075, 0.078, 0.081, 0.084, 0.087], dtype=tf.float64
        ),
        "dividends": tf.constant(
            [0.012, 0.0125, 0.013, 0.0135, 0.014], dtype=tf.float64
        ),
        "stock_buyback": tf.constant(
            [0.010, 0.0105, 0.011, 0.0115, 0.012], dtype=tf.float64
        ),
        "opex": tf.constant([0.18, 0.185, 0.19, 0.195, 0.20], dtype=tf.float64),
        "tax": tf.constant([0.012, 0.0125, 0.013, 0.0135, 0.014], dtype=tf.float64),
        "eff_st_debt": tf.constant(
            [0.12, 0.122, 0.124, 0.126, 0.128], dtype=tf.float64
        ),
        "current_lt_debt": tf.constant(
            [0.10, 0.101, 0.102, 0.103, 0.104], dtype=tf.float64
        ),
        "non_current_liabilities": tf.constant(
            [0.28, 0.283, 0.286, 0.289, 0.292], dtype=tf.float64
        ),
        "interest_payment": tf.constant(
            [0.021, 0.0215, 0.022, 0.0225, 0.023], dtype=tf.float64
        ),
        "ms_return": tf.constant(
            [0.0058, 0.0060, 0.0062, 0.0064, 0.0066], dtype=tf.float64
        ),
        "equity": tf.constant([0.55, 0.57, 0.59, 0.61, 0.63], dtype=tf.float64),
        "inflation": tf.constant([0.020, 0.021, 0.020, 0.019, 0.020], dtype=tf.float64),
        "years": tf.constant([2018, 2019, 2020, 2021, 2022], dtype=tf.float64),
    }


def test_simple_opex_is_not_stochastic(model):
    """SimpleOpEx should report is_stochastic = False."""
    assert model.opex_module.is_stochastic is False


def test_forecast_step_identities(model, mock_state, mock_inputs):
    """Balance sheet identity should hold with SimpleOpEx."""
    state_next = model.forecast_step(mock_state, mock_inputs, use_mean_opex=True)
    assert float(tf.math.abs(state_next["check"]).numpy()) < 1e-4
    assert float(tf.math.abs(state_next["liquidity_check"]).numpy()) < 1e-4


def test_forecast_step_outputs_finite(model, mock_state, mock_inputs):
    """All forecast_step outputs should be finite with SimpleOpEx."""
    state_next = model.forecast_step(mock_state, mock_inputs, use_mean_opex=True)
    for key, value in state_next.items():
        assert tf.math.is_finite(value), f"{key} is not finite"


def test_forecast_step_deterministic(model, mock_state, mock_inputs):
    """SimpleOpEx should produce identical results across calls (no sampling)."""
    r1 = model.forecast_step(mock_state, mock_inputs, use_mean_opex=False)
    r2 = model.forecast_step(mock_state, mock_inputs, use_mean_opex=False)
    for key in r1:
        assert float(r1[key]) == pytest.approx(
            float(r2[key]), abs=1e-12
        ), f"{key} differs between calls"


def _build_training_data(d):
    """Build HistoricalTrainingData from the mock_historical dict."""
    from financial_forecast.types import HistoricalTrainingData

    return HistoricalTrainingData(
        sales=d["sales"],
        purchases=d["purchases"],
        cogs=d["cogs"],
        nca=d["nca"],
        depreciation=d["depreciation"],
        advance_payments_sales=d["adv_pay_sales"],
        advance_payments_purchases=d["adv_pay_purch"],
        accounts_receivable=d["ar"],
        accounts_payable=d["ap"],
        inventory=d["inventory"],
        cash=d["cash"],
        ims=d["ims"],
        net_income=d["net_income"],
        dividends=d["dividends"],
        stock_buyback=d["stock_buyback"],
        opex=d["opex"],
        tax=d["tax"],
        effective_st_debt=d["eff_st_debt"],
        current_lt_debt=d.get("current_lt_debt", d["eff_st_debt"]),
        non_current_liabilities=d.get("non_current_liabilities", d["eff_st_debt"]),
        interest_payment=d.get("interest_payment", d["eff_st_debt"]),
        ms_return=d.get("ms_return", d["eff_st_debt"]),
        equity=d.get("equity", d["eff_st_debt"]),
        inflation=d["inflation"],
        years=d["years"],
    )


def test_training_executes(model, mock_historical):
    """Training with SimpleOpEx should complete without errors."""
    data = _build_training_data(mock_historical)
    PolicyTrainer(epochs=2).train(
        model,
        data,
        loss_scale_mode="std",
        show_plot=False,
        plot_every=1,
    )
    # Parameters should be finite after training
    assert tf.math.is_finite(tf.cast(model.opex_module.variable_opex_pct, tf.float64))
    assert tf.math.is_finite(tf.cast(model.opex_module.baseline_opex, tf.float64))


def test_mc_forecast_auto_reduces_samples(model, mock_state):
    """MC forecast should auto-reduce to 1 sample for deterministic OpEx."""
    n_years = 3
    sales = tf.fill([n_years], tf.constant(1.20, dtype=tf.float64))
    cum_inf = tf.cast(tf.math.cumprod(1.0 + tf.fill([n_years], 0.02)), tf.float64)
    years = tf.cast(tf.range(2019, 2019 + n_years), tf.float64)

    trajectories = model.trajectory_simulator.run(
        model,
        mock_state,
        sales,
        cum_inf,
        years,
    )

    # Should have been auto-reduced to 1 sample
    for key, arr in trajectories.items():
        assert arr.shape[0] == 1, f"{key}: expected 1 sample, got {arr.shape[0]}"
        assert arr.dtype == tf.float64


def test_save_load_simple_opex(model, tmp_path):
    """Save/load should preserve SimpleOpEx parameters."""
    model.opex_module.variable_opex_pct.assign(0.123)
    model.opex_module.baseline_opex.assign(-456.0)

    save_path = tmp_path / "simple_opex_params.npz"
    model.save_parameters(str(save_path))

    # Change values
    model.opex_module.variable_opex_pct.assign(0.0)
    model.opex_module.baseline_opex.assign(0.0)

    model.load_parameters(str(save_path))
    assert float(model.opex_module.variable_opex_pct.numpy()) == pytest.approx(
        0.123, abs=1e-12
    )
    assert float(model.opex_module.baseline_opex.numpy()) == pytest.approx(
        -456.0, abs=1e-12
    )


def test_simple_opex_loss_finite(model):
    """SimpleOpEx loss should be finite and non-negative."""
    sales = tf.constant([1.0, 1.1, 1.2], dtype=tf.float64)
    opex = tf.constant([0.2, 0.21, 0.22], dtype=tf.float64)
    cum_inf = tf.constant([1.0, 1.02, 1.04], dtype=tf.float64)
    scale = tf.constant(1.0, dtype=tf.float64)

    loss = model.opex_module.loss(opex, sales, cum_inf, scale)
    assert tf.math.is_finite(loss)
    assert float(loss.numpy()) >= 0.0
