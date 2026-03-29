"""Tests for BaseFinancialModel with prepare() and ForecastPipeline.

Verifies that the simple model configuration produces valid forecasts
without any gradient training, using the prepare → pipeline flow.

Run:
    python -m pytest -q tests/test_simple_model_forecast.py
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


@pytest.fixture
def model():
    """Create and prepare a BaseFinancialModel with mock data."""
    tf.random.set_seed(42)
    m = BaseFinancialModel(
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
    m.base_year = 2018
    m.amount_scale = 1.0
    m.opex_module.variable_opex_pct.assign(0.22)
    m.opex_module.baseline_opex.assign(0.0)
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


def test_forecast_step_identities(model, mock_state, mock_inputs):
    """Balance sheet identity should hold without training."""
    state_next = model.forecast_step(
        mock_state,
        mock_inputs,
        use_mean_opex=True,
    )
    assert float(tf.math.abs(state_next["check"]).numpy()) < 1e-4
    assert float(tf.math.abs(state_next["liquidity_check"]).numpy()) < 1e-4


def test_forecast_step_outputs_finite(model, mock_state, mock_inputs):
    """All outputs should be finite with default policy parameters."""
    state_next = model.forecast_step(
        mock_state,
        mock_inputs,
        use_mean_opex=True,
    )
    for key, value in state_next.items():
        assert tf.math.is_finite(value), f"{key} is not finite"


def test_forecast_step_deterministic(model, mock_state, mock_inputs):
    """Should produce identical results across calls (no sampling)."""
    r1 = model.forecast_step(mock_state, mock_inputs, use_mean_opex=False)
    r2 = model.forecast_step(mock_state, mock_inputs, use_mean_opex=False)
    for key in r1:
        assert float(r1[key]) == pytest.approx(
            float(r2[key]),
            abs=1e-12,
        ), f"{key} differs between calls"


def test_trajectory_simulation_no_training(model, mock_state):
    """DeterministicSimulator should run without any prior training."""
    n_years = 3
    sales = tf.fill([n_years], tf.constant(1.20, dtype=tf.float64))
    cum_inf = tf.cast(
        tf.math.cumprod(1.0 + tf.fill([n_years], 0.02)),
        tf.float64,
    )
    years = tf.cast(tf.range(2019, 2019 + n_years), tf.float64)

    trajectories = model.trajectory_simulator.run(
        model,
        mock_state,
        sales,
        cum_inf,
        years,
    )

    for key, arr in trajectories.items():
        assert arr.shape[0] == 1
        assert arr.shape[1] == n_years
        assert arr.dtype == tf.float64
        assert tf.reduce_all(tf.math.is_finite(arr)), f"{key} has non-finite values"


def test_trajectory_balance_sheet_identity(model, mock_state):
    """Balance sheet identity should hold across all forecast years."""
    n_years = 5
    sales = tf.constant(
        [1.20, 1.25, 1.30, 1.35, 1.40],
        dtype=tf.float64,
    )
    cum_inf = tf.cast(
        tf.math.cumprod(1.0 + tf.fill([n_years], 0.02)),
        tf.float64,
    )
    years = tf.cast(tf.range(2019, 2019 + n_years), tf.float64)

    trajectories = model.trajectory_simulator.run(
        model,
        mock_state,
        sales,
        cum_inf,
        years,
    )

    check = trajectories["check"]
    max_abs_check = float(tf.reduce_max(tf.abs(check)).numpy())
    assert (
        max_abs_check < 1e-4
    ), f"Balance sheet identity violated: max |check| = {max_abs_check}"


def test_prepare_sets_model_state():
    """prepare() should set base_year, amount_scale, and initial state."""
    m = BaseFinancialModel(
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

    n = 5
    fs = {
        "years": tf.constant([2018, 2019, 2020, 2021, 2022], dtype=tf.float64),
        "sales": tf.constant([1e11, 1.1e11, 1.2e11, 1.3e11, 1.4e11], dtype=tf.float64),
        "purchases": tf.constant([5e10] * n, dtype=tf.float64),
        "cogs": tf.constant([5e10] * n, dtype=tf.float64),
        "nca": tf.constant([2e11] * n, dtype=tf.float64),
        "depreciation": tf.constant([1e10] * n, dtype=tf.float64),
        "advance_payments_purchases": tf.constant([2e9] * n, dtype=tf.float64),
        "accounts_receivable": tf.constant([1.5e10] * n, dtype=tf.float64),
        "accounts_payable": tf.constant([2e10] * n, dtype=tf.float64),
        "advance_payments_sales": tf.constant([2e9] * n, dtype=tf.float64),
        "cash": tf.constant([1e10] * n, dtype=tf.float64),
        "ims": tf.constant([1e10] * n, dtype=tf.float64),
        "inventory": tf.constant([3e9] * n, dtype=tf.float64),
        "current_liabilities": tf.constant([4e10] * n, dtype=tf.float64),
        "non_current_liabilities": tf.constant([1e11] * n, dtype=tf.float64),
        "equity": tf.constant([5e10] * n, dtype=tf.float64),
        "net_income": tf.constant([2e10] * n, dtype=tf.float64),
        "dividends": tf.constant([3e9] * n, dtype=tf.float64),
        "stock_buyback": tf.constant([5e9] * n, dtype=tf.float64),
        "opex": tf.constant([2e10] * n, dtype=tf.float64),
        "tax": tf.constant([5e9] * n, dtype=tf.float64),
        "current_lt_debt": tf.constant([1e10] * n, dtype=tf.float64),
        "interest_payment": tf.constant([3e9] * n, dtype=tf.float64),
        "ms_return": tf.constant([5e8] * n, dtype=tf.float64),
    }

    m.prepare(fs)

    assert m.base_year == 2018
    assert m.amount_scale is not None
    assert m._initial_state is not None


def test_model_is_not_trainable(model):
    """BaseFinancialModel should not have training/serialization methods."""
    assert not hasattr(model, "save_parameters")
    assert not hasattr(model, "load_parameters")
    assert not hasattr(model, "train")
