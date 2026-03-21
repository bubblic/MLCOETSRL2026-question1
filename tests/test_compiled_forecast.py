"""Tests for the compiled (tf.while_loop) Monte Carlo forecast path.

Verifies:
- Deterministic equivalence with the dict-based forecast_step.
- Shape and dtype of compiled trajectory output.
- Finiteness of all trajectory values.
- Balance sheet identity: Assets = Liabilities + Equity.
"""

import pytest
import tensorflow as tf

from financial_forecast.inference.monte_carlo_forecast import run_monte_carlo_forecast
from financial_forecast.inference.state_index import (
    initial_state_to_batched,
    DIAGNOSTIC_KEYS,
)
from financial_forecast.models.trainable_financial_model import TrainableFinancialModel


@pytest.fixture
def model():
    tf.random.set_seed(42)
    m = TrainableFinancialModel()
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


def test_deterministic_equivalence(model, mock_state):
    """Dict-based and compiled paths must agree with mean OpEx and zero noise."""
    n_years = 3
    sales_vals = [1.20, 1.25, 1.30]
    year_vals = [2023.0, 2024.0, 2025.0]
    cum_inf_vals = [1.04, 1.07, 1.10]

    # --- Dict-based path ---
    dict_results = []
    state = mock_state.copy()
    for step in range(n_years):
        inputs = {
            "sales_t": tf.constant(sales_vals[step], dtype=tf.float64),
            "year": tf.constant(year_vals[step], dtype=tf.float64),
            "cum_inflation": tf.constant(cum_inf_vals[step], dtype=tf.float64),
        }
        state = model.forecast_step(state, inputs, use_mean_opex=True)
        dict_results.append(state)

    # --- Compiled path ---
    batched_state = initial_state_to_batched(mock_state, 1)

    compiled_diags = []
    for step in range(n_years):
        sales_t = tf.constant([sales_vals[step]], dtype=tf.float64)
        cum_inf = tf.constant(cum_inf_vals[step], dtype=tf.float64)
        opex = model.opex_module.predict(sales_t, cum_inf, use_mean=True)
        batched_state, diagnostics = model.forecast_step_compiled(
            batched_state,
            sales_t,
            tf.constant(year_vals[step], dtype=tf.float64),
            opex,
        )
        compiled_diags.append(diagnostics)

    # --- Compare ---
    for step in range(n_years):
        dr = dict_results[step]
        cd = compiled_diags[step]
        for i, key in enumerate(DIAGNOSTIC_KEYS):
            if key == "total_assets":
                dict_val = float(
                    sum(
                        dr[k]
                        for k in [
                            "nca",
                            "advance_payments_purchases",
                            "accounts_receivable",
                            "inventory",
                            "cash",
                            "investment_in_market_securities",
                        ]
                    )
                )
            else:
                dict_val = float(dr[key])
            compiled_val = float(cd[0, i])
            assert (
                abs(dict_val - compiled_val) < 1e-10
            ), f"Step {step}, {key}: dict={dict_val}, compiled={compiled_val}"


def test_compiled_shapes_and_dtypes(model, mock_state):
    """Compiled trajectories must have shape [n_samples, n_years] and dtype float64."""
    n_years, n_samples = 5, 4
    sales = tf.fill([n_years], tf.constant(1.20, dtype=tf.float64))
    cum_inf = tf.cast(tf.math.cumprod(1.0 + tf.fill([n_years], 0.02)), tf.float64)
    years = tf.cast(tf.range(2019, 2019 + n_years), tf.float64)

    trajectories = run_monte_carlo_forecast(
        model,
        mock_state,
        sales,
        cum_inf,
        years,
        n_samples=n_samples,
    )

    for key, arr in trajectories.items():
        assert arr.shape == (n_samples, n_years), f"{key}: shape {arr.shape}"
        assert arr.dtype == tf.float64, f"{key}: dtype {arr.dtype}"


def test_compiled_finiteness(model, mock_state):
    """All compiled trajectory values must be finite."""
    n_years, n_samples = 10, 5
    sales = tf.fill([n_years], tf.constant(1.20, dtype=tf.float64))
    cum_inf = tf.cast(tf.math.cumprod(1.0 + tf.fill([n_years], 0.02)), tf.float64)
    years = tf.cast(tf.range(2019, 2019 + n_years), tf.float64)

    trajectories = run_monte_carlo_forecast(
        model,
        mock_state,
        sales,
        cum_inf,
        years,
        n_samples=n_samples,
    )

    for key, arr in trajectories.items():
        assert tf.reduce_all(tf.math.is_finite(arr)), f"{key} has non-finite values"


def test_compiled_balance_sheet_identity(model, mock_state):
    """Assets must equal Liabilities + Equity for every sample and year."""
    n_years, n_samples = 5, 10
    sales = tf.fill([n_years], tf.constant(1.20, dtype=tf.float64))
    cum_inf = tf.cast(tf.math.cumprod(1.0 + tf.fill([n_years], 0.02)), tf.float64)
    years = tf.cast(tf.range(2019, 2019 + n_years), tf.float64)

    trajectories = run_monte_carlo_forecast(
        model,
        mock_state,
        sales,
        cum_inf,
        years,
        n_samples=n_samples,
    )

    total_assets = trajectories["total_assets"]
    total_liab_equity = (
        trajectories["accounts_payable"]
        + trajectories["advance_payments_sales"]
        + trajectories["effective_st_debt"]
        + trajectories["current_lt_debt"]
        + trajectories["non_current_liabilities"]
        + trajectories["equity"]
    )
    mismatch = tf.abs(total_assets - total_liab_equity)
    max_mismatch = float(tf.reduce_max(mismatch))
    assert max_mismatch < 1e-6, f"Balance sheet mismatch: {max_mismatch}"
