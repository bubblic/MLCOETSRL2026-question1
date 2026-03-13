"""Pytest suite for the refactored trainable financial model pipeline.

This suite focuses on:
- Accounting identity integrity in single-step forecasting.
- Bayesian OpEx variational sampling sanity.
- Parameter persistence roundtrip (save/load).
- Bijector-implied economic bounds.
- Short-horizon training execution stability.
- Monte Carlo trajectory finite-value stability.


Run the script by:
python -m pytest -q test_trainable_financial_model_taxanomaly_effstdebt_bayesian_vi_refactored.py
"""

import pytest
import tensorflow as tf

from financial_model_pipeline.forecast import run_monte_carlo_forecast
from financial_model_pipeline.model import TrainableFinancialModel


@pytest.fixture
def model():
    """Yield a fresh model instance with deterministic random seeds."""
    tf.random.set_seed(7)
    return TrainableFinancialModel(base_year=2018)


@pytest.fixture
def mock_historical_data():
    """Yield compact synthetic, positive, float64-friendly historical series.

    Values are intentionally small and smooth to keep gradients stable while still
    preserving realistic ratio relationships (for example, COGS/Sales in (0, 1)).
    """
    sales = tf.constant([1.00, 1.05, 1.10, 1.15, 1.20], dtype=tf.float64)
    cogs = tf.constant([0.58, 0.60, 0.62, 0.64, 0.66], dtype=tf.float64)
    inventory = tf.constant([0.03, 0.032, 0.034, 0.036, 0.038], dtype=tf.float64)

    # Keep purchases coherent with the inventory identity:
    # purchases_t = cogs_t + (inventory_t - inventory_{t-1})
    purchases_first = tf.expand_dims(cogs[0], axis=0)
    purchases_rest = cogs[1:] + (inventory[1:] - inventory[:-1])
    purchases = tf.concat([purchases_first, purchases_rest], axis=0)

    nca = tf.constant([0.90, 0.92, 0.95, 0.98, 1.01], dtype=tf.float64)
    depreciation = tf.constant([0.045, 0.046, 0.047, 0.048, 0.049], dtype=tf.float64)
    adv_pay_sales = tf.constant([0.020, 0.021, 0.022, 0.023, 0.024], dtype=tf.float64)
    adv_pay_purch = tf.constant([0.040, 0.041, 0.042, 0.043, 0.044], dtype=tf.float64)
    ar = tf.constant([0.16, 0.165, 0.170, 0.175, 0.180], dtype=tf.float64)
    ap = tf.constant([0.25, 0.255, 0.260, 0.265, 0.270], dtype=tf.float64)
    cash = tf.constant([0.11, 0.112, 0.114, 0.116, 0.118], dtype=tf.float64)
    ims = tf.constant([0.12, 0.123, 0.126, 0.129, 0.132], dtype=tf.float64)
    net_income = tf.constant([0.075, 0.078, 0.081, 0.084, 0.087], dtype=tf.float64)
    dividends = tf.constant([0.012, 0.0125, 0.013, 0.0135, 0.014], dtype=tf.float64)
    stock_buyback = tf.constant([0.010, 0.0105, 0.011, 0.0115, 0.012], dtype=tf.float64)
    opex = tf.constant([0.18, 0.185, 0.19, 0.195, 0.20], dtype=tf.float64)
    tax = tf.constant([0.012, 0.0125, 0.013, 0.0135, 0.014], dtype=tf.float64)
    eff_st_debt = tf.constant([0.12, 0.122, 0.124, 0.126, 0.128], dtype=tf.float64)
    current_lt_debt = tf.constant([0.10, 0.101, 0.102, 0.103, 0.104], dtype=tf.float64)
    non_current_liabilities = tf.constant(
        [0.28, 0.283, 0.286, 0.289, 0.292], dtype=tf.float64
    )
    interest_payment = tf.constant([0.021, 0.0215, 0.022, 0.0225, 0.023], dtype=tf.float64)
    ms_return = tf.constant([0.0058, 0.0060, 0.0062, 0.0064, 0.0066], dtype=tf.float64)
    equity = tf.constant([0.55, 0.57, 0.59, 0.61, 0.63], dtype=tf.float64)
    inflation = tf.constant([0.020, 0.021, 0.020, 0.019, 0.020], dtype=tf.float64)
    tax_onetime_payments = tf.zeros(5, dtype=tf.float64)
    years = tf.constant([2018, 2019, 2020, 2021, 2022], dtype=tf.float64)

    return {
        "sales": sales,
        "purchases": purchases,
        "cogs": cogs,
        "nca": nca,
        "depreciation": depreciation,
        "adv_pay_sales": adv_pay_sales,
        "adv_pay_purch": adv_pay_purch,
        "ar": ar,
        "ap": ap,
        "inventory": inventory,
        "cash": cash,
        "ims": ims,
        "net_income": net_income,
        "dividends": dividends,
        "stock_buyback": stock_buyback,
        "opex": opex,
        "tax": tax,
        "eff_st_debt": eff_st_debt,
        "current_lt_debt": current_lt_debt,
        "non_current_liabilities": non_current_liabilities,
        "interest_payment": interest_payment,
        "ms_return": ms_return,
        "equity": equity,
        "inflation": inflation,
        "tax_onetime_payments": tax_onetime_payments,
        "years": years,
    }


@pytest.fixture
def mock_forecast_state():
    """Yield a valid prior-period state dictionary for forecast_step."""
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
        # Keep prior balance sheet exactly closed:
        # Assets (1.45) = Liabilities (0.76) + Equity (0.69).
        "equity": tf.constant(0.69, dtype=tf.float64),
        "net_income": tf.constant(0.08, dtype=tf.float64),
        "dividends": tf.constant(0.013, dtype=tf.float64),
    }


@pytest.fixture
def mock_forecast_inputs():
    """Yield valid per-period input dictionary for forecast_step."""
    return {
        "sales_t": tf.constant(1.20, dtype=tf.float64),
        "year": tf.constant(2023.0, dtype=tf.float64),
        "cum_inflation": tf.constant(1.04, dtype=tf.float64),
    }


def test_forecast_step_identities(model, mock_forecast_state, mock_forecast_inputs):
    """Assets-(Liabilities+Equity) and liquidity closure should be near zero."""
    state_next = model.forecast_step(
        mock_forecast_state,
        mock_forecast_inputs,
        use_mean_opex=True,
    )

    assert float(tf.math.abs(state_next["check"]).numpy()) < 1e-4
    assert float(tf.math.abs(state_next["liquidity_check"]).numpy()) < 1e-4


def test_sample_opex_params(model):
    """Posterior samples and KL term should be finite scalar float64 tensors."""
    var_opex_sample, base_opex_sample = model.sample_opex_params()
    kl_div = model.get_opex_kl_divergence()

    assert isinstance(var_opex_sample, tf.Tensor)
    assert isinstance(base_opex_sample, tf.Tensor)
    assert isinstance(kl_div, tf.Tensor)

    assert var_opex_sample.dtype == tf.float64
    assert base_opex_sample.dtype == tf.float64
    assert kl_div.dtype == tf.float64

    # Scalar samples/penalty are expected for this VI parameterization.
    assert var_opex_sample.shape.rank == 0
    assert base_opex_sample.shape.rank == 0
    assert kl_div.shape.rank == 0

    assert tf.math.is_finite(var_opex_sample)
    assert tf.math.is_finite(base_opex_sample)
    assert tf.math.is_finite(kl_div)


def test_save_load_parameters(model, tmp_path):
    """Saving and reloading should restore original parameter values exactly."""
    save_path = tmp_path / "params_test.npz"

    original_asset_growth = float(model.asset_growth.numpy())
    model.save_parameters(str(save_path))

    # Change parameter after save to verify that load performs a true restore.
    model.asset_growth.assign(
        tf.constant(original_asset_growth + 0.5, dtype=tf.float64)
    )
    assert float(model.asset_growth.numpy()) != pytest.approx(original_asset_growth)

    model.load_parameters(str(save_path))
    assert float(model.asset_growth.numpy()) == pytest.approx(
        original_asset_growth, rel=0.0, abs=1e-12
    )


def test_parameter_bounds(model):
    """Transformed variables should respect their economic constraints."""
    softplus_params = [
        model.asset_growth,
        model.asset_maintain,
        model.depreciation_rate,
        model.advance_payments_sales_pct,
        model.advance_payments_purchases_pct,
        model.account_receivables_pct,
        model.account_payables_pct,
        model.inventory_pct,
        model.q_var_opex_scale,
        model.q_base_opex_scale,
        model.noise_sigma,
        model.avg_short_term_interest_pct,
        model.avg_long_term_interest_pct,
        model.market_securities_return_pct,
    ]
    for param in softplus_params:
        assert float(param.numpy()) >= 0.0

    sigmoid_params = [
        model.income_tax_pct,
        model.dividend_payout_ratio_pct,
        model.dividend_adjustment_speed,
    ]
    for param in sigmoid_params:
        value = float(param.numpy())
        assert 0.0 <= value <= 1.0

    # Shift(1.001) + Softplus bijector imposes a strict lower bound > 1.001.
    assert float(model.avg_maturity_years.numpy()) > 1.001


def test_training_step_execution(
    model,
    mock_historical_data,
    mock_forecast_state,
    mock_forecast_inputs,
):
    """Both training loops should run for a few epochs without numerical failure."""
    d = mock_historical_data

    # Simple policy training: very short run just for execution and finite outputs.
    model.train_simple_policies(
        historical_sales=d["sales"],
        historical_purchases=d["purchases"],
        historical_cogs=d["cogs"],
        historical_nca=d["nca"],
        historical_depreciation=d["depreciation"],
        historical_adv_pay_sales=d["adv_pay_sales"],
        historical_adv_pay_purch=d["adv_pay_purch"],
        historical_ar=d["ar"],
        historical_ap=d["ap"],
        historical_inventory=d["inventory"],
        historical_cash=d["cash"],
        historical_ims=d["ims"],
        historical_net_income=d["net_income"],
        historical_dividends=d["dividends"],
        historical_stock_buyback=d["stock_buyback"],
        historical_opex=d["opex"],
        historical_tax=d["tax"],
        historical_eff_st_debt=d["eff_st_debt"],
        historical_tax_onetime_payments=d["tax_onetime_payments"],
        historical_inflation=d["inflation"],
        historical_years=d["years"],
        epochs=2,
        plot_vi=False,
        plot_every=1,
        show_plot=False,
    )

    # Structural parameter training: short run and check for finite post-training state.
    model.train_structural_parameters(
        historical_sales=d["sales"],
        historical_nca=d["nca"],
        historical_adv_pay_sales=d["adv_pay_sales"],
        historical_adv_pay_purch=d["adv_pay_purch"],
        historical_ar=d["ar"],
        historical_ap=d["ap"],
        historical_inventory=d["inventory"],
        historical_cash=d["cash"],
        historical_ims=d["ims"],
        historical_net_income=d["net_income"],
        historical_dividends=d["dividends"],
        historical_stock_buyback=d["stock_buyback"],
        historical_opex=d["opex"],
        historical_tax=d["tax"],
        historical_effective_st_debt=d["eff_st_debt"],
        historical_current_lt_debt=d["current_lt_debt"],
        historical_non_current_liabilities=d["non_current_liabilities"],
        historical_interest_payment=d["interest_payment"],
        historical_ms_return=d["ms_return"],
        historical_equity=d["equity"],
        historical_tax_onetime_payments=d["tax_onetime_payments"],
        historical_inflation=d["inflation"],
        historical_years=d["years"],
        epochs=2,
        plot_every=1,
        show_plot=False,
    )

    critical_params = [
        model.asset_growth,
        model.depreciation_rate,
        model.income_tax_pct,
        model.avg_short_term_interest_pct,
        model.avg_long_term_interest_pct,
        model.avg_maturity_years,
    ]
    for param in critical_params:
        assert tf.math.is_finite(tf.cast(param, tf.float64))

    # One deterministic forecast call post-training should also remain finite.
    state_next = model.forecast_step(
        mock_forecast_state,
        mock_forecast_inputs,
        use_mean_opex=True,
    )
    for value in state_next.values():
        assert tf.math.is_finite(tf.cast(value, tf.float64))


def test_monte_carlo_stability(model, mock_forecast_state):
    """Monte Carlo trajectories should stay finite for debt and earnings paths."""
    n_years = 15
    sales_forecast = tf.fill([n_years], tf.constant(1.20, dtype=tf.float64))
    inflation_forecast = tf.fill([n_years], tf.constant(0.02, dtype=tf.float64))
    cum_inf_forecast = tf.cast(tf.math.cumprod(1.0 + inflation_forecast), tf.float64)
    forecast_years = tf.cast(tf.range(2019, 2019 + n_years), tf.float64)

    trajectories = run_monte_carlo_forecast(
        model=model,
        initial_state=mock_forecast_state,
        sales_forecast=sales_forecast,
        cum_inf_forecast=cum_inf_forecast,
        forecast_years=forecast_years,
        n_samples=2,
    )

    # Explicitly assert requested key trajectories.
    assert tf.reduce_all(tf.math.is_finite(trajectories["effective_st_debt"]))
    assert tf.reduce_all(tf.math.is_finite(trajectories["net_income"]))

    # Guardrail: all tracked outputs should stay finite in this short stress run.
    for arr in trajectories.values():
        assert tf.reduce_all(tf.math.is_finite(arr))
