"""Tests for TrainableFinancialModel with BayesianOpEx.

This suite focuses on:
- Accounting identity integrity in single-step forecasting.
- Bayesian OpEx variational sampling sanity.
- Parameter persistence roundtrip (save/load).
- Bijector-implied economic bounds.
- Short-horizon training execution stability.
- Monte Carlo trajectory finite-value stability.


Run the script by:
python -m pytest -q tests/test_trainable_financial_model_bayesianopex.py
"""

import pytest
import tensorflow as tf

from financial_forecast.inference.state_index import (
    initial_state_to_batched,
    DIAGNOSTIC_KEYS,
)
from financial_forecast.models.trainable_financial_model import TrainableFinancialModel
from financial_forecast.models.opex import BayesianOpEx
from financial_forecast.models.liquidity import TrendLiquidityPolicy
from financial_forecast.models.dividends import LintnerDividendPolicy
from financial_forecast.models.buyback import BaselineBuybackPolicy
from financial_forecast.models.purchases import TrendCostRatioPolicy
from financial_forecast.models.debt import TrendDebtPolicy
from financial_forecast.models.capex import CapexPolicy
from financial_forecast.models.working_capital import WorkingCapitalPolicy
from financial_forecast.models.tax import TaxWithAnomalies, SimpleTax
from financial_forecast.inference.trajectory_simulator import MonteCarloSimulator
from financial_forecast.training.policy_trainer import PolicyTrainer
from financial_forecast.training.structural_trainer import StructuralTrainer


@pytest.fixture
def model():
    """Yield a fresh model instance with deterministic random seeds."""
    tf.random.set_seed(7)
    m = TrainableFinancialModel(
        opex_module=BayesianOpEx(),
        trajectory_simulator=MonteCarloSimulator(n_samples=2),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=SimpleTax(),
    )
    m.base_year = 2018
    m.amount_scale = 1.0
    return m


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
    interest_payment = tf.constant(
        [0.021, 0.0215, 0.022, 0.0225, 0.023], dtype=tf.float64
    )
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
    var_opex_sample, base_opex_sample = model.opex_module.sample()
    kl_div = model.opex_module.kl_divergence()

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

    original_asset_growth = float(model.balance_sheet.capex_policy.asset_growth.numpy())
    model.save_parameters(str(save_path))

    # Change parameter after save to verify that load performs a true restore.
    model.balance_sheet.capex_policy.asset_growth.assign(
        tf.constant(original_asset_growth + 0.5, dtype=tf.float64)
    )
    assert float(
        model.balance_sheet.capex_policy.asset_growth.numpy()
    ) != pytest.approx(original_asset_growth)

    model.load_parameters(str(save_path))
    assert float(
        model.balance_sheet.capex_policy.asset_growth.numpy()
    ) == pytest.approx(original_asset_growth, rel=0.0, abs=1e-12)


def test_parameter_bounds(model):
    """Transformed variables should respect their economic constraints."""
    softplus_params = [
        model.balance_sheet.capex_policy.asset_growth,
        model.balance_sheet.capex_policy.asset_maintain,
        model.balance_sheet.capex_policy.depreciation_rate,
        model.balance_sheet.working_capital.advance_payments_sales_pct,
        model.balance_sheet.working_capital.advance_payments_purchases_pct,
        model.balance_sheet.working_capital.account_receivables_pct,
        model.balance_sheet.working_capital.account_payables_pct,
        model.balance_sheet.working_capital.inventory_cogs_pct,
        model.opex_module.q_var_opex_scale,
        model.opex_module.q_base_opex_scale,
        model.opex_module.noise_sigma,
        model.income_statement.avg_short_term_interest_pct,
        model.income_statement.avg_long_term_interest_pct,
        model.income_statement.market_securities_return_pct,
    ]
    for param in softplus_params:
        assert float(param.numpy()) >= 0.0

    sigmoid_params = [
        model.tax_module.income_tax_pct,
        model.cash_budget.dividend_policy.dividend_payout_ratio_pct,
        model.cash_budget.dividend_policy.dividend_adjustment_speed,
    ]
    for param in sigmoid_params:
        value = float(param.numpy())
        assert 0.0 <= value <= 1.0

    # Shift(1.001) + Softplus bijector imposes a strict lower bound > 1.001.
    assert float(model.cash_budget.debt_policy.avg_maturity_years.numpy()) > 1.001


def _build_training_data(d):
    """Build HistoricalTrainingData from mock_historical_data dict."""
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
        current_lt_debt=d["current_lt_debt"],
        non_current_liabilities=d["non_current_liabilities"],
        interest_payment=d["interest_payment"],
        ms_return=d["ms_return"],
        equity=d["equity"],
        inflation=d["inflation"],
        years=d["years"],
    )


def test_training_step_execution(
    model,
    mock_historical_data,
    mock_forecast_state,
    mock_forecast_inputs,
):
    """Both training loops should run for a few epochs without numerical failure."""
    data = _build_training_data(mock_historical_data)

    PolicyTrainer().train(
        model,
        data,
        loss_scale_mode="std",
        show_plot=False,
        epochs=2,
        plot_every=1,
    )

    StructuralTrainer().train(
        model,
        data,
        loss_scale_mode="std",
        show_plot=False,
        epochs=2,
        plot_every=1,
    )

    critical_params = [
        model.balance_sheet.capex_policy.asset_growth,
        model.balance_sheet.capex_policy.depreciation_rate,
        model.tax_module.income_tax_pct,
        model.income_statement.avg_short_term_interest_pct,
        model.income_statement.avg_long_term_interest_pct,
        model.cash_budget.debt_policy.avg_maturity_years,
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

    trajectories = model.trajectory_simulator.run(
        model,
        initial_state=mock_forecast_state,
        sales_forecast=sales_forecast,
        cum_inf_forecast=cum_inf_forecast,
        forecast_years=forecast_years,
    )

    # Explicitly assert requested key trajectories.
    assert tf.reduce_all(tf.math.is_finite(trajectories["effective_st_debt"]))
    assert tf.reduce_all(tf.math.is_finite(trajectories["net_income"]))

    # Guardrail: all tracked outputs should stay finite in this short stress run.
    for arr in trajectories.values():
        assert tf.reduce_all(tf.math.is_finite(arr))


def test_forecast_step_output_dtypes_are_float64(
    model, mock_forecast_state, mock_forecast_inputs
):
    """All forecast_step output values should be float64 to prevent dtype mismatch."""
    state_next = model.forecast_step(
        mock_forecast_state, mock_forecast_inputs, use_mean_opex=True
    )
    for key, value in state_next.items():
        assert hasattr(value, "dtype"), f"{key} is not a tensor"
        assert (
            value.dtype == tf.float64
        ), f"{key} has dtype {value.dtype}, expected float64"


def test_forecast_step_output_shapes_are_scalar(
    model, mock_forecast_state, mock_forecast_inputs
):
    """All forecast_step output values should be scalar (rank 0)."""
    state_next = model.forecast_step(
        mock_forecast_state, mock_forecast_inputs, use_mean_opex=True
    )
    for key, value in state_next.items():
        assert hasattr(value, "shape"), f"{key} is not a tensor"
        assert value.shape.rank == 0, f"{key} has rank {value.shape.rank}, expected 0"


def test_forecast_step_does_not_mutate_input_state(
    model, mock_forecast_state, mock_forecast_inputs
):
    """forecast_step must not modify the input state dictionary in-place."""
    original_values = {k: float(v.numpy()) for k, v in mock_forecast_state.items()}
    model.forecast_step(mock_forecast_state, mock_forecast_inputs, use_mean_opex=True)
    for k, orig_val in original_values.items():
        assert (
            float(mock_forecast_state[k].numpy()) == orig_val
        ), f"Input state key '{k}' was mutated by forecast_step"


def test_tensor_immutability_inflation_forecast():
    """Building inflation forecast via tf.concat should work; item assignment should not.

    This test guards against the exact bug where tf.fill + item assignment was used
    instead of tf.concat to build a forecast tensor.
    """
    n = 10
    last_hist = tf.constant(0.025, dtype=tf.float64)
    # Correct approach: tf.concat
    forecast = tf.concat(
        [
            tf.expand_dims(last_hist, 0),
            tf.fill([n - 1], tf.constant(0.03, dtype=tf.float64)),
        ],
        axis=0,
    )
    assert forecast.shape == (n,)
    assert forecast.dtype == tf.float64
    assert float(forecast[0].numpy()) == pytest.approx(0.025)
    assert float(forecast[1].numpy()) == pytest.approx(0.03)

    # Verify that item assignment on EagerTensor raises TypeError
    immutable_tensor = tf.fill([n], tf.constant(0.03, dtype=tf.float64))
    with pytest.raises(TypeError):
        immutable_tensor[0] = last_hist


def test_kl_divergence_is_nonnegative(model):
    """KL divergence between variational posterior and prior must be >= 0."""
    kl = model.opex_module.kl_divergence()
    assert float(kl.numpy()) >= 0.0


def test_gradient_flows_through_forecast_step(
    model, mock_forecast_state, mock_forecast_inputs
):
    """Trainable parameters should receive non-None, finite gradients through forecast_step."""
    with tf.GradientTape() as tape:
        state_next = model.forecast_step(
            mock_forecast_state, mock_forecast_inputs, use_mean_opex=True
        )
        loss = tf.square(state_next["net_income"])
    grads = tape.gradient(loss, model.trainable_variables)
    non_none_grads = [g for g in grads if g is not None]
    assert len(non_none_grads) > 0, "No gradients flowed through forecast_step"
    for g in non_none_grads:
        assert tf.reduce_all(tf.math.is_finite(g)), "Gradient contains inf or NaN"


def test_monte_carlo_trajectory_shapes(model, mock_forecast_state):
    """Monte Carlo trajectories should have shape (n_samples, n_years)."""
    n_years = 5
    n_samples = model.trajectory_simulator.n_samples  # from fixture
    sales_forecast = tf.fill([n_years], tf.constant(1.20, dtype=tf.float64))
    inflation_forecast = tf.fill([n_years], tf.constant(0.02, dtype=tf.float64))
    cum_inf_forecast = tf.cast(tf.math.cumprod(1.0 + inflation_forecast), tf.float64)
    forecast_years = tf.cast(tf.range(2019, 2019 + n_years), tf.float64)

    trajectories = model.trajectory_simulator.run(
        model,
        initial_state=mock_forecast_state,
        sales_forecast=sales_forecast,
        cum_inf_forecast=cum_inf_forecast,
        forecast_years=forecast_years,
    )

    for key, arr in trajectories.items():
        assert (
            arr.shape[0] == n_samples
        ), f"{key} has {arr.shape[0]} samples, expected {n_samples}"
        assert (
            arr.shape[1] == n_years
        ), f"{key} has {arr.shape[1]} years, expected {n_years}"


def test_monte_carlo_trajectory_dtypes(model, mock_forecast_state):
    """Monte Carlo trajectory tensors should all be float64."""
    n_years = 3
    sales_forecast = tf.fill([n_years], tf.constant(1.20, dtype=tf.float64))
    inflation_forecast = tf.fill([n_years], tf.constant(0.02, dtype=tf.float64))
    cum_inf_forecast = tf.cast(tf.math.cumprod(1.0 + inflation_forecast), tf.float64)
    forecast_years = tf.cast(tf.range(2019, 2019 + n_years), tf.float64)

    trajectories = model.trajectory_simulator.run(
        model,
        initial_state=mock_forecast_state,
        sales_forecast=sales_forecast,
        cum_inf_forecast=cum_inf_forecast,
        forecast_years=forecast_years,
    )

    for key, arr in trajectories.items():
        assert arr.dtype == tf.float64, f"{key} has dtype {arr.dtype}, expected float64"


def test_forecast_step_with_zero_sales(model, mock_forecast_state):
    """Edge case: zero sales should not cause NaN or inf."""
    inputs = {
        "sales_t": tf.constant(0.0, dtype=tf.float64),
        "year": tf.constant(2023.0, dtype=tf.float64),
        "cum_inflation": tf.constant(1.0, dtype=tf.float64),
    }
    state_next = model.forecast_step(mock_forecast_state, inputs, use_mean_opex=True)
    for key, value in state_next.items():
        assert tf.math.is_finite(value), f"{key} is not finite with zero sales"


def test_save_load_nonexistent_path_raises(model):
    """Loading from a nonexistent path should raise an error."""
    with pytest.raises(Exception):
        model.load_parameters("nonexistent_path_abc123.npz")


def test_deterministic_equivalence(model, mock_forecast_state):
    """Dict-based and compiled paths must agree with mean OpEx and zero noise."""
    n_years = 3
    sales_vals = [1.20, 1.25, 1.30]
    year_vals = [2023.0, 2024.0, 2025.0]
    cum_inf_vals = [1.04, 1.07, 1.10]

    # --- Dict-based path ---
    dict_results = []
    state = mock_forecast_state.copy()
    for step in range(n_years):
        inputs = {
            "sales_t": tf.constant(sales_vals[step], dtype=tf.float64),
            "year": tf.constant(year_vals[step], dtype=tf.float64),
            "cum_inflation": tf.constant(cum_inf_vals[step], dtype=tf.float64),
        }
        state = model.forecast_step(state, inputs, use_mean_opex=True)
        dict_results.append(state)

    # --- Compiled path ---
    batched_state = initial_state_to_batched(mock_forecast_state, 1)

    compiled_diags = []
    for step in range(n_years):
        sales_t = tf.constant([sales_vals[step]], dtype=tf.float64)
        cum_inf = tf.constant(cum_inf_vals[step], dtype=tf.float64)
        batched_state, diagnostics = model.forecast_step_compiled(
            batched_state,
            sales_t,
            tf.constant(year_vals[step], dtype=tf.float64),
            cum_inf,
            use_mean_opex=True,
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


def test_monte_carlo_with_tax_anomalies(mock_forecast_state):
    """MC forecast with TaxWithAnomalies must work for future years beyond anomaly data."""
    tf.random.set_seed(42)
    # Create model with tax anomalies keyed by historical years
    tax_data = {2018: 1.5e9, 2020: -0.5e9, 2022: 5.0e9}
    m = TrainableFinancialModel(
        opex_module=BayesianOpEx(),
        trajectory_simulator=MonteCarloSimulator(n_samples=3),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=TaxWithAnomalies(tax_data),
    )
    m.base_year = 2018
    m.amount_scale = 1.0
    # prepare_for_training must be called to build the lookup tensor
    training_years = tf.constant([2018, 2019, 2020, 2021, 2022], dtype=tf.float64)
    m.tax_module.prepare_for_training(m.amount_scale, training_years)

    # Forecast into future years BEYOND the anomaly data range
    n_years = 10
    sales_forecast = tf.fill([n_years], tf.constant(1.20, dtype=tf.float64))
    cum_inf_forecast = tf.cast(
        tf.math.cumprod(1.0 + tf.fill([n_years], 0.02)),
        tf.float64,
    )
    # Forecast starts at 2023 — all years are beyond the anomaly dict
    forecast_years = tf.cast(tf.range(2023, 2023 + n_years), tf.float64)

    trajectories = m.trajectory_simulator.run(
        m,
        initial_state=mock_forecast_state,
        sales_forecast=sales_forecast,
        cum_inf_forecast=cum_inf_forecast,
        forecast_years=forecast_years,
    )

    for key, arr in trajectories.items():
        assert tf.reduce_all(
            tf.math.is_finite(arr)
        ), f"{key} has non-finite values in MC forecast with tax anomalies"


def test_tax_anomaly_affects_historical_forecast(mock_forecast_state):
    """Forecast step for a year WITH an anomaly should differ from one WITHOUT."""
    tf.random.set_seed(42)
    tax_data = {2020: 5.0e9}
    m = TrainableFinancialModel(
        opex_module=BayesianOpEx(),
        trajectory_simulator=MonteCarloSimulator(n_samples=2),
        capex_policy=CapexPolicy(),
        working_capital=WorkingCapitalPolicy(),
        liquidity_policy=TrendLiquidityPolicy(),
        dividend_policy=LintnerDividendPolicy(),
        buyback_policy=BaselineBuybackPolicy(),
        purchases_policy=TrendCostRatioPolicy(),
        debt_policy=TrendDebtPolicy(),
        tax_module=TaxWithAnomalies(tax_data),
    )
    m.base_year = 2018
    m.amount_scale = 1e11
    training_years = tf.constant([2018, 2019, 2020, 2021], dtype=tf.float64)
    m.tax_module.prepare_for_training(m.amount_scale, training_years)

    inputs = {
        "sales_t": tf.constant(1.20, dtype=tf.float64),
        "cum_inflation": tf.constant(1.04, dtype=tf.float64),
    }

    # Year with anomaly
    inputs_2020 = {**inputs, "year": tf.constant(2020.0, dtype=tf.float64)}
    result_2020 = m.forecast_step(mock_forecast_state, inputs_2020, use_mean_opex=True)

    # Year without anomaly (same sales/inflation)
    inputs_2021 = {**inputs, "year": tf.constant(2021.0, dtype=tf.float64)}
    result_2021 = m.forecast_step(mock_forecast_state, inputs_2021, use_mean_opex=True)

    tax_2020 = float(result_2020["tax"])
    tax_2021 = float(result_2021["tax"])

    # 2020 should have higher tax due to the $5B anomaly
    assert tax_2020 > tax_2021, (
        f"Tax in anomaly year 2020 ({tax_2020}) should exceed "
        f"non-anomaly year 2021 ({tax_2021})"
    )
