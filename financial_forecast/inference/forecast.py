"""Monte Carlo forecast execution helpers.

This module runs forward simulations from an initial balance-sheet state and
returns per-year trajectory arrays for each forecasted metric.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

TrajectoryStore = Dict[str, List[List[float]]]
"""Nested list structure holding per-sample, per-year scalar values."""

StateDict = Dict[str, Any]
"""Dictionary representing a single-period balance-sheet state."""

_STATE_METRICS: Dict[str, str] = {
    "net_income": "net_income",
    "equity": "equity",
    "nca": "nca",
    "advance_payments_purchases": "advance_payments_purchases",
    "accounts_receivable": "accounts_receivable",
    "inventory": "inventory",
    "cash": "cash",
    "investment_in_market_securities": "investment_in_market_securities",
    "effective_st_debt": "effective_st_debt",
    "non_current_liabilities": "non_current_liabilities",
    "accounts_payable": "accounts_payable",
    "advance_payments_sales": "advance_payments_sales",
    "depreciation": "depreciation",
    "cogs": "cogs",
    "opex": "opex",
    "tax": "tax",
    "ms_return": "ms_return",
    "interest_payment": "interest_payment",
    "dividends": "dividends",
    "stock_buyback": "stock_buyback",
    "current_lt_debt": "current_lt_debt",
    "new_long_term_loan": "new_long_term_loan",
    "equity_financing": "equity_financing",
    "liquidity_deficit_st": "liquidity_deficit_st",
}


def _build_forecast_inputs(
    sales_value: float, year_value: float, cumulative_inflation: float
) -> Dict[str, tf.Tensor]:
    """Build the per-step model input dictionary.

    Args:
        sales_value: Sales amount for the forecast step (scaled).
        year_value: Calendar year of the forecast step.
        cumulative_inflation: Cumulative inflation factor up to this step.

    Returns:
        Dictionary with ``sales_t``, ``year``, and ``cum_inflation`` tensor
        entries.
    """
    return {
        "sales_t": tf.constant(sales_value),
        "year": tf.constant(float(year_value), dtype=tf.float64),
        "cum_inflation": tf.constant(cumulative_inflation),
    }


def _extract_total_assets(state: Mapping[str, Any]) -> float:
    """Compute total assets from the forecast state.

    Args:
        state: Model state dictionary containing individual asset tensors.

    Returns:
        Total assets as a Python float.
    """
    total_assets = (
        state["nca"]
        + state["advance_payments_purchases"]
        + state["accounts_receivable"]
        + state["inventory"]
        + state["cash"]
        + state["investment_in_market_securities"]
    )
    return float(total_assets.numpy())


def _extract_scalar_metrics(state: Mapping[str, Any]) -> Dict[str, float]:
    """Extract scalar numpy values from the state for tracked metrics.

    Args:
        state: Model state dictionary after a single forecast step.

    Returns:
        Dictionary mapping metric names to Python float values, including
        a computed ``total_assets`` entry.
    """
    metrics: Dict[str, float] = {"total_assets": _extract_total_assets(state)}
    for output_name, state_key in _STATE_METRICS.items():
        metrics[output_name] = float(state[state_key].numpy())
    return metrics


def _initialize_trajectory_store() -> TrajectoryStore:
    """Create an empty trajectory container.

    Returns:
        Dictionary with one empty list per tracked metric key.
    """
    keys = ["total_assets", *list(_STATE_METRICS.keys())]
    return {key: [] for key in keys}


def _append_sample(store: TrajectoryStore, sample: Dict[str, List[float]]) -> None:
    """Append one simulation sample into the trajectory store.

    Args:
        store: Trajectory store accumulating all samples.
        sample: Single-sample dictionary mapping metric names to lists of
            per-year float values.
    """
    for key, values in sample.items():
        store[key].append(values)


def _to_tf_trajectories(store: TrajectoryStore) -> Dict[str, tf.Tensor]:
    """Convert trajectory lists to TensorFlow tensors.

    Args:
        store: Completed trajectory store with all samples collected.

    Returns:
        Dictionary mapping metric names to rank-2 ``tf.Tensor`` objects
        with shape ``[n_samples, n_years]``.
    """
    return {key: tf.constant(values, dtype=tf.float64) for key, values in store.items()}


def _summarize_trajectories(
    name: str, trajectories: tf.Tensor, amount_scale: float
) -> None:
    """Print mean and 95% interval summary for one metric.

    Args:
        name: Human-readable label for the metric.
        trajectories: Tensor of shape ``[n_samples, n_years]``.
        amount_scale: Multiplier to convert scaled values back to USD.
    """
    mean_vals = tf.reduce_mean(trajectories, axis=0)
    lower_bound = tfp.stats.percentile(trajectories, 2.5, axis=0)
    upper_bound = tfp.stats.percentile(trajectories, 97.5, axis=0)

    print(f"\n{name}")
    print(f"{'Year':<5} | {'Mean':<15} | {'2.5% CI':<15} | {'97.5% CI':<15}")
    print("-" * 60)
    for idx in range(len(mean_vals)):
        mean_usd = float(mean_vals[idx]) * amount_scale
        lower_usd = float(lower_bound[idx]) * amount_scale
        upper_usd = float(upper_bound[idx]) * amount_scale
        print(
            f"{idx + 1:<5} | {mean_usd:<15.2e} | {lower_usd:<15.2e} | {upper_usd:<15.2e}"
        )


def run_monte_carlo_forecast(
    model: Any,
    initial_state: StateDict,
    sales_forecast: tf.Tensor,
    cum_inf_forecast: tf.Tensor,
    forecast_years: tf.Tensor,
    n_samples: int = 1000,
) -> Dict[str, tf.Tensor]:
    """Run Monte Carlo trajectories for all forecast years.

    The model structure and equations are unchanged; this function only
    orchestrates repeated calls to ``model.forecast_step``.

    Args:
        model: Trained financial model exposing ``forecast_step`` and
            ``sample_opex_params`` methods.
        initial_state: Balance-sheet state at the start of the forecast
            horizon.
        sales_forecast: 1-D tensor of forecasted sales (scaled).
        cum_inf_forecast: 1-D tensor of cumulative inflation factors.
        forecast_years: 1-D tensor of calendar years for each step.
        n_samples: Number of Monte Carlo simulation paths.

    Returns:
        Dictionary mapping metric names to rank-2 tensors of shape
        ``[n_samples, n_years]``.
    """
    print(f"\n--- Running Monte Carlo Forecast ({n_samples} samples) ---")

    # NOTE: This loop uses Python lists rather than tf.TensorArray because each
    # forecast_step call returns a Python dict with eager tensor values and uses
    # Bayesian sampling (model.sample_opex_params) that is inherently eager.
    # Converting to tf.TensorArray + tf.while_loop would require the model's
    # forecast_step to operate on flat tensor signatures, which conflicts with
    # the dict-based BayesianFinancialModel interface.  The final conversion to
    # tf.constant tensors in _to_tf_trajectories still enables vectorized
    # summary statistics via tf.reduce_mean / tfp.stats.percentile.
    store = _initialize_trajectory_store()
    for _ in range(n_samples):
        current_state = initial_state.copy()
        var_opex_sample, base_opex_sample = model.sample_opex_params()
        sample_store: Dict[str, List[float]] = {key: [] for key in store}

        for step in range(len(sales_forecast)):
            inputs = _build_forecast_inputs(
                sales_value=sales_forecast[step],
                year_value=forecast_years[step],
                cumulative_inflation=cum_inf_forecast[step],
            )
            current_state = model.forecast_step(
                current_state,
                inputs,
                use_mean_opex=False,
                sampled_var_opex=var_opex_sample,
                sampled_base_opex=base_opex_sample,
            )
            metrics = _extract_scalar_metrics(current_state)
            for key, value in metrics.items():
                sample_store[key].append(value)

        _append_sample(store, sample_store)

    trajectories = _to_tf_trajectories(store)
    amount_scale = model.amount_scale

    _summarize_trajectories("Net Income", trajectories["net_income"], amount_scale)
    _summarize_trajectories("Total Assets", trajectories["total_assets"], amount_scale)
    _summarize_trajectories(
        "Assets: Non-current Assets", trajectories["nca"], amount_scale
    )
    _summarize_trajectories(
        "Assets: Advance Payments (Purchases)",
        trajectories["advance_payments_purchases"],
        amount_scale,
    )
    _summarize_trajectories(
        "Assets: Accounts Receivable", trajectories["accounts_receivable"], amount_scale
    )
    _summarize_trajectories(
        "Assets: Inventory", trajectories["inventory"], amount_scale
    )
    _summarize_trajectories("Assets: Cash", trajectories["cash"], amount_scale)
    _summarize_trajectories(
        "Assets: Investment in Market Securities",
        trajectories["investment_in_market_securities"],
        amount_scale,
    )
    _summarize_trajectories(
        "Effective ST Debt", trajectories["effective_st_debt"], amount_scale
    )
    _summarize_trajectories(
        "Non-current Liabilities",
        trajectories["non_current_liabilities"],
        amount_scale,
    )
    _summarize_trajectories("Equity", trajectories["equity"], amount_scale)
    _summarize_trajectories(
        "Accounts Payable", trajectories["accounts_payable"], amount_scale
    )
    _summarize_trajectories(
        "Advance Payments (Sales)", trajectories["advance_payments_sales"], amount_scale
    )
    _summarize_trajectories("Depreciation", trajectories["depreciation"], amount_scale)
    _summarize_trajectories("COGS", trajectories["cogs"], amount_scale)
    _summarize_trajectories("OpEx", trajectories["opex"], amount_scale)
    _summarize_trajectories("Tax", trajectories["tax"], amount_scale)
    _summarize_trajectories(
        "Return on Market Securities", trajectories["ms_return"], amount_scale
    )
    _summarize_trajectories(
        "Interest Payment", trajectories["interest_payment"], amount_scale
    )
    _summarize_trajectories("Dividends", trajectories["dividends"], amount_scale)
    _summarize_trajectories(
        "Stock Buyback", trajectories["stock_buyback"], amount_scale
    )
    _summarize_trajectories(
        "Current Portion of LT debt", trajectories["current_lt_debt"], amount_scale
    )
    _summarize_trajectories(
        "New Long-Term Loan", trajectories["new_long_term_loan"], amount_scale
    )
    _summarize_trajectories(
        "Equity Financing", trajectories["equity_financing"], amount_scale
    )

    n_years = trajectories["total_assets"].shape[1]
    scale = amount_scale
    mean_total_assets = tf.reduce_mean(trajectories["total_assets"], axis=0)
    mean_total_liabilities = (
        tf.reduce_mean(trajectories["accounts_payable"], axis=0)
        + tf.reduce_mean(trajectories["advance_payments_sales"], axis=0)
        + tf.reduce_mean(trajectories["effective_st_debt"], axis=0)
        + tf.reduce_mean(trajectories["current_lt_debt"], axis=0)
        + tf.reduce_mean(trajectories["non_current_liabilities"], axis=0)
    )
    mean_equity = tf.reduce_mean(trajectories["equity"], axis=0)
    mean_total_liab_equity = mean_total_liabilities + mean_equity
    mean_check = mean_total_assets - mean_total_liab_equity
    year_labels = [f"FY{int(forecast_years[idx])}" for idx in range(n_years)]

    rows = [
        ("ASSETS", None),
        ("  Non-Current Assets", tf.reduce_mean(trajectories["nca"], axis=0)),
        (
            "  Adv Payments (Purch)",
            tf.reduce_mean(trajectories["advance_payments_purchases"], axis=0),
        ),
        ("  Accounts Receivable", tf.reduce_mean(trajectories["accounts_receivable"], axis=0)),
        ("  Inventory", tf.reduce_mean(trajectories["inventory"], axis=0)),
        ("  Cash", tf.reduce_mean(trajectories["cash"], axis=0)),
        (
            "  Invest in Mkt Sec",
            tf.reduce_mean(trajectories["investment_in_market_securities"], axis=0),
        ),
        ("TOTAL ASSETS", mean_total_assets),
        ("", None),
        ("LIABILITIES", None),
        ("  Accounts Payable", tf.reduce_mean(trajectories["accounts_payable"], axis=0)),
        (
            "  Adv Payments (Sales)",
            tf.reduce_mean(trajectories["advance_payments_sales"], axis=0),
        ),
        ("  Effective ST Debt", tf.reduce_mean(trajectories["effective_st_debt"], axis=0)),
        ("  Current LT Debt", tf.reduce_mean(trajectories["current_lt_debt"], axis=0)),
        (
            "  Non-Current Liabilities",
            tf.reduce_mean(trajectories["non_current_liabilities"], axis=0),
        ),
        ("TOTAL LIABILITIES", mean_total_liabilities),
        ("", None),
        ("EQUITY", mean_equity),
        ("", None),
        ("TOTAL LIAB + EQUITY", mean_total_liab_equity),
        ("", None),
        ("INCOME STATEMENT", None),
        ("  Net Income", tf.reduce_mean(trajectories["net_income"], axis=0)),
        ("", None),
        ("CHECK: Assets-(L+E)", mean_check),
    ]

    col_width = 14
    label_width = 26
    header = f"{'':>{label_width}}" + "".join(
        f"{label:>{col_width}}" for label in year_labels
    )
    print("\n" + "=" * len(header))
    print("FORECAST BALANCE SHEET — Mean across Monte Carlo samples (USD)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, data in rows:
        if data is None:
            print(f"{label:>{label_width}}")
            continue
        values_str = "".join(f"{float(value) * scale:>{col_width},.0f}" for value in data)
        print(f"{label:>{label_width}}{values_str}")
    print("-" * len(header))

    max_abs_check = float(tf.reduce_max(tf.abs(mean_check * scale)))
    print(f"\nBalance Sheet Identity Check (Assets = Liabilities + Equity):")
    print(f"  Max absolute mismatch across years (mean): ${max_abs_check:,.2f}")
    if max_abs_check < 1.0:
        print("  PASS: Balance sheet identity holds (mismatch < $1).")
    elif max_abs_check < 1000.0:
        print(
            "  PASS: Balance sheet identity holds within rounding (mismatch < $1,000)."
        )
    else:
        print("  WARNING: Balance sheet mismatch detected!")
        for idx in range(n_years):
            check_value = float(mean_check[idx]) * scale
            if abs(check_value) >= 1000.0:
                print(
                    f"    {year_labels[idx]}: Assets - (Liab+Eq) = ${check_value:,.2f}"
                )

    total_liab_equity_all = (
        trajectories["accounts_payable"]
        + trajectories["advance_payments_sales"]
        + trajectories["effective_st_debt"]
        + trajectories["current_lt_debt"]
        + trajectories["non_current_liabilities"]
        + trajectories["equity"]
    )
    check_all = trajectories["total_assets"] - total_liab_equity_all
    max_abs_check_all = float(tf.reduce_max(tf.abs(check_all))) * scale
    mean_abs_check_all = float(tf.reduce_mean(tf.abs(check_all))) * scale
    print(f"\n  Per-sample check (across all {n_samples} samples x {n_years} years):")
    print(f"    Max absolute mismatch:  ${max_abs_check_all:,.2f}")
    print(f"    Mean absolute mismatch: ${mean_abs_check_all:,.2f}")

    return trajectories
