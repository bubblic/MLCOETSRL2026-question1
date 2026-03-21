"""Monte Carlo forecast execution helpers.

This module runs forward simulations from an initial balance-sheet state and
returns per-year trajectory arrays for each forecasted metric.  The simulation
uses a ``tf.while_loop`` compiled via ``@tf.function`` for performance.
"""

from __future__ import annotations

from typing import Any, Dict
from financial_forecast.inference.state_index import (
    initial_state_to_batched,
    DIAGNOSTIC_KEYS,
)

import tensorflow as tf
import tensorflow_probability as tfp

StateDict = Dict[str, Any]
"""Dictionary representing a single-period balance-sheet state."""


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


def _print_forecast_report(
    trajectories: Dict[str, tf.Tensor],
    amount_scale: float,
    forecast_years,
    n_samples: int,
) -> None:
    """Print summary tables and balance sheet for forecast trajectories.

    Args:
        trajectories: Dict mapping metric names to ``[n_samples, n_years]``
            tensors.
        amount_scale: Multiplier to convert scaled values back to USD.
        forecast_years: 1-D tensor of calendar years.
        n_samples: Number of Monte Carlo simulation paths.
    """
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
        (
            "  Accounts Receivable",
            tf.reduce_mean(trajectories["accounts_receivable"], axis=0),
        ),
        ("  Inventory", tf.reduce_mean(trajectories["inventory"], axis=0)),
        ("  Cash", tf.reduce_mean(trajectories["cash"], axis=0)),
        (
            "  Invest in Mkt Sec",
            tf.reduce_mean(trajectories["investment_in_market_securities"], axis=0),
        ),
        ("TOTAL ASSETS", mean_total_assets),
        ("", None),
        ("LIABILITIES", None),
        (
            "  Accounts Payable",
            tf.reduce_mean(trajectories["accounts_payable"], axis=0),
        ),
        (
            "  Adv Payments (Sales)",
            tf.reduce_mean(trajectories["advance_payments_sales"], axis=0),
        ),
        (
            "  Effective ST Debt",
            tf.reduce_mean(trajectories["effective_st_debt"], axis=0),
        ),
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
    print("FORECAST BALANCE SHEET \u2014 Mean across Monte Carlo samples (USD)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, data in rows:
        if data is None:
            print(f"{label:>{label_width}}")
            continue
        values_str = "".join(
            f"{float(value) * scale:>{col_width},.0f}" for value in data
        )
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


def run_monte_carlo_forecast(
    model,
    initial_state: StateDict,
    sales_forecast: tf.Tensor,
    cum_inf_forecast: tf.Tensor,
    forecast_years: tf.Tensor,
    n_samples: int = 1000,
) -> Dict[str, tf.Tensor]:
    """Run Monte Carlo forecast using a compiled ``tf.while_loop``.

    Batches all samples into a single ``[n_samples, 14]`` state tensor and
    iterates over forecast years with ``tf.while_loop`` inside
    ``@tf.function``.  All stochastic values are pre-sampled in eager mode
    before entering the compiled graph.

    Args:
        model: Trained :class:`TrainableFinancialModel`.
        initial_state: Balance-sheet state dict at the forecast start.
        sales_forecast: 1-D tensor of forecasted sales (scaled).
        cum_inf_forecast: 1-D tensor of cumulative inflation factors.
        forecast_years: 1-D tensor of calendar years.
        n_samples: Number of Monte Carlo simulation paths.

    Returns:
        Dictionary mapping metric names to rank-2 tensors of shape
        ``[n_samples, n_years]``.
    """

    print(f"\n--- Running Compiled Monte Carlo Forecast ({n_samples} samples) ---")

    n_years = len(sales_forecast)

    # Pre-sample all stochastic OpEx values in eager mode before graph entry.
    model.opex_module.prepare_mc(n_samples, n_years)

    # Batch initial state: [n_samples, 14]
    state0 = initial_state_to_batched(initial_state, n_samples)

    # Materialize forecast inputs as float64 tensors
    sales_arr = tf.cast(sales_forecast, tf.float64)
    cum_inf_arr = tf.cast(cum_inf_forecast, tf.float64)
    years_arr = tf.cast(forecast_years, tf.float64)

    @tf.function
    def _run_loop(state0, sales_arr, cum_inf_arr, years_arr):
        n_steps = tf.shape(sales_arr)[0]
        diag_ta = tf.TensorArray(
            dtype=tf.float64,
            size=n_steps,
            dynamic_size=False,
        )

        def body(step, state, diag_ta):
            sales_t = tf.ones_like(state[:, 0]) * sales_arr[step]
            opex = model.opex_module.compute_mc_step(
                sales_t,
                cum_inf_arr[step],
                step,
            )
            new_state, diagnostics = model.forecast_step_compiled(
                state,
                sales_t,
                years_arr[step],
                opex,
            )
            diag_ta = diag_ta.write(step, diagnostics)
            return step + 1, new_state, diag_ta

        def cond(step, state, diag_ta):
            return step < n_steps

        _, _, diag_ta = tf.while_loop(
            cond,
            body,
            loop_vars=[tf.constant(0), state0, diag_ta],
        )
        return diag_ta.stack()

    # Execute compiled loop: [n_years, n_samples, N_DIAGNOSTIC]
    all_diag = _run_loop(state0, sales_arr, cum_inf_arr, years_arr)

    # Transpose to [n_samples, n_years, N_DIAGNOSTIC] and unpack
    all_diag = tf.transpose(all_diag, perm=[1, 0, 2])

    trajectories: Dict[str, tf.Tensor] = {}
    for i, key in enumerate(DIAGNOSTIC_KEYS):
        trajectories[key] = all_diag[:, :, i]

    _print_forecast_report(trajectories, model.amount_scale, forecast_years, n_samples)

    return trajectories
