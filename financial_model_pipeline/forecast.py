"""Monte Carlo forecast execution helpers.

This module runs forward simulations from an initial balance-sheet state and
returns per-year trajectory arrays for each forecasted metric.
"""

from typing import Any, Dict, List, Mapping

import numpy as np
import tensorflow as tf

TrajectoryStore = Dict[str, List[List[float]]]
StateDict = Dict[str, Any]

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
    """Build the per-step model input dictionary."""
    return {
        "sales_t": tf.constant(sales_value),
        "year": tf.constant(float(year_value), dtype=tf.float64),
        "cum_inflation": tf.constant(cumulative_inflation),
    }


def _extract_total_assets(state: Mapping[str, Any]) -> float:
    """Compute total assets from the forecast state."""
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
    """Extract scalar numpy values from the state for tracked metrics."""
    metrics: Dict[str, float] = {"total_assets": _extract_total_assets(state)}
    for output_name, state_key in _STATE_METRICS.items():
        metrics[output_name] = float(state[state_key].numpy())
    return metrics


def _initialize_trajectory_store() -> TrajectoryStore:
    """Create an empty trajectory container."""
    keys = ["total_assets", *list(_STATE_METRICS.keys())]
    return {key: [] for key in keys}


def _append_sample(store: TrajectoryStore, sample: Dict[str, List[float]]) -> None:
    """Append one simulation sample into the trajectory store."""
    for key, values in sample.items():
        store[key].append(values)


def _to_numpy_trajectories(store: TrajectoryStore) -> Dict[str, np.ndarray]:
    """Convert trajectory lists to numpy arrays."""
    return {key: np.array(values) for key, values in store.items()}


def _summarize_trajectories(
    name: str, trajectories: np.ndarray, amount_scale: float
) -> None:
    """Print mean and 95% interval summary for one metric."""
    mean_vals = np.mean(trajectories, axis=0)
    lower_bound = np.percentile(trajectories, 2.5, axis=0)
    upper_bound = np.percentile(trajectories, 97.5, axis=0)

    print(f"\n{name}")
    print(f"{'Year':<5} | {'Mean':<15} | {'2.5% CI':<15} | {'97.5% CI':<15}")
    print("-" * 60)
    for idx in range(len(mean_vals)):
        mean_usd = mean_vals[idx] * amount_scale
        lower_usd = lower_bound[idx] * amount_scale
        upper_usd = upper_bound[idx] * amount_scale
        print(
            f"{idx + 1:<5} | {mean_usd:<15.2e} | {lower_usd:<15.2e} | {upper_usd:<15.2e}"
        )


def run_monte_carlo_forecast(
    model: Any,
    initial_state: StateDict,
    sales_forecast: np.ndarray,
    cum_inf_forecast: np.ndarray,
    forecast_years: np.ndarray,
    n_samples: int = 1000,
) -> Dict[str, np.ndarray]:
    """Run Monte Carlo trajectories for all forecast years.

    The model structure and equations are unchanged; this function only
    orchestrates repeated calls to `model.forecast_step`.
    """
    print(f"\n--- Running Monte Carlo Forecast ({n_samples} samples) ---")

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

    trajectories = _to_numpy_trajectories(store)
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
    mean_total_assets = np.mean(trajectories["total_assets"], axis=0)
    mean_total_liabilities = (
        np.mean(trajectories["accounts_payable"], axis=0)
        + np.mean(trajectories["advance_payments_sales"], axis=0)
        + np.mean(trajectories["effective_st_debt"], axis=0)
        + np.mean(trajectories["current_lt_debt"], axis=0)
        + np.mean(trajectories["non_current_liabilities"], axis=0)
    )
    mean_equity = np.mean(trajectories["equity"], axis=0)
    mean_total_liab_equity = mean_total_liabilities + mean_equity
    mean_check = mean_total_assets - mean_total_liab_equity
    year_labels = [f"FY{int(forecast_years[idx])}" for idx in range(n_years)]

    rows = [
        ("ASSETS", None),
        ("  Non-Current Assets", np.mean(trajectories["nca"], axis=0)),
        (
            "  Adv Payments (Purch)",
            np.mean(trajectories["advance_payments_purchases"], axis=0),
        ),
        ("  Accounts Receivable", np.mean(trajectories["accounts_receivable"], axis=0)),
        ("  Inventory", np.mean(trajectories["inventory"], axis=0)),
        ("  Cash", np.mean(trajectories["cash"], axis=0)),
        (
            "  Invest in Mkt Sec",
            np.mean(trajectories["investment_in_market_securities"], axis=0),
        ),
        ("TOTAL ASSETS", mean_total_assets),
        ("", None),
        ("LIABILITIES", None),
        ("  Accounts Payable", np.mean(trajectories["accounts_payable"], axis=0)),
        (
            "  Adv Payments (Sales)",
            np.mean(trajectories["advance_payments_sales"], axis=0),
        ),
        ("  Effective ST Debt", np.mean(trajectories["effective_st_debt"], axis=0)),
        ("  Current LT Debt", np.mean(trajectories["current_lt_debt"], axis=0)),
        (
            "  Non-Current Liabilities",
            np.mean(trajectories["non_current_liabilities"], axis=0),
        ),
        ("TOTAL LIABILITIES", mean_total_liabilities),
        ("", None),
        ("EQUITY", mean_equity),
        ("", None),
        ("TOTAL LIAB + EQUITY", mean_total_liab_equity),
        ("", None),
        ("INCOME STATEMENT", None),
        ("  Net Income", np.mean(trajectories["net_income"], axis=0)),
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
        values_str = "".join(f"{value * scale:>{col_width},.0f}" for value in data)
        print(f"{label:>{label_width}}{values_str}")
    print("-" * len(header))

    max_abs_check = np.max(np.abs(mean_check * scale))
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
            check_value = mean_check[idx] * scale
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
    max_abs_check_all = np.max(np.abs(check_all)) * scale
    mean_abs_check_all = np.mean(np.abs(check_all)) * scale
    print(f"\n  Per-sample check (across all {n_samples} samples x {n_years} years):")
    print(f"    Max absolute mismatch:  ${max_abs_check_all:,.2f}")
    print(f"    Mean absolute mismatch: ${mean_abs_check_all:,.2f}")

    return trajectories


"""Monte Carlo forecast execution helpers."""

import numpy as np


def run_monte_carlo_forecast(
    model,
    initial_state,
    sales_forecast,
    cum_inf_forecast,
    forecast_years,
    n_samples=1000,
):
    print(f"\n--- Running Monte Carlo Forecast ({n_samples} samples) ---")

    def print_markdown_table(title, year_labels, rows, scale):
        print(f"\n{title}")
        header = "| Line Item | " + " | ".join(year_labels) + " |"
        separator = "| --- | " + " | ".join(["---:"] * len(year_labels)) + " |"
        print(header)
        print(separator)

        for label, data in rows:
            if data is None:
                print(f"| **{label}** | " + " | ".join([""] * len(year_labels)) + " |")
                continue

            values = [f"${float(v) * scale:,.0f}" for v in np.asarray(data)]
            print(f"| {label} | " + " | ".join(values) + " |")

    # Store trajectories
    # Shape: [Samples, Years]
    ni_trajectories = []
    equity_trajectories = []
    assets_trajectories = []
    nca_trajectories = []
    adv_pay_purch_trajectories = []
    ar_trajectories = []
    inv_trajectories = []
    cash_trajectories = []
    ims_trajectories = []
    effective_st_debt_trajectories = []
    non_current_liabilities_trajectories = []
    ap_trajectories = []
    aps_trajectories = []
    depreciation_trajectories = []
    cogs_trajectories = []
    opex_trajectories = []
    tax_trajectories = []
    ms_return_trajectories = []
    interest_payment_trajectories = []
    dividends_trajectories = []
    stock_buyback_trajectories = []
    current_lt_debt_trajectories = []
    new_lt_loan_trajectories = []
    equity_financing_trajectories = []
    liq_deficit_st_trajectories = []

    for i in range(n_samples):
        current_state = initial_state.copy()
        # Sample this trajectory's structural OpEx parameters once
        var_opex_sample, base_opex_sample = model.sample_opex_params()
        sample_ni = []
        sample_equity = []
        sample_assets = []
        sample_nca = []
        sample_adv_pp = []
        sample_ar = []
        sample_inv = []
        sample_cash = []
        sample_ims = []
        sample_eff_st = []
        sample_ncl = []
        sample_ap = []
        sample_aps = []
        sample_depr = []
        sample_cogs = []
        sample_opex = []
        sample_tax = []
        sample_ms_return = []
        sample_interest_payment = []
        sample_div = []
        sample_bb = []
        sample_curr_lt_debt = []
        sample_new_lt = []
        sample_ef = []
        sample_liq_deficit_st = []

        for t in range(len(sales_forecast)):
            inputs = {
                "sales_t": tf.constant(sales_forecast[t]),
                "year": tf.constant(float(forecast_years[t]), dtype=tf.float64),
                "cum_inflation": tf.constant(cum_inf_forecast[t]),
            }
            # use_mean_opex=False triggers sampling
            current_state = model.forecast_step(
                current_state,
                inputs,
                use_mean_opex=False,
                sampled_var_opex=var_opex_sample,
                sampled_base_opex=base_opex_sample,
            )

            sample_ni.append(current_state["net_income"].numpy())
            sample_equity.append(current_state["equity"].numpy())
            total_assets = (
                current_state["nca"]
                + current_state["advance_payments_purchases"]
                + current_state["accounts_receivable"]
                + current_state["inventory"]
                + current_state["cash"]
                + current_state["investment_in_market_securities"]
            )
            sample_assets.append(total_assets.numpy())
            sample_nca.append(current_state["nca"].numpy())
            sample_adv_pp.append(current_state["advance_payments_purchases"].numpy())
            sample_ar.append(current_state["accounts_receivable"].numpy())
            sample_inv.append(current_state["inventory"].numpy())
            sample_cash.append(current_state["cash"].numpy())
            sample_ims.append(current_state["investment_in_market_securities"].numpy())
            sample_eff_st.append(current_state["effective_st_debt"].numpy())
            sample_ncl.append(current_state["non_current_liabilities"].numpy())
            sample_ap.append(current_state["accounts_payable"].numpy())
            sample_aps.append(current_state["advance_payments_sales"].numpy())
            sample_depr.append(current_state["depreciation"].numpy())
            sample_cogs.append(current_state["cogs"].numpy())
            sample_opex.append(current_state["opex"].numpy())
            sample_tax.append(current_state["tax"].numpy())
            sample_ms_return.append(current_state["ms_return"].numpy())
            sample_interest_payment.append(current_state["interest_payment"].numpy())
            sample_div.append(current_state["dividends"].numpy())
            sample_bb.append(current_state["stock_buyback"].numpy())
            sample_curr_lt_debt.append(current_state["current_lt_debt"].numpy())
            sample_new_lt.append(current_state["new_long_term_loan"].numpy())
            sample_ef.append(current_state["equity_financing"].numpy())
            sample_liq_deficit_st.append(current_state["liquidity_deficit_st"].numpy())

        ni_trajectories.append(sample_ni)
        equity_trajectories.append(sample_equity)
        assets_trajectories.append(sample_assets)
        nca_trajectories.append(sample_nca)
        adv_pay_purch_trajectories.append(sample_adv_pp)
        ar_trajectories.append(sample_ar)
        inv_trajectories.append(sample_inv)
        cash_trajectories.append(sample_cash)
        ims_trajectories.append(sample_ims)
        effective_st_debt_trajectories.append(sample_eff_st)
        non_current_liabilities_trajectories.append(sample_ncl)
        ap_trajectories.append(sample_ap)
        aps_trajectories.append(sample_aps)
        depreciation_trajectories.append(sample_depr)
        cogs_trajectories.append(sample_cogs)
        opex_trajectories.append(sample_opex)
        tax_trajectories.append(sample_tax)
        ms_return_trajectories.append(sample_ms_return)
        interest_payment_trajectories.append(sample_interest_payment)
        dividends_trajectories.append(sample_div)
        stock_buyback_trajectories.append(sample_bb)
        current_lt_debt_trajectories.append(sample_curr_lt_debt)
        new_lt_loan_trajectories.append(sample_new_lt)
        equity_financing_trajectories.append(sample_ef)
        liq_deficit_st_trajectories.append(sample_liq_deficit_st)

    ni_trajectories = np.array(ni_trajectories)
    equity_trajectories = np.array(equity_trajectories)
    assets_trajectories = np.array(assets_trajectories)
    nca_trajectories = np.array(nca_trajectories)
    adv_pay_purch_trajectories = np.array(adv_pay_purch_trajectories)
    ar_trajectories = np.array(ar_trajectories)
    inv_trajectories = np.array(inv_trajectories)
    cash_trajectories = np.array(cash_trajectories)
    ims_trajectories = np.array(ims_trajectories)
    effective_st_debt_trajectories = np.array(effective_st_debt_trajectories)
    non_current_liabilities_trajectories = np.array(
        non_current_liabilities_trajectories
    )
    ap_trajectories = np.array(ap_trajectories)
    aps_trajectories = np.array(aps_trajectories)
    depreciation_trajectories = np.array(depreciation_trajectories)
    cogs_trajectories = np.array(cogs_trajectories)
    opex_trajectories = np.array(opex_trajectories)
    tax_trajectories = np.array(tax_trajectories)
    ms_return_trajectories = np.array(ms_return_trajectories)
    interest_payment_trajectories = np.array(interest_payment_trajectories)
    dividends_trajectories = np.array(dividends_trajectories)
    stock_buyback_trajectories = np.array(stock_buyback_trajectories)
    current_lt_debt_trajectories = np.array(current_lt_debt_trajectories)
    new_lt_loan_trajectories = np.array(new_lt_loan_trajectories)
    equity_financing_trajectories = np.array(equity_financing_trajectories)
    liq_deficit_st_trajectories = np.array(liq_deficit_st_trajectories)

    # Calculate Statistics
    def summarize_trajectories(name, trajectories):
        mean_vals = np.mean(trajectories, axis=0)
        lower_bound = np.percentile(trajectories, 2.5, axis=0)
        upper_bound = np.percentile(trajectories, 97.5, axis=0)

        print(f"\n{name}")
        print(f"{'Year':<5} | {'Mean':<15} | {'2.5% CI':<15} | {'97.5% CI':<15}")
        print("-" * 60)
        for t in range(len(mean_vals)):
            mean_usd = mean_vals[t] * model.amount_scale
            lower_usd = lower_bound[t] * model.amount_scale
            upper_usd = upper_bound[t] * model.amount_scale
            print(
                f"{t+1:<5} | {mean_usd:<15.2e} | {lower_usd:<15.2e} | {upper_usd:<15.2e}"
            )

    summarize_trajectories("Net Income", ni_trajectories)
    summarize_trajectories("Total Assets", assets_trajectories)
    summarize_trajectories("Assets: Non-current Assets", nca_trajectories)
    summarize_trajectories(
        "Assets: Advance Payments (Purchases)", adv_pay_purch_trajectories
    )
    summarize_trajectories("Assets: Accounts Receivable", ar_trajectories)
    summarize_trajectories("Assets: Inventory", inv_trajectories)
    summarize_trajectories("Assets: Cash", cash_trajectories)
    summarize_trajectories("Assets: Investment in Market Securities", ims_trajectories)
    summarize_trajectories("Effective ST Debt", effective_st_debt_trajectories)
    summarize_trajectories(
        "Non-current Liabilities", non_current_liabilities_trajectories
    )
    summarize_trajectories("Equity", equity_trajectories)
    summarize_trajectories("Accounts Payable", ap_trajectories)
    summarize_trajectories("Advance Payments (Sales)", aps_trajectories)
    summarize_trajectories("Depreciation", depreciation_trajectories)
    summarize_trajectories("COGS", cogs_trajectories)
    summarize_trajectories("OpEx", opex_trajectories)
    summarize_trajectories("Tax", tax_trajectories)
    summarize_trajectories("Return on Market Securities", ms_return_trajectories)
    summarize_trajectories("Interest Payment", interest_payment_trajectories)
    summarize_trajectories("Dividends", dividends_trajectories)
    summarize_trajectories("Stock Buyback", stock_buyback_trajectories)
    summarize_trajectories("Current Portion of LT debt", current_lt_debt_trajectories)
    summarize_trajectories("New Long-Term Loan", new_lt_loan_trajectories)
    summarize_trajectories("Equity Financing", equity_financing_trajectories)

    # --- Comprehensive Balance Sheet Table (Mean Values) ---
    n_years = assets_trajectories.shape[1]
    scale = model.amount_scale

    # Compute mean trajectories (in scaled units, i.e. billions)
    mean_nca = np.mean(nca_trajectories, axis=0)
    mean_adv_pp = np.mean(adv_pay_purch_trajectories, axis=0)
    mean_ar = np.mean(ar_trajectories, axis=0)
    mean_inv = np.mean(inv_trajectories, axis=0)
    mean_cash = np.mean(cash_trajectories, axis=0)
    mean_ims = np.mean(ims_trajectories, axis=0)
    mean_total_assets = np.mean(assets_trajectories, axis=0)

    mean_ap = np.mean(ap_trajectories, axis=0)
    mean_aps = np.mean(aps_trajectories, axis=0)
    mean_eff_st = np.mean(effective_st_debt_trajectories, axis=0)
    mean_curr_lt = np.mean(current_lt_debt_trajectories, axis=0)
    mean_ncl = np.mean(non_current_liabilities_trajectories, axis=0)
    mean_equity = np.mean(equity_trajectories, axis=0)
    mean_total_liabilities = mean_ap + mean_aps + mean_eff_st + mean_curr_lt + mean_ncl
    mean_total_liab_equity = mean_total_liabilities + mean_equity
    mean_check = mean_total_assets - mean_total_liab_equity

    # Build year labels from forecast_years
    year_labels = [f"FY{int(forecast_years[t])}" for t in range(n_years)]

    # Row definitions: (label, data_array)
    balance_sheet_rows = [
        ("ASSETS", None),
        ("Non-Current Assets", mean_nca),
        ("Advance Payments (Purchases)", mean_adv_pp),
        ("Accounts Receivable", mean_ar),
        ("Inventory", mean_inv),
        ("Cash", mean_cash),
        ("Investment in Market Securities", mean_ims),
        ("TOTAL ASSETS", mean_total_assets),
        ("", None),
        ("LIABILITIES", None),
        ("Accounts Payable", mean_ap),
        ("Advance Payments (Sales)", mean_aps),
        ("Effective ST Debt", mean_eff_st),
        ("Current LT Debt", mean_curr_lt),
        ("Non-Current Liabilities", mean_ncl),
        ("TOTAL LIABILITIES", mean_total_liabilities),
        ("", None),
        ("EQUITY", None),
        ("Equity", mean_equity),
        ("", None),
        ("TOTAL LIAB + EQUITY", mean_total_liab_equity),
        ("CHECK: Assets-(L+E)", mean_check),
    ]
    print_markdown_table(
        "FORECAST BALANCE SHEET — Mean across Monte Carlo samples (USD)",
        year_labels,
        balance_sheet_rows,
        scale,
    )

    # --- Income Statement (requested formula view) ---
    mean_sales = np.asarray(sales_forecast, dtype=np.float64)
    mean_depr = np.mean(depreciation_trajectories, axis=0)
    mean_cogs = np.mean(cogs_trajectories, axis=0)
    mean_opex = np.mean(opex_trajectories, axis=0)
    mean_interest = np.mean(interest_payment_trajectories, axis=0)
    mean_ms_return = np.mean(ms_return_trajectories, axis=0)
    mean_ebit = mean_sales - mean_cogs - mean_opex - mean_depr
    mean_ebt = mean_ebit - mean_interest + mean_ms_return
    eff_tax = float(model.income_tax_pct.numpy())
    payout_ratio = float(model.dividend_payout_ratio_pct.numpy())
    dividend_adjustment_speed = float(model.dividend_adjustment_speed.numpy())
    mean_income_taxes_formula = mean_ebt * eff_tax
    mean_net_income_formula = mean_ebt - mean_income_taxes_formula
    # Use simulated dividend outputs directly because the model applies
    # Lintner smoothing with prior-year NI and prior-year dividends.
    mean_dividends_prev = np.mean(dividends_trajectories, axis=0)
    mean_cre = np.cumsum(mean_net_income_formula)

    income_statement_rows = [
        ("Revenue (Sales_t)", mean_sales),
        ("COGS", mean_cogs),
        ("OpEx", mean_opex),
        ("Depreciation & Amortization", mean_depr),
        ("EBIT = Revenue - COGS - OpEx - Depreciation", mean_ebit),
        ("Interest Payments", mean_interest),
        ("ST Investment Returns", mean_ms_return),
        ("EBT = EBIT - Interest + ST Returns", mean_ebt),
        (f"Income Taxes = EBT * %EffTax ({eff_tax:.2%})", mean_income_taxes_formula),
        ("Net Income = EBT - Income Taxes", mean_net_income_formula),
        (
            "Dividends (model output, "
            f"Lintner smoothed: payout={payout_ratio:.2%}, "
            f"alpha={dividend_adjustment_speed:.2f})",
            mean_dividends_prev,
        ),
        ("CRE (cumulated retained earnings, forecast cumulative)", mean_cre),
    ]
    print_markdown_table(
        "FORECAST INCOME STATEMENT — Mean across Monte Carlo samples (USD)",
        year_labels,
        income_statement_rows,
        scale,
    )

    # --- Cash Budget (5 modules, requested decomposition) ---
    mean_equity_financing = np.mean(equity_financing_trajectories, axis=0)
    mean_new_lt_loan = np.mean(new_lt_loan_trajectories, axis=0)
    mean_stock_buyback = np.mean(stock_buyback_trajectories, axis=0)

    # CapEx formula in the model: asset_maintain * depreciation + sales_t * asset_growth
    mean_capex = (
        float(model.asset_maintain.numpy()) * mean_depr
        + float(model.asset_growth.numpy()) * mean_sales
    )

    prev_eff_st = np.concatenate(
        ([float(initial_state["effective_st_debt"].numpy())], mean_eff_st[:-1])
    )
    prev_curr_lt = np.concatenate(
        ([float(initial_state["current_lt_debt"].numpy())], mean_curr_lt[:-1])
    )
    prev_ncl = np.concatenate(
        ([float(initial_state["non_current_liabilities"].numpy())], mean_ncl[:-1])
    )
    mean_st_principal = prev_eff_st
    mean_lt_principal = prev_curr_lt
    mean_st_interest = (
        float(model.avg_short_term_interest_pct.numpy()) * mean_st_principal
    )
    mean_lt_interest = float(model.avg_long_term_interest_pct.numpy()) * (
        prev_ncl + prev_curr_lt
    )

    operating_inflows = mean_sales
    operating_outflows = mean_cogs + mean_opex + mean_income_taxes_formula
    operating_ncb = operating_inflows - operating_outflows

    investing_inflows = np.zeros_like(mean_sales)
    investing_outflows = mean_capex
    investing_ncb = investing_inflows - investing_outflows

    financing_inflows = mean_eff_st + mean_new_lt_loan
    financing_outflows = (
        mean_st_principal + mean_st_interest + mean_lt_principal + mean_lt_interest
    )
    financing_ncb = financing_inflows - financing_outflows

    owners_inflows = mean_equity_financing
    owners_outflows = mean_dividends_prev + mean_stock_buyback
    owners_ncb = owners_inflows - owners_outflows

    discretionary_inflows = mean_ms_return
    discretionary_outflows = np.zeros_like(mean_sales)
    discretionary_ncb = discretionary_inflows - discretionary_outflows

    cash_budget_rows = [
        ("MODULE 1: Operating Activities", None),
        ("Inflows from Sales", operating_inflows),
        ("Total Inflows (Operating)", operating_inflows),
        ("Payments for Purchases (COGS)", mean_cogs),
        ("Operational Expenses (OpEx)", mean_opex),
        ("Income Tax", mean_income_taxes_formula),
        ("Total Outflows (Operating)", operating_outflows),
        ("Operating Net Cash Balance", operating_ncb),
        ("", None),
        ("MODULE 2: Investing Activities", None),
        ("Total Inflows (Investing)", investing_inflows),
        ("Investment in Fixed Assets (CapEx)", investing_outflows),
        ("Total Outflows (Investing)", investing_outflows),
        ("Investing Net Cash Balance", investing_ncb),
        ("", None),
        ("MODULE 3: External Financing", None),
        ("ST Loan", mean_eff_st),
        ("LT Loan", mean_new_lt_loan),
        ("Total Inflows (Financing)", financing_inflows),
        ("ST Principal Payment", mean_st_principal),
        ("ST Interest", mean_st_interest),
        ("LT Principal Payment", mean_lt_principal),
        ("LT Interest", mean_lt_interest),
        ("Total Outflows (Financing)", financing_outflows),
        ("Financing Net Cash Balance", financing_ncb),
        ("", None),
        ("MODULE 4: Transactions with Owners", None),
        ("Invested Equity (Equity Financing)", owners_inflows),
        ("Total Inflows (Owners)", owners_inflows),
        ("Dividends from Last Year", mean_dividends_prev),
        ("Stock Buyback", mean_stock_buyback),
        ("Total Outflows (Owners)", owners_outflows),
        ("Owners' Transaction Net Cash Balance", owners_ncb),
        ("", None),
        ("MODULE 5: Discretionary Transactions", None),
        ("Return from ST Investments", discretionary_inflows),
        ("Total Inflows (Discretionary)", discretionary_inflows),
        ("Total Outflows (Discretionary)", discretionary_outflows),
        ("Discretionary Transaction Net Cash Balance", discretionary_ncb),
    ]
    print_markdown_table(
        "FORECAST CASH BUDGET — Mean across Monte Carlo samples (USD)",
        year_labels,
        cash_budget_rows,
        scale,
    )

    # --- Balance Sheet Identity Check ---
    max_abs_check = np.max(np.abs(mean_check * scale))
    print(f"\nBalance Sheet Identity Check (Assets = Liabilities + Equity):")
    print(f"  Max absolute mismatch across years (mean): ${max_abs_check:,.2f}")
    if max_abs_check < 1.0:
        print("  PASS: Balance sheet identity holds (mismatch < $1).")
    elif max_abs_check < 1000.0:
        print(
            "  PASS: Balance sheet identity holds within rounding (mismatch < $1,000)."
        )
    else:
        print(f"  WARNING: Balance sheet mismatch detected!")
        for t in range(n_years):
            check_val = mean_check[t] * scale
            if abs(check_val) >= 1000.0:
                print(f"    {year_labels[t]}: Assets - (Liab+Eq) = ${check_val:,.2f}")

    # --- Per-sample balance sheet identity check ---
    total_liab_equity_all = (
        ap_trajectories
        + aps_trajectories
        + effective_st_debt_trajectories
        + current_lt_debt_trajectories
        + non_current_liabilities_trajectories
        + equity_trajectories
    )
    check_all = assets_trajectories - total_liab_equity_all
    max_abs_check_all = np.max(np.abs(check_all)) * scale
    mean_abs_check_all = np.mean(np.abs(check_all)) * scale
    print(f"\n  Per-sample check (across all {n_samples} samples x {n_years} years):")
    print(f"    Max absolute mismatch:  ${max_abs_check_all:,.2f}")
    print(f"    Mean absolute mismatch: ${mean_abs_check_all:,.2f}")

    return {
        "net_income": ni_trajectories,
        "total_assets": assets_trajectories,
        "nca": nca_trajectories,
        "advance_payments_purchases": adv_pay_purch_trajectories,
        "accounts_receivable": ar_trajectories,
        "inventory": inv_trajectories,
        "cash": cash_trajectories,
        "investment_in_market_securities": ims_trajectories,
        "accounts_payable": ap_trajectories,
        "advance_payments_sales": aps_trajectories,
        "effective_st_debt": effective_st_debt_trajectories,
        "current_lt_debt": current_lt_debt_trajectories,
        "non_current_liabilities": non_current_liabilities_trajectories,
        "equity": equity_trajectories,
        "depreciation": depreciation_trajectories,
        "cogs": cogs_trajectories,
        "opex": opex_trajectories,
        "tax": tax_trajectories,
        "ms_return": ms_return_trajectories,
        "interest_payment": interest_payment_trajectories,
        "dividends": dividends_trajectories,
        "stock_buyback": stock_buyback_trajectories,
        "new_long_term_loan": new_lt_loan_trajectories,
        "equity_financing": equity_financing_trajectories,
        "liquidity_deficit_st": liq_deficit_st_trajectories,
    }
