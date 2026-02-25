"""Top-level orchestration for training and forecast runs."""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf

from historical_data import get_apple_historical_data

from .forecast import run_monte_carlo_forecast
from .model import TrainableFinancialModel
from .plotting import plot_historical_and_forecast, plot_opex_fit_with_aleatoric_noise


def _as_float64_constant(value: float) -> tf.Tensor:
    """Create a TensorFlow float64 scalar constant."""
    return tf.constant(value, dtype=tf.float64)


def _build_state_from_index(
    index: int,
    nca: np.ndarray,
    advance_payments_purchases: np.ndarray,
    accounts_receivable: np.ndarray,
    inventory: np.ndarray,
    cash: np.ndarray,
    investment_in_market_securities: np.ndarray,
    accounts_payable: np.ndarray,
    advance_payments_sales: np.ndarray,
    effective_st_debt: np.ndarray,
    current_lt_debt: np.ndarray,
    non_current_liabilities: np.ndarray,
    equity: np.ndarray,
    net_income: np.ndarray,
    dividends: np.ndarray,
) -> Dict[str, tf.Tensor]:
    """Build model state dictionary for a specific historical index."""
    return {
        "nca": _as_float64_constant(nca[index]),
        "advance_payments_purchases": _as_float64_constant(
            advance_payments_purchases[index]
        ),
        "accounts_receivable": _as_float64_constant(accounts_receivable[index]),
        "inventory": _as_float64_constant(inventory[index]),
        "cash": _as_float64_constant(cash[index]),
        "investment_in_market_securities": _as_float64_constant(
            investment_in_market_securities[index]
        ),
        "accounts_payable": _as_float64_constant(accounts_payable[index]),
        "advance_payments_sales": _as_float64_constant(advance_payments_sales[index]),
        "effective_st_debt": _as_float64_constant(effective_st_debt[index]),
        "current_lt_debt": _as_float64_constant(current_lt_debt[index]),
        "non_current_liabilities": _as_float64_constant(non_current_liabilities[index]),
        "equity": _as_float64_constant(equity[index]),
        "net_income": _as_float64_constant(net_income[index]),
        "dividends": _as_float64_constant(dividends[index]),
    }


def _validate_historical_data(data: Mapping[str, np.ndarray]) -> None:
    """Validate required historical data keys."""
    required_keys = {
        "sales",
        "purchases",
        "cogs",
        "nca",
        "depreciation",
        "advance_payments_purchases",
        "accounts_receivable",
        "accounts_payable",
        "advance_payments_sales",
        "cash",
        "ims",
        "inventory",
        "current_liabilities",
        "non_current_liabilities",
        "equity",
        "net_income",
        "dividends",
        "stock_buyback",
        "opex",
        "tax",
        "inflation",
        "current_lt_debt",
        "interest_payment",
        "ms_return",
        "tax_onetime_payments",
    }
    missing = sorted(required_keys.difference(data.keys()))
    if missing:
        raise ValueError(
            "historical_data is missing required keys: " + ", ".join(missing)
        )


def run_training_and_forecast(
    historical_data: Mapping[str, np.ndarray],
    sales_forecast_usd: Sequence[float],
    inflation_forecast: Sequence[float],
    use_trained_parameters: bool = False,
    parameters_path: str = "trained_parameters.npz",
    use_inflation: bool = True,
    include_tax_anomalies: bool = True,
    simple_policy_epochs: int = 25000,
    structural_epochs: int = 20000,
    monte_carlo_samples: int = 1000,
) -> None:
    """Run model training (optional), forecast simulation, and plotting.

    This function preserves the historical training/forecast flow and output
    artifacts while centralizing orchestration in one place. Callers may
    optionally provide preloaded historical data and forecast assumptions.
    """
    if simple_policy_epochs < 1:
        raise ValueError("simple_policy_epochs must be >= 1")
    if structural_epochs < 1:
        raise ValueError("structural_epochs must be >= 1")
    if monte_carlo_samples < 1:
        raise ValueError("monte_carlo_samples must be >= 1")

    model = TrainableFinancialModel(base_year=2018)

    # --- 1. LOAD HISTORICAL DATA FROM APPLE (2018-2025) ---
    data = historical_data
    _validate_historical_data(data)
    sales_hist = data["sales"]
    purchases_hist = data["purchases"]
    cogs_hist = data["cogs"]
    nca_hist = data["nca"]
    depr_hist = data["depreciation"]
    advance_payments_purchases_hist = data["advance_payments_purchases"]
    accounts_receivable_hist = data["accounts_receivable"]
    accounts_payable_hist = data["accounts_payable"]
    advance_payments_sales_hist = data["advance_payments_sales"]
    cash_hist = data["cash"]
    investment_in_market_securities_hist = data["ims"]
    inventory_hist = data["inventory"]
    current_liabilities_hist = data["current_liabilities"]
    non_current_liabilities_hist = data["non_current_liabilities"]
    equity_hist = data["equity"]
    net_income_hist = data["net_income"]
    dividends_hist = data["dividends"]
    stock_buyback_hist = data["stock_buyback"]
    opex_hist = data["opex"]
    tax_hist = data["tax"]
    tax_onetime_payments_hist = (
        data["tax_onetime_payments"]
        if include_tax_anomalies
        else np.zeros(len(sales_hist))
    )
    inflation_hist = data["inflation"] if use_inflation else np.zeros(len(sales_hist))
    current_lt_debt_hist = data["current_lt_debt"]
    effective_st_debt_hist = (
        current_liabilities_hist
        - accounts_payable_hist
        - advance_payments_sales_hist
        - current_lt_debt_hist
    )
    interest_payment_hist = data["interest_payment"]
    ms_return_hist = data["ms_return"]

    # --- 2. SCALE INPUTS AND TARGETS TO BILLIONS FOR TRAINING STABILITY ---
    amount_scale = model.amount_scale
    sales_hist_bil = sales_hist / amount_scale
    purchases_hist_bil = purchases_hist / amount_scale
    cogs_hist_bil = cogs_hist / amount_scale
    nca_hist_bil = nca_hist / amount_scale
    depr_hist_bil = depr_hist / amount_scale
    advance_payments_sales_hist_bil = advance_payments_sales_hist / amount_scale
    advance_payments_purchases_hist_bil = advance_payments_purchases_hist / amount_scale
    accounts_receivable_hist_bil = accounts_receivable_hist / amount_scale
    accounts_payable_hist_bil = accounts_payable_hist / amount_scale
    inventory_hist_bil = inventory_hist / amount_scale
    cash_hist_bil = cash_hist / amount_scale
    investment_in_market_securities_hist_bil = (
        investment_in_market_securities_hist / amount_scale
    )
    net_income_hist_bil = net_income_hist / amount_scale
    dividends_hist_bil = dividends_hist / amount_scale
    stock_buyback_hist_bil = stock_buyback_hist / amount_scale
    opex_hist_bil = opex_hist / amount_scale
    tax_hist_bil = tax_hist / amount_scale
    tax_onetime_payments_hist_bil = tax_onetime_payments_hist / amount_scale
    effective_st_debt_hist_bil = effective_st_debt_hist / amount_scale
    current_lt_debt_hist_bil = current_lt_debt_hist / amount_scale
    non_current_liabilities_hist_bil = non_current_liabilities_hist / amount_scale
    equity_hist_bil = equity_hist / amount_scale
    interest_payment_hist_bil = interest_payment_hist / amount_scale
    ms_return_hist_bil = ms_return_hist / amount_scale

    if use_trained_parameters:
        model.load_parameters(parameters_path)
    else:
        # --- 2. TRAIN THE MODEL ---
        # We feed in the historical arrays from 2018-2024, and leave 2025 for forecast testing.
        # Historical years: FY2018..FY2024 (training), FY2025 held out for testing
        n_train = len(sales_hist_bil[:-1])
        train_years = np.arange(
            model.base_year, model.base_year + n_train, dtype=np.float64
        )
        model.train_simple_policies(
            sales_hist_bil[:-1],
            purchases_hist_bil[:-1],
            cogs_hist_bil[:-1],
            nca_hist_bil[:-1],
            depr_hist_bil[:-1],
            advance_payments_sales_hist_bil[:-1],
            advance_payments_purchases_hist_bil[:-1],
            accounts_receivable_hist_bil[:-1],
            accounts_payable_hist_bil[:-1],
            inventory_hist_bil[:-1],
            cash_hist_bil[:-1],
            investment_in_market_securities_hist_bil[:-1],
            net_income_hist_bil[:-1],
            dividends_hist_bil[:-1],
            stock_buyback_hist_bil[:-1],
            opex_hist_bil[:-1],
            tax_hist_bil[:-1],
            historical_tax_onetime_payments=tax_onetime_payments_hist_bil[:-1],
            historical_eff_st_debt=effective_st_debt_hist_bil[:-1],
            historical_inflation=inflation_hist[:-1],
            historical_years=train_years,
            epochs=simple_policy_epochs,
            show_plot=False,
            loss_scale_mode="std",
        )

        # --- 3. TRAIN STRUCTURAL PARAMETERS ---
        # We still only feed in the historical arrays from FY2018-FY2024, and leave FY2025 for forecast testing.
        model.train_structural_parameters(
            sales_hist_bil[:-1],
            nca_hist_bil[:-1],
            advance_payments_sales_hist_bil[:-1],
            advance_payments_purchases_hist_bil[:-1],
            accounts_receivable_hist_bil[:-1],
            accounts_payable_hist_bil[:-1],
            inventory_hist_bil[:-1],
            cash_hist_bil[:-1],
            investment_in_market_securities_hist_bil[:-1],
            net_income_hist_bil[:-1],
            dividends_hist_bil[:-1],
            stock_buyback_hist_bil[:-1],
            opex_hist_bil[:-1],
            tax_hist_bil[:-1],
            effective_st_debt_hist_bil[:-1],
            current_lt_debt_hist_bil[:-1],
            non_current_liabilities_hist_bil[:-1],
            interest_payment_hist_bil[:-1],
            ms_return_hist_bil[:-1],
            equity_hist_bil[:-1],
            historical_tax_onetime_payments=tax_onetime_payments_hist_bil[:-1],
            historical_inflation=inflation_hist[:-1],
            historical_years=train_years,
            epochs=structural_epochs,
            loss_scale_mode="std",
        )
        model.save_parameters(parameters_path)

    # --- 4. PLOT OPEX FIT (Mean + Aleatoric Sigma) ---
    historical_years = np.arange(1, len(opex_hist_bil) + 1)

    # Posterior prediction by Gaussian Confidence Interval
    plot_opex_fit_with_aleatoric_noise(
        model,
        historical_years,
        sales_hist_bil,
        opex_hist_bil,
        inflation_hist,
        show_plot=False,
        use_gaussian_ci=True,
    )

    # Posterior prediction by sampling (Monte Carlo)
    plot_opex_fit_with_aleatoric_noise(
        model,
        historical_years,
        sales_hist_bil,
        opex_hist_bil,
        inflation_hist,
        show_plot=False,
        use_gaussian_ci=False,
    )

    # --- 5. RUN FORECAST (Using new parameters) ---
    # Initial State (t=0) 2024 Apple Balance Sheet
    state = _build_state_from_index(
        index=-2,
        nca=nca_hist_bil,
        advance_payments_purchases=advance_payments_purchases_hist_bil,
        accounts_receivable=accounts_receivable_hist_bil,
        inventory=inventory_hist_bil,
        cash=cash_hist_bil,
        investment_in_market_securities=investment_in_market_securities_hist_bil,
        accounts_payable=accounts_payable_hist_bil,
        advance_payments_sales=advance_payments_sales_hist_bil,
        effective_st_debt=effective_st_debt_hist_bil,
        current_lt_debt=current_lt_debt_hist_bil,
        non_current_liabilities=non_current_liabilities_hist_bil,
        equity=equity_hist_bil,
        net_income=net_income_hist_bil,
        dividends=dividends_hist_bil,
    )

    # Forecast Drivers: Sales is the sole exogenous driver.
    # Purchases are derived inside forecast_step from the learned cost ratio.
    n_hist = len(sales_hist)  # e.g. 8 for FY2018-FY2025
    sales_forecast_usd_array = np.asarray(sales_forecast_usd, dtype=np.float64)
    if sales_forecast_usd_array.ndim != 1 or sales_forecast_usd_array.size == 0:
        raise ValueError("sales_forecast_usd must be a non-empty 1D sequence")
    sales_forecast = sales_forecast_usd_array / amount_scale
    n_forecast_years = int(sales_forecast.size)
    # Forecast starts at FY2025 (last historical year) and continues forward
    last_hist_year = model.base_year + n_hist - 1  # FY2025
    forecast_years = np.arange(
        last_hist_year, last_hist_year + n_forecast_years, dtype=np.float64
    )

    # Continue inflation compounding from the historical baseline. The last year is the start of the forecasted years
    cum_inf_hist = np.cumprod(1 + inflation_hist)
    last_historical_cum_inf = cum_inf_hist[-2]

    inflation_forecast_array = np.asarray(inflation_forecast, dtype=np.float64)
    if (
        inflation_forecast_array.ndim != 1
        or inflation_forecast_array.size != n_forecast_years
    ):
        raise ValueError(
            "inflation_forecast must be a 1D sequence with the same length as sales_forecast_usd"
        )
    if not use_inflation:
        inflation_forecast_array = np.zeros_like(inflation_forecast_array)
    cum_inf_forecast = last_historical_cum_inf * np.cumprod(
        1 + inflation_forecast_array
    )

    # --- Execute Monte Carlo Forecast ---
    forecast_trajectories = run_monte_carlo_forecast(
        model,
        state,
        sales_forecast,
        cum_inf_forecast,
        forecast_years,
        n_samples=monte_carlo_samples,
    )

    # --- 6. COMPUTE ONE-STEP-AHEAD HISTORICAL FIT ---
    # For each year t+1, use actual state at t and predict state at t+1
    # This shows how well the model's learned parameters fit the historical data.
    n_hist_points = len(sales_hist)
    hist_fit_keys = [
        "net_income",
        "total_assets",
        "nca",
        "advance_payments_purchases",
        "accounts_receivable",
        "inventory",
        "cash",
        "investment_in_market_securities",
        "accounts_payable",
        "advance_payments_sales",
        "effective_st_debt",
        "non_current_liabilities",
        "equity",
        "depreciation",
        "cogs",
        "opex",
        "tax",
        "ms_return",
        "interest_payment",
        "dividends",
        "stock_buyback",
        "new_long_term_loan",
        "equity_financing",
        "liquidity_deficit_st",
    ]

    historical_fit = {k: [] for k in hist_fit_keys}
    historical_fit_years = []

    for t in range(n_hist_points - 1):  # t = 0..6, predicting index t+1
        # Actual state at year t
        state_t = _build_state_from_index(
            index=t,
            nca=nca_hist_bil,
            advance_payments_purchases=advance_payments_purchases_hist_bil,
            accounts_receivable=accounts_receivable_hist_bil,
            inventory=inventory_hist_bil,
            cash=cash_hist_bil,
            investment_in_market_securities=investment_in_market_securities_hist_bil,
            accounts_payable=accounts_payable_hist_bil,
            advance_payments_sales=advance_payments_sales_hist_bil,
            effective_st_debt=effective_st_debt_hist_bil,
            current_lt_debt=current_lt_debt_hist_bil,
            non_current_liabilities=non_current_liabilities_hist_bil,
            equity=equity_hist_bil,
            net_income=net_income_hist_bil,
            dividends=dividends_hist_bil,
        )

        sales_t1 = sales_hist_bil[t + 1]

        inputs_t = {
            "sales_t": _as_float64_constant(sales_t1),
            "year": _as_float64_constant(float(model.base_year + t + 1)),
            "cum_inflation": _as_float64_constant(cum_inf_hist[t + 1]),
            "tax_onetime_payment": _as_float64_constant(
                tax_onetime_payments_hist_bil[t + 1]
            ),
        }

        pred = model.forecast_step(state_t, inputs_t, use_mean_opex=True)

        # Collect predicted values (convert back to USD)
        historical_fit["net_income"].append(
            float(pred["net_income"].numpy()) * amount_scale
        )
        historical_fit["nca"].append(float(pred["nca"].numpy()) * amount_scale)
        historical_fit["advance_payments_purchases"].append(
            float(pred["advance_payments_purchases"].numpy()) * amount_scale
        )
        historical_fit["accounts_receivable"].append(
            float(pred["accounts_receivable"].numpy()) * amount_scale
        )
        historical_fit["inventory"].append(
            float(pred["inventory"].numpy()) * amount_scale
        )
        historical_fit["cash"].append(float(pred["cash"].numpy()) * amount_scale)
        historical_fit["investment_in_market_securities"].append(
            float(pred["investment_in_market_securities"].numpy()) * amount_scale
        )
        historical_fit["accounts_payable"].append(
            float(pred["accounts_payable"].numpy()) * amount_scale
        )
        historical_fit["advance_payments_sales"].append(
            float(pred["advance_payments_sales"].numpy()) * amount_scale
        )
        historical_fit["effective_st_debt"].append(
            float(pred["effective_st_debt"].numpy()) * amount_scale
        )
        historical_fit["non_current_liabilities"].append(
            float(pred["non_current_liabilities"].numpy()) * amount_scale
        )
        historical_fit["equity"].append(float(pred["equity"].numpy()) * amount_scale)
        historical_fit["depreciation"].append(
            float(pred["depreciation"].numpy()) * amount_scale
        )
        historical_fit["cogs"].append(float(pred["cogs"].numpy()) * amount_scale)
        historical_fit["opex"].append(float(pred["opex"].numpy()) * amount_scale)
        historical_fit["tax"].append(float(pred["tax"].numpy()) * amount_scale)
        historical_fit["ms_return"].append(
            float(pred["ms_return"].numpy()) * amount_scale
        )
        historical_fit["interest_payment"].append(
            float(pred["interest_payment"].numpy()) * amount_scale
        )
        historical_fit["dividends"].append(
            float(pred["dividends"].numpy()) * amount_scale
        )
        historical_fit["stock_buyback"].append(
            float(pred["stock_buyback"].numpy()) * amount_scale
        )
        historical_fit["new_long_term_loan"].append(
            float(pred["new_long_term_loan"].numpy()) * amount_scale
        )
        historical_fit["equity_financing"].append(
            float(pred["equity_financing"].numpy()) * amount_scale
        )
        historical_fit["liquidity_deficit_st"].append(
            float(pred["liquidity_deficit_st"].numpy()) * amount_scale
        )

        total_assets_pred = (
            pred["nca"]
            + pred["advance_payments_purchases"]
            + pred["accounts_receivable"]
            + pred["inventory"]
            + pred["cash"]
            + pred["investment_in_market_securities"]
        )
        historical_fit["total_assets"].append(
            float(total_assets_pred.numpy()) * amount_scale
        )

        historical_fit_years.append(model.base_year + t + 1)

    # Convert to numpy arrays
    for k in historical_fit:
        historical_fit[k] = np.array(historical_fit[k])
    historical_fit_years = np.array(historical_fit_years)

    # --- 7. PLOT ALL ELEMENTS: HISTORICAL + FIT + FORECAST ---
    historical_years = np.arange(model.base_year, model.base_year + n_hist_points)

    n_forecast_steps = len(sales_forecast)
    forecast_year_start = model.base_year + n_hist_points - 1  # FY2025
    plot_forecast_years = np.arange(
        forecast_year_start, forecast_year_start + n_forecast_steps
    )

    # Build historical data dict (in USD, not scaled)
    total_assets_hist = (
        nca_hist
        + advance_payments_purchases_hist
        + accounts_receivable_hist
        + inventory_hist
        + cash_hist
        + investment_in_market_securities_hist
    )
    historical_data = {
        "net_income": net_income_hist,
        "total_assets": total_assets_hist,
        "nca": nca_hist,
        "advance_payments_purchases": advance_payments_purchases_hist,
        "accounts_receivable": accounts_receivable_hist,
        "inventory": inventory_hist,
        "cash": cash_hist,
        "investment_in_market_securities": investment_in_market_securities_hist,
        "accounts_payable": accounts_payable_hist,
        "advance_payments_sales": advance_payments_sales_hist,
        "non_current_liabilities": non_current_liabilities_hist,
        "equity": equity_hist,
        "depreciation": depr_hist,
        "cogs": cogs_hist,
        "opex": opex_hist,
        "tax": tax_hist,
        "ms_return": ms_return_hist,
        "interest_payment": interest_payment_hist,
        "dividends": dividends_hist,
        "stock_buyback": stock_buyback_hist,
        "effective_st_debt": effective_st_debt_hist,
    }

    # Sales forecast in USD for the forecasted years
    sales_forecast_usd = sales_forecast * amount_scale

    plot_historical_and_forecast(
        historical_years=historical_years,
        forecast_years=plot_forecast_years,
        historical_data=historical_data,
        forecast_trajectories=forecast_trajectories,
        amount_scale=amount_scale,
        sales_hist_usd=sales_hist,
        sales_forecast_usd=sales_forecast_usd,
        historical_fit=historical_fit,
        historical_fit_years=historical_fit_years,
        show_plot=False,
    )
