"""Plotting utilities for OpEx fit and forecast diagnostics.

All plotting functions preserve prior chart content and output filenames while
providing clearer function contracts and documentation.
"""

from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from .io_utils import _get_training_results_path

tfd = tfp.distributions


def plot_opex_fit_with_aleatoric_noise(
    model: Any,
    historical_years: tf.Tensor,
    historical_sales_bil: tf.Tensor,
    historical_opex_bil: tf.Tensor,
    historical_inflation: Optional[tf.Tensor],
    n_samples: int = 2000,
    lower_q: float = 5.0,
    upper_q: float = 95.0,
    show_plot: bool = False,
    use_gaussian_ci: bool = False,
) -> None:
    """Plot historical OpEx against model fit with predictive uncertainty."""
    if historical_inflation is None:
        historical_inflation = tf.zeros_like(historical_sales_bil)
    cum_inf = tf.math.cumprod(1 + historical_inflation)

    mean_var_opex = model.q_var_opex_loc.numpy()
    mean_base_opex = model.q_base_opex_loc.numpy()
    sigma_opex = model.noise_sigma.numpy()
    sales_offset = model.sales_offset.numpy()

    # Center sales using the offset from training
    historical_sales_bil_centered = historical_sales_bil - sales_offset
    mean_opex_bil = (mean_base_opex * cum_inf) + (
        mean_var_opex * historical_sales_bil_centered
    )

    if use_gaussian_ci:
        # Analytical Gaussian predictive intervals (exact for linear-Gaussian model)
        var_var = float(model.q_var_opex_scale.numpy()) ** 2
        var_base = float(model.q_base_opex_scale.numpy()) ** 2
        var_noise = float(sigma_opex) ** 2
        cum_inf_tf = tf.cast(cum_inf, dtype=tf.float64)
        # Use centered sales for variance calculation
        sales_tf = tf.cast(historical_sales_bil_centered, dtype=tf.float64)
        std_opex_bil = tf.sqrt(
            (cum_inf_tf**2) * var_base + (sales_tf**2) * var_var + var_noise
        )
        z_low = float(tfd.Normal(0.0, 1.0).quantile(lower_q / 100.0))
        z_up = float(tfd.Normal(0.0, 1.0).quantile(upper_q / 100.0))
        lower_opex_bil = mean_opex_bil + z_low * std_opex_bil
        upper_opex_bil = mean_opex_bil + z_up * std_opex_bil
    else:
        # Posterior predictive samples with aleatoric noise (sigma)
        q_var = tfd.Normal(loc=model.q_var_opex_loc, scale=model.q_var_opex_scale)
        q_base = tfd.Normal(loc=model.q_base_opex_loc, scale=model.q_base_opex_scale)
        var_samples = q_var.sample(n_samples)  # [S]
        base_samples = q_base.sample(n_samples)  # [S]

        var_samples = tf.reshape(var_samples, (-1, 1))
        base_samples = tf.reshape(base_samples, (-1, 1))
        # Use centered sales for sampling
        sales = tf.reshape(
            tf.convert_to_tensor(historical_sales_bil_centered, dtype=tf.float64),
            (1, -1),
        )
        cum_inf_t = tf.reshape(tf.convert_to_tensor(cum_inf, dtype=tf.float64), (1, -1))
        noise = tf.random.normal(
            shape=(n_samples, len(historical_sales_bil)),
            mean=0.0,
            stddev=sigma_opex,
            dtype=tf.float64,
        )
        opex_samples_bil = (base_samples * cum_inf_t) + (var_samples * sales) + noise

        lower_opex_bil = tfp.stats.percentile(opex_samples_bil, lower_q, axis=0)
        upper_opex_bil = tfp.stats.percentile(opex_samples_bil, upper_q, axis=0)

    amount_scale = model.amount_scale
    mean_opex_usd = mean_opex_bil * amount_scale
    upper_opex_usd = upper_opex_bil * amount_scale
    lower_opex_usd = lower_opex_bil * amount_scale
    opex_hist_usd = historical_opex_bil * amount_scale
    sales_hist_usd = historical_sales_bil * amount_scale

    plt.figure(figsize=(10, 5))
    plt.plot(
        historical_years,
        opex_hist_usd,
        "o-",
        label="Historical OpEx",
        color="black",
    )
    plt.plot(
        historical_years,
        mean_opex_usd,
        "o-",
        label="Mean OpEx (learned)",
        color="tab:blue",
    )
    plt.plot(
        historical_years,
        lower_opex_usd,
        "--",
        label=f"Posterior predictive {lower_q:.0f}%",
        color="tab:blue",
        alpha=0.8,
    )
    plt.plot(
        historical_years,
        upper_opex_usd,
        "--",
        label=f"Posterior predictive {upper_q:.0f}%",
        color="tab:blue",
        alpha=0.8,
    )
    plt.title("OpEx vs Year with Learned Probabilistic Linear Regression")
    plt.xlabel("Year")
    plt.ylabel("OpEx (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "gaussian_ci" if use_gaussian_ci else "monte_carlo"
    plot_path = _get_training_results_path(
        f"opex_probabilistic_fit_{timestamp}_{tag}.png"
    )
    plt.savefig(plot_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()

    # --- OpEx vs Sales (separate figure) ---
    # Add x-axis padding to visualize extrapolation beyond training range
    x_min = float(tf.reduce_min(sales_hist_usd))
    x_max = float(tf.reduce_max(sales_hist_usd))
    x_span = x_max - x_min if x_max > x_min else max(abs(x_max), 1.0)
    x_pad = 0.5 * x_span
    x_left = x_min - x_pad
    x_right = x_max + x_pad

    # Extend the regression lines to the padded range
    sales_grid_usd = tf.linspace(tf.constant(x_left, dtype=tf.float64), tf.constant(x_right, dtype=tf.float64), 200)
    sales_grid_bil = sales_grid_usd / amount_scale
    # Center the sales grid using the offset
    sales_grid_bil_centered = sales_grid_bil - sales_offset
    cum_inf_mean = float(tf.reduce_mean(cum_inf))
    mean_opex_grid_bil = (mean_base_opex * cum_inf_mean) + (
        mean_var_opex * sales_grid_bil_centered
    )
    mean_opex_grid_usd = mean_opex_grid_bil * amount_scale

    if use_gaussian_ci:
        var_var = float(model.q_var_opex_scale.numpy()) ** 2
        var_base = float(model.q_base_opex_scale.numpy()) ** 2
        var_noise = float(sigma_opex) ** 2
        # Use centered sales for variance calculation
        std_opex_grid_bil = tf.sqrt(
            (cum_inf_mean**2) * var_base
            + (sales_grid_bil_centered**2) * var_var
            + var_noise
        )
        z_low = float(tfd.Normal(0.0, 1.0).quantile(lower_q / 100.0))
        z_up = float(tfd.Normal(0.0, 1.0).quantile(upper_q / 100.0))
        lower_opex_grid_bil = mean_opex_grid_bil + z_low * std_opex_grid_bil
        upper_opex_grid_bil = mean_opex_grid_bil + z_up * std_opex_grid_bil
    else:
        q_var = tfd.Normal(loc=model.q_var_opex_loc, scale=model.q_var_opex_scale)
        q_base = tfd.Normal(loc=model.q_base_opex_loc, scale=model.q_base_opex_scale)
        var_samples = q_var.sample(n_samples)
        base_samples = q_base.sample(n_samples)
        var_samples = tf.reshape(var_samples, (-1, 1))
        base_samples = tf.reshape(base_samples, (-1, 1))
        # Use centered sales for grid sampling
        sales_grid_t = tf.reshape(
            tf.convert_to_tensor(sales_grid_bil_centered, dtype=tf.float64), (1, -1)
        )
        cum_inf_grid_t = tf.reshape(
            tf.fill(sales_grid_bil.shape, tf.constant(cum_inf_mean, dtype=tf.float64)),
            (1, -1),
        )
        noise_grid = tf.random.normal(
            shape=(n_samples, len(sales_grid_bil)),
            mean=0.0,
            stddev=sigma_opex,
            dtype=tf.float64,
        )
        opex_samples_grid_bil = (
            (base_samples * cum_inf_grid_t) + (var_samples * sales_grid_t) + noise_grid
        )
        lower_opex_grid_bil = tfp.stats.percentile(opex_samples_grid_bil, lower_q, axis=0)
        upper_opex_grid_bil = tfp.stats.percentile(opex_samples_grid_bil, upper_q, axis=0)
    lower_opex_grid_usd = lower_opex_grid_bil * amount_scale
    upper_opex_grid_usd = upper_opex_grid_bil * amount_scale

    plt.figure(figsize=(10, 5))
    plt.scatter(
        sales_hist_usd,
        opex_hist_usd,
        label="Historical OpEx",
        color="black",
        zorder=3,
    )
    # Per-data-point predictions using actual per-year cum_inf (matches vs-Year plot)
    plt.scatter(
        sales_hist_usd,
        mean_opex_usd,
        label="Mean OpEx per data point (learned)",
        color="tab:blue",
        marker="x",
        s=80,
        zorder=4,
    )
    plt.plot(
        sales_grid_usd,
        mean_opex_grid_usd,
        "-",
        label="Mean OpEx trend (avg. inflation)",
        color="tab:blue",
        alpha=0.5,
    )
    plt.plot(
        sales_grid_usd,
        lower_opex_grid_usd,
        "--",
        label=f"Posterior predictive {lower_q:.0f}%",
        color="tab:blue",
        alpha=0.8,
    )
    plt.plot(
        sales_grid_usd,
        upper_opex_grid_usd,
        "--",
        label=f"Posterior predictive {upper_q:.0f}%",
        color="tab:blue",
        alpha=0.8,
    )
    plt.xlim(x_left, x_right)
    plt.title("OpEx vs Sales with Learned Probabilistic Linear Regression")
    plt.xlabel("Sales (USD)")
    plt.ylabel("OpEx (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "gaussian_ci" if use_gaussian_ci else "monte_carlo"
    plot_path = _get_training_results_path(f"opex_vs_sales_fit_{timestamp}_{tag}.png")
    plt.savefig(plot_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_historical_and_forecast(
    historical_years: tf.Tensor,
    forecast_years: tf.Tensor,
    historical_data: Dict[str, tf.Tensor],
    forecast_trajectories: Dict[str, tf.Tensor],
    amount_scale: float,
    sales_hist_usd: Optional[tf.Tensor] = None,
    sales_forecast_usd: Optional[tf.Tensor] = None,
    historical_fit: Optional[Dict[str, tf.Tensor]] = None,
    historical_fit_years: Optional[tf.Tensor] = None,
    show_plot: bool = False,
) -> None:
    """
    Plots all financial elements from historical period through forecast period,
    optionally overlaying the model's one-step-ahead fitted values on historical data.

    Args:
        historical_years: array of year labels for historical data (e.g., [2018, ..., 2025])
        forecast_years: array of year labels for forecast data (e.g., [2025, ..., 2033])
        historical_data: dict of {name: array_in_usd} for historical data
        forecast_trajectories: dict of {name: array[n_samples, n_years] in scaled units}
        amount_scale: scaling factor to convert scaled units back to USD
        sales_hist_usd: optional array of historical sales in USD
        sales_forecast_usd: optional array of deterministic sales forecast in USD
        historical_fit: optional dict of {name: array_in_usd} for model-fitted historical values
        historical_fit_years: optional array of year labels for fitted values
        show_plot: whether to call plt.show()
    """
    elements = list(forecast_trajectories.keys())
    n_elements = len(elements)

    # Compute mean, 2.5%, 97.5% for each element
    forecast_stats = {}
    for name in elements:
        trajs = forecast_trajectories[name]
        forecast_stats[name] = {
            "mean": tf.reduce_mean(trajs, axis=0) * amount_scale,
            "lower": tfp.stats.percentile(trajs, 2.5, axis=0) * amount_scale,
            "upper": tfp.stats.percentile(trajs, 97.5, axis=0) * amount_scale,
        }

    # Layout: add 1 for sales if provided
    total_plots = n_elements + (1 if sales_forecast_usd is not None else 0)
    ncols = 3
    nrows = (total_plots + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    axs = axs.flatten()

    # Readable display names
    display_names = {
        "net_income": "Net Income",
        "total_assets": "Total Assets",
        "nca": "Non-Current Assets",
        "advance_payments_purchases": "Advance Payments (Purchases)",
        "accounts_receivable": "Accounts Receivable",
        "inventory": "Inventory",
        "cash": "Cash",
        "investment_in_market_securities": "Investment in Market Securities",
        "accounts_payable": "Accounts Payable",
        "advance_payments_sales": "Advance Payments (Sales)",
        "current_liabilities": "Current Liabilities",
        "non_current_liabilities": "Non-Current Liabilities",
        "equity": "Stockholders' Equity",
        "depreciation": "Depreciation",
        "dividends": "Dividends",
        "stock_buyback": "Stock Buyback",
        "ms_return": "Return on Market Securities",
        "interest_payment": "Interest Payment",
        "new_short_term_loan": "New Short-Term Loan",
        "new_long_term_loan": "New Long-Term Loan",
        "equity_financing": "Equity Financing",
        "liquidity_deficit_st": "Liquidity Deficit (Short-Term)",
        "cogs": "COGS",
        "opex": "OpEx",
        "tax": "Tax",
    }

    ax_idx = 0

    # Plot Sales (deterministic — exogenous input, no model fit)
    if sales_forecast_usd is not None:
        ax = axs[ax_idx]
        if sales_hist_usd is not None:
            ax.plot(
                historical_years,
                sales_hist_usd,
                "ko-",
                label="Historical",
                markersize=5,
                linewidth=1.5,
            )
        ax.plot(
            forecast_years,
            sales_forecast_usd,
            "s-",
            color="tab:blue",
            label="Forecast",
            markersize=5,
            linewidth=1.5,
        )
        ax.set_title(
            "Sales (Revenue) [Exogenous Input]", fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("USD")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.tick_params(axis="x", rotation=45)
        ax_idx += 1

    # Plot each forecasted element
    for name in elements:
        ax = axs[ax_idx]
        stats = forecast_stats[name]
        label = display_names.get(name, name)

        # Historical actual data
        if name in historical_data and historical_data[name] is not None:
            ax.plot(
                historical_years,
                historical_data[name],
                "ko-",
                label="Historical",
                markersize=5,
                linewidth=1.5,
            )

        # Model fit on historical data (one-step-ahead predictions)
        if (
            historical_fit is not None
            and historical_fit_years is not None
            and name in historical_fit
        ):
            ax.plot(
                historical_fit_years,
                historical_fit[name],
                "^--",
                color="tab:red",
                label="Model Fit (1-step)",
                markersize=5,
                linewidth=1.2,
                alpha=0.85,
            )

        # Forecast mean + 95% CI
        ax.plot(
            forecast_years,
            stats["mean"],
            "s-",
            color="tab:blue",
            label="Forecast Mean",
            markersize=5,
            linewidth=1.5,
        )
        ax.fill_between(
            forecast_years,
            stats["lower"],
            stats["upper"],
            color="tab:blue",
            alpha=0.2,
            label="95% CI",
        )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("USD")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.tick_params(axis="x", rotation=45)
        ax_idx += 1

    # Hide unused axes
    for i in range(ax_idx, len(axs)):
        axs[i].set_visible(False)

    fig.suptitle(
        "Financial Model: Historical Fit & Monte Carlo Forecast",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = _get_training_results_path(f"all_elements_forecast_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {plot_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()
