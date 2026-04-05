"""Plotting utilities for forecast diagnostics."""

from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from financial_forecast.training.io_utils import get_training_results_path

tfd = tfp.distributions


def plot_historical_and_forecast(
    historical_years: tf.Tensor,
    forecast_years: tf.Tensor,
    historical_data: Dict[str, tf.Tensor],
    forecast_trajectories: Dict[str, tf.Tensor],
    amount_scale: float,
    sales_hist_usd: Optional[tf.Tensor] = None,
    sales_forecast_usd: Optional[tf.Tensor] = None,
    historical_fit: Optional[Dict[str, tf.Tensor]] = None,
    historical_fit_lower: Optional[Dict[str, tf.Tensor]] = None,
    historical_fit_upper: Optional[Dict[str, tf.Tensor]] = None,
    historical_fit_years: Optional[tf.Tensor] = None,
    show_plot: bool = False,
) -> None:
    """Plot all financial elements from historical period through forecast.

    Optionally overlays the model's one-step-ahead fitted values on
    historical data alongside Monte Carlo forecast bands.

    Args:
        historical_years: Year labels for historical data (e.g.
            ``[2018, ..., 2025]``).
        forecast_years: Year labels for forecast data (e.g.
            ``[2025, ..., 2033]``).
        historical_data: Mapping of ``{name: array_in_usd}`` for
            historical actuals.
        forecast_trajectories: Mapping of
            ``{name: array[n_samples, n_years]}`` in scaled units.
        amount_scale: Factor to convert scaled units back to USD.
        sales_hist_usd: Optional historical sales in USD.
        sales_forecast_usd: Optional deterministic sales forecast in USD.
        historical_fit: Optional mapping of ``{name: array_in_usd}`` for
            model-fitted historical values (mean series).
        historical_fit_lower: Optional mapping of
            ``{name: array_in_usd}`` with the 2.5% bound of the 1-step
            MC fit.  When provided together with ``historical_fit_upper``,
            a shaded 95% credible band is drawn around the fit mean.
        historical_fit_upper: Optional mapping of
            ``{name: array_in_usd}`` with the 97.5% bound of the 1-step
            MC fit.
        historical_fit_years: Optional year labels for fitted values.
        show_plot: Whether to call ``plt.show()`` after saving.
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

    # Plot Sales (deterministic -- exogenous input, no model fit)
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
        has_mc_bounds = (
            historical_fit_lower is not None
            and historical_fit_upper is not None
            and historical_fit_lower is not None
            and name in (historical_fit_lower or {})
            and name in (historical_fit_upper or {})
        )
        if (
            historical_fit is not None
            and historical_fit_years is not None
            and name in historical_fit
        ):
            fit_label = (
                "Model Fit (1-step MC mean)"
                if has_mc_bounds
                else "Model Fit (1-step)"
            )
            ax.plot(
                historical_fit_years,
                historical_fit[name],
                "^--",
                color="tab:red",
                label=fit_label,
                markersize=5,
                linewidth=1.2,
                alpha=0.85,
            )
            if has_mc_bounds:
                ax.fill_between(
                    historical_fit_years,
                    historical_fit_lower[name],
                    historical_fit_upper[name],
                    color="tab:red",
                    alpha=0.2,
                    label="1-step 95% CI",
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
    plot_path = get_training_results_path(f"all_elements_forecast_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {plot_path}")
    if show_plot:
        plt.show()
    else:
        plt.close()
