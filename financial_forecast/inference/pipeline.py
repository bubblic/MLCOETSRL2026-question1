"""Simulation and plotting orchestrator for prepared financial models.

Takes a model that has already been prepared (and optionally trained),
generates forecast drivers via pluggable sales/inflation forecast models,
runs the trajectory simulator, computes one-step-ahead historical fit,
plots results, and exports a self-contained JSON report.

Example::

    model = BaseFinancialModel(...)
    model.prepare(data.financial_statements, data.inflation)
    ForecastPipeline(model, forecast_years=10).run()
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from financial_forecast.inference.plotting import plot_historical_and_forecast
from financial_forecast.inference.state_index import (
    DIAGNOSTIC_KEYS,
    RECURRENT_KEYS,
)
from financial_forecast.inference.forecast_driver_models import (
    SalesForecastModel,
    InflationForecastModel,
)
from financial_forecast.models.opex import BayesianOpEx
from financial_forecast.training.io_utils import get_training_results_path
from financial_forecast.data.loader import HistoricalDataLoader
from financial_forecast.reporting.table_formatter import MarkdownTableFormatter


class ForecastPipeline:
    """Runs trajectory simulation and plots results for a prepared model.

    The model must have been prepared via :meth:`prepare` before
    constructing the pipeline.  The pipeline generates forecast drivers
    (sales and inflation trajectories) via pluggable forecast models,
    runs the trajectory simulator (which calls
    ``model.forecast_step_compiled`` as the evolution function),
    computes one-step-ahead historical fit, and plots everything.

    Args:
        model: A prepared :class:`BaseFinancialModel` or
            :class:`TrainableFinancialModel`.
        data: The :class:`HistoricalDataLoader` used to prepare the
            model.  Used to format historical tables in the JSON report.
        sales_forecast: An initialized sales forecast model (e.g.
            :class:`LinearSalesForecast`).
        inflation_forecast: An initialized inflation forecast model
            (e.g. :class:`ConstantInflationForecast`).
        show_plot: Whether to call ``plt.show()`` after saving plots.
    """

    def __init__(
        self,
        model: tf.Module,
        data: HistoricalDataLoader,
        sales_forecast: SalesForecastModel,
        inflation_forecast: InflationForecastModel,
        show_plot: bool = False,
    ):
        self.model = model
        self._show_plot = show_plot
        self._data = data
        self._formatter = MarkdownTableFormatter()

        d = model.historical_data
        n_hist = len(d["sales"])
        n_fc = sales_forecast.n_years

        # Scaled sales forecast
        self._sales_forecast_usd = sales_forecast.forecast
        self._sales_forecast = self._sales_forecast_usd / model.amount_scale

        # Year labels
        last_hist_year = model.base_year + n_hist - 1
        self._forecast_years = tf.cast(
            tf.range(last_hist_year, last_hist_year + n_fc),
            dtype=tf.float64,
        )

        # Cumulative inflation for forecast period
        cum_inf_hist = tf.math.cumprod(1 + d["inflation"])
        last_cum_inf = cum_inf_hist[-(model.test_years + 1)]
        self._cum_inf_forecast = last_cum_inf * tf.math.cumprod(
            1 + inflation_forecast.forecast
        )

    def run(self) -> None:
        """Execute: trajectory forecast, historical fit, plot, export JSON."""
        trajectories = self._run_trajectory_forecast()
        fit_mean, fit_lower, fit_upper, fit_years = self._compute_historical_fit()
        self._print_backtest_table(fit_mean, fit_lower, fit_upper, fit_years)
        self._plot_results(
            trajectories, fit_mean, fit_lower, fit_upper, fit_years
        )
        self._export_report_json(trajectories)

    def _export_report_json(
        self,
        trajectories: Dict[str, tf.Tensor],
    ) -> str:
        """Export a self-contained JSON report from already-computed trajectories.

        Uses ``data`` and ``formatter`` from construction, and reads
        ``parameters_path`` from the model.

        Args:
            trajectories: Dict from :meth:`_run_trajectory_forecast`.

        Returns:
            Path to the saved JSON file.
        """
        scale = self.model.amount_scale

        historical_table = self._formatter.format_historical(self._data)
        forecast_table = self._formatter.format_forecast(
            trajectories, self._forecast_years, scale
        )

        n_years = trajectories["total_assets"].shape[1]
        forecast_year_labels = [int(self._forecast_years[i]) for i in range(n_years)]

        report = {
            "generated_at": datetime.now().isoformat(),
            "company": self._data.company,
            "parameters_path": getattr(self.model, "parameters_path", None),
            "forecast_years": forecast_year_labels,
            "amount_scale": float(scale),
            "base_year": int(self.model.base_year),
            "n_monte_carlo_samples": int(self.model.trajectory_simulator.n_samples),
            "historical_table": historical_table,
            "forecast_table": forecast_table,
        }

        out_path = get_training_results_path("forecast_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nForecast report saved to: {out_path}")
        return out_path

    def _print_backtest_table(
        self,
        fit_mean: Dict[str, tf.Tensor],
        fit_lower: Optional[Dict[str, tf.Tensor]],
        fit_upper: Optional[Dict[str, tf.Tensor]],
        fit_years: tf.Tensor,
    ) -> None:
        """Print markdown table comparing actual vs predicted for test years."""
        model = self.model
        test_years = model.test_years
        if test_years == 0:
            return

        d = model.historical_data
        n_hist = len(d["sales"])
        is_mc = fit_lower is not None

        total_assets_hist = (
            d["nca"]
            + d["advance_payments_purchases"]
            + d["accounts_receivable"]
            + d["inventory"]
            + d["cash"]
            + d["ims"]
        )

        # (display_label, fit_key, hist_tensor)  —  None fit_key = section header
        backtest_items = [
            ("ASSETS", None, None),
            ("Non-Current Assets", "nca", d["nca"]),
            ("Adv Payments (Purchases)", "advance_payments_purchases", d["advance_payments_purchases"]),
            ("Accounts Receivable", "accounts_receivable", d["accounts_receivable"]),
            ("Inventory", "inventory", d["inventory"]),
            ("Cash", "cash", d["cash"]),
            ("Invest in Mkt Securities", "investment_in_market_securities", d["ims"]),
            ("**Total Assets**", "total_assets", total_assets_hist),
            ("", None, None),
            ("LIABILITIES", None, None),
            ("Accounts Payable", "accounts_payable", d["accounts_payable"]),
            ("Adv Payments (Sales)", "advance_payments_sales", d["advance_payments_sales"]),
            ("Effective ST Debt", "effective_st_debt", d["effective_st_debt"]),
            ("Non-Current Liabilities", "non_current_liabilities", d["non_current_liabilities"]),
            ("**Equity**", "equity", d["equity"]),
            ("", None, None),
            ("INCOME STATEMENT", None, None),
            ("COGS", "cogs", d["cogs"]),
            ("OpEx", "opex", d["opex"]),
            ("Depreciation", "depreciation", d["depreciation"]),
            ("Interest Payments", "interest_payment", d["interest_payment"]),
            ("ST Investment Returns", "ms_return", d["ms_return"]),
            ("Tax", "tax", d["tax"]),
            ("**Net Income**", "net_income", d["net_income"]),
            ("", None, None),
            ("CASH FLOW", None, None),
            ("Dividends", "dividends", d["dividends"]),
            ("Stock Buyback", "stock_buyback", d["stock_buyback"]),
        ]

        n_fit = len(fit_years)
        dollar = lambda v: f"${v:,.0f}"

        for ty in range(test_years):
            fit_idx = n_fit - test_years + ty
            hist_idx = n_hist - test_years + ty
            year_label = f"FY{int(fit_years[fit_idx])}"

            print(f"\nBACKTEST \u2014 {year_label} (Out-of-Sample)\n")

            if is_mc:
                n_data_cols = 5
                header = "| Line Item | Actual | Predicted (Mean) | Lower (2.5%) | Upper (97.5%) | Error % |"
                sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
            else:
                n_data_cols = 3
                header = "| Line Item | Actual | Predicted | Error % |"
                sep = "| --- | ---: | ---: | ---: |"

            print(header)
            print(sep)

            for label, fit_key, hist_tensor in backtest_items:
                if fit_key is None:
                    blank = " | ".join([""] * n_data_cols)
                    if label:
                        print(f"| **{label}** | {blank} |")
                    else:
                        print(f"| | {blank} |")
                    continue

                actual = float(hist_tensor[hist_idx])
                predicted = float(fit_mean[fit_key][fit_idx])

                if np.isnan(actual):
                    if is_mc:
                        lower_v = float(fit_lower[fit_key][fit_idx])
                        upper_v = float(fit_upper[fit_key][fit_idx])
                        print(
                            f"| {label} | N/A | {dollar(predicted)}"
                            f" | {dollar(lower_v)} | {dollar(upper_v)} | N/A |"
                        )
                    else:
                        print(f"| {label} | N/A | {dollar(predicted)} | N/A |")
                    continue

                if abs(actual) > 1.0:
                    error_pct = (predicted - actual) / abs(actual) * 100
                    error_str = f"{error_pct:+.1f}%"
                else:
                    error_str = "N/A"

                if is_mc:
                    lower_v = float(fit_lower[fit_key][fit_idx])
                    upper_v = float(fit_upper[fit_key][fit_idx])
                    print(
                        f"| {label} | {dollar(actual)} | {dollar(predicted)}"
                        f" | {dollar(lower_v)} | {dollar(upper_v)} | {error_str} |"
                    )
                else:
                    print(
                        f"| {label} | {dollar(actual)}"
                        f" | {dollar(predicted)} | {error_str} |"
                    )

    def _run_trajectory_forecast(self) -> Dict[str, tf.Tensor]:
        """Run the model's trajectory simulator."""
        model = self.model
        return model.trajectory_simulator.run(
            model,
            model.initial_state,
            self._sales_forecast,
            self._cum_inf_forecast,
            self._forecast_years,
        )

    def _compute_historical_fit(
        self,
    ) -> Tuple[
        Dict[str, tf.Tensor],
        Optional[Dict[str, tf.Tensor]],
        Optional[Dict[str, tf.Tensor]],
        tf.Tensor,
    ]:
        """One-step-ahead predictions on historical data.

        When the model's trajectory simulator is a
        :class:`MonteCarloSimulator` with ``n_samples > 1`` *and* the
        OpEx module is a :class:`BayesianOpEx`, each transition is
        evaluated on a tiled batched state with stochastic OpEx, and
        the returned tensors summarize the MC draws as a mean with
        2.5%/97.5% credible bounds.  Otherwise, the fit is a
        single-sample deterministic point estimate (``use_mean_opex=True``)
        and the lower/upper returns are ``None``.

        Returns:
            Tuple ``(fit_mean, fit_lower, fit_upper, fit_years)``.
            *fit_mean* is a dict of USD tensors (one per historical
            transition).  *fit_lower* and *fit_upper* are the matching
            2.5%/97.5% bounds when MC is active, or ``None`` when the
            fit is deterministic.  *fit_years* is a 1-D year tensor.
        """
        model = self.model
        s = model.scaled_data
        d = model.historical_data
        scale = model.amount_scale
        cum_inf_hist = tf.math.cumprod(1 + d["inflation"])
        n_hist = len(d["sales"])
        f64 = lambda v: tf.constant(float(v), dtype=tf.float64)

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

        # Decide whether to run MC.  MC only makes sense if the
        # trajectory simulator is configured with multiple samples AND
        # the OpEx module is stochastic (BayesianOpEx); otherwise all
        # samples would collapse to the same value.
        n_mc_samples = int(model.trajectory_simulator.n_samples)
        use_mc = n_mc_samples > 1 and isinstance(
            model.opex_module, BayesianOpEx
        )

        fit_mean = {k: [] for k in hist_fit_keys}
        fit_lower = {k: [] for k in hist_fit_keys} if use_mc else None
        fit_upper = {k: [] for k in hist_fit_keys} if use_mc else None
        fit_years: list = []

        if use_mc:
            # Pre-sample the Bayesian OpEx posterior + aleatoric noise
            # once, covering every historical transition.  Indexed by
            # ``year - start_year`` inside ``compute_mc_step``.
            model.opex_module.prepare_mc(
                n_samples=n_mc_samples,
                n_years=n_hist,
                start_year=float(model.base_year + 1),
            )
            diag_index = {k: i for i, k in enumerate(DIAGNOSTIC_KEYS)}

        for t in range(n_hist - 1):
            state_dict = model.build_state_from_index(t)
            sales_t_val = f64(s["sales"][t + 1])
            year_val = f64(float(model.base_year + t + 1))
            cum_inf_val = f64(cum_inf_hist[t + 1])

            if not use_mc:
                inputs_t = {
                    "sales_t": sales_t_val,
                    "year": year_val,
                    "cum_inflation": cum_inf_val,
                }
                pred = model.forecast_step(
                    state_dict,
                    inputs_t,
                    use_mean_opex=True,
                )
                for key in hist_fit_keys:
                    if key == "total_assets":
                        val = sum(
                            pred[k]
                            for k in [
                                "nca",
                                "advance_payments_purchases",
                                "accounts_receivable",
                                "inventory",
                                "cash",
                                "investment_in_market_securities",
                            ]
                        )
                    else:
                        val = pred[key]
                    fit_mean[key].append(float(val.numpy()) * scale)
            else:
                # Tile the historical scalar state to [n_mc_samples, 14]
                # and dispatch to the compiled batched forecast path
                # with stochastic OpEx.
                state_row = tf.stack(
                    [
                        tf.cast(state_dict[k], tf.float64)
                        for k in RECURRENT_KEYS
                    ]
                )
                state_batch = tf.broadcast_to(
                    state_row[tf.newaxis, :],
                    [n_mc_samples, state_row.shape[0]],
                )
                sales_batch = tf.fill([n_mc_samples], sales_t_val)
                _, diagnostics = model.forecast_step_compiled(
                    state_batch,
                    sales_batch,
                    year_val,
                    cum_inf_val,
                    False,  # use_mean_opex
                )
                diag_np = diagnostics.numpy() * scale
                total_assets_samples = sum(
                    diag_np[:, diag_index[k]]
                    for k in [
                        "nca",
                        "advance_payments_purchases",
                        "accounts_receivable",
                        "inventory",
                        "cash",
                        "investment_in_market_securities",
                    ]
                )
                for key in hist_fit_keys:
                    if key == "total_assets":
                        samples = total_assets_samples
                    else:
                        samples = diag_np[:, diag_index[key]]
                    fit_mean[key].append(float(np.mean(samples)))
                    fit_lower[key].append(float(np.percentile(samples, 2.5)))
                    fit_upper[key].append(float(np.percentile(samples, 97.5)))

            fit_years.append(model.base_year + t + 1)

        for k in fit_mean:
            fit_mean[k] = tf.constant(fit_mean[k], dtype=tf.float64)
        if use_mc:
            for k in fit_lower:
                fit_lower[k] = tf.constant(fit_lower[k], dtype=tf.float64)
                fit_upper[k] = tf.constant(fit_upper[k], dtype=tf.float64)

        fit_years_tensor = tf.constant(fit_years, dtype=tf.float64)
        return fit_mean, fit_lower, fit_upper, fit_years_tensor

    def _plot_results(
        self,
        trajectories: Dict[str, tf.Tensor],
        historical_fit: Dict[str, tf.Tensor],
        historical_fit_lower: Optional[Dict[str, tf.Tensor]],
        historical_fit_upper: Optional[Dict[str, tf.Tensor]],
        fit_years: tf.Tensor,
    ) -> None:
        """Plot historical actuals, model fit, and forecast trajectories."""
        model = self.model
        d = model.historical_data
        scale = model.amount_scale
        n_hist = len(d["sales"])

        hist_years = tf.cast(
            tf.range(model.base_year, model.base_year + n_hist),
            dtype=tf.float64,
        )

        total_assets_hist = (
            d["nca"]
            + d["advance_payments_purchases"]
            + d["accounts_receivable"]
            + d["inventory"]
            + d["cash"]
            + d["ims"]
        )
        hist_data = {
            "net_income": d["net_income"],
            "total_assets": total_assets_hist,
            "nca": d["nca"],
            "advance_payments_purchases": d["advance_payments_purchases"],
            "accounts_receivable": d["accounts_receivable"],
            "inventory": d["inventory"],
            "cash": d["cash"],
            "investment_in_market_securities": d["ims"],
            "accounts_payable": d["accounts_payable"],
            "advance_payments_sales": d["advance_payments_sales"],
            "non_current_liabilities": d["non_current_liabilities"],
            "equity": d["equity"],
            "depreciation": d["depreciation"],
            "cogs": d["cogs"],
            "opex": d["opex"],
            "tax": d["tax"],
            "ms_return": d["ms_return"],
            "interest_payment": d["interest_payment"],
            "dividends": d["dividends"],
            "stock_buyback": d["stock_buyback"],
            "effective_st_debt": d["effective_st_debt"],
        }

        plot_historical_and_forecast(
            historical_years=hist_years,
            forecast_years=self._forecast_years,
            historical_data=hist_data,
            forecast_trajectories=trajectories,
            amount_scale=scale,
            sales_hist_usd=d["sales"],
            sales_forecast_usd=self._sales_forecast_usd,
            historical_fit=historical_fit,
            historical_fit_lower=historical_fit_lower,
            historical_fit_upper=historical_fit_upper,
            historical_fit_years=fit_years,
            show_plot=self._show_plot,
        )
