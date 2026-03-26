"""Simulation and plotting orchestrator for prepared financial models.

Takes a model that has already been prepared (and optionally trained),
generates forecast drivers via pluggable sales/inflation forecast models,
runs the trajectory simulator, computes one-step-ahead historical fit,
and plots results.

Example::

    model = BaseFinancialModel(...)
    model.prepare(data.financial_statements, data.inflation)
    ForecastPipeline(model, forecast_years=10).run()
"""

from typing import Dict

import tensorflow as tf

from financial_forecast.inference.plotting import plot_historical_and_forecast
from financial_forecast.inference.state_index import DIAGNOSTIC_KEYS
from financial_forecast.inference.forecast_driver_models import (
    SalesForecastModel,
    InflationForecastModel,
)


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
        sales_forecast: An initialized sales forecast model (e.g.
            :class:`LinearSalesForecast`).
        inflation_forecast: An initialized inflation forecast model
            (e.g. :class:`ConstantInflationForecast`).
        show_plot: Whether to call ``plt.show()`` after saving plots.
    """

    def __init__(
        self,
        model,
        sales_forecast: SalesForecastModel,
        inflation_forecast: InflationForecastModel,
        show_plot: bool = False,
    ):
        self.model = model
        self._show_plot = show_plot

        d = model._d
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
        last_cum_inf = cum_inf_hist[-(model._test_years + 1)]
        self._cum_inf_forecast = last_cum_inf * tf.math.cumprod(
            1 + inflation_forecast.forecast
        )

    def run(self) -> None:
        """Execute: trajectory forecast, historical fit, plot."""
        trajectories = self._run_trajectory_forecast()
        historical_fit, fit_years = self._compute_historical_fit()
        self._plot_results(trajectories, historical_fit, fit_years)

    def _run_trajectory_forecast(self) -> Dict[str, tf.Tensor]:
        """Run the model's trajectory simulator."""
        model = self.model
        return model.trajectory_simulator.run(
            model,
            model._initial_state,
            self._sales_forecast,
            self._cum_inf_forecast,
            self._forecast_years,
        )

    def _compute_historical_fit(self):
        """One-step-ahead predictions on historical data.

        Returns:
            Tuple ``(fit_dict, fit_years)`` where *fit_dict* maps metric
            names to USD tensors and *fit_years* is a 1-D year tensor.
        """
        model = self.model
        s = model._s
        d = model._d
        scale = model.amount_scale
        inflation = d["inflation"]
        cum_inf_hist = tf.math.cumprod(1 + inflation)

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
        fit = {k: [] for k in hist_fit_keys}
        fit_years = []

        for t in range(n_hist - 1):
            state_t = model._build_state_from_index(t)
            inputs_t = {
                "sales_t": f64(s["sales"][t + 1]),
                "year": f64(float(model.base_year + t + 1)),
                "cum_inflation": f64(cum_inf_hist[t + 1]),
            }
            pred = model.forecast_step(
                state_t,
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
                fit[key].append(float(val.numpy()) * scale)
            fit_years.append(model.base_year + t + 1)

        for k in fit:
            fit[k] = tf.constant(fit[k], dtype=tf.float64)
        fit_years = tf.constant(fit_years, dtype=tf.float64)
        return fit, fit_years

    def _plot_results(
        self,
        trajectories,
        historical_fit,
        fit_years,
    ) -> None:
        """Plot historical actuals, model fit, and forecast trajectories."""
        model = self.model
        d = model._d
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
            historical_fit_years=fit_years,
            show_plot=self._show_plot,
        )
