"""Top-level orchestration for training and forecast runs.

Provides :class:`ForecastPipeline`, a model-agnostic orchestrator that
wires together data preparation, model training, Monte Carlo forecasting,
and result plotting.  All financial logic is delegated to the model and
helper modules; this class only handles orchestration and data plumbing.

Dependency flow::

    pipeline  ->  models/base          (model interface)
              ->  training/*_trainer   (parameter optimisation)
              ->  inference/forecast   (Monte Carlo simulation)
              ->  inference/plotting   (visualisation)

Example::

    from financial_forecast.data.loader import HistoricalDataLoader

    data = HistoricalDataLoader("aapl", include_inflation=True)
    ForecastPipeline(
        model=TrainableFinancialModel(opex_module=BayesianOpEx()),
        trainers=[PolicyTrainer(), StructuralTrainer()],
        financial_statements=data.financial_statements,
        inflation=data.inflation,
    ).run()
"""

from typing import Dict, List, Mapping, Optional, Sequence

import tensorflow as tf

from financial_forecast.inference.plotting import plot_historical_and_forecast
from financial_forecast.training.base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# Module-level helpers (stateless, used by the pipeline internally)
# ---------------------------------------------------------------------------


def _as_float64_constant(value: float) -> tf.Tensor:
    """Create a TensorFlow float64 scalar constant.

    Args:
        value: The scalar value to wrap.

    Returns:
        A rank-0 ``tf.Tensor`` with dtype ``float64``.
    """
    return tf.constant(value, dtype=tf.float64)


def _build_state_from_index(
    index: int,
    nca: tf.Tensor,
    advance_payments_purchases: tf.Tensor,
    accounts_receivable: tf.Tensor,
    inventory: tf.Tensor,
    cash: tf.Tensor,
    investment_in_market_securities: tf.Tensor,
    accounts_payable: tf.Tensor,
    advance_payments_sales: tf.Tensor,
    effective_st_debt: tf.Tensor,
    current_lt_debt: tf.Tensor,
    non_current_liabilities: tf.Tensor,
    equity: tf.Tensor,
    net_income: tf.Tensor,
    dividends: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    """Build a model state dictionary for a specific historical index.

    Each balance-sheet item is extracted at *index* from the corresponding
    historical tensor and wrapped in a ``tf.float64`` scalar constant.

    Args:
        index: Position along the time axis to slice from each tensor.
        nca: Non-current assets tensor.
        advance_payments_purchases: Advance payments (purchases) tensor.
        accounts_receivable: Accounts receivable tensor.
        inventory: Inventory tensor.
        cash: Cash tensor.
        investment_in_market_securities: Market-securities investment tensor.
        accounts_payable: Accounts payable tensor.
        advance_payments_sales: Advance payments (sales) tensor.
        effective_st_debt: Effective short-term debt tensor.
        current_lt_debt: Current portion of long-term debt tensor.
        non_current_liabilities: Non-current liabilities tensor.
        equity: Stockholders' equity tensor.
        net_income: Net income tensor.
        dividends: Dividends tensor.

    Returns:
        Dictionary mapping state-variable names to scalar ``tf.Tensor``
        values at the requested *index*.
    """
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


_REQUIRED_KEYS = {
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
    "current_lt_debt",
    "interest_payment",
    "ms_return",
}

_SUPPLEMENTAL_KEYS = {
    "effective_st_debt",
}


# ---------------------------------------------------------------------------
# ForecastPipeline
# ---------------------------------------------------------------------------


class ForecastPipeline:
    """Model-agnostic orchestrator for training, forecasting, and plotting.

    Mirrors the ``StockPredictor`` pattern: accepts pre-built components
    via dependency injection and exposes a single :meth:`run` entry point.

    If ``sales_forecast_usd`` or ``inflation_forecast`` are not provided,
    the pipeline auto-generates them from the historical data:

    - **Sales**: linear extrapolation of the average year-over-year delta.
    - **Inflation**: last observed rate for year 1, then 3 % constant.

    Args:
        model: A trained or untrained financial model (e.g.
            :class:`TrainableFinancialModel`).
        trainers: Ordered list of :class:`BaseTrainer` instances.  Each
            trainer's :meth:`train` is called in sequence during the
            training phase.
        financial_statements: Mapping of historical financial series in
            USD (e.g. from ``data.financial_statements``).
        inflation: Optional 1-D tensor of annual inflation rates.  If
            ``None``, inflation effects are excluded.
        forecast_years: Number of years to forecast.  Defaults to 10.
        test_years: Number of historical years held out for testing.
            Defaults to 1.
        monte_carlo_samples: Number of Monte Carlo simulation paths.
        sales_forecast_usd: Optional explicit sales forecast (USD).  If
            ``None``, auto-generated from historical trend.
        inflation_forecast: Optional explicit inflation forecast.  If
            ``None``, auto-generated from last observed rate + 3 %.
        parameters_save_path: Path for saving/loading trained parameters.
        use_trained_parameters: If ``True``, load parameters from disk
            instead of training.
    """

    def __init__(
        self,
        model,
        trainers: List[BaseTrainer],
        financial_statements: Mapping[str, tf.Tensor],
        inflation: Optional[tf.Tensor] = None,
        forecast_years: int = 10,
        test_years: int = 1,
        monte_carlo_samples: int = 1000,
        sales_forecast_usd: Optional[tf.Tensor] = None,
        inflation_forecast: Optional[tf.Tensor] = None,
        parameters_save_path: str = "trained_parameters.npz",
        use_trained_parameters: bool = False,
    ):
        self.model = model
        self.trainers = trainers
        self.forecast_years = forecast_years
        self.test_years = test_years
        self.monte_carlo_samples = monte_carlo_samples
        self.parameters_save_path = parameters_save_path
        self.use_trained_parameters = use_trained_parameters
        self._inflation = inflation

        # Store raw data; will be validated and scaled in _prepare_data()
        self._raw = dict(financial_statements)
        self._sales_forecast_usd = sales_forecast_usd
        self._inflation_forecast = inflation_forecast

        # Populated by _prepare_data()
        self._d: Dict[str, tf.Tensor] = {}  # raw USD arrays
        self._s: Dict[str, tf.Tensor] = {}  # scaled (billions) arrays

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full pipeline: prepare, train, forecast, plot."""
        self._prepare_data()
        self._train_or_load()
        trajectories = self._run_monte_carlo_forecast()
        historical_fit, fit_years = self._compute_historical_fit()
        self._plot_results(trajectories, historical_fit, fit_years)

    # ------------------------------------------------------------------
    # Private pipeline stages
    # ------------------------------------------------------------------

    def _prepare_data(self) -> None:
        """Validate historical data, build forecast drivers, scale to billions."""
        missing = sorted(_REQUIRED_KEYS.difference(self._raw.keys()))
        if missing:
            raise ValueError(
                "historical_data is missing required keys: " + ", ".join(missing)
            )

        d = self._d
        for k in _REQUIRED_KEYS:
            d[k] = self._raw[k]

        # Supplementary quantities
        d["inflation"] = (
            self._inflation
            if self._inflation is not None
            else tf.zeros(len(d["sales"]), dtype=tf.float64)
        )
        d["effective_st_debt"] = (
            d["current_liabilities"]
            - d["accounts_payable"]
            - d["advance_payments_sales"]
            - d["current_lt_debt"]
        )

        # Scale to billions for training stability
        mean_sales = float(tf.reduce_mean(d["sales"]))
        scale = 10 ** int(tf.math.floor(tf.math.log(mean_sales) / tf.math.log(10.0)))
        for key in _REQUIRED_KEYS | _SUPPLEMENTAL_KEYS:
            self._s[key] = d[key] / scale

        # Configure model with data-derived settings
        self.model.prepare_for_training(
            self._raw["years"],
            scale,
        )

        self.model.opex_module.prepare_for_training(
            scale,
            self._s["sales"],
            self._s["opex"],
            self._d["inflation"],
        )

        # Let the tax module scale its stored data and build year-keyed adjustments
        t = len(d["sales"]) - self.test_years
        training_years = tf.cast(
            tf.range(self.model.base_year, self.model.base_year + t),
            dtype=tf.float64,
        )
        self.model.tax_module.prepare_for_training(scale, training_years)

        # Build forecast drivers (auto-generate if not supplied)
        if self._sales_forecast_usd is None:
            self._sales_forecast_usd = self._build_default_sales_forecast()
        if self._inflation_forecast is None:
            self._inflation_forecast = self._build_default_inflation_forecast()

    def _build_default_sales_forecast(self) -> tf.Tensor:
        """Linear extrapolation of the historical average annual delta.

        Returns:
            1-D float64 tensor of length ``self.forecast_years``.
        """
        sales = self._raw["sales"]
        avg_growth = tf.reduce_mean(sales[1:] - sales[:-1])
        return tf.constant(
            [float(sales[-1] + avg_growth * i) for i in range(self.forecast_years)],
            dtype=tf.float64,
        )

    def _build_default_inflation_forecast(self) -> tf.Tensor:
        """Last observed inflation rate for year 1, then 3 % constant.

        Returns:
            1-D float64 tensor of length ``self.forecast_years``.
        """
        inflation = self._d["inflation"]
        if tf.reduce_all(inflation == 0.0):
            return tf.zeros([self.forecast_years], dtype=tf.float64)
        return tf.concat(
            [
                inflation[-1:],
                tf.fill(
                    [self.forecast_years - 1],
                    tf.constant(0.03, dtype=tf.float64),
                ),
            ],
            axis=0,
        )

    def _train_or_load(self) -> None:
        """Train the model via each trainer in order, or load saved parameters."""
        model = self.model
        s = self._s
        d = self._d
        inflation = d["inflation"]

        if self.use_trained_parameters:
            model.load_parameters(self.parameters_save_path)
            return

        # Training window: all years except the last `test_years` held out
        t = len(s["sales"]) - self.test_years  # number of training years
        train_years = tf.cast(
            tf.range(model.base_year, model.base_year + t), dtype=tf.float64
        )

        # --- Train policy + Bayesian OpEx (first trainer) ---
        self.trainers[0].train(
            model,
            historical_sales=s["sales"][:t],
            historical_purchases=s["purchases"][:t],
            historical_cogs=s["cogs"][:t],
            historical_nca=s["nca"][:t],
            historical_depreciation=s["depreciation"][:t],
            historical_adv_pay_sales=s["advance_payments_sales"][:t],
            historical_adv_pay_purch=s["advance_payments_purchases"][:t],
            historical_ar=s["accounts_receivable"][:t],
            historical_ap=s["accounts_payable"][:t],
            historical_inventory=s["inventory"][:t],
            historical_cash=s["cash"][:t],
            historical_ims=s["ims"][:t],
            historical_net_income=s["net_income"][:t],
            historical_dividends=s["dividends"][:t],
            historical_stock_buyback=s["stock_buyback"][:t],
            historical_opex=s["opex"][:t],
            historical_tax=s["tax"][:t],
            historical_eff_st_debt=s["effective_st_debt"][:t],
            historical_inflation=inflation[:t],
            historical_years=train_years,
            show_plot=False,
            loss_scale_mode="std",
        )

        # --- Train structural parameters (second trainer) ---
        self.trainers[1].train(
            model,
            historical_sales=s["sales"][:t],
            historical_nca=s["nca"][:t],
            historical_adv_pay_sales=s["advance_payments_sales"][:t],
            historical_adv_pay_purch=s["advance_payments_purchases"][:t],
            historical_ar=s["accounts_receivable"][:t],
            historical_ap=s["accounts_payable"][:t],
            historical_inventory=s["inventory"][:t],
            historical_cash=s["cash"][:t],
            historical_ims=s["ims"][:t],
            historical_net_income=s["net_income"][:t],
            historical_dividends=s["dividends"][:t],
            historical_stock_buyback=s["stock_buyback"][:t],
            historical_opex=s["opex"][:t],
            historical_tax=s["tax"][:t],
            historical_effective_st_debt=s["effective_st_debt"][:t],
            historical_current_lt_debt=s["current_lt_debt"][:t],
            historical_non_current_liabilities=s["non_current_liabilities"][:t],
            historical_interest_payment=s["interest_payment"][:t],
            historical_ms_return=s["ms_return"][:t],
            historical_equity=s["equity"][:t],
            historical_inflation=inflation[:t],
            historical_years=train_years,
            loss_scale_mode="std",
        )
        model.save_parameters(self.parameters_save_path)

    def _run_monte_carlo_forecast(self) -> Dict[str, tf.Tensor]:
        """Build initial state and run Monte Carlo simulation.

        Returns:
            Dict mapping metric names to ``[n_samples, n_years]`` tensors.
        """
        model = self.model
        s = self._s
        scale = model.amount_scale

        # Initial state: second-to-last historical year (FY2024)
        state = _build_state_from_index(
            index=-2,
            nca=s["nca"],
            advance_payments_purchases=s["advance_payments_purchases"],
            accounts_receivable=s["accounts_receivable"],
            inventory=s["inventory"],
            cash=s["cash"],
            investment_in_market_securities=s["ims"],
            accounts_payable=s["accounts_payable"],
            advance_payments_sales=s["advance_payments_sales"],
            effective_st_debt=s["effective_st_debt"],
            current_lt_debt=s["current_lt_debt"],
            non_current_liabilities=s["non_current_liabilities"],
            equity=s["equity"],
            net_income=s["net_income"],
            dividends=s["dividends"],
        )

        # Forecast drivers
        sales_forecast_usd = tf.constant(self._sales_forecast_usd, dtype=tf.float64)
        if sales_forecast_usd.ndim != 1 or tf.size(sales_forecast_usd) == 0:
            raise ValueError("sales_forecast_usd must be a non-empty 1D sequence")
        sales_forecast = sales_forecast_usd / scale
        n_fc = int(tf.size(sales_forecast))

        n_hist = len(self._d["sales"])
        last_hist_year = model.base_year + n_hist - 1
        forecast_years = tf.cast(
            tf.range(last_hist_year, last_hist_year + n_fc), dtype=tf.float64
        )

        # Continue inflation compounding from historical baseline
        inflation = self._d["inflation"]
        cum_inf_hist = tf.math.cumprod(1 + inflation)
        last_cum_inf = cum_inf_hist[-2]

        inf_fc = tf.constant(self._inflation_forecast, dtype=tf.float64)
        if inf_fc.ndim != 1 or tf.size(inf_fc) != n_fc:
            raise ValueError(
                "inflation_forecast must be a 1D sequence with the same "
                "length as sales_forecast_usd"
            )
        if self._inflation is None:
            inf_fc = tf.zeros_like(inf_fc)
        cum_inf_forecast = last_cum_inf * tf.math.cumprod(1 + inf_fc)

        # Store for use in _plot_results
        self._forecast_years = forecast_years
        self._sales_forecast_scaled = sales_forecast

        return model.trajectory_simulator.run(
            model,
            state,
            sales_forecast,
            cum_inf_forecast,
            forecast_years,
            n_samples=self.monte_carlo_samples,
        )

    def _compute_historical_fit(self):
        """Run one-step-ahead predictions on the training window.

        Returns:
            Tuple of ``(historical_fit, historical_fit_years)`` where
            *historical_fit* maps metric names to USD tensors and
            *historical_fit_years* is a 1-D year tensor.
        """
        model = self.model
        s = self._s
        d = self._d
        scale = model.amount_scale
        inflation = d["inflation"]
        cum_inf_hist = tf.math.cumprod(1 + inflation)

        n_hist = len(d["sales"])
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
            state_t = _build_state_from_index(
                index=t,
                nca=s["nca"],
                advance_payments_purchases=s["advance_payments_purchases"],
                accounts_receivable=s["accounts_receivable"],
                inventory=s["inventory"],
                cash=s["cash"],
                investment_in_market_securities=s["ims"],
                accounts_payable=s["accounts_payable"],
                advance_payments_sales=s["advance_payments_sales"],
                effective_st_debt=s["effective_st_debt"],
                current_lt_debt=s["current_lt_debt"],
                non_current_liabilities=s["non_current_liabilities"],
                equity=s["equity"],
                net_income=s["net_income"],
                dividends=s["dividends"],
            )
            inputs_t = {
                "sales_t": _as_float64_constant(s["sales"][t + 1]),
                "year": _as_float64_constant(float(model.base_year + t + 1)),
                "cum_inflation": _as_float64_constant(cum_inf_hist[t + 1]),
            }
            pred = model.forecast_step(state_t, inputs_t, use_mean_opex=True)

            # Collect predicted values (convert back to USD)
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

        # Convert to tensors
        for k in fit:
            fit[k] = tf.constant(fit[k], dtype=tf.float64)
        fit_years = tf.constant(fit_years, dtype=tf.float64)
        return fit, fit_years

    def _plot_results(self, trajectories, historical_fit, fit_years) -> None:
        """Plot historical actuals, model fit, and forecast trajectories."""
        model = self.model
        d = self._d
        scale = model.amount_scale
        n_hist = len(d["sales"])

        hist_years = tf.cast(
            tf.range(model.base_year, model.base_year + n_hist), dtype=tf.float64
        )
        n_fc = len(self._sales_forecast_scaled)
        fc_start = model.base_year + n_hist - 1
        plot_fc_years = tf.cast(tf.range(fc_start, fc_start + n_fc), dtype=tf.float64)

        # Build historical data dict (USD, not scaled)
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
            forecast_years=plot_fc_years,
            historical_data=hist_data,
            forecast_trajectories=trajectories,
            amount_scale=scale,
            sales_hist_usd=d["sales"],
            sales_forecast_usd=self._sales_forecast_scaled * scale,
            historical_fit=historical_fit,
            historical_fit_years=fit_years,
            show_plot=False,
        )
