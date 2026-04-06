"""Forecast engine modules: single-point and Monte Carlo.

Provides an abstract base class and two implementations:

- ``DeterministicSimulator``: one deterministic trajectory (no sampling).
- ``MonteCarloSimulator``: N stochastic trajectories via ``tf.while_loop``.

The model auto-selects the engine based on ``opex_module.is_stochastic``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import tensorflow as tf
import tensorflow_probability as tfp

from financial_forecast.inference.state_index import (
    initial_state_to_batched,
    DIAGNOSTIC_KEYS,
)
from financial_forecast.types import RecurrentState, validate_recurrent_state


class TrajectorySimulator(ABC):
    """Abstract base class for forecast engines.

    Provides shared reporting methods inherited by all engines.
    Subclasses must define :attr:`n_samples`.
    """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Number of Monte Carlo samples (1 for deterministic)."""

    @abstractmethod
    def run(
        self,
        model,
        initial_state: RecurrentState,
        sales_forecast: tf.Tensor,
        cum_inf_forecast: tf.Tensor,
        forecast_years: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Run the forecast and return trajectory tensors.

        Args:
            model: Trained ``TrainableFinancialModel``.
            initial_state: :class:`RecurrentState` dict at forecast start.
            sales_forecast: 1-D tensor of forecasted sales (scaled).
            cum_inf_forecast: 1-D tensor of cumulative inflation factors.
            forecast_years: 1-D tensor of calendar years.

        Returns:
            Dict mapping metric names to ``[n_samples, n_years]`` tensors.
        """

    def _summarize_trajectories(
        self,
        name: str,
        trajectories: tf.Tensor,
        amount_scale: float,
    ) -> None:
        """Print mean and 95% interval summary for one metric."""
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
                f"{idx + 1:<5} | {mean_usd:<15.2e} | "
                f"{lower_usd:<15.2e} | {upper_usd:<15.2e}"
            )

    def _print_forecast_report(
        self,
        trajectories: Dict[str, tf.Tensor],
        amount_scale: float,
        forecast_years: tf.Tensor,
        n_samples: int,
    ) -> None:
        """Print summary tables and balance sheet for forecast trajectories."""
        s = self._summarize_trajectories
        s("Net Income", trajectories["net_income"], amount_scale)
        s("Total Assets", trajectories["total_assets"], amount_scale)
        s("Assets: Non-current Assets", trajectories["nca"], amount_scale)
        s(
            "Assets: Advance Payments (Purchases)",
            trajectories["advance_payments_purchases"],
            amount_scale,
        )
        s(
            "Assets: Accounts Receivable",
            trajectories["accounts_receivable"],
            amount_scale,
        )
        s("Assets: Inventory", trajectories["inventory"], amount_scale)
        s("Assets: Cash", trajectories["cash"], amount_scale)
        s(
            "Assets: Investment in Market Securities",
            trajectories["investment_in_market_securities"],
            amount_scale,
        )
        s("Effective ST Debt", trajectories["effective_st_debt"], amount_scale)
        s(
            "Non-current Liabilities",
            trajectories["non_current_liabilities"],
            amount_scale,
        )
        s("Equity", trajectories["equity"], amount_scale)
        s("Accounts Payable", trajectories["accounts_payable"], amount_scale)
        s(
            "Advance Payments (Sales)",
            trajectories["advance_payments_sales"],
            amount_scale,
        )
        s("Depreciation", trajectories["depreciation"], amount_scale)
        s("COGS", trajectories["cogs"], amount_scale)
        s("OpEx", trajectories["opex"], amount_scale)
        s("Tax", trajectories["tax"], amount_scale)
        s("Return on Market Securities", trajectories["ms_return"], amount_scale)
        s("Interest Payment", trajectories["interest_payment"], amount_scale)
        s("Dividends", trajectories["dividends"], amount_scale)
        s("Stock Buyback", trajectories["stock_buyback"], amount_scale)
        s("Current Portion of LT debt", trajectories["current_lt_debt"], amount_scale)
        s("New Long-Term Loan", trajectories["new_long_term_loan"], amount_scale)
        s("Equity Financing", trajectories["equity_financing"], amount_scale)

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
            (
                "  Current LT Debt",
                tf.reduce_mean(trajectories["current_lt_debt"], axis=0),
            ),
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

        header = "| Line Item | " + " | ".join(year_labels) + " |"
        sep = "| --- | " + " | ".join("---:" for _ in year_labels) + " |"
        blank_cells = " | ".join([""] * n_years)
        print("\nFORECAST BALANCE SHEET \u2014 Mean across Monte Carlo samples (USD)\n")
        print(header)
        print(sep)
        for label, data in rows:
            label = label.strip()
            if data is None:
                if label:
                    print(f"| **{label}** | {blank_cells} |")
                else:
                    print(f"| | {blank_cells} |")
                continue
            vals = " | ".join(f"${float(v) * scale:,.0f}" for v in data)
            if label.isupper():
                print(f"| **{label}** | {vals} |")
            else:
                print(f"| {label} | {vals} |")

        max_abs_check = float(tf.reduce_max(tf.abs(mean_check * scale)))
        print("\nBalance Sheet Identity Check " "(Assets = Liabilities + Equity):")
        print("  Max absolute mismatch across years (mean): " f"${max_abs_check:,.2f}")
        if max_abs_check < 1.0:
            print("  PASS: Balance sheet identity holds (mismatch < $1).")
        elif max_abs_check < 1000.0:
            print(
                "  PASS: Balance sheet identity holds within rounding "
                "(mismatch < $1,000)."
            )
        else:
            print("  WARNING: Balance sheet mismatch detected!")
            for idx in range(n_years):
                check_value = float(mean_check[idx]) * scale
                if abs(check_value) >= 1000.0:
                    print(
                        f"    {year_labels[idx]}: "
                        f"Assets - (Liab+Eq) = ${check_value:,.2f}"
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
        print(
            f"\n  Per-sample check "
            f"(across all {n_samples} samples x {n_years} years):"
        )
        print(f"    Max absolute mismatch:  ${max_abs_check_all:,.2f}")
        print(f"    Mean absolute mismatch: ${mean_abs_check_all:,.2f}")


class DeterministicSimulator(TrajectorySimulator):
    """Single deterministic trajectory — no sampling.

    Runs one trajectory with mean OpEx parameters.
    """

    @property
    def n_samples(self) -> int:
        return 1

    def run(
        self,
        model: tf.Module,
        initial_state: RecurrentState,
        sales_forecast: tf.Tensor,
        cum_inf_forecast: tf.Tensor,
        forecast_years: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        validate_recurrent_state(initial_state, "DeterministicSimulator.run")
        print("\n--- Running Single-Point Forecast (deterministic) ---")

        state0 = initial_state_to_batched(initial_state, 1)
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
                new_state, diagnostics = model.forecast_step_compiled(
                    state,
                    sales_t,
                    years_arr[step],
                    cum_inf_arr[step],
                    use_mean_opex=True,
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

        all_diag = _run_loop(state0, sales_arr, cum_inf_arr, years_arr)
        all_diag = tf.transpose(all_diag, perm=[1, 0, 2])

        trajectories: Dict[str, tf.Tensor] = {}
        for i, key in enumerate(DIAGNOSTIC_KEYS):
            trajectories[key] = all_diag[:, :, i]

        self._print_forecast_report(
            trajectories,
            model.amount_scale,
            forecast_years,
            1,
        )
        return trajectories


class MonteCarloSimulator(TrajectorySimulator):
    """Monte Carlo forecast with N stochastic trajectories.

    Uses ``tf.while_loop`` compiled via ``@tf.function``.
    All stochastic values are pre-sampled in eager mode.

    Args:
        n_samples: Number of Monte Carlo simulation paths.
    """

    def __init__(self, n_samples: int = 1000):
        self._n_samples = n_samples

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def run(
        self,
        model: tf.Module,
        initial_state: RecurrentState,
        sales_forecast: tf.Tensor,
        cum_inf_forecast: tf.Tensor,
        forecast_years: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        validate_recurrent_state(initial_state, "MonteCarloSimulator.run")
        n_samples = self.n_samples
        print(f"\n--- Running Monte Carlo Forecast ({n_samples} samples) ---")

        n_years = len(sales_forecast)
        start_year = float(forecast_years[0])
        model.opex_module.prepare_mc(n_samples, n_years, start_year)
        state0 = initial_state_to_batched(initial_state, n_samples)

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
                new_state, diagnostics = model.forecast_step_compiled(
                    state,
                    sales_t,
                    years_arr[step],
                    cum_inf_arr[step],
                    use_mean_opex=False,
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

        all_diag = _run_loop(state0, sales_arr, cum_inf_arr, years_arr)
        all_diag = tf.transpose(all_diag, perm=[1, 0, 2])

        trajectories: Dict[str, tf.Tensor] = {}
        for i, key in enumerate(DIAGNOSTIC_KEYS):
            trajectories[key] = all_diag[:, :, i]

        self._print_forecast_report(
            trajectories,
            model.amount_scale,
            forecast_years,
            n_samples,
        )
        return trajectories
