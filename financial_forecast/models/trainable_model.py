"""Trainable financial forecasting model with optimizable parameters.

This module provides :class:`TrainableFinancialModel`, a concrete
implementation of :class:`BaseFinancialModel` that uses ``tf.Variable``
values for all model parameters.  The trainable variables can be
optimized against historical financial data using the included training
methods:

- :meth:`train_simple_policies` fits policy parameters (asset growth,
  working-capital ratios, OpEx structure, shareholder returns) via
  closed-form ratio losses.
- :meth:`train_structural_parameters` fits structural parameters
  (interest rates, debt maturity, securities returns, financing mix) via
  full state-transition gradients.

The forecast logic itself is inherited from :class:`BaseFinancialModel`.
"""

from __future__ import annotations

from typing import Any, Dict

import tensorflow as tf

from financial_forecast.types import EconomicInputs, FinancialState
from financial_forecast.models.base import BaseFinancialModel


class TrainableFinancialModel(BaseFinancialModel):
    """Financial model with trainable ``tf.Variable`` parameters.

    Extends :class:`BaseFinancialModel` with gradient-based training
    capabilities.  Parameters are initialized to reasonable defaults
    and can then be refined using :meth:`train_simple_policies` and
    :meth:`train_structural_parameters`.

    Example:
        >>> model = TrainableFinancialModel()
        >>> model.train_simple_policies(historical_data, epochs=5000)
        >>> model.train_structural_parameters(historical_data, epochs=5000)
        >>> forecast = model.forecast_step(initial_state, inputs)
    """

    def _initialize_parameters(self) -> None:
        """Initialize all model parameters as trainable ``tf.Variable`` values.

        Parameters are grouped into policy parameters (optimized by
        :meth:`train_simple_policies`) and structural parameters
        (optimized by :meth:`train_structural_parameters`).
        """
        # --- Policy Parameters ---
        self.asset_growth = tf.Variable(0.0076, name="asset_growth", dtype=tf.float64)
        self.depreciation_rate = tf.Variable(0.055, name="depr_rate", dtype=tf.float64)
        self.advance_payments_sales_pct = tf.Variable(
            0.020614523, name="advance_payments_sales_pct", dtype=tf.float64
        )
        self.advance_payments_purchases_pct = tf.Variable(
            0.073525733,
            name="advance_payments_purchases_pct",
            dtype=tf.float64,
        )
        self.account_receivables_pct = tf.Variable(
            0.159111366, name="account_receivables_pct", dtype=tf.float64
        )
        self.account_payables_pct = tf.Variable(
            0.35014191, name="account_payables_pct", dtype=tf.float64
        )
        self.inventory_pct = tf.Variable(0.0165, name="inventory_pct", dtype=tf.float64)
        self.total_liquidity_pct = tf.Variable(
            0.16, name="total_liquidity_pct", dtype=tf.float64
        )
        self.cash_pct_of_liquidity = tf.Variable(
            0.487, name="cash_pct_of_liquidity", dtype=tf.float64
        )
        self.income_tax_pct = tf.Variable(
            0.147, name="income_tax_pct", dtype=tf.float64
        )
        self.variable_opex_pct = tf.Variable(
            0.222168147, name="variable_opex_pct", dtype=tf.float64
        )
        self.baseline_opex = tf.Variable(
            -30306718214.0, name="baseline_opex", dtype=tf.float64
        )
        self.dividend_payout_ratio_pct = tf.Variable(
            0.15, name="dividend_payout_ratio_pct", dtype=tf.float64
        )
        self.stock_buyback_pct = tf.Variable(
            7.5, name="stock_buyback_pct", dtype=tf.float64
        )

        # --- Structural Parameters ---
        self.avg_short_term_interest_pct = tf.Variable(
            0.6, name="avg_short_term_interest_pct", dtype=tf.float64
        )
        self.avg_long_term_interest_pct = tf.Variable(
            0.06, name="avg_long_term_interest_pct", dtype=tf.float64
        )
        self.avg_maturity_years = tf.Variable(
            3.0, name="avg_maturity_years", dtype=tf.float64
        )
        self.market_securities_return_pct = tf.Variable(
            0.05, name="market_securities_return_pct", dtype=tf.float64
        )
        self.equity_financing_pct = tf.Variable(
            0.15, name="equity_financing_pct", dtype=tf.float64
        )

    def train_simple_policies(
        self,
        historical_data: Dict[str, tf.Tensor],
        learning_rate: float = 0.0001,
        epochs: int = 5000,
    ) -> None:
        """Train policy parameters using historical data alignment.

        Minimizes the sum of squared residuals between observed financial
        ratios and model-implied ratios for each policy parameter.

        Args:
            historical_data: Dictionary of historical time-series tensors
                with keys such as ``"sales"``, ``"nca"``, ``"depr"``,
                ``"accounts_receivable"``, etc.
            learning_rate: Adam optimizer learning rate.
            epochs: Number of optimization iterations.
        """
        data = {
            k: tf.convert_to_tensor(v, dtype=tf.float64)
            for k, v in historical_data.items()
        }

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        vars_to_train = [
            self.asset_growth,
            self.depreciation_rate,
            self.advance_payments_sales_pct,
            self.advance_payments_purchases_pct,
            self.account_receivables_pct,
            self.account_payables_pct,
            self.inventory_pct,
            self.total_liquidity_pct,
            self.cash_pct_of_liquidity,
            self.income_tax_pct,
            self.variable_opex_pct,
            self.baseline_opex,
            self.dividend_payout_ratio_pct,
            self.stock_buyback_pct,
        ]

        print(f"Training policies on {len(data['sales'])} periods...")
        cum_inf = tf.math.cumprod(
            1 + data.get("inflation", tf.zeros_like(data["sales"]))
        )

        # Use @tf.function for the inner training step to enable graph-mode
        # optimization, reducing Python overhead and allowing TensorFlow to
        # fuse operations for faster execution.
        @tf.function
        def _policy_train_step(data_tensors, cum_inf_tensor):
            """Single policy training step traced as a TensorFlow graph."""
            with tf.GradientTape() as tape:
                losses = [
                    tf.reduce_mean(
                        tf.square(
                            (data_tensors["nca"][1:] - data_tensors["nca"][:-1])
                            - data_tensors["sales"][1:] * self.asset_growth
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["depr"][1:]
                            - data_tensors["nca"][:-1] * self.depreciation_rate
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["advance_payments_sales"][:-1]
                            - data_tensors["sales"][1:]
                            * self.advance_payments_sales_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["advance_payments_purchases"][:-1]
                            - data_tensors["purchases"][1:]
                            * self.advance_payments_purchases_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["accounts_receivable"]
                            - data_tensors["sales"] * self.account_receivables_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["accounts_payable"]
                            - data_tensors["purchases"] * self.account_payables_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["inventory"]
                            - data_tensors["sales"] * self.inventory_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            (
                                data_tensors["cash"]
                                + data_tensors["investment_in_market_securities"]
                            )
                            - data_tensors["sales"] * self.total_liquidity_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["cash"]
                            - (
                                data_tensors["cash"]
                                + data_tensors["investment_in_market_securities"]
                            )
                            * self.cash_pct_of_liquidity
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["tax"]
                            - data_tensors["net_income"] / (1 / self.income_tax_pct - 1)
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["opex"]
                            - (
                                self.baseline_opex * cum_inf_tensor
                                + data_tensors["sales"] * self.variable_opex_pct
                            )
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["dividends"][1:]
                            - data_tensors["net_income"][:-1]
                            * self.dividend_payout_ratio_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data_tensors["stock_buyback"]
                            - data_tensors["depr"] * self.stock_buyback_pct
                        )
                    ),
                ]
                total_loss = tf.add_n(losses)

            grads = tape.gradient(total_loss, vars_to_train)
            optimizer.apply_gradients(zip(grads, vars_to_train))
            return total_loss

        for i in range(epochs):
            total_loss = _policy_train_step(data, cum_inf)
            self._apply_constraints()

            if i % 1000 == 0:
                print(f"Epoch {i}: Policy Loss = {total_loss.numpy():.4e}")

    # NOTE: forecast_step (inherited from BaseFinancialModel) is not decorated
    # with @tf.function because it accepts/returns Python dataclasses and dicts,
    # which would cause excessive retracing. The training inner loops below are
    # the primary performance bottleneck and benefit most from graph compilation.

    def train_structural_parameters(
        self,
        historical_data: Dict[str, tf.Tensor],
        learning_rate: float = 0.0001,
        epochs: int = 5000,
    ) -> None:
        """Train structural parameters using state transition gradients.

        Runs single-step forecasts from each historical state and
        minimizes the squared error between predicted and actual values
        for net income, current/non-current liabilities, and equity.

        Args:
            historical_data: Dictionary of historical time-series tensors.
            learning_rate: Adam optimizer learning rate.
            epochs: Number of optimization iterations.
        """
        data = {
            k: tf.convert_to_tensor(v, dtype=tf.float64)
            for k, v in historical_data.items()
        }
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        vars_to_train = [
            self.avg_short_term_interest_pct,
            self.avg_long_term_interest_pct,
            self.avg_maturity_years,
            self.market_securities_return_pct,
            self.equity_financing_pct,
        ]

        num_transitions = len(data["sales"]) - 2
        cum_inf = tf.math.cumprod(
            1 + data.get("inflation", tf.zeros_like(data["sales"]))
        )

        print(f"Training structural parameters on {num_transitions} " f"transitions...")

        # NOTE: The structural training step is not wrapped with @tf.function
        # because it calls forecast_step in a Python loop with dict/dataclass
        # state, which would cause retracing on every iteration. The inner loop
        # over transitions is small (typically 5-6 years of historical data).
        for i in range(epochs):
            with tf.GradientTape() as tape:
                total_loss = 0.0
                for t in range(num_transitions):
                    prev_state = {
                        "nca": data["nca"][t],
                        "advance_payments_purchases": data[
                            "advance_payments_purchases"
                        ][t],
                        "accounts_receivable": data["accounts_receivable"][t],
                        "inventory": data["inventory"][t],
                        "cash": data["cash"][t],
                        "investment_in_market_securities": data[
                            "investment_in_market_securities"
                        ][t],
                        "accounts_payable": data["accounts_payable"][t],
                        "advance_payments_sales": data["advance_payments_sales"][t],
                        "current_liabilities": data["current_liabilities"][t],
                        "non_current_liabilities": data["non_current_liabilities"][t],
                        "equity": data["equity"][t],
                        "net_income": data["net_income"][t],
                    }
                    inputs = EconomicInputs(
                        sales_t=data["sales"][t + 1],
                        purchases_t=data["purchases"][t + 1],
                        sales_t_plus_1=data["sales"][t + 2],
                        purchases_t_plus_1=data["purchases"][t + 2],
                        cum_inflation=cum_inf[t + 1],
                    )
                    pred = self.forecast_step(prev_state, inputs)

                    losses = [
                        tf.square(pred.net_income - data["net_income"][t + 1]),
                        tf.square(
                            pred.current_liabilities
                            - data["current_liabilities"][t + 1]
                        ),
                        tf.square(
                            pred.non_current_liabilities
                            - data["non_current_liabilities"][t + 1]
                        ),
                        tf.square(pred.equity - data["equity"][t + 1]),
                    ]
                    total_loss += tf.add_n(losses) / 1e18

            grads = tape.gradient(total_loss, vars_to_train)
            optimizer.apply_gradients(zip(grads, vars_to_train))
            self._apply_constraints()

            if i % 1000 == 0:
                print(f"Epoch {i}: Structural Loss = {total_loss.numpy():.4e}")

    @tf.function
    def _apply_constraints(self) -> None:
        """Ensure parameters remain within economically valid ranges.

        Clips each trainable variable to its admissible domain after
        each gradient update.  For example, percentages are clipped to
        [0, 1], maturity must exceed 1 year, and growth rates must be
        non-negative.

        Decorated with ``@tf.function`` to compile the constraint
        assignments into a single graph operation, avoiding Python
        overhead on every training step.
        """
        # Use float64 constants to match variable dtypes (required by
        # @tf.function's strict type checking in graph mode).
        _zero = tf.constant(0.0, dtype=tf.float64)
        _one = tf.constant(1.0, dtype=tf.float64)
        _min_maturity = tf.constant(1.001, dtype=tf.float64)

        self.asset_growth.assign(tf.maximum(_zero, self.asset_growth))
        self.depreciation_rate.assign(tf.maximum(_zero, self.depreciation_rate))
        self.advance_payments_sales_pct.assign(
            tf.maximum(_zero, self.advance_payments_sales_pct)
        )
        self.advance_payments_purchases_pct.assign(
            tf.maximum(_zero, self.advance_payments_purchases_pct)
        )
        self.account_receivables_pct.assign(
            tf.maximum(_zero, self.account_receivables_pct)
        )
        self.account_payables_pct.assign(tf.maximum(_zero, self.account_payables_pct))
        self.inventory_pct.assign(tf.maximum(_zero, self.inventory_pct))
        self.total_liquidity_pct.assign(tf.maximum(_zero, self.total_liquidity_pct))
        self.cash_pct_of_liquidity.assign(
            tf.clip_by_value(self.cash_pct_of_liquidity, _zero, _one)
        )
        self.income_tax_pct.assign(tf.clip_by_value(self.income_tax_pct, _zero, _one))
        self.variable_opex_pct.assign(tf.maximum(_zero, self.variable_opex_pct))
        self.dividend_payout_ratio_pct.assign(
            tf.clip_by_value(self.dividend_payout_ratio_pct, _zero, _one)
        )
        self.stock_buyback_pct.assign(tf.maximum(_zero, self.stock_buyback_pct))
        self.avg_short_term_interest_pct.assign(
            tf.maximum(_zero, self.avg_short_term_interest_pct)
        )
        self.avg_long_term_interest_pct.assign(
            tf.maximum(_zero, self.avg_long_term_interest_pct)
        )
        self.avg_maturity_years.assign(
            tf.maximum(_min_maturity, self.avg_maturity_years)
        )
        self.market_securities_return_pct.assign(
            tf.maximum(_zero, self.market_securities_return_pct)
        )
        self.equity_financing_pct.assign(
            tf.clip_by_value(self.equity_financing_pct, _zero, _one)
        )
