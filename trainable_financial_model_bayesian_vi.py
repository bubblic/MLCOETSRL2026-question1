"""
Trainable Financial Model with Bayesian OpEx (Variational Inference).

This module implements a financial forecasting model that uses Bayesian inference
to handle uncertainty in Operating Expenses (OpEx). It combines deterministic
policy parameters with probabilistic OpEx parameters using TensorFlow Probability.

Software Engineering Principles:
- Probabilistic Programming: Integration of VI for uncertainty quantification.
- Modularity: Separation of deterministic and stochastic components.
- Consistency: Standardized field names across the model pipeline.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Union, Any, Optional

tfd = tfp.distributions
tfb = tfp.bijectors


@dataclass(frozen=True)
class FinancialState:
    """Represents the financial state of a company at a specific point in time."""

    nca: Any
    advance_payments_purchases: Any
    accounts_receivable: Any
    inventory: Any
    cash: Any
    investment_in_market_securities: Any
    accounts_payable: Any
    advance_payments_sales: Any
    current_liabilities: Any
    non_current_liabilities: Any
    equity: Any
    net_income: Any

    # Diagnostics
    liquidity_check: Any = 0.0
    balance_sheet_check: Any = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinancialState":
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass(frozen=True)
class EconomicInputs:
    """External drivers for a period."""

    sales_t: Any
    purchases_t: Any
    sales_t_plus_1: Any
    purchases_t_plus_1: Any
    cum_inflation: Any


class TrainableFinancialModel(tf.Module):
    """
    A trainable model with Bayesian OpEx parameters.
    """

    def __init__(self):
        super().__init__()
        self._initialize_deterministic_params()
        self._initialize_bayesian_params()
        self._initialize_structural_params()

    def _initialize_deterministic_params(self):
        """Initialize parameters trained via simple regression."""
        self.asset_growth = tf.Variable(0.0076, name="asset_growth", dtype=tf.float64)
        self.depreciation_rate = tf.Variable(0.055, name="depr_rate", dtype=tf.float64)
        self.advance_payments_sales_pct = tf.Variable(
            0.0206, name="adv_ps_pct", dtype=tf.float64
        )
        self.advance_payments_purchases_pct = tf.Variable(
            0.0735, name="adv_pp_pct", dtype=tf.float64
        )
        self.account_receivables_pct = tf.Variable(
            0.1591, name="ar_pct", dtype=tf.float64
        )
        self.account_payables_pct = tf.Variable(0.3501, name="ap_pct", dtype=tf.float64)
        self.inventory_pct = tf.Variable(0.0165, name="inv_pct", dtype=tf.float64)
        self.total_liquidity_pct = tf.Variable(0.16, name="tl_pct", dtype=tf.float64)
        self.cash_pct_of_liquidity = tf.Variable(
            0.487, name="cash_pct", dtype=tf.float64
        )
        self.income_tax_pct = tf.Variable(0.147, name="tax_pct", dtype=tf.float64)
        self.dividend_payout_ratio_pct = tf.Variable(
            0.15, name="div_pct", dtype=tf.float64
        )
        self.stock_buyback_pct = tf.Variable(7.5, name="bb_pct", dtype=tf.float64)

    def _initialize_bayesian_params(self):
        """Initialize Variational Inference parameters for OpEx."""
        self.q_var_opex_loc = tf.Variable(0.22, dtype=tf.float64, name="q_var_opex_loc")
        self.q_var_opex_scale = tfp.util.TransformedVariable(
            0.01, tfb.Softplus(), dtype=tf.float64, name="q_var_opex_scale"
        )
        self.q_base_opex_loc = tf.Variable(
            -3.0e10, dtype=tf.float64, name="q_base_opex_loc"
        )
        self.q_base_opex_scale = tfp.util.TransformedVariable(
            1.0e9, tfb.Softplus(), dtype=tf.float64, name="q_base_opex_scale"
        )
        self.noise_sigma = tfp.util.TransformedVariable(
            1.0e9, tfb.Softplus(), dtype=tf.float64, name="noise_sigma"
        )

    def _initialize_structural_params(self):
        """Initialize parameters trained via gradient descent through transitions."""
        self.avg_short_term_interest_pct = tf.Variable(
            0.6, name="st_int_pct", dtype=tf.float64
        )
        self.avg_long_term_interest_pct = tf.Variable(
            0.06, name="lt_int_pct", dtype=tf.float64
        )
        self.avg_maturity_years = tf.Variable(3.0, name="avg_m", dtype=tf.float64)
        self.market_securities_return_pct = tf.Variable(
            0.05, name="ms_ret_pct", dtype=tf.float64
        )
        self.equity_financing_pct = tf.Variable(
            0.15, name="equity_fin_pct", dtype=tf.float64
        )

    def sample_opex_params(self):
        """Sample parameters from the variational posterior."""
        q_var = tfd.Normal(self.q_var_opex_loc, self.q_var_opex_scale)
        q_base = tfd.Normal(self.q_base_opex_loc, self.q_base_opex_scale)
        return q_var.sample(), q_base.sample()

    def get_opex_kl_divergence(self):
        """Calculate KL divergence for OpEx parameters."""
        prior_var = tfd.Normal(tf.constant(0.20, dtype=tf.float64), 0.1)
        prior_base = tfd.Normal(tf.constant(-3.0e10, dtype=tf.float64), 1.0e10)
        q_var = tfd.Normal(self.q_var_opex_loc, self.q_var_opex_scale)
        q_base = tfd.Normal(self.q_base_opex_loc, self.q_base_opex_scale)
        return tfd.kl_divergence(q_var, prior_var) + tfd.kl_divergence(
            q_base, prior_base
        )

    def forecast_step(
        self,
        state: Union[FinancialState, Dict[str, Any]],
        inputs: EconomicInputs,
        use_mean_opex: bool = False,
    ) -> FinancialState:
        """Single period forecast with Bayesian components."""
        if isinstance(state, dict):
            state = FinancialState.from_dict(state)

        # 1. Assets
        assets = self._evolve_assets(state, inputs)

        # 2. Income Statement (Bayesian)
        income = self._calculate_income_statement(state, inputs, assets, use_mean_opex)

        # 3. Liquidity and Financing
        financing = self._manage_liquidity_and_financing(state, inputs, assets, income)

        # 4. Final Assembly
        return self._assemble_state(state, assets, income, financing, inputs)

    def _evolve_assets(
        self, state: FinancialState, inputs: EconomicInputs
    ) -> Dict[str, Any]:
        depr = state.nca * self.depreciation_rate
        capex = depr + (inputs.sales_t * self.asset_growth)
        nca_curr = state.nca - depr + capex
        adv_pp = inputs.purchases_t_plus_1 * self.advance_payments_purchases_pct
        ar = inputs.sales_t * self.account_receivables_pct
        inv = inputs.sales_t * self.inventory_pct
        total_liq = inputs.sales_t * self.total_liquidity_pct
        cash = total_liq * self.cash_pct_of_liquidity
        ims = total_liq * (1 - self.cash_pct_of_liquidity)

        return {
            "nca": nca_curr,
            "depreciation": depr,
            "capex": capex,
            "advance_payments_purchases": adv_pp,
            "accounts_receivable": ar,
            "inventory": inv,
            "cash": cash,
            "investment_in_market_securities": ims,
            "total_liquidity": total_liq,
        }

    def _calculate_income_statement(
        self,
        state: FinancialState,
        inputs: EconomicInputs,
        assets: Dict[str, Any],
        use_mean: bool,
    ) -> Dict[str, Any]:
        cogs = state.inventory + inputs.purchases_t - assets["inventory"]

        if use_mean:
            var_opex, base_opex, noise = self.q_var_opex_loc, self.q_base_opex_loc, 0.0
        else:
            var_opex, base_opex = self.sample_opex_params()
            noise = tfd.Normal(0.0, self.noise_sigma).sample()

        opex = (base_opex * inputs.cum_inflation + inputs.sales_t * var_opex) + noise
        ebitda = inputs.sales_t - cogs - opex

        # Debt Service (t-1)
        p_lt = state.non_current_liabilities / (self.avg_maturity_years - 1)
        i_lt = (
            self.avg_long_term_interest_pct
            * state.non_current_liabilities
            / (1 - 1 / self.avg_maturity_years)
        )
        p_st = state.current_liabilities - p_lt
        i_st = self.avg_short_term_interest_pct * p_st

        ms_ret = (
            state.investment_in_market_securities * self.market_securities_return_pct
        )
        ebt = ebitda - assets["depreciation"] - (i_st + i_lt) + ms_ret
        tax = ebt * self.income_tax_pct
        ni = ebt - tax

        return {
            "net_income": ni,
            "opex": opex,
            "ms_return": ms_ret,
            "principal_st": p_st,
            "principal_lt": p_lt,
            "interest_st": i_st,
            "interest_lt": i_lt,
        }

    def _manage_liquidity_and_financing(
        self,
        state: FinancialState,
        inputs: EconomicInputs,
        assets: Dict[str, Any],
        income: Dict[str, Any],
    ) -> Dict[str, Any]:
        adv_sales_curr = inputs.sales_t_plus_1 * self.advance_payments_sales_pct
        sales_in = (
            inputs.sales_t * (1 - self.account_receivables_pct)
            - state.advance_payments_sales
            + state.accounts_receivable
            + adv_sales_curr
        )
        purch_out = (
            inputs.purchases_t * (1 - self.account_payables_pct)
            - state.advance_payments_purchases
            + state.accounts_payable
            + assets["advance_payments_purchases"]
        )

        tax = (income["net_income"] / (1 - self.income_tax_pct)) * self.income_tax_pct
        op_nlb = sales_in - (purch_out + income["opex"] + tax)

        prev_total_liq = state.cash + state.investment_in_market_securities
        liq_def_st = (
            assets["total_liquidity"]
            - prev_total_liq
            - income["ms_return"]
            - op_nlb
            + income["principal_st"]
            + income["interest_st"]
        )
        new_st = tf.maximum(0.0, liq_def_st)

        divs = state.net_income * self.dividend_payout_ratio_pct
        bb = assets["depreciation"] * self.stock_buyback_pct
        liq_def_lt = (
            liq_def_st
            - new_st
            + assets["capex"]
            + income["principal_lt"]
            + income["interest_lt"]
            + divs
            + bb
        )

        lt_fin = tf.maximum(0.0, liq_def_lt)
        eq_fin = lt_fin * self.equity_financing_pct
        new_lt = lt_fin * (1 - self.equity_financing_pct)

        fin_nlb = (
            new_st
            + new_lt
            - income["principal_st"]
            - income["principal_lt"]
            - (income["interest_st"] + income["interest_lt"])
        )
        own_nlb = eq_fin - divs - bb
        total_nlb = op_nlb - assets["capex"] + fin_nlb + income["ms_return"] + own_nlb

        return {
            "new_st": new_st,
            "new_lt": new_lt,
            "equity_financing": eq_fin,
            "dividends": divs,
            "buybacks": bb,
            "adv_sales": adv_sales_curr,
            "total_nlb": total_nlb,
        }

    def _assemble_state(
        self,
        prev: FinancialState,
        assets: Dict[str, Any],
        income: Dict[str, Any],
        fin: Dict[str, Any],
        inputs: EconomicInputs,
    ) -> FinancialState:
        ap = inputs.purchases_t * self.account_payables_pct
        ncl = (
            (fin["new_lt"] + prev.non_current_liabilities)
            * (self.avg_maturity_years - 1)
            / self.avg_maturity_years
        )
        cl = fin["new_st"] + ncl / (self.avg_maturity_years - 1)
        equity = (
            prev.equity
            + fin["equity_financing"]
            + income["net_income"]
            - fin["dividends"]
            - fin["buybacks"]
        )

        total_assets = (
            assets["nca"]
            + assets["advance_payments_purchases"]
            + assets["accounts_receivable"]
            + assets["inventory"]
            + assets["cash"]
            + assets["investment_in_market_securities"]
        )
        total_liab_eq = ap + fin["adv_sales"] + cl + ncl + equity
        liq_check = (
            (prev.cash + prev.investment_in_market_securities)
            + fin["total_nlb"]
            - assets["total_liquidity"]
        )

        return FinancialState(
            nca=assets["nca"],
            advance_payments_purchases=assets["advance_payments_purchases"],
            accounts_receivable=assets["accounts_receivable"],
            inventory=assets["inventory"],
            cash=assets["cash"],
            investment_in_market_securities=assets["investment_in_market_securities"],
            accounts_payable=ap,
            advance_payments_sales=fin["adv_sales"],
            current_liabilities=cl,
            non_current_liabilities=ncl,
            equity=equity,
            net_income=income["net_income"],
            liquidity_check=liq_check,
            balance_sheet_check=total_assets - total_liab_eq,
        )

    def train_simple_policies(
        self, historical_data: Dict[str, np.ndarray], learning_rate=0.0001, epochs=5000
    ):
        """Train deterministic and Bayesian OpEx parameters."""
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
            self.dividend_payout_ratio_pct,
            self.stock_buyback_pct,
            self.q_var_opex_loc,
            self.q_var_opex_scale.trainable_variables[0],
            self.q_base_opex_loc,
            self.q_base_opex_scale.trainable_variables[0],
            self.noise_sigma.trainable_variables[0],
        ]

        print(f"Training parameters on {len(data['sales'])} periods...")
        cum_inf = tf.math.cumprod(
            1 + data.get("inflation", tf.zeros_like(data["sales"]))
        )

        for i in range(epochs):
            with tf.GradientTape() as tape:
                # Deterministic Losses
                losses = [
                    tf.reduce_mean(
                        tf.square(
                            (data["nca"][1:] - data["nca"][:-1])
                            - data["sales"][1:] * self.asset_growth
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data["depr"][1:] - data["nca"][:-1] * self.depreciation_rate
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data["advance_payments_sales"][:-1]
                            - data["sales"][1:] * self.advance_payments_sales_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data["advance_payments_purchases"][:-1]
                            - data["purchases"][1:]
                            * self.advance_payments_purchases_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data["accounts_receivable"]
                            - data["sales"] * self.account_receivables_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data["accounts_payable"]
                            - data["purchases"] * self.account_payables_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data["inventory"] - data["sales"] * self.inventory_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            (data["cash"] + data["investment_in_market_securities"])
                            - data["sales"] * self.total_liquidity_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data["cash"]
                            - (data["cash"] + data["investment_in_market_securities"])
                            * self.cash_pct_of_liquidity
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data["tax"] - data["net_income"] * self.income_tax_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data["dividends"][1:]
                            - data["net_income"][:-1] * self.dividend_payout_ratio_pct
                        )
                    ),
                    tf.reduce_mean(
                        tf.square(
                            data["stock_buyback"]
                            - data["depr"] * self.stock_buyback_pct
                        )
                    ),
                ]

                # Bayesian OpEx Loss
                v_samp, b_samp = self.sample_opex_params()
                pred_opex = (b_samp * cum_inf) + (v_samp * data["sales"])
                residuals_scaled = (data["opex"] - pred_opex) / 1e10
                sigma_scaled = self.noise_sigma / 1e10
                nll = -tf.reduce_sum(
                    tfd.Normal(0.0, sigma_scaled).log_prob(residuals_scaled)
                )
                losses.append(nll + self.get_opex_kl_divergence())

                total_loss = tf.add_n(losses)

            grads = tape.gradient(total_loss, vars_to_train)
            optimizer.apply_gradients(zip(grads, vars_to_train))
            self._apply_constraints()

            if i % 1000 == 0:
                print(
                    f"Epoch {i}: Total Loss = {total_loss.numpy():.4e} | Noise Sigma = {self.noise_sigma.numpy():.2e}"
                )

    def train_structural_parameters(
        self, historical_data: Dict[str, np.ndarray], learning_rate=0.0001, epochs=5000
    ):
        """Train structural parameters through state transitions."""
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

        for i in range(epochs):
            with tf.GradientTape() as tape:
                total_loss = 0.0
                for t in range(num_transitions):
                    prev_state = {
                        k: data[k][t]
                        for k in [
                            "nca",
                            "advance_payments_purchases",
                            "accounts_receivable",
                            "inventory",
                            "cash",
                            "investment_in_market_securities",
                            "accounts_payable",
                            "advance_payments_sales",
                            "current_liabilities",
                            "non_current_liabilities",
                            "equity",
                            "net_income",
                        ]
                    }
                    inputs = EconomicInputs(
                        sales_t=data["sales"][t + 1],
                        purchases_t=data["purchases"][t + 1],
                        sales_t_plus_1=data["sales"][t + 2],
                        purchases_t_plus_1=data["purchases"][t + 2],
                        cum_inflation=cum_inf[t + 1],
                    )
                    pred = self.forecast_step(prev_state, inputs, use_mean_opex=True)

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

    def _apply_constraints(self):
        """Clip variables to valid economic ranges."""
        self.asset_growth.assign(tf.maximum(0.0, self.asset_growth))
        self.depreciation_rate.assign(tf.maximum(0.0, self.depreciation_rate))
        self.advance_payments_sales_pct.assign(
            tf.maximum(0.0, self.advance_payments_sales_pct)
        )
        self.advance_payments_purchases_pct.assign(
            tf.maximum(0.0, self.advance_payments_purchases_pct)
        )
        self.account_receivables_pct.assign(
            tf.maximum(0.0, self.account_receivables_pct)
        )
        self.account_payables_pct.assign(tf.maximum(0.0, self.account_payables_pct))
        self.inventory_pct.assign(tf.maximum(0.0, self.inventory_pct))
        self.total_liquidity_pct.assign(tf.maximum(0.0, self.total_liquidity_pct))
        self.cash_pct_of_liquidity.assign(
            tf.clip_by_value(self.cash_pct_of_liquidity, 0.0, 1.0)
        )
        self.income_tax_pct.assign(tf.clip_by_value(self.income_tax_pct, 0.0, 1.0))
        self.dividend_payout_ratio_pct.assign(
            tf.clip_by_value(self.dividend_payout_ratio_pct, 0.0, 1.0)
        )
        self.stock_buyback_pct.assign(tf.maximum(0.0, self.stock_buyback_pct))
        self.avg_maturity_years.assign(tf.maximum(1.001, self.avg_maturity_years))
        self.equity_financing_pct.assign(
            tf.clip_by_value(self.equity_financing_pct, 0.0, 1.0)
        )


def run_monte_carlo_forecast(
    model, initial_state, sales_f, purch_f, cum_inf_f, n_samples=1000
):
    """Run Monte Carlo simulation using the Bayesian model."""
    print(f"\n--- Running Monte Carlo Forecast ({n_samples} samples) ---")
    ni_trajectories = []

    for _ in range(n_samples):
        curr_state = initial_state
        sample_ni = []
        for t in range(len(sales_f) - 1):
            inputs = EconomicInputs(
                sales_t=tf.constant(sales_f[t]),
                purchases_t=tf.constant(purch_f[t]),
                sales_t_plus_1=tf.constant(sales_f[t + 1]),
                purchases_t_plus_1=tf.constant(purch_f[t + 1]),
                cum_inflation=tf.constant(cum_inf_f[t]),
            )
            curr_state = model.forecast_step(curr_state, inputs, use_mean_opex=False)
            sample_ni.append(curr_state.net_income.numpy())
        ni_trajectories.append(sample_ni)

    ni_trajectories = np.array(ni_trajectories)
    mean_ni = np.mean(ni_trajectories, axis=0)
    lower = np.percentile(ni_trajectories, 2.5, axis=0)
    upper = np.percentile(ni_trajectories, 97.5, axis=0)

    print(f"{'Year':<5} | {'Mean NI':<15} | {'2.5% CI':<15} | {'97.5% CI':<15}")
    print("-" * 60)
    for t in range(len(mean_ni)):
        print(f"{t+1:<5} | {mean_ni[t]:<15.2e} | {lower[t]:<15.2e} | {upper[t]:<15.2e}")


def run_training_and_forecast():
    """Main pipeline for training and Bayesian forecasting."""
    model = TrainableFinancialModel()

    # Historical data (same as deterministic model)
    sales = [
        2.65595e11,
        2.60174e11,
        2.74515e11,
        3.65817e11,
        3.94328e11,
        3.83285e11,
        3.91035e11,
        4.16161e11,
    ]
    inv_raw = [4855e6, 3956e6, 4106e6, 4061e6, 6580e6, 4946e6, 6331e6, 7286e6, 5718e6]
    cogs_raw = [
        1.52853e11,
        1.49235e11,
        1.58503e11,
        2.01697e11,
        2.12442e11,
        2.02618e11,
        1.98907e11,
        2.09262e11,
    ]
    purch = [cogs_raw[i] + inv_raw[i + 1] - inv_raw[i] for i in range(len(cogs_raw))]

    historical_data = {
        "sales": np.array(sales),
        "purchases": np.array(purch),
        "inventory": np.array(inv_raw[1:]),
        "depr": np.array(
            [10903e6, 12547e6, 11056e6, 11284e6, 11104e6, 11519e6, 11445e6, 11698e6]
        ),
        "nca": np.array(
            [
                2.34386e11,
                1.75697e11,
                1.80175e11,
                2.16166e11,
                2.1735e11,
                2.09017e11,
                2.11993e11,
                2.11284e11,
            ]
        ),
        "advance_payments_purchases": np.array(
            [12087e6, 12352e6, 11264e6, 14111e6, 21223e6, 14695e6, 14287e6, 14585e6]
        ),
        "accounts_receivable": np.array(
            [48995e6, 45804e6, 37445e6, 51506e6, 60932e6, 60985e6, 66243e6, 72957e6]
        ),
        "cash": np.array(
            [25913e6, 48844e6, 38016e6, 34940e6, 23646e6, 29965e6, 29943e6, 35934e6]
        ),
        "investment_in_market_securities": np.array(
            [40388e6, 51713e6, 52927e6, 27699e6, 24658e6, 31590e6, 35228e6, 18763e6]
        ),
        "accounts_payable": np.array(
            [55888e6, 46236e6, 42296e6, 54763e6, 64115e6, 62611e6, 68960e6, 69860e6]
        ),
        "advance_payments_sales": np.array(
            [5966e6, 5522e6, 6643e6, 7612e6, 7912e6, 8061e6, 8249e6, 9055e6]
        ),
        "current_liabilities": np.array(
            [55012e6, 53960e6, 56453e6, 63106e6, 81955e6, 74636e6, 99183e6, 86716e6]
        ),
        "non_current_liabilities": np.array(
            [
                1.41712e11,
                1.4231e11,
                1.53157e11,
                1.62431e11,
                1.48101e11,
                1.45129e11,
                1.31638e11,
                1.19877e11,
            ]
        ),
        "equity": np.array(
            [1.07147e11, 90488e6, 65339e6, 63090e6, 50672e6, 62146e6, 56950e6, 73733e6]
        ),
        "net_income": np.array(
            [59531e6, 55256e6, 57411e6, 94680e6, 99803e6, 96995e6, 93736e6, 112010e6]
        ),
        "dividends": np.array(
            [13712e6, 14119e6, 14081e6, 14467e6, 14841e6, 15025e6, 15234e6, 15421e6]
        ),
        "stock_buyback": np.array(
            [72738e6, 66897e6, 72358e6, 85971e6, 89402e6, 77550e6, 94949e6, 90711e6]
        ),
        "opex": np.array(
            [30941e6, 34462e6, 38668e6, 43887e6, 51345e6, 54847e6, 57467e6, 62151e6]
        ),
        "tax": np.array(
            [13372e6, 10481e6, 9680e6, 14527e6, 19300e6, 16741e6, 29749e6, 20719e6]
        ),
        "inflation": np.array([0.024, 0.018, 0.012, 0.047, 0.08, 0.041, 0.029, 0.027]),
    }

    # Training
    train_data = {k: v[:-1] for k, v in historical_data.items()}
    model.train_simple_policies(train_data)
    model.train_structural_parameters(train_data)

    print("-" * 50)
    print("Training Complete.")
    print(f"Final %AG: {model.asset_growth.numpy():.5f}")
    print(f"Final %Depr: {model.depreciation_rate.numpy():.5f}")
    print(f"Final %AdvPS: {model.advance_payments_sales_pct.numpy():.5f}")
    print(f"Final %AdvPP: {model.advance_payments_purchases_pct.numpy():.5f}")
    print(f"Final %AR: {model.account_receivables_pct.numpy():.5f}")
    print(f"Final %AP: {model.account_payables_pct.numpy():.5f}")
    print(f"Final %Inv: {model.inventory_pct.numpy():.5f}")
    print(f"Final %TL: {model.total_liquidity_pct.numpy():.5f}")
    print(f"Final %Cash: {model.cash_pct_of_liquidity.numpy():.5f}")
    print(f"Final %IT: {model.income_tax_pct.numpy():.5f}")
    print(f"Final %PR: {model.dividend_payout_ratio_pct.numpy():.5f}")
    print(f"Final %BB: {model.stock_buyback_pct.numpy():.5f}")
    print(
        f"Bayesian OpEx Variable %: Mean={model.q_var_opex_loc.numpy():.4f}, Std={model.q_var_opex_scale.numpy():.4f}"
    )
    print(
        f"Bayesian OpEx Baseline:   Mean={model.q_base_opex_loc.numpy():.2e}, Std={model.q_base_opex_scale.numpy():.2e}"
    )

    print("-" * 50)
    print("Structural Training Complete.")
    print(f"Final %AvgSTInt: {model.avg_short_term_interest_pct.numpy():.5f}")
    print(f"Final %AvgLTInt: {model.avg_long_term_interest_pct.numpy():.5f}")
    print(f"Final AvgM: {model.avg_maturity_years.numpy():.5f}")
    print(f"Final %MSReturn: {model.market_securities_return_pct.numpy():.5f}")
    print(f"Final %EF: {model.equity_financing_pct.numpy():.5f}")
    print("-" * 50)

    # Forecasting
    idx_2024 = -2
    state_2024 = FinancialState.from_dict(
        {k: v[idx_2024] for k, v in historical_data.items()}
    )

    s_growth = historical_data["sales"][-1] / historical_data["sales"][-2]
    p_growth = historical_data["purchases"][-1] / historical_data["purchases"][-2]
    s_forecast = [historical_data["sales"][-1] * (s_growth**i) for i in range(4)]
    p_forecast = [historical_data["purchases"][-1] * (p_growth**i) for i in range(4)]
    inf_forecast = [0.03] * 4
    cum_inf_forecast = np.cumprod(1 + np.array(inf_forecast))

    run_monte_carlo_forecast(
        model, state_2024, s_forecast, p_forecast, cum_inf_forecast
    )


if __name__ == "__main__":
    run_training_and_forecast()
