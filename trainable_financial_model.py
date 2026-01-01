"""
Trainable Financial Model using TensorFlow.

This module provides a financial forecasting model that implements the
Cash Budget construction logic (Pareja, 2009). The model includes trainable
parameters that can be optimized using historical financial data.

Software Engineering Principles Applied:
- Modularity: Separation of concerns between state, inputs, and model logic.
- Readability: Meaningful names and PEP 257 compliant docstrings.
- Reusability: Dataclasses for consistent data handling.
- Maintainability: Modularized forecasting steps and clean training loops.
"""

import tensorflow as tf
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any


@dataclass(frozen=True)
class FinancialState:
    """
    Represents the financial state of a company at a specific point in time.
    All values are typically tf.Tensor (float64) or float.
    """

    nca: Any  # Non-current assets
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

    # Diagnostic and flow fields
    liquidity_check: Any = 0.0
    balance_sheet_check: Any = 0.0
    st_loan_issued: Any = 0.0
    lt_loan_issued: Any = 0.0
    st_principal_paid: Any = 0.0
    lt_principal_paid: Any = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinancialState":
        """Create a FinancialState from a dictionary, filtering unknown keys."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass(frozen=True)
class EconomicInputs:
    """
    Represents external economic inputs and company-specific drivers for a period.
    """

    sales_t: Any
    purchases_t: Any
    sales_t_plus_1: Any
    purchases_t_plus_1: Any
    cum_inflation: Any


class TrainableFinancialModel(tf.Module):
    """
    A modular financial forecasting model with trainable parameters.

    Implements the Cash Budget construction logic (Pareja, 2009),
    supporting both policy parameter and structural parameter training.
    """

    def __init__(self):
        super().__init__()
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize all trainable variables with default values."""
        # Policy Parameters
        self.asset_growth = tf.Variable(0.0076, name="asset_growth", dtype=tf.float64)
        self.depreciation_rate = tf.Variable(0.055, name="depr_rate", dtype=tf.float64)
        self.advance_payments_sales_pct = tf.Variable(
            0.020614523, name="advance_payments_sales_pct", dtype=tf.float64
        )
        self.advance_payments_purchases_pct = tf.Variable(
            0.073525733, name="advance_payments_purchases_pct", dtype=tf.float64
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

        # Structural Parameters
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

    def forecast_step(
        self, state: Union[FinancialState, Dict[str, Any]], inputs: EconomicInputs
    ) -> FinancialState:
        """
        Perform a single period financial forecast step.

        Args:
            state: The previous financial state (t-1).
            inputs: Economic inputs for the current period (t).

        Returns:
            The predicted financial state at time t.
        """
        # Ensure state is a FinancialState object for easier access
        if isinstance(state, dict):
            # Map legacy/abbreviated keys to dataclass field names
            mapping = {
                "adv_pp": "advance_payments_purchases",
                "adv_ps": "advance_payments_sales",
                "ar": "accounts_receivable",
                "inv": "inventory",
                "ap": "accounts_payable",
                "cl": "current_liabilities",
                "ncl": "non_current_liabilities",
                "ni": "net_income",
                "ims": "investment_in_market_securities",
            }
            mapped_state = {mapping.get(k, k): v for k, v in state.items()}
            state = FinancialState.from_dict(mapped_state)

        # 1. Assets Evolution
        asset_updates = self._evolve_assets(state, inputs)

        # 2. Income Statement
        income_stmt = self._calculate_income_statement(state, inputs, asset_updates)

        # 3. Liquidity and Financing
        financing = self._manage_liquidity_and_financing(
            state, inputs, asset_updates, income_stmt
        )

        # 4. Final state assembly and integrity checks
        return self._assemble_and_check_state(
            state, asset_updates, income_stmt, financing, inputs
        )

    def _evolve_assets(
        self, state: FinancialState, inputs: EconomicInputs
    ) -> Dict[str, Any]:
        """Calculate evolution of asset accounts."""
        depr = state.nca * self.depreciation_rate
        capex = depr + (inputs.sales_t * self.asset_growth)
        nca_curr = state.nca - depr + capex

        adv_pp_curr = inputs.purchases_t_plus_1 * self.advance_payments_purchases_pct
        ar_curr = inputs.sales_t * self.account_receivables_pct
        inv_curr = inputs.sales_t * self.inventory_pct

        total_liq_curr = inputs.sales_t * self.total_liquidity_pct
        cash_curr = total_liq_curr * self.cash_pct_of_liquidity
        ims_curr = total_liq_curr * (1 - self.cash_pct_of_liquidity)

        return {
            "nca": nca_curr,
            "depreciation": depr,
            "capex": capex,
            "advance_payments_purchases": adv_pp_curr,
            "accounts_receivable": ar_curr,
            "inventory": inv_curr,
            "cash": cash_curr,
            "investment_in_market_securities": ims_curr,
            "total_liquidity": total_liq_curr,
        }

    def _calculate_income_statement(
        self, state: FinancialState, inputs: EconomicInputs, assets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate current period income statement."""
        cogs = state.inventory + inputs.purchases_t - assets["inventory"]
        opex = (
            self.baseline_opex * inputs.cum_inflation
            + inputs.sales_t * self.variable_opex_pct
        )
        ebitda = inputs.sales_t - cogs - opex

        # Debt servicing based on PREVIOUS debt levels (avoids circularity)
        prin_lt_due = state.non_current_liabilities / (self.avg_maturity_years - 1)
        int_lt = (
            self.avg_long_term_interest_pct
            * state.non_current_liabilities
            / (1 - 1 / self.avg_maturity_years)
        )

        prin_st_due = state.current_liabilities - prin_lt_due
        int_st = self.avg_short_term_interest_pct * prin_st_due

        ms_return = (
            state.investment_in_market_securities * self.market_securities_return_pct
        )

        ebt = ebitda - assets["depreciation"] - (int_st + int_lt) + ms_return
        tax = ebt * self.income_tax_pct
        ni_curr = ebt - tax

        return {
            "cogs": cogs,
            "opex": opex,
            "ebitda": ebitda,
            "net_income": ni_curr,
            "tax": tax,
            "interest_total": int_st + int_lt,
            "principal_st": prin_st_due,
            "principal_lt": prin_lt_due,
            "ms_return": ms_return,
            "interest_st": int_st,
            "interest_lt": int_lt,
        }

    def _manage_liquidity_and_financing(
        self,
        state: FinancialState,
        inputs: EconomicInputs,
        assets: Dict[str, Any],
        income: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Manage cash budget and determine new financing needs."""
        # Operating flows
        adv_sales_curr = inputs.sales_t_plus_1 * self.advance_payments_sales_pct
        sales_cash_in = (
            inputs.sales_t * (1 - self.account_receivables_pct)
            - state.advance_payments_sales
            + state.accounts_receivable
            + adv_sales_curr
        )

        purch_cash_out = (
            inputs.purchases_t * (1 - self.account_payables_pct)
            - state.advance_payments_purchases
            + state.accounts_payable
            + assets["advance_payments_purchases"]
        )
        op_outflows = purch_cash_out + income["opex"] + income["tax"]
        op_nlb = sales_cash_in - op_outflows

        # Short-term deficit
        prev_total_liq = state.cash + state.investment_in_market_securities
        liq_deficit_st = (
            assets["total_liquidity"]
            - prev_total_liq
            - income["ms_return"]
            - op_nlb
            + income["principal_st"]
            + income["interest_st"]
        )
        new_st_loan = tf.maximum(0.0, liq_deficit_st)

        # Long-term deficit
        divs = state.net_income * self.dividend_payout_ratio_pct
        bb = assets["depreciation"] * self.stock_buyback_pct

        liq_deficit_lt = (
            liq_deficit_st
            - new_st_loan
            + assets["capex"]
            + income["principal_lt"]
            + income["interest_lt"]
            + divs
            + bb
        )

        lt_fin_needed = tf.maximum(0.0, liq_deficit_lt)
        equity_fin = lt_fin_needed * self.equity_financing_pct
        new_lt_loan = lt_fin_needed * (1 - self.equity_financing_pct)

        # Total Financing NLB
        fin_nlb = (
            new_st_loan
            + new_lt_loan
            - income["principal_st"]
            - income["principal_lt"]
            - income["interest_total"]
        )
        owners_nlb = equity_fin - divs - bb

        total_nlb = (
            op_nlb - assets["capex"] + fin_nlb + income["ms_return"] + owners_nlb
        )

        return {
            "new_st_loan": new_st_loan,
            "new_lt_loan": new_lt_loan,
            "equity_financing": equity_fin,
            "dividends": divs,
            "buybacks": bb,
            "advance_payments_sales": adv_sales_curr,
            "total_nlb": total_nlb,
        }

    def _assemble_and_check_state(
        self,
        prev_state: FinancialState,
        assets: Dict[str, Any],
        income: Dict[str, Any],
        financing: Dict[str, Any],
        inputs: EconomicInputs,
    ) -> FinancialState:
        """Assemble final state and verify identities."""
        ap_curr = inputs.purchases_t * self.account_payables_pct

        ncl_curr = (
            (financing["new_lt_loan"] + prev_state.non_current_liabilities)
            * (self.avg_maturity_years - 1)
            / self.avg_maturity_years
        )

        cl_curr = financing["new_st_loan"] + ncl_curr / (self.avg_maturity_years - 1)

        equity_curr = (
            prev_state.equity
            + financing["equity_financing"]
            + income["net_income"]
            - financing["dividends"]
            - financing["buybacks"]
        )

        # Identity Checks
        total_assets = (
            assets["nca"]
            + assets["advance_payments_purchases"]
            + assets["accounts_receivable"]
            + assets["inventory"]
            + assets["cash"]
            + assets["investment_in_market_securities"]
        )
        total_liab_eq = (
            ap_curr
            + financing["advance_payments_sales"]
            + cl_curr
            + ncl_curr
            + equity_curr
        )

        prev_total_liq = prev_state.cash + prev_state.investment_in_market_securities
        liq_check = prev_total_liq + financing["total_nlb"] - assets["total_liquidity"]

        return FinancialState(
            nca=assets["nca"],
            advance_payments_purchases=assets["advance_payments_purchases"],
            accounts_receivable=assets["accounts_receivable"],
            inventory=assets["inventory"],
            cash=assets["cash"],
            investment_in_market_securities=assets["investment_in_market_securities"],
            accounts_payable=ap_curr,
            advance_payments_sales=financing["advance_payments_sales"],
            current_liabilities=cl_curr,
            non_current_liabilities=ncl_curr,
            equity=equity_curr,
            net_income=income["net_income"],
            liquidity_check=liq_check,
            balance_sheet_check=total_assets - total_liab_eq,
            st_loan_issued=financing["new_st_loan"],
            lt_loan_issued=financing["new_lt_loan"],
            st_principal_paid=income["principal_st"],
            lt_principal_paid=income["principal_lt"],
        )

    def train_simple_policies(
        self, historical_data: Dict[str, np.ndarray], learning_rate=0.0001, epochs=5000
    ):
        """Train policy parameters using historical data alignment."""
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

        for i in range(epochs):
            with tf.GradientTape() as tape:
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
                            data["opex"]
                            - (
                                self.baseline_opex * cum_inf
                                + data["sales"] * self.variable_opex_pct
                            )
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
                total_loss = tf.add_n(losses)

            grads = tape.gradient(total_loss, vars_to_train)
            optimizer.apply_gradients(zip(grads, vars_to_train))
            self._apply_constraints()

            if i % 1000 == 0:
                print(f"Epoch {i}: Policy Loss = {total_loss.numpy():.4e}")

    def train_structural_parameters(
        self, historical_data: Dict[str, np.ndarray], learning_rate=0.0001, epochs=5000
    ):
        """Train structural parameters using state transition gradients."""
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

        print(f"Training structural parameters on {num_transitions} transitions...")

        for i in range(epochs):
            with tf.GradientTape() as tape:
                total_loss = 0.0
                for t in range(num_transitions):
                    # Construct state with full descriptive names
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
                    total_loss += tf.add_n(losses) / 1e18  # Normalization

            grads = tape.gradient(total_loss, vars_to_train)
            optimizer.apply_gradients(zip(grads, vars_to_train))
            self._apply_constraints()

            if i % 1000 == 0:
                print(f"Epoch {i}: Structural Loss = {total_loss.numpy():.4e}")

    def _apply_constraints(self):
        """Ensure parameters remain within economically valid ranges."""
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
        self.variable_opex_pct.assign(tf.maximum(0.0, self.variable_opex_pct))
        self.dividend_payout_ratio_pct.assign(
            tf.clip_by_value(self.dividend_payout_ratio_pct, 0.0, 1.0)
        )
        self.stock_buyback_pct.assign(tf.maximum(0.0, self.stock_buyback_pct))
        self.avg_short_term_interest_pct.assign(
            tf.maximum(0.0, self.avg_short_term_interest_pct)
        )
        self.avg_long_term_interest_pct.assign(
            tf.maximum(0.0, self.avg_long_term_interest_pct)
        )
        self.avg_maturity_years.assign(tf.maximum(1.001, self.avg_maturity_years))
        self.market_securities_return_pct.assign(
            tf.maximum(0.0, self.market_securities_return_pct)
        )
        self.equity_financing_pct.assign(
            tf.clip_by_value(self.equity_financing_pct, 0.0, 1.0)
        )


def run_training_and_forecast():
    """Main execution function for training and backtesting."""
    model = TrainableFinancialModel()

    # Organized Historical Data (Apple 2022-2025)
    sales = np.array(
        [
            2.65595e11,
            2.60174e11,
            2.74515e11,
            3.65817e11,
            3.94328e11,
            3.83285e11,
            3.91035e11,
            4.16161e11,
        ]
    )
    inv_raw = np.array(
        [
            4855000000,
            3956000000,
            4106000000,
            4061000000,
            6580000000,
            4946000000,
            6331000000,
            7286000000,
            5718000000,
        ]
    )
    cogs_raw = np.array(
        [
            1.52853e11,
            1.49235e11,
            1.58503e11,
            2.01697e11,
            2.12442e11,
            2.02618e11,
            1.98907e11,
            2.09262e11,
        ]
    )
    purchases = cogs_raw + inv_raw[1:] - inv_raw[:-1]

    historical_data = {
        "sales": sales,
        "purchases": purchases,
        "inventory": inv_raw[1:],
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

    # Training (using all data except last for backtesting)
    train_data = {k: v[:-1] for k, v in historical_data.items()}
    model.train_simple_policies(train_data)
    model.train_structural_parameters(train_data)

    # Backtesting 2025
    print("\\n" + "=" * 90)
    print("BACKTESTING: Forecasted 2025 vs Actual 2025 Balance Sheet")
    print("=" * 90)

    # Initial state (2024)
    idx_2024 = -2
    initial_state = FinancialState(
        nca=historical_data["nca"][idx_2024],
        advance_payments_purchases=historical_data["adv_pp"][idx_2024],
        accounts_receivable=historical_data["ar"][idx_2024],
        inventory=historical_data["inv"][idx_2024],
        cash=historical_data["cash"][idx_2024],
        investment_in_market_securities=historical_data["ims"][idx_2024],
        accounts_payable=historical_data["ap"][idx_2024],
        advance_payments_sales=historical_data["adv_ps"][idx_2024],
        current_liabilities=historical_data["cl"][idx_2024],
        non_current_liabilities=historical_data["ncl"][idx_2024],
        equity=historical_data["equity"][idx_2024],
        net_income=historical_data["ni"][idx_2024],
    )

    # Inputs for 2025
    s_growth = historical_data["sales"][-1] / historical_data["sales"][-2]
    p_growth = historical_data["purchases"][-1] / historical_data["purchases"][-2]

    inputs_2025 = EconomicInputs(
        sales_t=historical_data["sales"][-1],
        purchases_t=historical_data["purchases"][-1],
        sales_t_plus_1=historical_data["sales"][-1] * s_growth,
        purchases_t_plus_1=historical_data["purchases"][-1] * p_growth,
        cum_inflation=np.prod(1 + historical_data["inflation"]),
    )

    forecast_2025 = model.forecast_step(initial_state, inputs_2025)

    # Metrics comparison
    print(
        f"\\n{'Item':<35} | {'Actual (B)':>12} | {'Forecast (B)':>12} | {'Error %':>10}"
    )
    print("-" * 80)

    items_to_show = [
        "nca",
        "inventory",
        "cash",
        "equity",
        "net_income",
        "current_liabilities",
    ]
    actuals = historical_data
    for item in items_to_show:
        actual_val = actuals[item if item != "ims" else "ims"][-1]
        forecast_val = (
            getattr(forecast_2025, item).numpy()
            if hasattr(getattr(forecast_2025, item), "numpy")
            else getattr(forecast_2025, item)
        )
        err = (forecast_val - actual_val) / abs(actual_val) * 100
        print(
            f"{item:<35} | {actual_val/1e9:>12.2f} | {forecast_val/1e9:>12.2f} | {err:>9.1f}%"
        )

    print("=" * 90)
    print(f"Balance Sheet Check: {forecast_2025.balance_sheet_check.numpy():.2f}")
    print(f"Liquidity Check:     {forecast_2025.liquidity_check.numpy():.2f}")


if __name__ == "__main__":
    run_training_and_forecast()
