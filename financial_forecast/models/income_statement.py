"""Income statement model — COGS, interest, tax, and net income.

Owns interest rate and market securities return parameters.
OpEx and tax are injected as external modules.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from financial_forecast.inference.state_index import (
    R_INV,
    R_EFF_ST_DEBT,
    R_CUR_LT_DEBT,
    R_NCL,
    R_IMS,
)

tfb = tfp.bijectors


class IncomeStatementModel(tf.Module):
    """Computes the income statement from asset evolution results.

    Owns interest rate and market securities return parameters.
    Tax computation is delegated to the tax module passed at call time.
    """

    def __init__(self, name="income_statement"):
        super().__init__(name=name)

        self.avg_short_term_interest_pct = tfp.util.TransformedVariable(
            initial_value=0.1,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="avg_short_term_interest_pct",
        )
        self.avg_long_term_interest_pct = tfp.util.TransformedVariable(
            initial_value=0.06,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="avg_long_term_interest_pct",
        )
        self.market_securities_return_pct = tfp.util.TransformedVariable(
            initial_value=0.05,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="market_securities_return_pct",
        )

    def init_from_data(self, s):
        """Initialize interest/return rates from historical averages."""
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        self.market_securities_return_pct.assign(
            _f64(
                max(
                    _EPS,
                    float(tf.reduce_mean(s["ms_return"] / tf.maximum(s["ims"], _EPS))),
                )
            )
        )

    def calculate_income(self, state, assets, sales_t, opex, tax_module, year):
        """Compute income statement.

        Args:
            state: ``[n_samples, 14]`` recurrent state tensor.
            assets: Dict from ``BalanceSheetModel.evolve_assets()``.
            sales_t: ``[n_samples]`` sales.
            opex: ``[n_samples]`` pre-computed operating expenses.
            tax_module: Tax module with ``compute(ebt, year)`` method.
            year: Scalar calendar year for tax anomaly lookup.

        Returns:
            Dict with income statement items.
        """
        inv_prev = state[:, R_INV]
        eff_st_debt_prev = state[:, R_EFF_ST_DEBT]
        cur_lt_debt_prev = state[:, R_CUR_LT_DEBT]
        ncl_prev = state[:, R_NCL]
        ims_prev = state[:, R_IMS]

        cogs = inv_prev + assets["purchases_t"] - assets["inv_curr"]
        ebitda = sales_t - cogs - opex

        principal_lt = cur_lt_debt_prev
        interest_lt = self.avg_long_term_interest_pct * (ncl_prev + cur_lt_debt_prev)
        principal_st = eff_st_debt_prev
        interest_st = self.avg_short_term_interest_pct * principal_st

        ms_return = ims_prev * self.market_securities_return_pct
        ebt = ebitda - assets["depreciation"] - (interest_st + interest_lt) + ms_return
        tax = tax_module.compute(ebt, year=year)
        ni_curr = ebt - tax

        return {
            "cogs": cogs,
            "opex": opex,
            "ms_return": ms_return,
            "tax": tax,
            "ni_curr": ni_curr,
            "interest_st": interest_st,
            "interest_lt": interest_lt,
            "principal_st": principal_st,
            "principal_lt": principal_lt,
        }

    def print_summary(self):
        """Print learned parameters."""
        print(f"Final %AvgSTInt: {self.avg_short_term_interest_pct.numpy():.5f}")
        print(f"Final %AvgLTInt: {self.avg_long_term_interest_pct.numpy():.5f}")
        print(f"Final %MSReturn: {self.market_securities_return_pct.numpy():.5f}")
