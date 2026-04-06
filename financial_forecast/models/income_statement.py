"""Income statement model — COGS, OpEx, interest, tax, and net income.

Owns the OpEx module, interest rate parameters, and market securities
return.  Tax is injected at call time.
"""

from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

from financial_forecast.models.opex import OpExModule
from financial_forecast.models.tax import SimpleTax
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

    Owns the OpEx module, interest rate, and market securities return
    parameters.  Tax computation is delegated to the tax module passed
    at call time.
    """

    def __init__(
        self,
        opex_module: OpExModule,
        name: str = "income_statement",
    ):
        super().__init__(name=name)
        self.opex_module = opex_module

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
        self.market_securities_return_pct = tf.Variable(
            0.05,
            dtype=tf.float64,
            name="market_securities_return_pct",
        )

    @property
    def structural_trainable_variables(self) -> list:
        """Interest rate and MS return variables for the structural phase.

        Excludes OpEx variables, which are optimized in the policy phase.
        """
        return [
            *self.avg_short_term_interest_pct.trainable_variables,
            *self.avg_long_term_interest_pct.trainable_variables,
            self.market_securities_return_pct,
        ]

    def init_from_data(self, s: Dict[str, tf.Tensor]) -> None:
        """Initialize interest/return rates from historical averages."""
        _f64 = lambda v: tf.constant(v, dtype=tf.float64)
        _EPS = 1e-12
        ratios = s["ms_return"] / tf.maximum(s["ims"], _EPS)
        finite_mask = tf.math.is_finite(ratios)
        if tf.reduce_any(finite_mask):
            mean_ratio = tf.reduce_mean(tf.boolean_mask(ratios, finite_mask))
            self.market_securities_return_pct.assign(_f64(float(mean_ratio)))

    def calculate_income(
        self,
        state: tf.Tensor,
        assets: Dict[str, tf.Tensor],
        sales_t: tf.Tensor,
        cum_inflation: tf.Tensor,
        tax_module: SimpleTax,
        year: tf.Tensor,
        use_mean_opex: bool = True,
    ) -> Dict[str, tf.Tensor]:
        """Compute income statement.

        Args:
            state: ``[n_samples, 14]`` recurrent state tensor.
            assets: Dict from ``BalanceSheetModel.evolve_assets()``.
            sales_t: ``[n_samples]`` sales.
            cum_inflation: Scalar cumulative inflation factor.
            tax_module: Tax module with ``compute(ebt, year)`` method.
            year: Scalar calendar year.
            use_mean_opex: If ``True``, use deterministic/mean OpEx.
                If ``False``, use pre-sampled MC values.

        Returns:
            Dict with income statement items.
        """
        inv_prev = state[:, R_INV]
        eff_st_debt_prev = state[:, R_EFF_ST_DEBT]
        cur_lt_debt_prev = state[:, R_CUR_LT_DEBT]
        ncl_prev = state[:, R_NCL]
        ims_prev = state[:, R_IMS]

        # OpEx — computed by the owned module
        if use_mean_opex:
            opex = self.opex_module.predict(
                sales_t,
                cum_inflation,
                use_mean=True,
            )
        else:
            opex = self.opex_module.compute_mc_step(
                sales_t,
                cum_inflation,
                year,
            )

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

    def print_summary(self) -> None:
        """Print learned parameters."""
        print(f"Final %AvgSTInt: " f"{self.avg_short_term_interest_pct.numpy():.5f}")
        print(f"Final %AvgLTInt: " f"{self.avg_long_term_interest_pct.numpy():.5f}")
        print(f"Final %MSReturn: " f"{self.market_securities_return_pct.numpy():.5f}")
