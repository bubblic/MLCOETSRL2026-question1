"""Bayesian financial model with variational inference for operating expenses.

This model extends :class:`BaseFinancialModel` with:

- **TransformedVariable** parameters with bijector-based constraints.
- **Bayesian OpEx** via variational inference (Normal posterior, wide prior).
- **Time-varying** logit-linear trends for liquidity, cash split, cost ratio,
  equity financing mix, and short-term debt.
- **Lintner dividend smoothing** and policy-driven stock buybacks.

The ``forecast_step`` method overrides the base class to implement the full
Bayesian Cash Budget construction, while ``_initialize_parameters`` sets up
all three parameter layers (policy, Bayesian OpEx, structural).

Training and serialization are handled externally by
:class:`~financial_forecast.training.policy_trainer.PolicyTrainer`,
:class:`~financial_forecast.training.structural_trainer.StructuralTrainer`,
and :mod:`financial_forecast.serialization`.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from financial_forecast.models.base import BaseFinancialModel
from financial_forecast.models.tax import SimpleTax, TaxWithAnomalies
from financial_forecast.serialization.parameter_io import (
    save_parameters as _save_parameters,
    load_parameters as _load_parameters,
)
from financial_forecast.inference.state_index import (
    R_NCA,
    R_ADV_PP,
    R_AR,
    R_INV,
    R_CASH,
    R_IMS,
    R_AP,
    R_ADV_PS,
    R_EFF_ST_DEBT,
    R_CUR_LT_DEBT,
    R_NCL,
    R_EQUITY,
    R_NET_INCOME,
    R_DIVIDENDS,
    RECURRENT_KEYS,
    DIAGNOSTIC_KEYS,
)

tfb = tfp.bijectors


class TrainableFinancialModel(BaseFinancialModel):
    """Bayesian financial model using variational inference for OpEx estimation.

    Combines deterministic policy parameters with Bayesian variational
    inference for operating expenses. Structural parameters (interest rates,
    maturity, financing mix) form a third trainable layer.

    All constrained parameters use bijector-based reparameterization
    (Softplus for non-negative, Sigmoid for [0,1]-bounded, Chain for
    lower-bounded).

    Attributes:
        base_year: The fiscal year corresponding to ``t = 0`` in all
            logit-linear time-trend parameters.  Set by the pipeline
            from the historical data's ``"years"`` field.
    """

    def __init__(self, opex_module, tax_anomalies=None, name=None):
        """Create a new trainable financial model.

        Args:
            opex_module: OpEx module instance (``SimpleOpEx`` or
                ``BayesianOpEx``).
            tax_anomalies: Optional 1-D tensor of one-time tax payment
                amounts in USD.  If ``None``, uses :class:`SimpleTax`;
                otherwise :class:`TaxWithAnomalies`.
            name: Optional name for the underlying ``tf.Module``.
        """
        self.amount_scale = None
        self.base_year = None
        self.opex_module = opex_module
        if tax_anomalies is not None:
            self.tax_module = TaxWithAnomalies(tax_anomalies)
        else:
            self.tax_module = SimpleTax()

        super().__init__(
            name=name
        )  # calls _initialize_parameters() for the rest of parameters

    def prepare_for_training(self, years, amount_scale):
        """Configure data-derived model settings.

        Called by the pipeline after computing ``amount_scale``.

        Args:
            years: 1-D tensor of fiscal year labels.
            amount_scale: USD-to-scaled-units conversion factor.
            scaled_sales: 1-D tensor of historical sales (already scaled).
            scaled_opex: 1-D tensor of historical OpEx (already scaled).
            inflation: 1-D tensor of annual inflation rates (or zeros).
        """
        self.base_year = int(years[0])
        self.amount_scale = amount_scale

    # ------------------------------------------------------------------
    # Abstract method implementation (required by BaseFinancialModel)
    # ------------------------------------------------------------------

    def _initialize_parameters(self) -> None:
        """Initialize policy, Bayesian OpEx, and structural parameters."""

        # ===================== LAYER 1: POLICY PARAMETERS =====================
        # These are trainable with simple linear regression.
        # Non-negative params use Softplus bijector (constraint by construction).
        # [0,1]-bounded params use Sigmoid bijector.
        self.asset_growth = tfp.util.TransformedVariable(
            initial_value=0.0076,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="asset_growth",
        )  # %AG
        self.asset_maintain = tfp.util.TransformedVariable(
            initial_value=0.99,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="asset_maintain",
        )  # %AM (maintenance capex multiplier on depreciation)
        self.depreciation_rate = tfp.util.TransformedVariable(
            initial_value=0.055,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="depr_rate",
        )  # %Depr
        self.advance_payments_sales_pct = tfp.util.TransformedVariable(
            initial_value=0.0206,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="adv_ps",
        )  # %AdvPS
        self.advance_payments_purchases_pct = tfp.util.TransformedVariable(
            initial_value=0.0735,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="adv_pp",
        )  # %AdvPP
        self.account_receivables_pct = tfp.util.TransformedVariable(
            initial_value=0.1591,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="ar_pct",
        )  # %AR
        self.account_payables_pct = tfp.util.TransformedVariable(
            initial_value=0.3501,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="ap_pct",
        )  # %AP
        self.inventory_pct = tfp.util.TransformedVariable(
            initial_value=0.0165,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="inv_pct",
        )  # %Inv

        # --- Total Liquidity Logit-Linear Model with Baseline ---
        # %TL(t) = sigmoid(tl_alpha + tl_beta * t), where t = year - base_year
        # TL_t = tl_baseline + sales_t * %TL(t)
        # Sigmoid ensures %TL stays in (0, 1), preventing negative liquidity
        # even when the historical trend is decreasing.
        # tl_baseline captures a fixed liquidity floor independent of sales.
        # Initialize alpha to inverse_sigmoid(0.16) ≈ -1.66
        self.tl_alpha = tf.Variable(-1.66, dtype=tf.float64, name="tl_alpha")
        self.tl_beta = tf.Variable(0.0, dtype=tf.float64, name="tl_beta")
        self.tl_baseline = tf.Variable(0.0, dtype=tf.float64, name="tl_baseline")

        # --- Cash % of Liquidity Logit-Linear Model ---
        # %Cash(t) = sigmoid(cash_alpha + cash_beta * t), where t = year - base_year
        # Cash_t = TL_t * %Cash(t)
        # Sigmoid ensures %Cash stays in (0, 1), so IMS = TL - Cash >= 0.
        # Initialize alpha to inverse_sigmoid(0.487) ≈ -0.05
        self.cash_alpha = tf.Variable(-0.05, dtype=tf.float64, name="cash_alpha")
        self.cash_beta = tf.Variable(0.0, dtype=tf.float64, name="cash_beta")

        # --- Dividend Smoothing (Lintner Model) ---
        # D_t = α * (PayoutRatio * NI_t) + (1 - α) * D_{t-1}
        # α=1.0 → pure payout ratio (no smoothing), α=0.0 → constant dividends
        self.dividend_payout_ratio_pct = tfp.util.TransformedVariable(
            initial_value=0.15,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="div_pct",
        )  # %PR
        self.dividend_adjustment_speed = tfp.util.TransformedVariable(
            initial_value=0.01,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="div_adj_speed",
        )  # α

        # --- Stock Buyback: Baseline + Ratio * Depreciation ---
        # BB(t) = sb_baseline + sb_ratio * depreciation(t)
        self.sb_baseline = tf.Variable(0.0, dtype=tf.float64, name="sb_baseline")
        self.sb_ratio = tf.Variable(1.0, dtype=tf.float64, name="sb_ratio")

        # --- Cost Ratio Parameters (Logit-Linear Trend) ---
        # logit(CR_t) = alpha + beta * t  =>  CR_t = sigmoid(alpha + beta * t),
        # where t = year - base_year
        # Purchases are derived: P_t = Sales_t * CR_t + (Inv_target_t - Inv_{t-1})
        # COGS simplifies to: Sales_t * CR_t
        self.cost_ratio_alpha = tf.Variable(
            0.35, dtype=tf.float64, name="cost_ratio_alpha"
        )
        self.cost_ratio_beta = tf.Variable(
            -0.05, dtype=tf.float64, name="cost_ratio_beta"
        )

        # =============== LAYER 2: STRUCTURAL PARAMETERS =======================
        # These are trained with gradient descent on state-transition losses,
        # using the trained policy parameters from Layer 1 as fixed inputs.
        # Non-negative params use Softplus, [0,1]-bounded use Sigmoid,
        # avg_maturity_years uses Shift(1.001) + Softplus to enforce > 1.001.
        self.avg_short_term_interest_pct = tfp.util.TransformedVariable(
            initial_value=0.6,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="avg_short_term_interest_pct",
        )  # %AvgSTInt
        self.avg_long_term_interest_pct = tfp.util.TransformedVariable(
            initial_value=0.06,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="avg_long_term_interest_pct",
        )  # %AvgLTInt
        self.avg_maturity_years = tfp.util.TransformedVariable(
            initial_value=3.0,
            bijector=tfb.Chain(
                [tfb.Shift(tf.constant(1.001, dtype=tf.float64)), tfb.Softplus()]
            ),
            dtype=tf.float64,
            name="avg_maturity_years",
        )  # AvgM (always > 1.001)
        self.market_securities_return_pct = tfp.util.TransformedVariable(
            initial_value=0.05,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="market_securities_return_pct",
        )  # %MSReturn

        # --- Short-Term Debt % of Sales Logit-Linear Model ---
        # %STDebt(t) = sigmoid(st_debt_alpha + st_debt_beta * t),
        # where t = year - base_year
        # new_short_term_loan = sales * %STDebt(t)
        # Replaces deficit-driven ST borrowing: corporations maintain revolving
        # credit / commercial paper as treasury policy, not just to cover deficits.
        # Initialize alpha to inverse_sigmoid(0.17) ≈ -1.59 (Apple's avg ST debt/sales)
        self.st_debt_alpha = tf.Variable(-1.59, dtype=tf.float64, name="st_debt_alpha")
        self.st_debt_beta = tf.Variable(0.0, dtype=tf.float64, name="st_debt_beta")

        # --- Equity Financing % Logit-Linear Model ---
        # %EF(t) = sigmoid(ef_alpha + ef_beta * t), where t = year - base_year
        # Sigmoid ensures %EF stays in (0, 1), and allows the financing mix
        # to evolve over time (e.g., declining equity reliance as firm matures).
        # Initialize alpha to inverse_sigmoid(0.15) ≈ -1.73
        self.ef_alpha = tf.Variable(-1.73, dtype=tf.float64, name="ef_alpha")
        self.ef_beta = tf.Variable(0.0, dtype=tf.float64, name="ef_beta")

    # ------------------------------------------------------------------
    # Serialization (delegates to serialization module)
    # ------------------------------------------------------------------

    def save_parameters(self, path):
        """Save all model parameters to an .npz file.

        Args:
            path: Filesystem path for the output ``.npz`` file.
        """
        _save_parameters(self, path)

    def load_parameters(self, path):
        """Load model parameters from an .npz file.

        Args:
            path: Filesystem path to the ``.npz`` parameter file.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        _load_parameters(self, path)

    # ------------------------------------------------------------------
    # Forecast (overrides base class template)
    # ------------------------------------------------------------------

    @tf.function
    def forecast_step(
        self,
        state,
        inputs,
        use_mean_opex=False,
    ):
        """Advance the financial state by one period (graph-compiled).

        Thin wrapper around :meth:`forecast_step_compiled` that converts
        between the dict-based interface and packed tensor representation.

        Args:
            state: Dict at *t-1* with balance-sheet entries.
            inputs: Dict with ``sales_t``, ``year``, ``cum_inflation``.
            use_mean_opex: Use posterior mean (no sampling/noise).

        Returns:
            Dict mapping output keys to scalar tensors for period *t*.
        """
        state_tensor = tf.expand_dims(
            tf.stack([tf.cast(state[k], tf.float64) for k in RECURRENT_KEYS]),
            0,
        )
        sales_t = tf.reshape(inputs["sales_t"], [1])
        opex = self.opex_module.predict(
            sales_t,
            inputs["cum_inflation"],
            use_mean=use_mean_opex,
        )

        _, diagnostics = self.forecast_step_compiled(
            state_tensor,
            sales_t,
            inputs["year"],
            opex,
        )

        return {key: diagnostics[0, i] for i, key in enumerate(DIAGNOSTIC_KEYS)}

    def forecast_step_compiled(
        self,
        state,
        sales_t,
        year,
        opex,
    ):
        """Batched single-period forecast -- single source of truth.

        Orchestrates the four-stage Pareja Cash Budget construction,
        mirroring the base class decomposition but with trainable
        parameters and time-varying logit-linear trends.

        Args:
            state: ``[n_samples, 14]`` recurrent state tensor.
            sales_t: ``[n_samples]`` sales for this period.
            year: Scalar float64 calendar year.
            opex: ``[n_samples]`` pre-computed operating expenses.

        Returns:
            Tuple ``(new_state, diagnostics)`` where *new_state* has shape
            ``[n_samples, 14]`` and *diagnostics* has shape
            ``[n_samples, 27]``.
        """
        time_index = year - tf.constant(float(self.base_year), dtype=tf.float64)

        assets = self._evolve_assets_compiled(state, sales_t, time_index)
        income = self._calculate_income_compiled(
            state,
            assets,
            sales_t,
            opex,
            year,
        )
        financing = self._manage_liquidity_compiled(
            state,
            assets,
            income,
            sales_t,
            time_index,
        )
        return self._assemble_state_compiled(state, assets, income, financing)

    # ------------------------------------------------------------------
    # Compiled forecast sub-stages
    # ------------------------------------------------------------------

    def _evolve_assets_compiled(self, state, sales_t, time_index):
        """Evolve asset accounts and compute payout decisions from prior state."""
        nca_prev = state[:, R_NCA]
        inv_prev = state[:, R_INV]
        ni_prev = state[:, R_NET_INCOME]
        div_prev_actual = state[:, R_DIVIDENDS]

        depreciation = nca_prev * self.depreciation_rate
        stock_buyback = self.sb_baseline + self.sb_ratio * depreciation

        # Lintner dividend smoothing: D_t = alpha*(PR*NI_{t-1}) + (1-alpha)*D_{t-1}
        # alpha=1 -> pure payout ratio; alpha=0 -> constant dividends
        dividend_target = ni_prev * self.dividend_payout_ratio_pct
        dividends_prev = (
            self.dividend_adjustment_speed * dividend_target
            + (1.0 - self.dividend_adjustment_speed) * div_prev_actual
        )

        # Cost ratio CR_t = sigmoid(alpha + beta*t) follows a logit-linear trend.
        # Purchases are derived: P_t = Sales*CR_t + (Inv_target - Inv_prev)
        cost_ratio_t = tf.sigmoid(
            self.cost_ratio_alpha + self.cost_ratio_beta * time_index
        )

        # CapEx = maintenance (asset_maintain * depr) + growth (asset_growth * sales)
        capex = self.asset_maintain * depreciation + sales_t * self.asset_growth
        nca_curr = nca_prev - depreciation + capex

        ar_curr = sales_t * self.account_receivables_pct
        inv_curr = sales_t * self.inventory_pct
        # Inventory identity: COGS = Inv_prev + Purchases - Inv_curr
        purchases_t = sales_t * cost_ratio_t + (inv_curr - inv_prev)
        adv_pp_curr = purchases_t * self.advance_payments_purchases_pct

        # Total liquidity: baseline floor + sales * sigmoid(logit-linear trend)
        tl_pct = tf.sigmoid(self.tl_alpha + self.tl_beta * time_index)
        total_liquidity_curr = self.tl_baseline + sales_t * tl_pct

        # Cash split: sigmoid trend ensures cash fraction stays in (0, 1)
        cash_pct = tf.sigmoid(self.cash_alpha + self.cash_beta * time_index)
        cash_curr = total_liquidity_curr * cash_pct
        ims_curr = total_liquidity_curr - cash_curr

        return {
            "depreciation": depreciation,
            "capex": capex,
            "nca_curr": nca_curr,
            "ar_curr": ar_curr,
            "inv_curr": inv_curr,
            "purchases_t": purchases_t,
            "adv_pp_curr": adv_pp_curr,
            "total_liquidity_curr": total_liquidity_curr,
            "cash_curr": cash_curr,
            "ims_curr": ims_curr,
            "dividends_prev": dividends_prev,
            "stock_buyback_base": stock_buyback,
        }

    def _calculate_income_compiled(self, state, assets, sales_t, opex, year):
        """Compute income statement from asset evolution results."""
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
        tax = self.tax_module.compute(ebt, year=year)
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

    def _manage_liquidity_compiled(self, state, assets, income, sales_t, time_index):
        """
        Use Cash Budget to compute liquidity budget and financing decisions.

        Five-module structure: operating, investing, external, financing, owners
        """
        zero = tf.constant(0.0, dtype=tf.float64)
        ar_prev = state[:, R_AR]
        ap_prev = state[:, R_AP]
        adv_pp_prev = state[:, R_ADV_PP]
        adv_ps_prev = state[:, R_ADV_PS]
        cash_prev = state[:, R_CASH]
        ims_prev = state[:, R_IMS]

        # 1. Operating NLB: cash inflows from sales vs outflows for purchases/opex/tax
        sales_curr = sales_t * (1 - self.account_receivables_pct) - adv_ps_prev
        adv_ps_curr = sales_t * self.advance_payments_sales_pct
        inflows = sales_curr + ar_prev + adv_ps_curr

        purchases_curr = (
            assets["purchases_t"] * (1 - self.account_payables_pct) - adv_pp_prev
        )
        outflows = (
            purchases_curr
            + ap_prev
            + assets["adv_pp_curr"]
            + income["opex"]
            + income["tax"]
        )
        operating_nlb = inflows - outflows

        # 2. Capital expenditure outflow
        capex_nlb = -assets["capex"]

        # 3. Return on market securities
        external_investment_nlb = income["ms_return"]
        # 4. Financing: ST debt is policy-driven (logit-linear % of sales),
        # not deficit-driven — reflects corporate treasury/revolving credit policy
        st_debt_pct = tf.sigmoid(self.st_debt_alpha + self.st_debt_beta * time_index)
        eff_st_debt_curr = sales_t * st_debt_pct

        liquidity_deficit_st = (
            assets["total_liquidity_curr"]
            - (cash_prev + ims_prev)
            - operating_nlb
            + income["principal_st"]
            + income["interest_st"]
        )

        dividends_prev = assets["dividends_prev"]
        stock_buyback = assets["stock_buyback_base"]

        liquidity_deficit_lt = (
            liquidity_deficit_st
            - eff_st_debt_curr
            - external_investment_nlb
            - capex_nlb
            + income["principal_lt"]
            + income["interest_lt"]
            + dividends_prev
            + stock_buyback
        )
        long_term_financing = tf.maximum(zero, liquidity_deficit_lt)

        # Equity-financing mix: logit-linear trend for debt vs equity split
        ef_pct = tf.sigmoid(self.ef_alpha + self.ef_beta * time_index)
        new_lt_loan = long_term_financing * (1 - ef_pct)
        equity_financing = long_term_financing * ef_pct

        # Any surplus cash beyond the liquidity target funds additional buybacks,
        # ensuring the liquidity budget closes exactly.
        excess_cash_buyback = tf.maximum(zero, -liquidity_deficit_lt)
        stock_buyback = stock_buyback + excess_cash_buyback

        financing_nlb = (
            eff_st_debt_curr
            + new_lt_loan
            - income["principal_st"]
            - income["principal_lt"]
            - income["interest_st"]
            - income["interest_lt"]
        )
        # 5. Transaction with owners: equity issuance minus payouts
        transaction_with_owners_nlb = equity_financing - dividends_prev - stock_buyback
        total_nlb = (
            operating_nlb
            + capex_nlb
            + financing_nlb
            + external_investment_nlb
            + transaction_with_owners_nlb
        )
        liquidity_check = (
            (cash_prev + ims_prev) + total_nlb - assets["total_liquidity_curr"]
        )

        return {
            "adv_ps_curr": adv_ps_curr,
            "eff_st_debt_curr": eff_st_debt_curr,
            "new_lt_loan": new_lt_loan,
            "equity_financing": equity_financing,
            "dividends_prev": dividends_prev,
            "stock_buyback": stock_buyback,
            "liquidity_deficit_st": liquidity_deficit_st,
            "liquidity_check": liquidity_check,
        }

    def _assemble_state_compiled(self, state, assets, income, financing):
        """Evolve liabilities, check balance sheet, and pack output tensors."""
        ncl_prev = state[:, R_NCL]
        equity_prev = state[:, R_EQUITY]

        ap_curr = assets["purchases_t"] * self.account_payables_pct
        total_lt_liabilities = financing["new_lt_loan"] + ncl_prev
        ncl_curr = total_lt_liabilities * (1 - 1 / self.avg_maturity_years)
        cur_lt_debt_curr = total_lt_liabilities / self.avg_maturity_years

        equity_curr = (
            equity_prev
            + financing["equity_financing"]
            + income["ni_curr"]
            - financing["dividends_prev"]
            - financing["stock_buyback"]
        )

        # --- Total Assets & Balance Sheet Check ---
        total_assets = (
            assets["nca_curr"]
            + assets["adv_pp_curr"]
            + assets["ar_curr"]
            + assets["inv_curr"]
            + assets["cash_curr"]
            + assets["ims_curr"]
        )
        total_liab_equity = (
            ap_curr
            + financing["adv_ps_curr"]
            + financing["eff_st_debt_curr"]
            + cur_lt_debt_curr
            + ncl_curr
            + equity_curr
        )
        check = total_assets - total_liab_equity

        new_state = tf.stack(
            [
                assets["nca_curr"],
                assets["adv_pp_curr"],
                assets["ar_curr"],
                assets["inv_curr"],
                assets["cash_curr"],
                assets["ims_curr"],
                ap_curr,
                financing["adv_ps_curr"],
                financing["eff_st_debt_curr"],
                cur_lt_debt_curr,
                ncl_curr,
                equity_curr,
                income["ni_curr"],
                financing["dividends_prev"],
            ],
            axis=1,
        )

        diagnostics = tf.stack(
            [
                total_assets,
                assets["nca_curr"],
                assets["adv_pp_curr"],
                assets["ar_curr"],
                assets["inv_curr"],
                assets["cash_curr"],
                assets["ims_curr"],
                ap_curr,
                financing["adv_ps_curr"],
                financing["eff_st_debt_curr"],
                cur_lt_debt_curr,
                ncl_curr,
                equity_curr,
                income["ni_curr"],
                assets["depreciation"],
                income["cogs"],
                income["opex"],
                income["tax"],
                income["ms_return"],
                income["interest_lt"] + income["interest_st"],
                financing["dividends_prev"],
                financing["stock_buyback"],
                financing["new_lt_loan"],
                financing["equity_financing"],
                financing["liquidity_deficit_st"],
                financing["liquidity_check"],
                check,
            ],
            axis=1,
        )

        return new_state, diagnostics
