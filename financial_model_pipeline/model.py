"""Trainable financial model definition and training logic."""

import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from datetime import datetime

from .io_utils import TRAINING_RESULTS_DIR, _get_training_results_path

tfd = tfp.distributions
tfb = tfp.bijectors


def _as_float64_tensor(value):
    """Convert a value to a float64 TensorFlow tensor."""
    return tf.convert_to_tensor(value, dtype=tf.float64)


class TrainableFinancialModel(tf.Module):
    def __init__(self, base_year=2018):
        self.amount_scale = 1.0e11
        self.base_year = base_year  # t=0 corresponds to this fiscal year

        # --- Policy Parameters (Deterministic) ---
        ## These are trainable with simple linear regression
        ## Non-negative params use Softplus bijector (constraint by construction)
        ## [0,1]-bounded params use Sigmoid bijector
        self.asset_growth = tfp.util.TransformedVariable(
            initial_value=0.0076,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="asset_growth",
        )  # %AG
        self.asset_maintain = tfp.util.TransformedVariable(
            initial_value=1.0,
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
        # self.tl_baseline = tfp.util.TransformedVariable(
        #     initial_value=0.0,
        #     bijector=tfb.Softplus(),
        #     dtype=tf.float64,
        #     name="tl_baseline",
        # )

        # --- Cash % of Liquidity Logit-Linear Model ---
        # %Cash(t) = sigmoid(cash_alpha + cash_beta * t), where t = year - base_year
        # Cash_t = TL_t * %Cash(t)
        # Sigmoid ensures %Cash stays in (0, 1), so IMS = TL - Cash >= 0.
        # Initialize alpha to inverse_sigmoid(0.487) ≈ -0.05
        self.cash_alpha = tf.Variable(-0.05, dtype=tf.float64, name="cash_alpha")
        self.cash_beta = tf.Variable(0.0, dtype=tf.float64, name="cash_beta")
        self.income_tax_pct = tfp.util.TransformedVariable(
            initial_value=0.147,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="tax_pct",
        )  # %IT
        self.dividend_payout_ratio_pct = tfp.util.TransformedVariable(
            initial_value=0.15,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="div_pct",
        )  # %PR
        # --- Dividend Smoothing (Lintner Model) ---
        # D_t = α * (PayoutRatio * NI_t) + (1 - α) * D_{t-1}
        # α=1.0 → pure payout ratio (no smoothing), α=0.0 → constant dividends
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
        # logit(CR_t) = alpha + beta * t  =>  CR_t = sigmoid(alpha + beta * t), where t = year - base_year
        # Purchases are derived: P_t = Sales_t * CR_t + (Inv_target_t - Inv_{t-1})
        # COGS simplifies to: Sales_t * CR_t
        self.cost_ratio_alpha = tf.Variable(
            0.35, dtype=tf.float64, name="cost_ratio_alpha"
        )
        self.cost_ratio_beta = tf.Variable(
            -0.05, dtype=tf.float64, name="cost_ratio_beta"
        )

        # --- BAYESIAN OPEX PARAMETERS (Variational Inference) ---
        # We learn a distribution (Normal) defined by a Mean (loc) and StdDev (scale)

        # 1. Variable OpEx %
        # self.q_var_opex_loc = tf.Variable(0.22, dtype=tf.float64, name="q_var_opex_loc")
        self.q_var_opex_loc = tf.Variable(0.0, dtype=tf.float64, name="q_var_opex_loc")
        self.q_var_opex_scale = tfp.util.TransformedVariable(
            # initial_value=0.01,
            initial_value=1.0,
            bijector=tfb.Softplus(),  # Ensures scale is always positive
            dtype=tf.float64,
            name="q_var_opex_scale",
        )

        # 2. Baseline OpEx (Large negative number)
        self.q_base_opex_loc = tf.Variable(
            # -3.0e10 / self.amount_scale,
            0.0,
            dtype=tf.float64,
            name="q_base_opex_loc",
        )
        self.q_base_opex_scale = tfp.util.TransformedVariable(
            # initial_value=1.0e9 / self.amount_scale,
            initial_value=1.0,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="q_base_opex_scale",
        )

        # 3. Aleatoric Uncertainty (The inherent noise in the OpEx data)
        self.noise_sigma = tfp.util.TransformedVariable(
            # initial_value=1.0e9 / self.amount_scale,
            initial_value=1.0,  # The initial can be a normal distribution of mean 0, sigma 1 because the training data are scaled to be around the order of 1.
            # initial_value=1.0e9,  # assuming a big sigma, i.e., uniform distribution, does not lead to convergence since the likelihood will always stay constant.
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="noise_sigma",
        )

        # 4. Sales Offset (for centering sales data during OpEx training)
        self.sales_offset = tf.Variable(
            0.0,
            dtype=tf.float64,
            name="sales_offset",
            trainable=False,  # This is not trained, just stored for reference
        )

        # --- Structural Parameters ---
        ## These are trained with gradient descent with the trained variables from above and other data (sales, purchases, equity, liabilities, etc.) as inputs
        ## Non-negative params use Softplus, [0,1]-bounded use Sigmoid,
        ## avg_maturity_years uses Shift(1.001) + Softplus to enforce > 1.001
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
        # %STDebt(t) = sigmoid(st_debt_alpha + st_debt_beta * t), where t = year - base_year
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

    def save_parameters(self, path):
        params = {
            "asset_growth": float(self.asset_growth.numpy()),
            "asset_maintain": float(self.asset_maintain.numpy()),
            "depreciation_rate": float(self.depreciation_rate.numpy()),
            "advance_payments_sales_pct": float(
                self.advance_payments_sales_pct.numpy()
            ),
            "advance_payments_purchases_pct": float(
                self.advance_payments_purchases_pct.numpy()
            ),
            "account_receivables_pct": float(self.account_receivables_pct.numpy()),
            "account_payables_pct": float(self.account_payables_pct.numpy()),
            "inventory_pct": float(self.inventory_pct.numpy()),
            "tl_alpha": float(self.tl_alpha.numpy()),
            "tl_beta": float(self.tl_beta.numpy()),
            "tl_baseline": float(self.tl_baseline.numpy()),
            "cash_alpha": float(self.cash_alpha.numpy()),
            "cash_beta": float(self.cash_beta.numpy()),
            "income_tax_pct": float(self.income_tax_pct.numpy()),
            "dividend_payout_ratio_pct": float(self.dividend_payout_ratio_pct.numpy()),
            "dividend_adjustment_speed": float(self.dividend_adjustment_speed.numpy()),
            "sb_baseline": float(self.sb_baseline.numpy()),
            "sb_ratio": float(self.sb_ratio.numpy()),
            "q_var_opex_loc": float(self.q_var_opex_loc.numpy()),
            "q_var_opex_scale": float(self.q_var_opex_scale.numpy()),
            "q_base_opex_loc": float(self.q_base_opex_loc.numpy()),
            "q_base_opex_scale": float(self.q_base_opex_scale.numpy()),
            "noise_sigma": float(self.noise_sigma.numpy()),
            "sales_offset": float(self.sales_offset.numpy()),
            "avg_short_term_interest_pct": float(
                self.avg_short_term_interest_pct.numpy()
            ),
            "avg_long_term_interest_pct": float(
                self.avg_long_term_interest_pct.numpy()
            ),
            "avg_maturity_years": float(self.avg_maturity_years.numpy()),
            "market_securities_return_pct": float(
                self.market_securities_return_pct.numpy()
            ),
            "ef_alpha": float(self.ef_alpha.numpy()),
            "ef_beta": float(self.ef_beta.numpy()),
            "st_debt_alpha": float(self.st_debt_alpha.numpy()),
            "st_debt_beta": float(self.st_debt_beta.numpy()),
            "cost_ratio_alpha": float(self.cost_ratio_alpha.numpy()),
            "cost_ratio_beta": float(self.cost_ratio_beta.numpy()),
            "base_year": self.base_year,
        }
        np.savez(path, **params)

    def load_parameters(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parameter file not found: {path}")
        data = np.load(path)
        self.asset_growth.assign(data["asset_growth"])
        self.asset_maintain.assign(data["asset_maintain"])
        self.depreciation_rate.assign(data["depreciation_rate"])
        self.advance_payments_sales_pct.assign(data["advance_payments_sales_pct"])
        self.advance_payments_purchases_pct.assign(
            data["advance_payments_purchases_pct"]
        )
        self.account_receivables_pct.assign(data["account_receivables_pct"])
        self.account_payables_pct.assign(data["account_payables_pct"])
        self.inventory_pct.assign(data["inventory_pct"])
        self.tl_alpha.assign(data["tl_alpha"])
        self.tl_beta.assign(data["tl_beta"])
        self.tl_baseline.assign(data.get("tl_baseline", 0.0))
        self.cash_alpha.assign(data["cash_alpha"])
        self.cash_beta.assign(data["cash_beta"])
        self.income_tax_pct.assign(data["income_tax_pct"])
        self.dividend_payout_ratio_pct.assign(data["dividend_payout_ratio_pct"])
        self.dividend_adjustment_speed.assign(
            data.get("dividend_adjustment_speed", 1.0)
        )
        self.sb_baseline.assign(data.get("sb_baseline", 0.0))
        self.sb_ratio.assign(data.get("sb_ratio", 1.0))
        self.q_var_opex_loc.assign(data["q_var_opex_loc"])
        self.q_var_opex_scale.assign(data["q_var_opex_scale"])
        self.q_base_opex_loc.assign(data["q_base_opex_loc"])
        self.q_base_opex_scale.assign(data["q_base_opex_scale"])
        self.noise_sigma.assign(data["noise_sigma"])
        self.sales_offset.assign(data["sales_offset"])
        self.avg_short_term_interest_pct.assign(data["avg_short_term_interest_pct"])
        self.avg_long_term_interest_pct.assign(data["avg_long_term_interest_pct"])
        self.avg_maturity_years.assign(data["avg_maturity_years"])
        self.market_securities_return_pct.assign(data["market_securities_return_pct"])
        self.ef_alpha.assign(data["ef_alpha"])
        self.ef_beta.assign(data["ef_beta"])
        self.st_debt_alpha.assign(data.get("st_debt_alpha", -1.59))
        self.st_debt_beta.assign(data.get("st_debt_beta", 0.0))
        self.cost_ratio_alpha.assign(data["cost_ratio_alpha"])
        self.cost_ratio_beta.assign(data["cost_ratio_beta"])
        if "base_year" in data:
            self.base_year = int(data["base_year"])

    def sample_opex_params(self):
        """Samples from the variational posterior using Reparameterization Trick"""
        # Create distributions
        q_var_dist = tfd.Normal(loc=self.q_var_opex_loc, scale=self.q_var_opex_scale)
        q_base_dist = tfd.Normal(loc=self.q_base_opex_loc, scale=self.q_base_opex_scale)

        return q_var_dist.sample(), q_base_dist.sample()

    def get_opex_kl_divergence(self):
        """Calculates KL Divergence between Posterior (q) and Prior (p)"""
        # Define Priors (Fixed beliefs)
        # Prior: Variable OpEx is around 20% with some wiggle room
        # prior_var = tfd.Normal(loc=tf.constant(0.20, dtype=tf.float64), scale=0.1)
        # prior_var = tfd.Normal(loc=tf.constant(0.20, dtype=tf.float64), scale=0.5)

        # Prior: Baseline OpEx is around -30B with large wiggle room
        # prior_base = tfd.Normal(
        #     loc=tf.constant(-3.0e10 / self.amount_scale, dtype=tf.float64),
        #     scale=1.0e10 / self.amount_scale,
        # )

        ## Assuming uniform distribution by setting very wide Gaussian and mean 0 works well for VI inference's prior distributions
        prior_var = tfd.Normal(loc=tf.constant(0.0, dtype=tf.float64), scale=1.0e10)
        prior_base = tfd.Normal(
            loc=tf.constant(0.0, dtype=tf.float64),
            scale=1.0e10,
        )

        # Define Posteriors
        q_var = tfd.Normal(loc=self.q_var_opex_loc, scale=self.q_var_opex_scale)
        q_base = tfd.Normal(loc=self.q_base_opex_loc, scale=self.q_base_opex_scale)

        return tfd.kl_divergence(q_var, prior_var) + tfd.kl_divergence(
            q_base, prior_base
        )

    def train_simple_policies(
        self,
        historical_sales,
        historical_purchases,
        historical_cogs,
        historical_nca,
        historical_depreciation,
        historical_adv_pay_sales,
        historical_adv_pay_purch,
        historical_ar,
        historical_ap,
        historical_inventory,
        historical_cash,
        historical_ims,
        historical_net_income,
        historical_dividends,
        historical_stock_buyback,
        historical_opex,
        historical_tax,
        historical_eff_st_debt,
        historical_tax_onetime_payments=None,
        historical_inflation=None,
        historical_years=None,
        learning_rate=0.001,
        epochs=25000,
        plot_vi=True,
        plot_every=1000,
        show_plot=False,
        prior_strength_asset_maintain=1.0,
        loss_scale_mode="std",
    ):
        """
        Trains simple policy parameters using historical data.
        """

        # Convert inputs to tensors and ensure float64
        sales_tensor = _as_float64_tensor(historical_sales)
        purchases_tensor = _as_float64_tensor(historical_purchases)
        cogs_tensor = _as_float64_tensor(historical_cogs)
        nca_tensor = _as_float64_tensor(historical_nca)
        depr_tensor = _as_float64_tensor(historical_depreciation)
        adv_pay_sales_tensor = _as_float64_tensor(historical_adv_pay_sales)
        adv_pay_purch_tensor = _as_float64_tensor(historical_adv_pay_purch)
        ar_tensor = _as_float64_tensor(historical_ar)
        ap_tensor = _as_float64_tensor(historical_ap)
        inv_tensor = _as_float64_tensor(historical_inventory)
        cash_tensor = _as_float64_tensor(historical_cash)
        ims_tensor = _as_float64_tensor(historical_ims)
        ni_tensor = _as_float64_tensor(historical_net_income)
        div_tensor = _as_float64_tensor(historical_dividends)
        bb_tensor = _as_float64_tensor(historical_stock_buyback)
        opex_tensor = _as_float64_tensor(historical_opex)
        tax_tensor = _as_float64_tensor(historical_tax)
        if historical_tax_onetime_payments is None:
            tax_onetime_tensor = tf.zeros_like(tax_tensor)
        else:
            tax_onetime_tensor = _as_float64_tensor(historical_tax_onetime_payments)
        eff_st_debt_tensor = _as_float64_tensor(historical_eff_st_debt)

        if historical_inflation is None:
            historical_inflation = tf.zeros_like(sales_tensor)
        inf_tensor = _as_float64_tensor(historical_inflation)
        cum_inf_tensor = tf.math.cumprod(1 + inf_tensor)

        # --- Calculate and store sales offset for OpEx training ---
        # This centers the sales data around 0 for better numerical stability
        sales_offset_value = tf.reduce_mean(sales_tensor)
        self.sales_offset.assign(sales_offset_value)
        sales_tensor_centered = sales_tensor - sales_offset_value

        print(f"Sales offset for OpEx training: {sales_offset_value.numpy():.4e}")
        print(
            f"Sales range before centering: [{tf.reduce_min(sales_tensor).numpy():.4e}, {tf.reduce_max(sales_tensor).numpy():.4e}]"
        )
        print(
            f"Sales range after centering: [{tf.reduce_min(sales_tensor_centered).numpy():.4e}, {tf.reduce_max(sales_tensor_centered).numpy():.4e}]"
        )

        # --- Cost Ratio Training Data ---
        # Compute logit(CR) targets from historical COGS/Sales
        cost_ratio_hist = cogs_tensor / sales_tensor
        logit_cr_hist = tf.math.log(cost_ratio_hist / (1.0 - cost_ratio_hist))
        # Time indices: t = year - base_year (e.g., FY2018 -> 0, FY2019 -> 1, ...)
        if historical_years is not None:
            time_indices = tf.cast(historical_years, dtype=tf.float64) - tf.constant(
                float(self.base_year), dtype=tf.float64
            )
        else:
            time_indices = tf.cast(tf.range(len(historical_sales)), dtype=tf.float64)

        # --- Prepare Training Data & Alignment ---

        # 1. Asset Growth: (NCA_t - NCA_{t-1}) = sales_t * asset_growth
        delta_nca_true = nca_tensor[1:] - nca_tensor[:-1]
        sales_aligned_growth = sales_tensor[1:]

        # 2. Depreciation: depr_t = nca_{t-1} * depr_rate
        depr_true = depr_tensor[1:]
        nca_prev_aligned = nca_tensor[:-1]

        # 3. Advance Payments Sales: adv_ps_t = sales_t * adv_ps_pct
        adv_ps_true = adv_pay_sales_tensor

        # 4. Advance Payments Purchases: adv_pp_t = purchases_t * adv_pp_pct
        adv_pp_true = adv_pay_purch_tensor
        purchases_aligned_adv_pp = purchases_tensor

        # 5. Dividends (Lintner Smoothing):
        #    D_t = α * (NI_{t-1} * PayoutRatio) + (1 - α) * D_{t-1}
        div_true = div_tensor[1:]
        ni_prev_aligned = ni_tensor[:-1]
        div_prev_aligned = div_tensor[:-1]

        # --- Loss Scaling (robustness across heterogeneous magnitudes) ---
        eps = tf.constant(1e-12, dtype=tf.float64)
        if loss_scale_mode == "std":
            scale_growth = tf.math.reduce_std(delta_nca_true) + eps
            scale_depr = tf.math.reduce_std(depr_true) + eps
            scale_adv_ps = tf.math.reduce_std(adv_ps_true) + eps
            scale_adv_pp = tf.math.reduce_std(adv_pp_true) + eps
            scale_ar = tf.math.reduce_std(ar_tensor) + eps
            scale_ap = tf.math.reduce_std(ap_tensor) + eps
            scale_inv = tf.math.reduce_std(inv_tensor) + eps
            scale_tl = tf.math.reduce_std(cash_tensor + ims_tensor) + eps
            scale_cash = tf.math.reduce_std(cash_tensor) + eps
            scale_tax = tf.math.reduce_std(tax_tensor) + eps
            scale_div = tf.math.reduce_std(div_true) + eps
            scale_bb = tf.math.reduce_std(bb_tensor) + eps
            scale_cost_ratio = tf.math.reduce_std(logit_cr_hist) + eps
            scale_eff_st = tf.math.reduce_std(eff_st_debt_tensor) + eps
            scale_opex = tf.math.reduce_std(opex_tensor) + eps
        elif loss_scale_mode == "none":
            one = tf.constant(1.0, dtype=tf.float64)
            scale_growth = one
            scale_depr = one
            scale_adv_ps = one
            scale_adv_pp = one
            scale_ar = one
            scale_ap = one
            scale_inv = one
            scale_tl = one
            scale_cash = one
            scale_tax = one
            scale_div = one
            scale_bb = one
            scale_cost_ratio = one
            scale_eff_st = one
            scale_opex = one
        else:
            raise ValueError(
                f"Unsupported loss_scale_mode='{loss_scale_mode}'. Use 'std' or 'none'."
            )
        num_opex_obs = tf.cast(tf.size(opex_tensor), tf.float64)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        print(f"Training on {len(historical_sales)} years of historical data...")

        # --- Training Loop ---
        vars_to_train = [
            # Policy params (unconstrained underlying variables via bijectors)
            self.asset_growth.trainable_variables[0],
            self.asset_maintain.trainable_variables[0],
            self.depreciation_rate.trainable_variables[0],
            self.advance_payments_sales_pct.trainable_variables[0],
            self.advance_payments_purchases_pct.trainable_variables[0],
            self.account_receivables_pct.trainable_variables[0],
            self.account_payables_pct.trainable_variables[0],
            self.inventory_pct.trainable_variables[0],
            self.tl_alpha,
            self.tl_beta,
            self.tl_baseline,
            # self.tl_baseline.trainable_variables[0],
            self.cash_alpha,
            self.cash_beta,
            self.income_tax_pct.trainable_variables[0],
            self.dividend_payout_ratio_pct.trainable_variables[0],
            self.dividend_adjustment_speed.trainable_variables[0],
            self.sb_baseline,
            self.sb_ratio,
            # ST Debt Params (Logit-Linear) — include only if data exists
            self.st_debt_alpha,
            self.st_debt_beta,
            # Cost Ratio Params (Logit-Linear)
            self.cost_ratio_alpha,
            self.cost_ratio_beta,
            # Bayesian Params
            self.q_var_opex_loc,
            self.q_var_opex_scale.trainable_variables[0],
            self.q_base_opex_loc,
            self.q_base_opex_scale.trainable_variables[0],
            self.noise_sigma.trainable_variables[0],
        ]

        vi_history = {
            "epochs": [],
            "loss_vi": [],
            "q_var_opex_loc": [],
            "q_var_opex_scale": [],
            "q_base_opex_loc": [],
            "q_base_opex_scale": [],
            "noise_sigma": [],
        }

        simple_history = {
            "epochs": [],
            "loss_total": [],
            "loss_growth": [],
            "loss_depr": [],
            "loss_adv_ps": [],
            "loss_adv_pp": [],
            "loss_ar": [],
            "loss_ap": [],
            "loss_inv": [],
            "loss_tl": [],
            "loss_cash": [],
            "loss_tax": [],
            "loss_div": [],
            "loss_bb": [],
            "loss_cost_ratio": [],
            "loss_eff_st_debt": [],
            "loss_prior_am": [],
        }

        for i in range(epochs):
            with tf.GradientTape() as tape:
                # --- Deterministic Losses (MSE) ---
                loss_growth = tf.reduce_mean(
                    tf.square(
                        (
                            delta_nca_true
                            - (
                                (self.asset_maintain - 1) * depr_true
                                + sales_aligned_growth * self.asset_growth
                            )
                        )
                        / scale_growth
                    )
                )
                loss_depr = tf.reduce_mean(
                    tf.square(
                        (depr_true - nca_prev_aligned * self.depreciation_rate)
                        / scale_depr
                    )
                )
                loss_adv_ps = tf.reduce_mean(
                    tf.square(
                        (adv_ps_true - sales_tensor * self.advance_payments_sales_pct)
                        / scale_adv_ps
                    )
                )
                loss_adv_pp = tf.reduce_mean(
                    tf.square(
                        (
                            adv_pp_true
                            - purchases_aligned_adv_pp
                            * self.advance_payments_purchases_pct
                        )
                        / scale_adv_pp
                    )
                )
                loss_ar = tf.reduce_mean(
                    tf.square(
                        (ar_tensor - sales_tensor * self.account_receivables_pct)
                        / scale_ar
                    )
                )
                loss_ap = tf.reduce_mean(
                    tf.square(
                        (ap_tensor - purchases_tensor * self.account_payables_pct)
                        / scale_ap
                    )
                )
                loss_inv = tf.reduce_mean(
                    tf.square(
                        (inv_tensor - sales_tensor * self.inventory_pct) / scale_inv
                    )
                )
                # TL(t) = tl_baseline + sales_t * sigmoid(tl_alpha + tl_beta * t)
                tl_pct_t_logit = self.tl_alpha + self.tl_beta * time_indices
                tl_pct_t = tf.sigmoid(tl_pct_t_logit)
                loss_tl = tf.reduce_mean(
                    tf.square(
                        (
                            (cash_tensor + ims_tensor)
                            - (self.tl_baseline + sales_tensor * tl_pct_t)
                        )
                        / scale_tl
                    )
                )
                # %Cash(t) = sigmoid(cash_alpha + cash_beta * t) (logit-linear)
                cash_pct_t_logit = self.cash_alpha + self.cash_beta * time_indices
                cash_pct_t = tf.sigmoid(cash_pct_t_logit)
                loss_cash = tf.reduce_mean(
                    tf.square(
                        (cash_tensor - (cash_tensor + ims_tensor) * cash_pct_t)
                        / scale_cash
                    )
                )
                # Tax-loss compares against total tax (baseline model tax + one-time tax).
                tax_pred_total = (
                    ni_tensor / (1 / self.income_tax_pct - 1) + tax_onetime_tensor
                )
                loss_tax = tf.reduce_mean(
                    tf.square((tax_tensor - tax_pred_total) / scale_tax)
                )
                div_target = ni_prev_aligned * self.dividend_payout_ratio_pct
                div_pred = (
                    self.dividend_adjustment_speed * div_target
                    + (1.0 - self.dividend_adjustment_speed) * div_prev_aligned
                )
                loss_div = tf.reduce_mean(tf.square((div_true - div_pred) / scale_div))
                bb_pred = self.sb_baseline + self.sb_ratio * depr_tensor
                loss_bb = tf.reduce_mean(tf.square((bb_tensor - bb_pred) / scale_bb))

                # --- Cost Ratio Loss (Logit-Linear) ---
                # logit(CR_t) = alpha + beta * t
                logit_cr_pred = (
                    self.cost_ratio_alpha + self.cost_ratio_beta * time_indices
                )
                loss_cost_ratio = tf.reduce_mean(
                    tf.square((logit_cr_hist - logit_cr_pred) / scale_cost_ratio)
                )

                # --- ST Debt Loss (Logit-Linear) ---
                # %STDebt(t) = sigmoid(st_debt_alpha + st_debt_beta * t)
                # new_short_term_loan = sales * %STDebt(t)
                st_debt_pct_pred = tf.sigmoid(
                    self.st_debt_alpha + self.st_debt_beta * time_indices
                )
                loss_eff_st_debt = tf.reduce_mean(
                    tf.square(
                        (eff_st_debt_tensor - sales_tensor * st_debt_pct_pred)
                        / scale_eff_st
                    )
                )

                # --- Bayesian OpEx Loss ---
                # 1. Sample parameters
                var_opex_sample, base_opex_sample = self.sample_opex_params()

                # 2. Calculate the Raw Prediction (in Billion Dollars)
                # Use centered sales for training to make spread symmetric around y-axis
                pred_opex_raw = (base_opex_sample * cum_inf_tensor) + (
                    var_opex_sample * sales_tensor_centered
                )

                # 3. Calculate Residuals (The Error)
                residuals = (opex_tensor - pred_opex_raw) / scale_opex

                # 4. Calculate Likelihood
                likelihood_dist = tfd.Normal(
                    loc=0.0, scale=(self.noise_sigma / scale_opex)
                )
                neg_log_likelihood = -tf.reduce_sum(likelihood_dist.log_prob(residuals))

                # 5. KL Divergence
                kl = self.get_opex_kl_divergence()

                # 6. Final Sum (normalized by number of observations)
                loss_opex_bayes = (neg_log_likelihood + kl) / num_opex_obs

                # --- Prior / Regularization Losses ---
                # Quadratic prior on asset_maintain centered at 1.0:
                # Economically, asset_maintain ≈ 1.0 means capex fully replaces
                # depreciation (maintenance capex), with asset_growth capturing
                # incremental growth capex on top. Without this prior, the
                # optimizer can collapse asset_maintain → 0 and absorb everything
                # into asset_growth, which is economically implausible.
                prior_loss_am = prior_strength_asset_maintain * tf.square(
                    self.asset_maintain - 1.0
                )

                # --- Combined Loss (Heuristic: Normalize by scale to help Adam) ---
                # But for simplicity, we'll just sum them up for now.
                total_loss = (
                    loss_growth
                    + loss_depr
                    + loss_adv_ps
                    + loss_adv_pp
                    + loss_ar
                    + loss_ap
                    + loss_inv
                    + loss_tl
                    + loss_cash
                    + loss_tax
                    + loss_div
                    + loss_bb
                    + loss_cost_ratio
                    + loss_eff_st_debt
                    + loss_opex_bayes
                    + prior_loss_am
                )

            # Compute Gradients
            grads = tape.gradient(total_loss, vars_to_train)

            # Apply Gradients
            optimizer.apply_gradients(zip(grads, vars_to_train))

            if i % plot_every == 0:
                vi_history["epochs"].append(i)
                vi_history["loss_vi"].append(loss_opex_bayes.numpy())
                vi_history["q_var_opex_loc"].append(self.q_var_opex_loc.numpy())
                vi_history["q_var_opex_scale"].append(self.q_var_opex_scale.numpy())
                vi_history["q_base_opex_loc"].append(self.q_base_opex_loc.numpy())
                vi_history["q_base_opex_scale"].append(self.q_base_opex_scale.numpy())
                vi_history["noise_sigma"].append(self.noise_sigma.numpy())

                simple_history["epochs"].append(i)
                simple_history["loss_total"].append(total_loss.numpy())
                simple_history["loss_growth"].append(loss_growth.numpy())
                simple_history["loss_depr"].append(loss_depr.numpy())
                simple_history["loss_adv_ps"].append(loss_adv_ps.numpy())
                simple_history["loss_adv_pp"].append(loss_adv_pp.numpy())
                simple_history["loss_ar"].append(loss_ar.numpy())
                simple_history["loss_ap"].append(loss_ap.numpy())
                simple_history["loss_inv"].append(loss_inv.numpy())
                simple_history["loss_tl"].append(loss_tl.numpy())
                simple_history["loss_cash"].append(loss_cash.numpy())
                simple_history["loss_tax"].append(loss_tax.numpy())
                simple_history["loss_div"].append(loss_div.numpy())
                simple_history["loss_bb"].append(loss_bb.numpy())
                simple_history["loss_cost_ratio"].append(loss_cost_ratio.numpy())
                simple_history["loss_eff_st_debt"].append(loss_eff_st_debt.numpy())
                simple_history["loss_prior_am"].append(prior_loss_am.numpy())

                print(
                    f"Epoch {i}: Loss={total_loss.numpy():.4e} | "
                    f"OpEx VI Loss={loss_opex_bayes.numpy():.4e} | "
                    f"OpEx Noise={(self.noise_sigma.numpy() * self.amount_scale):.2e} | "
                    f"AM={self.asset_maintain.numpy():.4f} AG={self.asset_growth.numpy():.6f} "
                    f"Prior_AM={prior_loss_am.numpy():.4e}"
                )

        print("-" * 50)
        print("Training Complete.")
        print(f"Final %AG: {self.asset_growth.numpy():.5f}")
        print(f"Final %AM: {self.asset_maintain.numpy():.5f}")
        print(f"Final %Depr: {self.depreciation_rate.numpy():.5f}")
        print(f"Final %AdvPS: {self.advance_payments_sales_pct.numpy():.5f}")
        print(f"Final %AdvPP: {self.advance_payments_purchases_pct.numpy():.5f}")
        print(f"Final %AR: {self.account_receivables_pct.numpy():.5f}")
        print(f"Final %AP: {self.account_payables_pct.numpy():.5f}")
        print(f"Final %Inv: {self.inventory_pct.numpy():.5f}")
        print(
            f"Total Liquidity (baseline + logit-linear): baseline={self.tl_baseline.numpy():.4f}, "
            f"alpha={self.tl_alpha.numpy():.4f}, "
            f"beta={self.tl_beta.numpy():.6f}"
        )
        print(
            f"  => %TL at t=0: {tf.sigmoid(self.tl_alpha).numpy():.4f}, "
            f"%TL at t={len(historical_sales)-1}: "
            f"{tf.sigmoid(self.tl_alpha + self.tl_beta * (len(historical_sales)-1)).numpy():.4f}"
        )
        print(
            f"Cash % of Liquidity (logit-linear): alpha={self.cash_alpha.numpy():.4f}, "
            f"beta={self.cash_beta.numpy():.6f}"
        )
        print(
            f"  => %Cash at t=0: {tf.sigmoid(self.cash_alpha).numpy():.4f}, "
            f"%Cash at t={len(historical_sales)-1}: "
            f"{tf.sigmoid(self.cash_alpha + self.cash_beta * (len(historical_sales)-1)).numpy():.4f}"
        )
        print(f"Final %IT: {self.income_tax_pct.numpy():.5f}")
        print(f"Final %PR: {self.dividend_payout_ratio_pct.numpy():.5f}")
        print(f"Final DivAdjSpeed (α): {self.dividend_adjustment_speed.numpy():.5f}")
        print(
            f"Stock Buyback (baseline + ratio*depr): baseline={self.sb_baseline.numpy():.4f}, "
            f"ratio={self.sb_ratio.numpy():.6f}"
        )
        print(
            f"Effective ST Debt % of Sales (logit-linear): alpha={self.st_debt_alpha.numpy():.4f}, "
            f"beta={self.st_debt_beta.numpy():.6f}"
        )
        print(
            f"  => %EffSTDebt at t=0: {tf.sigmoid(self.st_debt_alpha).numpy():.4f}, "
            f"%EffSTDebt at t={len(historical_sales)-1}: "
            f"{tf.sigmoid(self.st_debt_alpha + self.st_debt_beta * (len(historical_sales)-1)).numpy():.4f}"
        )

        print(
            f"Cost Ratio (logit-linear): alpha={self.cost_ratio_alpha.numpy():.4f}, "
            f"beta={self.cost_ratio_beta.numpy():.4f}"
        )
        print(
            f"  => CR at t=0: {tf.sigmoid(self.cost_ratio_alpha).numpy():.4f}, "
            f"CR at t={len(historical_sales)-1}: "
            f"{tf.sigmoid(self.cost_ratio_alpha + self.cost_ratio_beta * (len(historical_sales)-1)).numpy():.4f}"
        )
        print(
            f"Bayesian OpEx Variable %: Mean={self.q_var_opex_loc.numpy():.4f}, Std={self.q_var_opex_scale.numpy():.4f}"
        )
        print(
            "Bayesian OpEx Baseline (USD):   "
            f"Mean={(self.q_base_opex_loc.numpy() * self.amount_scale):.2e}, "
            f"Std={(self.q_base_opex_scale.numpy() * self.amount_scale):.2e}"
        )
        print(
            "OpEx aleatoric uncertainty (USD): "
            f"{(self.noise_sigma.numpy() * self.amount_scale):.2e}"
        )

        print("-" * 50)

        if plot_vi and vi_history["epochs"]:
            epochs_hist = np.array(vi_history["epochs"])
            fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

            axs[0].plot(
                epochs_hist, vi_history["q_var_opex_loc"], label="q_var_opex_loc"
            )
            axs[0].plot(
                epochs_hist,
                vi_history["q_var_opex_scale"],
                label="q_var_opex_scale",
            )
            axs[0].set_ylabel("Variable OpEx %")
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)

            axs[1].plot(
                epochs_hist,
                np.array(vi_history["q_base_opex_loc"]) * self.amount_scale,
                label="q_base_opex_loc (USD)",
            )
            axs[1].plot(
                epochs_hist,
                np.array(vi_history["q_base_opex_scale"]) * self.amount_scale,
                label="q_base_opex_scale (USD)",
            )
            axs[1].set_ylabel("Baseline OpEx (USD)")
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)

            axs[2].plot(
                epochs_hist,
                np.array(vi_history["noise_sigma"]) * self.amount_scale,
                label="noise_sigma (USD)",
            )
            axs[2].set_ylabel("Noise Sigma (USD)")
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)

            axs[3].plot(epochs_hist, vi_history["loss_vi"], label="Loss_VI")
            axs[3].set_xlabel("Epoch")
            axs[3].set_ylabel("Loss_VI")
            axs[3].legend()
            axs[3].grid(True, alpha=0.3)

            fig.suptitle("Variational Inference Parameters and Loss Over Epochs")
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = _get_training_results_path(
                f"vi_training_diagnostics_{timestamp}.png"
            )
            plt.savefig(plot_path, dpi=150)
            if show_plot:
                plt.show()
            else:
                plt.close()

        # --- Simple Parameters Training Diagnostics ---
        if plot_vi and simple_history["epochs"]:
            epochs_hist = np.array(simple_history["epochs"])
            fig, axs = plt.subplots(4, 1, figsize=(10, 15))

            # Panel 1: Total loss
            axs[0].plot(
                epochs_hist,
                simple_history["loss_total"],
                label="Total Loss",
                color="black",
                linewidth=2,
            )
            axs[0].set_ylabel("Total Loss")
            axs[0].set_yscale("log")
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)

            # Panel 2: Individual simple parameter losses (balance sheet ratios)
            ratio_losses = [
                ("loss_ar", "%AR"),
                ("loss_ap", "%AP"),
                ("loss_inv", "%Inv"),
                ("loss_tl", "%TL"),
                ("loss_cash", "%Cash"),
                ("loss_adv_ps", "%AdvPS"),
                ("loss_adv_pp", "%AdvPP"),
                ("loss_tax", "%IT"),
                ("loss_div", "%PR"),
                ("loss_bb", "%BB"),
                ("loss_eff_st_debt", "%EffSTDebt"),
            ]
            for key, label in ratio_losses:
                axs[1].plot(epochs_hist, simple_history[key], label=label)
            axs[1].set_ylabel("Loss (MSE)")
            axs[1].set_yscale("log")
            axs[1].legend(ncol=3, fontsize=8)
            axs[1].grid(True, alpha=0.3)

            # Panel 3: Asset-related losses + cost ratio + prior
            structural_losses = [
                ("loss_growth", "%AG (Growth)"),
                ("loss_depr", "%Depr"),
                ("loss_cost_ratio", "Cost Ratio"),
                ("loss_prior_am", "Prior AM"),
            ]
            for key, label in structural_losses:
                axs[2].plot(epochs_hist, simple_history[key], label=label)
            axs[2].set_xlabel("Epoch")
            axs[2].set_ylabel("Loss (MSE)")
            axs[2].set_yscale("log")
            axs[2].legend(fontsize=8)
            axs[2].grid(True, alpha=0.3)

            # Panel 4: Final fitted logit(CR) vs historical logit(CR)
            if historical_years is not None:
                cr_x = np.array(historical_years)
                axs[3].set_xlabel("Year")
            else:
                cr_x = np.arange(len(historical_sales))
                axs[3].set_xlabel("Time Index")
            final_logit_cr_pred = (
                self.cost_ratio_alpha + self.cost_ratio_beta * time_indices
            )
            axs[3].plot(cr_x, logit_cr_hist.numpy(), marker="o", label="logit_cr_hist")
            axs[3].plot(
                cr_x,
                final_logit_cr_pred.numpy(),
                marker="x",
                linestyle="--",
                label="logit_cr_pred",
            )
            axs[3].set_ylabel("logit(CR)")
            axs[3].legend(fontsize=8)
            axs[3].grid(True, alpha=0.3)

            fig.suptitle("Simple Parameters Training Diagnostics")
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = _get_training_results_path(
                f"simple_training_diagnostics_{timestamp}.png"
            )
            plt.savefig(plot_path, dpi=150)
            if show_plot:
                plt.show()
            else:
                plt.close()

    def train_structural_parameters(
        self,
        historical_sales,
        historical_nca,
        historical_adv_pay_sales,
        historical_adv_pay_purch,
        historical_ar,
        historical_ap,
        historical_inventory,
        historical_cash,
        historical_ims,
        historical_net_income,
        historical_dividends,
        historical_stock_buyback,
        historical_opex,
        historical_tax,
        historical_effective_st_debt,
        historical_current_lt_debt,
        historical_non_current_liabilities,
        historical_interest_payment,
        historical_ms_return,
        historical_equity,
        historical_tax_onetime_payments=None,
        historical_inflation=None,
        historical_years=None,
        learning_rate=0.001,
        epochs=20000,
        plot_every=1000,
        gradient_clip_norm=5.0,
        show_plot=False,
        loss_scale_mode="std",
    ):
        """
        Trains structural parameters (interest rates, maturity, financing)
        using historical state transitions.
        Purchases are derived inside forecast_step from the learned cost ratio.
        """
        # NOTE: When calling forecast_step inside here, we need to pass use_mean_opex=True
        # because we want to learn structural parameters based on the "most likely" OpEx, not noisy samples.
        sales_t = _as_float64_tensor(historical_sales)
        nca_t = _as_float64_tensor(historical_nca)
        adv_ps_t = _as_float64_tensor(historical_adv_pay_sales)
        adv_pp_t = _as_float64_tensor(historical_adv_pay_purch)
        ar_t = _as_float64_tensor(historical_ar)
        ap_t = _as_float64_tensor(historical_ap)
        inv_t = _as_float64_tensor(historical_inventory)
        cash_t = _as_float64_tensor(historical_cash)
        ims_t = _as_float64_tensor(historical_ims)
        ni_t = _as_float64_tensor(historical_net_income)
        div_t = _as_float64_tensor(historical_dividends)
        eff_st_t = _as_float64_tensor(historical_effective_st_debt)
        curr_lt_t = _as_float64_tensor(historical_current_lt_debt)
        ncl_t = _as_float64_tensor(historical_non_current_liabilities)
        interest_t = _as_float64_tensor(historical_interest_payment)
        ms_return_t = _as_float64_tensor(historical_ms_return)
        equity_t = _as_float64_tensor(historical_equity)
        if historical_tax_onetime_payments is None:
            tax_onetime_t = tf.zeros_like(ni_t)
        else:
            tax_onetime_t = _as_float64_tensor(historical_tax_onetime_payments)

        if historical_inflation is None:
            historical_inflation = tf.zeros_like(sales_t)
        inf_t = _as_float64_tensor(historical_inflation)
        cum_inf_t = tf.math.cumprod(1 + inf_t)

        if historical_years is None:
            historical_years = np.arange(
                self.base_year, self.base_year + len(historical_sales)
            )
        years_t = _as_float64_tensor(historical_years)
        time_idx_t = years_t - tf.constant(float(self.base_year), dtype=tf.float64)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        eps = tf.constant(1e-12, dtype=tf.float64)

        def finite_std(values):
            """Std over finite values only; fallback to 1.0 if insufficient data."""
            finite_values = tf.boolean_mask(values, tf.math.is_finite(values))
            return tf.cond(
                tf.size(finite_values) > 1,
                lambda: tf.math.reduce_std(finite_values) + eps,
                lambda: tf.constant(1.0, dtype=tf.float64),
            )

        if loss_scale_mode == "std":
            scale_ni = tf.math.reduce_std(ni_t[1:]) + eps
            scale_eff_st = tf.math.reduce_std(eff_st_t[1:]) + eps
            scale_curr_lt = tf.math.reduce_std(curr_lt_t[1:]) + eps
            scale_ncl = tf.math.reduce_std(ncl_t[1:]) + eps
            scale_equity = tf.math.reduce_std(equity_t[1:]) + eps
            scale_interest = finite_std(interest_t[1:])
            scale_ms_return = tf.math.reduce_std(ms_return_t[1:]) + eps
        elif loss_scale_mode == "none":
            one = tf.constant(1.0, dtype=tf.float64)
            scale_ni = one
            scale_eff_st = one
            scale_curr_lt = one
            scale_ncl = one
            scale_equity = one
            scale_interest = one
            scale_ms_return = one
        else:
            raise ValueError(
                f"Unsupported loss_scale_mode='{loss_scale_mode}'. Use 'std' or 'none'."
            )

        vars_to_train = [
            self.avg_short_term_interest_pct.trainable_variables[0],
            self.avg_long_term_interest_pct.trainable_variables[0],
            self.avg_maturity_years.trainable_variables[0],
            self.market_securities_return_pct.trainable_variables[0],
            self.ef_alpha,
            self.ef_beta,
        ]

        structural_history = {
            "epochs": [],
            "loss_total": [],
            "loss_ni": [],
            # "loss_cl": [],
            "loss_interest": [],
            "loss_ms_return": [],
            "loss_curr_lt": [],
            "loss_ncl": [],
            "loss_equity": [],
        }

        print(f"Training structural parameters...")
        for i in range(epochs):
            with tf.GradientTape() as tape:
                total_loss = 0.0
                total_loss_ni = 0.0
                # total_loss_cl = 0.0
                total_loss_interest = 0.0
                total_loss_ms_return = 0.0
                total_loss_curr_lt = 0.0
                total_loss_ncl = 0.0
                total_loss_equity = 0.0
                num_transitions = len(historical_sales) - 1

                for t in range(num_transitions):
                    # State at t
                    state_prev = {
                        "nca": nca_t[t],
                        "advance_payments_purchases": adv_pp_t[t],
                        "accounts_receivable": ar_t[t],
                        "inventory": inv_t[t],
                        "cash": cash_t[t],
                        "investment_in_market_securities": ims_t[t],
                        "accounts_payable": ap_t[t],
                        "advance_payments_sales": adv_ps_t[t],
                        "effective_st_debt": eff_st_t[t],
                        "current_lt_debt": curr_lt_t[t],
                        "non_current_liabilities": ncl_t[t],
                        "equity": equity_t[t],
                        "net_income": ni_t[t],
                        "dividends": div_t[t],
                    }

                    # Inputs for predicting state at t+1
                    # Purchases are derived inside forecast_step from cost ratio
                    inputs_curr = {
                        "sales_t": sales_t[t + 1],
                        "year": years_t[t + 1],
                        "cum_inflation": cum_inf_t[t + 1],
                        "tax_onetime_payment": tax_onetime_t[t + 1],
                    }

                    # IMPORTANT: Use mean (deterministic) OpEx for structural training
                    state_pred = self.forecast_step(
                        state_prev,
                        inputs_curr,
                        use_mean_opex=True,
                    )

                    # Targets are values at t+1
                    loss_ni = tf.square(
                        (state_pred["net_income"] - ni_t[t + 1]) / scale_ni
                    )
                    loss_eff_st = tf.square(
                        (state_pred["effective_st_debt"] - eff_st_t[t + 1])
                        / scale_eff_st
                    )
                    loss_curr_lt = tf.square(
                        (state_pred["current_lt_debt"] - curr_lt_t[t + 1])
                        / scale_curr_lt
                    )
                    loss_ncl = tf.square(
                        (state_pred["non_current_liabilities"] - ncl_t[t + 1])
                        / scale_ncl
                    )
                    loss_equity = tf.square(
                        (state_pred["equity"] - equity_t[t + 1]) / scale_equity
                    )
                    valid_interest = tf.cast(
                        tf.math.is_finite(interest_t[t + 1]), tf.float64
                    )
                    # Replace missing target with prediction itself so residual=0
                    # and no arithmetic ever involves NaN.
                    interest_target = tf.where(
                        tf.math.is_finite(interest_t[t + 1]),
                        interest_t[t + 1],
                        state_pred["interest_payment"],
                    )
                    loss_interest = valid_interest * tf.square(
                        (state_pred["interest_payment"] - interest_target)
                        / scale_interest
                    )
                    loss_ms_return = tf.square(
                        (state_pred["ms_return"] - ms_return_t[t + 1]) / scale_ms_return
                    )

                    # Total loss to minimize
                    total_loss += (
                        loss_ni
                        + loss_eff_st
                        + loss_curr_lt
                        + loss_ncl
                        + loss_equity
                        + loss_interest
                        + loss_ms_return
                    )
                    total_loss_ni += loss_ni
                    # total_loss_cl += loss_cl
                    total_loss_curr_lt += loss_curr_lt
                    total_loss_ncl += loss_ncl
                    total_loss_equity += loss_equity
                    total_loss_interest += loss_interest
                    total_loss_ms_return += loss_ms_return

            grads = tape.gradient(total_loss, vars_to_train)
            if gradient_clip_norm is not None and gradient_clip_norm > 0:
                grads = [
                    None if g is None else tf.clip_by_norm(g, gradient_clip_norm)
                    for g in grads
                ]
            optimizer.apply_gradients(zip(grads, vars_to_train))

            if i % plot_every == 0:
                structural_history["epochs"].append(i)
                structural_history["loss_total"].append(total_loss.numpy())
                structural_history["loss_ni"].append(total_loss_ni.numpy())
                # structural_history["loss_cl"].append(total_loss_cl.numpy())
                structural_history["loss_interest"].append(total_loss_interest.numpy())
                structural_history["loss_ms_return"].append(
                    total_loss_ms_return.numpy()
                )
                structural_history["loss_curr_lt"].append(total_loss_curr_lt.numpy())
                structural_history["loss_ncl"].append(total_loss_ncl.numpy())
                structural_history["loss_equity"].append(total_loss_equity.numpy())

            if i % 1000 == 0:
                print(f"Epoch {i}: Structural Loss={total_loss.numpy():.4e}")

        print("Structural Training Complete.")
        print(f"Final %AvgSTInt: {self.avg_short_term_interest_pct.numpy():.5f}")
        print(f"Final %AvgLTInt: {self.avg_long_term_interest_pct.numpy():.5f}")
        print(f"Final AvgM: {self.avg_maturity_years.numpy():.5f}")
        print(f"Final %MSReturn: {self.market_securities_return_pct.numpy():.5f}")
        print(
            f"Equity Financing % (logit-linear): alpha={self.ef_alpha.numpy():.4f}, "
            f"beta={self.ef_beta.numpy():.6f}"
        )
        print(
            f"  => %EF at t=0: {tf.sigmoid(self.ef_alpha).numpy():.4f}, "
            f"%EF at t={num_transitions}: "
            f"{tf.sigmoid(self.ef_alpha + self.ef_beta * num_transitions).numpy():.4f}"
        )
        print("-" * 50)

        # --- Structural Parameters Training Diagnostics ---
        if structural_history["epochs"]:
            epochs_hist = np.array(structural_history["epochs"])
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Panel 1: Total loss
            axs[0].plot(
                epochs_hist,
                structural_history["loss_total"],
                label="Total Loss",
                color="black",
                linewidth=2,
            )
            axs[0].set_ylabel("Total Loss")
            axs[0].set_yscale("log")
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)

            # Panel 2: Component losses
            component_losses = [
                ("loss_ni", "Net Income"),
                ("loss_interest", "Interest Payment"),
                ("loss_ms_return", "Return on Market Securities Investment"),
                ("loss_curr_lt", "Current LT Debt"),
                ("loss_ncl", "Non-Current Liabilities"),
                ("loss_equity", "Equity"),
            ]
            for key, label in component_losses:
                axs[1].plot(epochs_hist, structural_history[key], label=label)
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Loss (SSE)")
            axs[1].set_yscale("log")
            axs[1].legend(fontsize=9)
            axs[1].grid(True, alpha=0.3)

            fig.suptitle("Structural Parameters Training Diagnostics")
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = _get_training_results_path(
                f"structural_training_diagnostics_{timestamp}.png"
            )
            plt.savefig(plot_path, dpi=150)
            if show_plot:
                plt.show()
            else:
                plt.close()

    def forecast_step(
        self,
        state,
        inputs,
        use_mean_opex=False,
        sampled_var_opex=None,
        sampled_base_opex=None,
    ):
        """
        Calculate t based on t-1 state and t inputs.
        Implements the logic of Pareja (09) Cash Budget construction.
        """
        # Unpack previous state (t-1)
        ## Assets
        nca_prev = state["nca"]  # Non-current assets
        advance_payments_purchases_prev = state["advance_payments_purchases"]
        accounts_receivable_prev = state["accounts_receivable"]
        inventory_prev = state["inventory"]
        cash_prev = state["cash"]
        investment_in_market_securities_prev = state["investment_in_market_securities"]

        ## Liabilities and Equity
        accounts_payable_prev = state["accounts_payable"]
        advance_payments_sales_prev = state["advance_payments_sales"]
        effective_st_debt_prev = state["effective_st_debt"]
        current_lt_debt_prev = state["current_lt_debt"]
        non_current_liabilities_prev = state["non_current_liabilities"]

        ## Equity
        equity_prev = state["equity"]
        ## Net Income
        net_income_prev = state["net_income"]
        ## Dividends (previous period, for Lintner smoothing)
        dividends_prev_actual = state["dividends"]

        # Unpack current inputs (t)
        sales_t = inputs["sales_t"]
        year = inputs["year"]
        cum_inflation = inputs["cum_inflation"]

        # Convert year to 0-based time index for trend models (e.g., FY2018 -> 0)
        time_index = year - tf.constant(float(self.base_year), dtype=tf.float64)

        depreciation = nca_prev * self.depreciation_rate
        stock_buyback = self.sb_baseline + self.sb_ratio * depreciation
        # Lintner dividend smoothing: D_t = α * (PR * NI_{t-1}) + (1-α) * D_{t-1}
        dividend_target = net_income_prev * self.dividend_payout_ratio_pct
        dividends_prev = (
            self.dividend_adjustment_speed * dividend_target
            + (1.0 - self.dividend_adjustment_speed) * dividends_prev_actual
        )

        # --- Derive purchases from cost ratio (Logit-Linear Model) ---
        # CR_t = sigmoid(alpha + beta * t)
        cost_ratio_t = tf.sigmoid(
            self.cost_ratio_alpha + self.cost_ratio_beta * time_index
        )

        # --- 1. Assets Evolution ---
        # 1.1. Non-current Assets (NCA)
        # Policy: Maintain NCA + Growth (Simplified for this model)
        # Investment required to maintain depreciated assets + grow
        capex = self.asset_maintain * depreciation + sales_t * self.asset_growth
        nca_curr = nca_prev - depreciation + capex

        # 1.3. Accounts Receivable (AR)
        accounts_receivable_curr = sales_t * self.account_receivables_pct

        # 1.4. Inventory (Inv)
        inventory_curr = sales_t * self.inventory_pct

        # 1.2. Derive Purchases from Cost Ratio and Inventory Identity
        # P_t = Sales_t * CR_t + (Inv_target_t - Inv_{t-1})
        purchases_t = sales_t * cost_ratio_t + (inventory_curr - inventory_prev)

        # 1.2b. Advance Payments (AdvPP) — based on this year's purchases (no lookahead)
        advance_payments_purchases_curr = (
            purchases_t * self.advance_payments_purchases_pct
        )

        # 1.5. Total Liquidity Target (TL) — baseline + logit-linear (sigmoid) in time
        # TL(t) = tl_baseline + sales_t * sigmoid(tl_alpha + tl_beta * t)
        tl_pct = tf.sigmoid(self.tl_alpha + self.tl_beta * time_index)
        total_liquidity_curr = self.tl_baseline + sales_t * tl_pct

        # 1.6. Cash Target (Cash) — logit-linear (sigmoid) cash fraction in time
        # %Cash(t) = sigmoid(cash_alpha + cash_beta * t)
        cash_pct = tf.sigmoid(self.cash_alpha + self.cash_beta * time_index)
        cash_curr = total_liquidity_curr * cash_pct

        # 1.7. Investment in Market Securities Target (IMS)
        investment_in_market_securities_curr = total_liquidity_curr - cash_curr

        # --- 2. Income Statement (IS) ---
        # Before we move on to Liabilities, we need to calculate Income Statement quantities and Liquidity Budget quantities, as they connect the assets to liabilities and equity.
        # Net income (NI) is calculated by first calculating EBITDA = Sales - COGS - OpEx
        # Then, EBT is calculated by EBITDA - Depreciation - loan interest payments + return from market securities.
        # Finally, NI is calculated by EBT - Tax.
        # COGS = Inv_{t-1} + P_t - Inv_t = Sales_t * CR_t (by construction)
        cogs = inventory_prev + purchases_t - inventory_curr

        # --- BAYESIAN OPEX CALCULATION ---
        if use_mean_opex:
            # For deterministic paths or structural training
            var_opex = self.q_var_opex_loc
            base_opex = self.q_base_opex_loc
            noise = 0.0
        else:
            # Use trajectory-fixed structural parameters and sample only annual noise
            var_opex = (
                sampled_var_opex
                if sampled_var_opex is not None
                else self.q_var_opex_loc
            )
            base_opex = (
                sampled_base_opex
                if sampled_base_opex is not None
                else self.q_base_opex_loc
            )
            noise = tfd.Normal(0.0, self.noise_sigma).sample()

        # Center sales by subtracting the offset used during training
        sales_t_centered = sales_t - self.sales_offset
        opex = (base_opex * cum_inflation) + (sales_t_centered * var_opex) + noise

        ebitda = sales_t - cogs - opex

        # Principals and interests are based on PREVIOUS debt (No Circularity)
        ## Long-term portion of current liability from last year is found by:
        ## last year's non-current liabilities / (Average maturity – 1)
        principal_lt = current_lt_debt_prev
        interest_lt = self.avg_long_term_interest_pct * (
            non_current_liabilities_prev + current_lt_debt_prev
        )

        ## Short-term portion of current liability from last year is found by:
        ## last year's current liabilities - last year's non-current liabilities / (Average maturity – 1)
        principal_st = effective_st_debt_prev
        interest_st = self.avg_short_term_interest_pct * principal_st

        ms_return = (
            investment_in_market_securities_prev * self.market_securities_return_pct
        )
        ebt = ebitda - depreciation - (interest_st + interest_lt) + ms_return
        tax_onetime_payment = tf.cast(
            inputs.get("tax_onetime_payment", 0.0), dtype=tf.float64
        )
        tax = ebt * self.income_tax_pct + tax_onetime_payment
        net_income_curr = ebt - tax

        # --- 3. Liquidity Budget (LB) ---
        # We need quantities from Liquidity Budget calculations before proceeding to liabilities.

        # 3.1. Operating Net Liquidity Balance (Operating NLB)
        # Inflows: Sales | Outflows: Purchases, OpEx, Tax, Interest

        # Sales: cash flow from current year's sales + accounts receivable from previous year + advance payment from this year's sales
        sales_curr = (
            sales_t * (1 - self.account_receivables_pct) - advance_payments_sales_prev
        )
        advance_payments_sales_curr = sales_t * self.advance_payments_sales_pct
        inflows = sales_curr + accounts_receivable_prev + advance_payments_sales_curr

        # Purchases: cash flow from current year's purchases + cash flow from previous year's purchases + cash flow from next year's purchases
        purchases_curr = (
            purchases_t * (1 - self.account_payables_pct)
            - advance_payments_purchases_prev
        )
        outflows = (
            purchases_curr
            + accounts_payable_prev
            + advance_payments_purchases_curr
            + opex
            + tax
        )
        operating_nlb = inflows - outflows

        # 3.2. Capital Expense Net Liquidity Balance (CapEx NLB)
        capex_nlb = -capex
        # 3.3. External Investment Net Liquidity Balance (External Investment NLB)
        ## Note: This only accounts for the return generated from the previous year's investment in market securities. It does not account for the investment in and out of market securities because that is done as a ratio of total liquidity balance.
        external_investment_nlb = ms_return

        # 3.4. Financing Net Liquidity Balance (Financing NLB)
        ## First, we need to figure out how much new short-term loan and long-term loan to issue this year.
        ## Note: The return from market securities investment is added to previous total liquidity balance because we always allocate a portion of total liquidity to market securities, instead of excess cash balance.

        # ST debt as policy-driven ratio of sales (logit-linear trend):
        # %STDebt(t) = sigmoid(st_debt_alpha + st_debt_beta * t)
        st_debt_pct = tf.sigmoid(self.st_debt_alpha + self.st_debt_beta * time_index)
        effective_st_debt_curr = sales_t * st_debt_pct

        # Diagnostic: what the old deficit-driven model would have computed
        liquidity_deficit_st = (
            total_liquidity_curr
            - (cash_prev + investment_in_market_securities_prev)
            - operating_nlb
            + principal_st
            + interest_st
        )

        ## New long-term loan is found by:
        liquidity_deficit_lt = (
            liquidity_deficit_st
            - effective_st_debt_curr
            - external_investment_nlb  # FIXED: Moved from short-term loan calculation to long-term loan calculation
            - capex_nlb
            + principal_lt
            + interest_lt
            + dividends_prev
            + stock_buyback
        )
        long_term_financing = tf.maximum(0.0, liquidity_deficit_lt)
        # %EF(t) = sigmoid(ef_alpha + ef_beta * t) — logit-linear equity financing mix
        ef_pct = tf.sigmoid(self.ef_alpha + self.ef_beta * time_index)
        new_long_term_loan = long_term_financing * (1 - ef_pct)
        equity_financing = long_term_financing * ef_pct

        # Any surplus cash beyond the liquidity target (negative deficit) is used for
        # additional stock buybacks, ensuring the liquidity budget closes exactly.
        excess_cash_buyback = tf.maximum(0.0, -liquidity_deficit_lt)
        stock_buyback = stock_buyback + excess_cash_buyback

        financing_nlb = (
            effective_st_debt_curr
            + new_long_term_loan
            - principal_st
            - principal_lt
            - interest_st
            - interest_lt
        )

        # 3.5. Transaction with Owners Net Liquidity Balance (Transaction with Owners NLB)
        transaction_with_owners_nlb = equity_financing - dividends_prev - stock_buyback

        # 3.6. Total Net Liquidity Balance (Total NLB)
        total_nlb = (
            operating_nlb
            + capex_nlb
            + financing_nlb
            + external_investment_nlb
            + transaction_with_owners_nlb
        )

        ## Check that the liquidity arrived in the Liquidity Budget matches the target liquidity
        liquidity_check = (
            (cash_prev + investment_in_market_securities_prev)
            + total_nlb
            - total_liquidity_curr
        )

        # --- 4. Liabilities Evolution ---
        # 4.1. Accounts Payable (AP)
        accounts_payable_curr = purchases_t * self.account_payables_pct

        # 4.2. Advance Payments Sales (AdvPS)
        # Already calculated in Liquidity Budget
        # advance_payments_sales_curr = sales_t * self.advance_payments_sales_pct

        # 4.3. Non-current Liabilities (NLiab)
        ## This is equal to the total long-term liabilities minus the effective principal due next year
        total_long_term_liabilities = new_long_term_loan + non_current_liabilities_prev

        non_current_liabilities_curr = total_long_term_liabilities * (
            1 - 1 / self.avg_maturity_years
        )

        # 4.4. Effective ST debt is simply effective_st_debt_curr
        # The current_lt_debt_curr is total_long_term_liabilities / self.avg_maturity_years
        current_lt_debt_curr = total_long_term_liabilities / self.avg_maturity_years

        # # 4.4. Current Liabilities (CLiab)
        # # This is equal to the new short-term plus the long-term liabilities' effective principal due next year
        # current_liabilities_curr = (
        #     new_short_term_loan + total_long_term_liabilities / self.avg_maturity_years
        # )

        # 4.5. Stockholders Equity (SE)
        equity_curr = (
            equity_prev
            + equity_financing
            + net_income_curr
            - dividends_prev
            - stock_buyback
        )

        # --- 5. Balance Sheet Identity Check ---
        # Assets = NCA + Advance Payments Purchases + Accounts Receivable + Inventory + Cash + Investment in Market Securities
        total_assets = (
            nca_curr
            + advance_payments_purchases_curr
            + accounts_receivable_curr
            + inventory_curr
            + cash_curr
            + investment_in_market_securities_curr
        )
        # Liabilities + Equity = Accounts Payable + Advance Payments Sales + Current Liabilities + Non-current Liabilities + Equity
        total_liab_equity = (
            accounts_payable_curr
            + advance_payments_sales_curr
            + current_lt_debt_curr
            + effective_st_debt_curr
            + non_current_liabilities_curr
            + equity_curr
        )
        # Check mismatch (Should be near zero if logic is consistent)
        check = total_assets - total_liab_equity

        return {
            "nca": nca_curr,
            "advance_payments_purchases": advance_payments_purchases_curr,
            "accounts_receivable": accounts_receivable_curr,
            "inventory": inventory_curr,
            "cash": cash_curr,
            "investment_in_market_securities": investment_in_market_securities_curr,
            "accounts_payable": accounts_payable_curr,
            "advance_payments_sales": advance_payments_sales_curr,
            "effective_st_debt": effective_st_debt_curr,
            "non_current_liabilities": non_current_liabilities_curr,
            "equity": equity_curr,
            "cogs": cogs,
            "opex": opex,
            "tax": tax,
            "ms_return": ms_return,
            "interest_payment": interest_lt + interest_st,
            "net_income": net_income_curr,
            "depreciation": depreciation,
            "dividends": dividends_prev,
            "stock_buyback": stock_buyback,
            "current_lt_debt": current_lt_debt_curr,
            "new_long_term_loan": new_long_term_loan,
            "equity_financing": equity_financing,
            "liquidity_deficit_st": liquidity_deficit_st,
            "liquidity_check": liquidity_check,
            "check": check,
        }


# --- Monte Carlo Forecast Execution ---
