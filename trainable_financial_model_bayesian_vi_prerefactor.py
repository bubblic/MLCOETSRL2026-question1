import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


tfd = tfp.distributions
tfb = tfp.bijectors


# --- 1. Define the Trainable Model ---
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
            initial_value=0.5,
            bijector=tfb.Sigmoid(),
            dtype=tf.float64,
            name="div_adj_speed",
        )  # α
        # --- Stock Buyback % Softplus-Linear Model ---
        # %BB(t) = softplus(sb_alpha + sb_beta * t), where t = year - base_year
        # Softplus ensures %BB stays positive (but can be > 1, since buybacks
        # can exceed depreciation). Time trend allows buyback policy to evolve.
        # Initialize alpha to inverse_softplus(7.5) ≈ 7.5 (for large x, softplus ≈ identity)
        self.sb_alpha = tf.Variable(7.5, dtype=tf.float64, name="sb_alpha")
        self.sb_beta = tf.Variable(0.0, dtype=tf.float64, name="sb_beta")

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
            "sb_alpha": float(self.sb_alpha.numpy()),
            "sb_beta": float(self.sb_beta.numpy()),
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
        self.sb_alpha.assign(data["sb_alpha"])
        self.sb_beta.assign(data["sb_beta"])
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
        historical_st_debt=None,
        historical_inflation=None,
        historical_years=None,
        learning_rate=0.001,
        epochs=30000,
        plot_vi=True,
        plot_every=1000,
        show_plot=False,
        prior_strength_asset_maintain=1.0,
    ):
        """
        Trains simple policy parameters using historical data.
        """

        # Convert inputs to tensors and ensure float64
        sales_tensor = tf.convert_to_tensor(historical_sales, dtype=tf.float64)
        purchases_tensor = tf.convert_to_tensor(historical_purchases, dtype=tf.float64)
        cogs_tensor = tf.convert_to_tensor(historical_cogs, dtype=tf.float64)
        nca_tensor = tf.convert_to_tensor(historical_nca, dtype=tf.float64)
        depr_tensor = tf.convert_to_tensor(historical_depreciation, dtype=tf.float64)
        adv_pay_sales_tensor = tf.convert_to_tensor(
            historical_adv_pay_sales, dtype=tf.float64
        )
        adv_pay_purch_tensor = tf.convert_to_tensor(
            historical_adv_pay_purch, dtype=tf.float64
        )
        ar_tensor = tf.convert_to_tensor(historical_ar, dtype=tf.float64)
        ap_tensor = tf.convert_to_tensor(historical_ap, dtype=tf.float64)
        inv_tensor = tf.convert_to_tensor(historical_inventory, dtype=tf.float64)
        cash_tensor = tf.convert_to_tensor(historical_cash, dtype=tf.float64)
        ims_tensor = tf.convert_to_tensor(historical_ims, dtype=tf.float64)
        ni_tensor = tf.convert_to_tensor(historical_net_income, dtype=tf.float64)
        div_tensor = tf.convert_to_tensor(historical_dividends, dtype=tf.float64)
        bb_tensor = tf.convert_to_tensor(historical_stock_buyback, dtype=tf.float64)
        opex_tensor = tf.convert_to_tensor(historical_opex, dtype=tf.float64)
        tax_tensor = tf.convert_to_tensor(historical_tax, dtype=tf.float64)
        if historical_st_debt is not None:
            st_debt_tensor = tf.convert_to_tensor(historical_st_debt, dtype=tf.float64)

        if historical_inflation is None:
            historical_inflation = tf.zeros_like(sales_tensor)
        inf_tensor = tf.convert_to_tensor(historical_inflation, dtype=tf.float64)
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

        # 3. Advance Payments Sales: adv_ps_t = sales_{t+1} * adv_ps_pct
        adv_ps_true = adv_pay_sales_tensor[:-1]
        sales_next_aligned = sales_tensor[1:]

        # 4. Advance Payments Purchases: adv_pp_t = purchases_t * adv_pp_pct
        adv_pp_true = adv_pay_purch_tensor
        purchases_aligned_adv_pp = purchases_tensor

        # 5. Dividends (Lintner Smoothing):
        #    D_t = α * (NI_{t-1} * PayoutRatio) + (1 - α) * D_{t-1}
        div_true = div_tensor[1:]
        ni_prev_aligned = ni_tensor[:-1]
        div_prev_aligned = div_tensor[:-1]

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
            self.sb_alpha,
            self.sb_beta,
            # ST Debt Params (Logit-Linear)
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
            "loss_st_debt": [],
            "loss_prior_am": [],
        }

        for i in range(epochs):
            with tf.GradientTape() as tape:
                # --- Deterministic Losses (MSE) ---
                loss_growth = tf.reduce_mean(
                    tf.square(
                        delta_nca_true
                        - (
                            (self.asset_maintain - 1) * depr_true
                            + sales_aligned_growth * self.asset_growth
                        )
                    )
                )
                loss_depr = tf.reduce_mean(
                    tf.square(depr_true - nca_prev_aligned * self.depreciation_rate)
                )
                loss_adv_ps = tf.reduce_mean(
                    tf.square(
                        adv_ps_true
                        - sales_next_aligned * self.advance_payments_sales_pct
                    )
                )
                loss_adv_pp = tf.reduce_mean(
                    tf.square(
                        adv_pp_true
                        - purchases_aligned_adv_pp * self.advance_payments_purchases_pct
                    )
                )
                loss_ar = tf.reduce_mean(
                    tf.square(ar_tensor - sales_tensor * self.account_receivables_pct)
                )
                loss_ap = tf.reduce_mean(
                    tf.square(ap_tensor - purchases_tensor * self.account_payables_pct)
                )
                loss_inv = tf.reduce_mean(
                    tf.square(inv_tensor - sales_tensor * self.inventory_pct)
                )
                # TL(t) = tl_baseline + sales_t * sigmoid(tl_alpha + tl_beta * t)
                tl_pct_t_logit = self.tl_alpha + self.tl_beta * time_indices
                tl_pct_t = tf.sigmoid(tl_pct_t_logit)
                loss_tl = tf.reduce_mean(
                    tf.square(
                        (cash_tensor + ims_tensor)
                        - (self.tl_baseline + sales_tensor * tl_pct_t)
                    )
                )
                # %Cash(t) = sigmoid(cash_alpha + cash_beta * t) (logit-linear)
                cash_pct_t_logit = self.cash_alpha + self.cash_beta * time_indices
                cash_pct_t = tf.sigmoid(cash_pct_t_logit)
                loss_cash = tf.reduce_mean(
                    tf.square(cash_tensor - (cash_tensor + ims_tensor) * cash_pct_t)
                )
                loss_tax = tf.reduce_mean(
                    tf.square(tax_tensor - ni_tensor * self.income_tax_pct)
                )
                div_target = ni_prev_aligned * self.dividend_payout_ratio_pct
                div_pred = (
                    self.dividend_adjustment_speed * div_target
                    + (1.0 - self.dividend_adjustment_speed) * div_prev_aligned
                )
                loss_div = tf.reduce_mean(tf.square(div_true - div_pred))
                # %BB(t) = softplus(sb_alpha + sb_beta * t) (softplus-linear)
                bb_pct_t = tf.math.softplus(self.sb_alpha + self.sb_beta * time_indices)
                loss_bb = tf.reduce_mean(tf.square(bb_tensor - depr_tensor * bb_pct_t))

                # --- Cost Ratio Loss (Logit-Linear) ---
                # logit(CR_t) = alpha + beta * t
                logit_cr_pred = (
                    self.cost_ratio_alpha + self.cost_ratio_beta * time_indices
                )
                loss_cost_ratio = tf.reduce_mean(
                    tf.square(logit_cr_hist - logit_cr_pred)
                )

                # --- ST Debt Loss (Logit-Linear) ---
                # %STDebt(t) = sigmoid(st_debt_alpha + st_debt_beta * t)
                # new_short_term_loan = sales * %STDebt(t)
                if historical_st_debt is not None:
                    st_debt_pct_pred = tf.sigmoid(
                        self.st_debt_alpha + self.st_debt_beta * time_indices
                    )
                    loss_st_debt = tf.reduce_mean(
                        tf.square(st_debt_tensor - sales_tensor * st_debt_pct_pred)
                    )
                else:
                    loss_st_debt = tf.constant(0.0, dtype=tf.float64)

                # --- Bayesian OpEx Loss ---
                # 1. Sample parameters
                var_opex_sample, base_opex_sample = self.sample_opex_params()

                # 2. Calculate the Raw Prediction (in Billion Dollars)
                # Use centered sales for training to make spread symmetric around y-axis
                pred_opex_raw = (base_opex_sample * cum_inf_tensor) + (
                    var_opex_sample * sales_tensor_centered
                )

                # 3. Calculate Residuals (The Error)
                residuals = opex_tensor - pred_opex_raw

                # 4. Calculate Likelihood
                likelihood_dist = tfd.Normal(loc=0.0, scale=self.noise_sigma)
                neg_log_likelihood = -tf.reduce_sum(likelihood_dist.log_prob(residuals))

                # 5. KL Divergence
                kl = self.get_opex_kl_divergence()

                # 6. Final Sum
                loss_opex_bayes = neg_log_likelihood + kl

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
                    + loss_st_debt
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
                simple_history["loss_st_debt"].append(loss_st_debt.numpy())
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
            f"Stock Buyback % (softplus-linear): alpha={self.sb_alpha.numpy():.4f}, "
            f"beta={self.sb_beta.numpy():.6f}"
        )
        print(
            f"  => %BB at t=0: {tf.math.softplus(self.sb_alpha).numpy():.4f}, "
            f"%BB at t={len(historical_sales)-1}: "
            f"{tf.math.softplus(self.sb_alpha + self.sb_beta * (len(historical_sales)-1)).numpy():.4f}"
        )
        print(
            f"ST Debt % of Sales (logit-linear): alpha={self.st_debt_alpha.numpy():.4f}, "
            f"beta={self.st_debt_beta.numpy():.6f}"
        )
        print(
            f"  => %STDebt at t=0: {tf.sigmoid(self.st_debt_alpha).numpy():.4f}, "
            f"%STDebt at t={len(historical_sales)-1}: "
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
            plt.savefig(f"vi_training_diagnostics_{timestamp}.png", dpi=150)
            if show_plot:
                plt.show()
            else:
                plt.close()

        # --- Simple Parameters Training Diagnostics ---
        if plot_vi and simple_history["epochs"]:
            epochs_hist = np.array(simple_history["epochs"])
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

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

            fig.suptitle("Simple Parameters Training Diagnostics")
            plt.tight_layout()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"simple_training_diagnostics_{timestamp}.png", dpi=150)
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
        historical_current_liabilities,
        historical_non_current_liabilities,
        historical_equity,
        historical_inflation=None,
        historical_years=None,
        learning_rate=0.001,
        epochs=20000,
        plot_every=1000,
        show_plot=False,
    ):
        """
        Trains structural parameters (interest rates, maturity, financing)
        using historical state transitions.
        Purchases are derived inside forecast_step from the learned cost ratio.
        """
        # NOTE: When calling forecast_step inside here, we need to pass use_mean_opex=True
        # because we want to learn structural parameters based on the "most likely" OpEx, not noisy samples.
        sales_t = tf.convert_to_tensor(historical_sales, dtype=tf.float64)
        nca_t = tf.convert_to_tensor(historical_nca, dtype=tf.float64)
        adv_ps_t = tf.convert_to_tensor(historical_adv_pay_sales, dtype=tf.float64)
        adv_pp_t = tf.convert_to_tensor(historical_adv_pay_purch, dtype=tf.float64)
        ar_t = tf.convert_to_tensor(historical_ar, dtype=tf.float64)
        ap_t = tf.convert_to_tensor(historical_ap, dtype=tf.float64)
        inv_t = tf.convert_to_tensor(historical_inventory, dtype=tf.float64)
        cash_t = tf.convert_to_tensor(historical_cash, dtype=tf.float64)
        ims_t = tf.convert_to_tensor(historical_ims, dtype=tf.float64)
        ni_t = tf.convert_to_tensor(historical_net_income, dtype=tf.float64)
        div_t = tf.convert_to_tensor(historical_dividends, dtype=tf.float64)
        cl_t = tf.convert_to_tensor(historical_current_liabilities, dtype=tf.float64)
        ncl_t = tf.convert_to_tensor(
            historical_non_current_liabilities, dtype=tf.float64
        )
        equity_t = tf.convert_to_tensor(historical_equity, dtype=tf.float64)

        if historical_inflation is None:
            historical_inflation = tf.zeros_like(sales_t)
        inf_t = tf.convert_to_tensor(historical_inflation, dtype=tf.float64)
        cum_inf_t = tf.math.cumprod(1 + inf_t)

        if historical_years is None:
            historical_years = np.arange(
                self.base_year, self.base_year + len(historical_sales)
            )
        years_t = tf.convert_to_tensor(historical_years, dtype=tf.float64)
        time_idx_t = years_t - tf.constant(float(self.base_year), dtype=tf.float64)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
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
            "loss_cl": [],
            "loss_ncl": [],
            "loss_equity": [],
        }

        print(f"Training structural parameters...")
        for i in range(epochs):
            with tf.GradientTape() as tape:
                total_loss = 0.0
                total_loss_ni = 0.0
                total_loss_cl = 0.0
                total_loss_ncl = 0.0
                total_loss_equity = 0.0
                # We need t+1 for the target and t+2 for the lookahead inputs in forecast_step
                num_transitions = len(historical_sales) - 2

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
                        "current_liabilities": cl_t[t],
                        "non_current_liabilities": ncl_t[t],
                        "equity": equity_t[t],
                        "net_income": ni_t[t],
                        "dividends": div_t[t],
                    }

                    # Inputs for predicting state at t+1
                    # Purchases are derived inside forecast_step from cost ratio
                    inputs_curr = {
                        "sales_t": sales_t[t + 1],
                        "sales_t_plus_1": sales_t[t + 2],
                        "year": years_t[t + 1],
                        "cum_inflation": cum_inf_t[t + 1],
                    }

                    # IMPORTANT: Use mean (deterministic) OpEx for structural training
                    state_pred = self.forecast_step(
                        state_prev,
                        inputs_curr,
                        use_mean_opex=True,
                    )

                    # Targets are values at t+1
                    loss_ni = tf.square(state_pred["net_income"] - ni_t[t + 1])
                    loss_cl = tf.square(state_pred["current_liabilities"] - cl_t[t + 1])
                    loss_ncl = tf.square(
                        state_pred["non_current_liabilities"] - ncl_t[t + 1]
                    )
                    loss_equity = tf.square(state_pred["equity"] - equity_t[t + 1])

                    # Total loss to minimize
                    total_loss += loss_ni + loss_cl + loss_ncl + loss_equity
                    total_loss_ni += loss_ni
                    total_loss_cl += loss_cl
                    total_loss_ncl += loss_ncl
                    total_loss_equity += loss_equity

            grads = tape.gradient(total_loss, vars_to_train)
            optimizer.apply_gradients(zip(grads, vars_to_train))

            if i % plot_every == 0:
                structural_history["epochs"].append(i)
                structural_history["loss_total"].append(total_loss.numpy())
                structural_history["loss_ni"].append(total_loss_ni.numpy())
                structural_history["loss_cl"].append(total_loss_cl.numpy())
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
                ("loss_cl", "Current Liabilities"),
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
            plt.savefig(f"structural_training_diagnostics_{timestamp}.png", dpi=150)
            if show_plot:
                plt.show()
            else:
                plt.close()

    def forecast_step(
        self,
        state,
        inputs,
        use_mean_opex=False,
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
        current_liabilities_prev = state["current_liabilities"]
        non_current_liabilities_prev = state["non_current_liabilities"]

        ## Equity
        equity_prev = state["equity"]
        ## Net Income
        net_income_prev = state["net_income"]
        ## Dividends (previous period, for Lintner smoothing)
        dividends_prev_actual = state["dividends"]

        # Unpack current inputs (t)
        sales_t = inputs["sales_t"]
        sales_t_plus_1 = inputs["sales_t_plus_1"]
        year = inputs["year"]
        cum_inflation = inputs["cum_inflation"]

        # Convert year to 0-based time index for trend models (e.g., FY2018 -> 0)
        time_index = year - tf.constant(float(self.base_year), dtype=tf.float64)

        depreciation = nca_prev * self.depreciation_rate
        # %BB(t) = softplus(sb_alpha + sb_beta * t) — softplus-linear buyback policy
        bb_pct = tf.math.softplus(self.sb_alpha + self.sb_beta * time_index)
        stock_buyback = depreciation * bb_pct
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
            # Sample for Monte Carlo forecasting
            var_opex, base_opex = self.sample_opex_params()
            noise = tfd.Normal(0.0, self.noise_sigma).sample()

        # Center sales by subtracting the offset used during training
        sales_t_centered = sales_t - self.sales_offset
        opex = (base_opex * cum_inflation) + (sales_t_centered * var_opex) + noise

        ebitda = sales_t - cogs - opex

        # Principals and interests are based on PREVIOUS debt (No Circularity)
        ## Long-term portion of current liability from last year is found by:
        ## last year's non-current liabilities / (Average maturity – 1)
        principal_lt = non_current_liabilities_prev / (self.avg_maturity_years - 1)
        interest_lt = (
            self.avg_long_term_interest_pct
            * non_current_liabilities_prev
            / (1 - 1 / self.avg_maturity_years)
        )

        ## Short-term portion of current liability from last year is found by:
        ## last year's current liabilities - last year's non-current liabilities / (Average maturity – 1)
        principal_st = current_liabilities_prev - principal_lt
        interest_st = self.avg_short_term_interest_pct * principal_st

        ms_return = (
            investment_in_market_securities_prev * self.market_securities_return_pct
        )
        ebt = ebitda - depreciation - (interest_st + interest_lt) + ms_return
        tax = ebt * self.income_tax_pct
        net_income_curr = ebt - tax

        # --- 3. Liquidity Budget (LB) ---
        # We need quantities from Liquidity Budget calculations before proceeding to liabilities.

        # 3.1. Operating Net Liquidity Balance (Operating NLB)
        # Inflows: Sales | Outflows: Purchases, OpEx, Tax, Interest

        # Sales: cash flow from current year's sales + accounts receivable from previous year + advance payment for next year's sales
        sales_curr = (
            sales_t * (1 - self.account_receivables_pct) - advance_payments_sales_prev
        )
        advance_payments_sales_curr = sales_t_plus_1 * self.advance_payments_sales_pct
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
        new_short_term_loan = sales_t * st_debt_pct

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
            - new_short_term_loan
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
            new_short_term_loan
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
        # advance_payments_sales_curr = sales_t_plus_1 * self.advance_payments_sales_pct

        # 4.3. Non-current Liabilities (NLiab)
        ## This is equal to the total long-term liabilities minus the effective principal due next year
        total_long_term_liabilities = new_long_term_loan + non_current_liabilities_prev

        non_current_liabilities_curr = total_long_term_liabilities * (
            1 - 1 / self.avg_maturity_years
        )

        # 4.4. Current Liabilities (CLiab)
        # This is equal to the new short-term plus the long-term liabilities' effective principal due next year
        current_liabilities_curr = (
            new_short_term_loan + total_long_term_liabilities / self.avg_maturity_years
        )

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
            + current_liabilities_curr
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
            "current_liabilities": current_liabilities_curr,
            "non_current_liabilities": non_current_liabilities_curr,
            "equity": equity_curr,
            "net_income": net_income_curr,
            "depreciation": depreciation,
            "dividends": dividends_prev,
            "stock_buyback": stock_buyback,
            "new_short_term_loan": new_short_term_loan,
            "new_long_term_loan": new_long_term_loan,
            "equity_financing": equity_financing,
            "liquidity_deficit_st": liquidity_deficit_st,
            "liquidity_check": liquidity_check,
            "check": check,
        }


# --- Monte Carlo Forecast Execution ---
def run_monte_carlo_forecast(
    model,
    initial_state,
    sales_forecast,
    cum_inf_forecast,
    forecast_years,
    n_samples=1000,
):
    print(f"\n--- Running Monte Carlo Forecast ({n_samples} samples) ---")

    # Store trajectories
    # Shape: [Samples, Years]
    ni_trajectories = []
    equity_trajectories = []
    assets_trajectories = []
    nca_trajectories = []
    adv_pay_purch_trajectories = []
    ar_trajectories = []
    inv_trajectories = []
    cash_trajectories = []
    ims_trajectories = []
    current_liabilities_trajectories = []
    non_current_liabilities_trajectories = []
    ap_trajectories = []
    aps_trajectories = []
    depreciation_trajectories = []
    dividends_trajectories = []
    stock_buyback_trajectories = []
    new_st_loan_trajectories = []
    new_lt_loan_trajectories = []
    equity_financing_trajectories = []
    liq_deficit_st_trajectories = []

    for i in range(n_samples):
        current_state = initial_state.copy()
        sample_ni = []
        sample_equity = []
        sample_assets = []
        sample_nca = []
        sample_adv_pp = []
        sample_ar = []
        sample_inv = []
        sample_cash = []
        sample_ims = []
        sample_cl = []
        sample_ncl = []
        sample_ap = []
        sample_aps = []
        sample_depr = []
        sample_div = []
        sample_bb = []
        sample_new_st = []
        sample_new_lt = []
        sample_ef = []
        sample_liq_deficit_st = []

        for t in range(len(sales_forecast) - 1):
            inputs = {
                "sales_t": tf.constant(sales_forecast[t]),
                "sales_t_plus_1": tf.constant(sales_forecast[t + 1]),
                "year": tf.constant(float(forecast_years[t]), dtype=tf.float64),
                "cum_inflation": tf.constant(cum_inf_forecast[t]),
            }
            # use_mean_opex=False triggers sampling
            current_state = model.forecast_step(
                current_state,
                inputs,
                use_mean_opex=False,
            )

            sample_ni.append(current_state["net_income"].numpy())
            sample_equity.append(current_state["equity"].numpy())
            total_assets = (
                current_state["nca"]
                + current_state["advance_payments_purchases"]
                + current_state["accounts_receivable"]
                + current_state["inventory"]
                + current_state["cash"]
                + current_state["investment_in_market_securities"]
            )
            sample_assets.append(total_assets.numpy())
            sample_nca.append(current_state["nca"].numpy())
            sample_adv_pp.append(current_state["advance_payments_purchases"].numpy())
            sample_ar.append(current_state["accounts_receivable"].numpy())
            sample_inv.append(current_state["inventory"].numpy())
            sample_cash.append(current_state["cash"].numpy())
            sample_ims.append(current_state["investment_in_market_securities"].numpy())
            sample_cl.append(current_state["current_liabilities"].numpy())
            sample_ncl.append(current_state["non_current_liabilities"].numpy())
            sample_ap.append(current_state["accounts_payable"].numpy())
            sample_aps.append(current_state["advance_payments_sales"].numpy())
            sample_depr.append(current_state["depreciation"].numpy())
            sample_div.append(current_state["dividends"].numpy())
            sample_bb.append(current_state["stock_buyback"].numpy())
            sample_new_st.append(current_state["new_short_term_loan"].numpy())
            sample_new_lt.append(current_state["new_long_term_loan"].numpy())
            sample_ef.append(current_state["equity_financing"].numpy())
            sample_liq_deficit_st.append(current_state["liquidity_deficit_st"].numpy())

        ni_trajectories.append(sample_ni)
        equity_trajectories.append(sample_equity)
        assets_trajectories.append(sample_assets)
        nca_trajectories.append(sample_nca)
        adv_pay_purch_trajectories.append(sample_adv_pp)
        ar_trajectories.append(sample_ar)
        inv_trajectories.append(sample_inv)
        cash_trajectories.append(sample_cash)
        ims_trajectories.append(sample_ims)
        current_liabilities_trajectories.append(sample_cl)
        non_current_liabilities_trajectories.append(sample_ncl)
        ap_trajectories.append(sample_ap)
        aps_trajectories.append(sample_aps)
        depreciation_trajectories.append(sample_depr)
        dividends_trajectories.append(sample_div)
        stock_buyback_trajectories.append(sample_bb)
        new_st_loan_trajectories.append(sample_new_st)
        new_lt_loan_trajectories.append(sample_new_lt)
        equity_financing_trajectories.append(sample_ef)
        liq_deficit_st_trajectories.append(sample_liq_deficit_st)

    ni_trajectories = np.array(ni_trajectories)
    equity_trajectories = np.array(equity_trajectories)
    assets_trajectories = np.array(assets_trajectories)
    nca_trajectories = np.array(nca_trajectories)
    adv_pay_purch_trajectories = np.array(adv_pay_purch_trajectories)
    ar_trajectories = np.array(ar_trajectories)
    inv_trajectories = np.array(inv_trajectories)
    cash_trajectories = np.array(cash_trajectories)
    ims_trajectories = np.array(ims_trajectories)
    current_liabilities_trajectories = np.array(current_liabilities_trajectories)
    non_current_liabilities_trajectories = np.array(
        non_current_liabilities_trajectories
    )
    ap_trajectories = np.array(ap_trajectories)
    aps_trajectories = np.array(aps_trajectories)
    depreciation_trajectories = np.array(depreciation_trajectories)
    dividends_trajectories = np.array(dividends_trajectories)
    stock_buyback_trajectories = np.array(stock_buyback_trajectories)
    new_st_loan_trajectories = np.array(new_st_loan_trajectories)
    new_lt_loan_trajectories = np.array(new_lt_loan_trajectories)
    equity_financing_trajectories = np.array(equity_financing_trajectories)
    liq_deficit_st_trajectories = np.array(liq_deficit_st_trajectories)

    # Calculate Statistics
    def summarize_trajectories(name, trajectories):
        mean_vals = np.mean(trajectories, axis=0)
        lower_bound = np.percentile(trajectories, 2.5, axis=0)
        upper_bound = np.percentile(trajectories, 97.5, axis=0)

        print(f"\n{name}")
        print(f"{'Year':<5} | {'Mean':<15} | {'2.5% CI':<15} | {'97.5% CI':<15}")
        print("-" * 60)
        for t in range(len(mean_vals)):
            mean_usd = mean_vals[t] * model.amount_scale
            lower_usd = lower_bound[t] * model.amount_scale
            upper_usd = upper_bound[t] * model.amount_scale
            print(
                f"{t+1:<5} | {mean_usd:<15.2e} | {lower_usd:<15.2e} | {upper_usd:<15.2e}"
            )

    summarize_trajectories("Net Income", ni_trajectories)
    summarize_trajectories("Total Assets", assets_trajectories)
    summarize_trajectories("Assets: Non-current Assets", nca_trajectories)
    summarize_trajectories(
        "Assets: Advance Payments (Purchases)", adv_pay_purch_trajectories
    )
    summarize_trajectories("Assets: Accounts Receivable", ar_trajectories)
    summarize_trajectories("Assets: Inventory", inv_trajectories)
    summarize_trajectories("Assets: Cash", cash_trajectories)
    summarize_trajectories("Assets: Investment in Market Securities", ims_trajectories)
    summarize_trajectories("Current Liabilities", current_liabilities_trajectories)
    summarize_trajectories(
        "Non-current Liabilities", non_current_liabilities_trajectories
    )
    summarize_trajectories("Equity", equity_trajectories)
    summarize_trajectories("Accounts Payable", ap_trajectories)
    summarize_trajectories("Advance Payments (Sales)", aps_trajectories)
    summarize_trajectories("Depreciation", depreciation_trajectories)
    summarize_trajectories("Dividends", dividends_trajectories)
    summarize_trajectories("Stock Buyback", stock_buyback_trajectories)
    summarize_trajectories("New Short-Term Loan", new_st_loan_trajectories)
    summarize_trajectories("New Long-Term Loan", new_lt_loan_trajectories)
    summarize_trajectories("Equity Financing", equity_financing_trajectories)

    # --- Comprehensive Balance Sheet Table (Mean Values) ---
    n_years = assets_trajectories.shape[1]
    scale = model.amount_scale

    # Compute mean trajectories (in scaled units, i.e. billions)
    mean_nca = np.mean(nca_trajectories, axis=0)
    mean_adv_pp = np.mean(adv_pay_purch_trajectories, axis=0)
    mean_ar = np.mean(ar_trajectories, axis=0)
    mean_inv = np.mean(inv_trajectories, axis=0)
    mean_cash = np.mean(cash_trajectories, axis=0)
    mean_ims = np.mean(ims_trajectories, axis=0)
    mean_total_assets = np.mean(assets_trajectories, axis=0)

    mean_ap = np.mean(ap_trajectories, axis=0)
    mean_aps = np.mean(aps_trajectories, axis=0)
    mean_cl = np.mean(current_liabilities_trajectories, axis=0)
    mean_ncl = np.mean(non_current_liabilities_trajectories, axis=0)
    mean_equity = np.mean(equity_trajectories, axis=0)
    mean_ni = np.mean(ni_trajectories, axis=0)

    mean_total_liabilities = mean_ap + mean_aps + mean_cl + mean_ncl
    mean_total_liab_equity = mean_total_liabilities + mean_equity
    mean_check = mean_total_assets - mean_total_liab_equity

    # Build year labels from forecast_years
    year_labels = [f"FY{int(forecast_years[t])}" for t in range(n_years)]

    # Row definitions: (label, data_array)
    rows = [
        ("ASSETS", None),
        ("  Non-Current Assets", mean_nca),
        ("  Adv Payments (Purch)", mean_adv_pp),
        ("  Accounts Receivable", mean_ar),
        ("  Inventory", mean_inv),
        ("  Cash", mean_cash),
        ("  Invest in Mkt Sec", mean_ims),
        ("TOTAL ASSETS", mean_total_assets),
        ("", None),
        ("LIABILITIES", None),
        ("  Accounts Payable", mean_ap),
        ("  Adv Payments (Sales)", mean_aps),
        ("  Current Liabilities", mean_cl),
        ("  Non-Current Liabilities", mean_ncl),
        ("TOTAL LIABILITIES", mean_total_liabilities),
        ("", None),
        ("EQUITY", mean_equity),
        ("", None),
        ("TOTAL LIAB + EQUITY", mean_total_liab_equity),
        ("", None),
        ("INCOME STATEMENT", None),
        ("  Net Income", mean_ni),
        ("", None),
        ("CHECK: Assets-(L+E)", mean_check),
    ]

    col_width = 14
    label_width = 26
    header = f"{'':>{label_width}}" + "".join(
        f"{yl:>{col_width}}" for yl in year_labels
    )

    print("\n" + "=" * len(header))
    print("FORECAST BALANCE SHEET — Mean across Monte Carlo samples (USD)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, data in rows:
        if data is None:
            print(f"{label:>{label_width}}")
        else:
            vals_str = "".join(f"{v * scale:>{col_width},.0f}" for v in data)
            print(f"{label:>{label_width}}{vals_str}")

    print("-" * len(header))

    # --- Balance Sheet Identity Check ---
    max_abs_check = np.max(np.abs(mean_check * scale))
    print(f"\nBalance Sheet Identity Check (Assets = Liabilities + Equity):")
    print(f"  Max absolute mismatch across years (mean): ${max_abs_check:,.2f}")
    if max_abs_check < 1.0:
        print("  PASS: Balance sheet identity holds (mismatch < $1).")
    elif max_abs_check < 1000.0:
        print(
            "  PASS: Balance sheet identity holds within rounding (mismatch < $1,000)."
        )
    else:
        print(f"  WARNING: Balance sheet mismatch detected!")
        for t in range(n_years):
            check_val = mean_check[t] * scale
            if abs(check_val) >= 1000.0:
                print(f"    {year_labels[t]}: Assets - (Liab+Eq) = ${check_val:,.2f}")

    # --- Per-sample balance sheet identity check ---
    total_liab_equity_all = (
        ap_trajectories
        + aps_trajectories
        + current_liabilities_trajectories
        + non_current_liabilities_trajectories
        + equity_trajectories
    )
    check_all = assets_trajectories - total_liab_equity_all
    max_abs_check_all = np.max(np.abs(check_all)) * scale
    mean_abs_check_all = np.mean(np.abs(check_all)) * scale
    print(f"\n  Per-sample check (across all {n_samples} samples x {n_years} years):")
    print(f"    Max absolute mismatch:  ${max_abs_check_all:,.2f}")
    print(f"    Mean absolute mismatch: ${mean_abs_check_all:,.2f}")

    return {
        "net_income": ni_trajectories,
        "total_assets": assets_trajectories,
        "nca": nca_trajectories,
        "advance_payments_purchases": adv_pay_purch_trajectories,
        "accounts_receivable": ar_trajectories,
        "inventory": inv_trajectories,
        "cash": cash_trajectories,
        "investment_in_market_securities": ims_trajectories,
        "accounts_payable": ap_trajectories,
        "advance_payments_sales": aps_trajectories,
        "current_liabilities": current_liabilities_trajectories,
        "non_current_liabilities": non_current_liabilities_trajectories,
        "equity": equity_trajectories,
        "depreciation": depreciation_trajectories,
        "dividends": dividends_trajectories,
        "stock_buyback": stock_buyback_trajectories,
        "new_short_term_loan": new_st_loan_trajectories,
        "new_long_term_loan": new_lt_loan_trajectories,
        "equity_financing": equity_financing_trajectories,
        "liquidity_deficit_st": liq_deficit_st_trajectories,
    }


def plot_opex_fit_with_aleatoric_noise(
    model,
    historical_years,
    historical_sales_bil,
    historical_opex_bil,
    historical_inflation,
    n_samples=2000,
    lower_q=5.0,
    upper_q=95.0,
    show_plot=False,
    use_gaussian_ci=False,
):
    if historical_inflation is None:
        historical_inflation = np.zeros_like(historical_sales_bil)
    cum_inf = np.cumprod(1 + historical_inflation)

    mean_var_opex = model.q_var_opex_loc.numpy()
    mean_base_opex = model.q_base_opex_loc.numpy()
    sigma_opex = model.noise_sigma.numpy()
    sales_offset = model.sales_offset.numpy()

    # Center sales using the offset from training
    historical_sales_bil_centered = historical_sales_bil - sales_offset
    mean_opex_bil = (mean_base_opex * cum_inf) + (
        mean_var_opex * historical_sales_bil_centered
    )

    if use_gaussian_ci:
        # Analytical Gaussian predictive intervals (exact for linear-Gaussian model)
        var_var = float(model.q_var_opex_scale.numpy()) ** 2
        var_base = float(model.q_base_opex_scale.numpy()) ** 2
        var_noise = float(sigma_opex) ** 2
        cum_inf_np = np.asarray(cum_inf, dtype=np.float64)
        # Use centered sales for variance calculation
        sales_np = np.asarray(historical_sales_bil_centered, dtype=np.float64)
        std_opex_bil = np.sqrt(
            (cum_inf_np**2) * var_base + (sales_np**2) * var_var + var_noise
        )
        z_low = float(tfd.Normal(0.0, 1.0).quantile(lower_q / 100.0))
        z_up = float(tfd.Normal(0.0, 1.0).quantile(upper_q / 100.0))
        lower_opex_bil = mean_opex_bil + z_low * std_opex_bil
        upper_opex_bil = mean_opex_bil + z_up * std_opex_bil
    else:
        # Posterior predictive samples with aleatoric noise (sigma)
        q_var = tfd.Normal(loc=model.q_var_opex_loc, scale=model.q_var_opex_scale)
        q_base = tfd.Normal(loc=model.q_base_opex_loc, scale=model.q_base_opex_scale)
        var_samples = q_var.sample(n_samples)  # [S]
        base_samples = q_base.sample(n_samples)  # [S]

        var_samples = tf.reshape(var_samples, (-1, 1))
        base_samples = tf.reshape(base_samples, (-1, 1))
        # Use centered sales for sampling
        sales = tf.reshape(
            tf.convert_to_tensor(historical_sales_bil_centered, dtype=tf.float64),
            (1, -1),
        )
        cum_inf_t = tf.reshape(tf.convert_to_tensor(cum_inf, dtype=tf.float64), (1, -1))
        noise = tf.random.normal(
            shape=(n_samples, len(historical_sales_bil)),
            mean=0.0,
            stddev=sigma_opex,
            dtype=tf.float64,
        )
        opex_samples_bil = (base_samples * cum_inf_t) + (var_samples * sales) + noise
        opex_samples_bil = opex_samples_bil.numpy()

        lower_opex_bil = np.percentile(opex_samples_bil, lower_q, axis=0)
        upper_opex_bil = np.percentile(opex_samples_bil, upper_q, axis=0)

    amount_scale = model.amount_scale
    mean_opex_usd = mean_opex_bil * amount_scale
    upper_opex_usd = upper_opex_bil * amount_scale
    lower_opex_usd = lower_opex_bil * amount_scale
    opex_hist_usd = historical_opex_bil * amount_scale
    sales_hist_usd = historical_sales_bil * amount_scale

    plt.figure(figsize=(10, 5))
    plt.plot(
        historical_years,
        opex_hist_usd,
        "o-",
        label="Historical OpEx",
        color="black",
    )
    plt.plot(
        historical_years,
        mean_opex_usd,
        "o-",
        label="Mean OpEx (learned)",
        color="tab:blue",
    )
    plt.plot(
        historical_years,
        lower_opex_usd,
        "--",
        label=f"Posterior predictive {lower_q:.0f}%",
        color="tab:blue",
        alpha=0.8,
    )
    plt.plot(
        historical_years,
        upper_opex_usd,
        "--",
        label=f"Posterior predictive {upper_q:.0f}%",
        color="tab:blue",
        alpha=0.8,
    )
    plt.title("OpEx vs Year with Learned Probabilistic Linear Regression")
    plt.xlabel("Year")
    plt.ylabel("OpEx (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "gaussian_ci" if use_gaussian_ci else "monte_carlo"
    plt.savefig(f"opex_probabilistic_fit_{timestamp}_{tag}.png", dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()

    # --- OpEx vs Sales (separate figure) ---
    # Add x-axis padding to visualize extrapolation beyond training range
    x_min = float(np.min(sales_hist_usd))
    x_max = float(np.max(sales_hist_usd))
    x_span = x_max - x_min if x_max > x_min else max(abs(x_max), 1.0)
    # x_pad = 10.0 * x_span
    # x_pad = 0.15 * x_span
    x_pad = 0.5 * x_span
    x_left = x_min - x_pad
    x_right = x_max + x_pad

    # Extend the regression lines to the padded range
    sales_grid_usd = np.linspace(x_left, x_right, 200)
    sales_grid_bil = sales_grid_usd / amount_scale
    # Center the sales grid using the offset
    sales_grid_bil_centered = sales_grid_bil - sales_offset
    cum_inf_mean = float(np.mean(cum_inf))
    mean_opex_grid_bil = (mean_base_opex * cum_inf_mean) + (
        mean_var_opex * sales_grid_bil_centered
    )
    mean_opex_grid_usd = mean_opex_grid_bil * amount_scale

    if use_gaussian_ci:
        var_var = float(model.q_var_opex_scale.numpy()) ** 2
        var_base = float(model.q_base_opex_scale.numpy()) ** 2
        var_noise = float(sigma_opex) ** 2
        # Use centered sales for variance calculation
        std_opex_grid_bil = np.sqrt(
            (cum_inf_mean**2) * var_base
            + (sales_grid_bil_centered**2) * var_var
            + var_noise
        )
        z_low = float(tfd.Normal(0.0, 1.0).quantile(lower_q / 100.0))
        z_up = float(tfd.Normal(0.0, 1.0).quantile(upper_q / 100.0))
        lower_opex_grid_bil = mean_opex_grid_bil + z_low * std_opex_grid_bil
        upper_opex_grid_bil = mean_opex_grid_bil + z_up * std_opex_grid_bil
    else:
        q_var = tfd.Normal(loc=model.q_var_opex_loc, scale=model.q_var_opex_scale)
        q_base = tfd.Normal(loc=model.q_base_opex_loc, scale=model.q_base_opex_scale)
        var_samples = q_var.sample(n_samples)
        base_samples = q_base.sample(n_samples)
        var_samples = tf.reshape(var_samples, (-1, 1))
        base_samples = tf.reshape(base_samples, (-1, 1))
        # Use centered sales for grid sampling
        sales_grid_t = tf.reshape(
            tf.convert_to_tensor(sales_grid_bil_centered, dtype=tf.float64), (1, -1)
        )
        cum_inf_grid_t = tf.reshape(
            tf.convert_to_tensor(
                np.full_like(sales_grid_bil, cum_inf_mean), dtype=tf.float64
            ),
            (1, -1),
        )
        noise_grid = tf.random.normal(
            shape=(n_samples, len(sales_grid_bil)),
            mean=0.0,
            stddev=sigma_opex,
            dtype=tf.float64,
        )
        opex_samples_grid_bil = (
            (base_samples * cum_inf_grid_t) + (var_samples * sales_grid_t) + noise_grid
        )
        opex_samples_grid_bil = opex_samples_grid_bil.numpy()
        lower_opex_grid_bil = np.percentile(opex_samples_grid_bil, lower_q, axis=0)
        upper_opex_grid_bil = np.percentile(opex_samples_grid_bil, upper_q, axis=0)
    lower_opex_grid_usd = lower_opex_grid_bil * amount_scale
    upper_opex_grid_usd = upper_opex_grid_bil * amount_scale

    plt.figure(figsize=(10, 5))
    plt.scatter(
        sales_hist_usd,
        opex_hist_usd,
        label="Historical OpEx",
        color="black",
        zorder=3,
    )
    # Per-data-point predictions using actual per-year cum_inf (matches vs-Year plot)
    plt.scatter(
        sales_hist_usd,
        mean_opex_usd,
        label="Mean OpEx per data point (learned)",
        color="tab:blue",
        marker="x",
        s=80,
        zorder=4,
    )
    plt.plot(
        sales_grid_usd,
        mean_opex_grid_usd,
        "-",
        label="Mean OpEx trend (avg. inflation)",
        color="tab:blue",
        alpha=0.5,
    )
    plt.plot(
        sales_grid_usd,
        lower_opex_grid_usd,
        "--",
        label=f"Posterior predictive {lower_q:.0f}%",
        color="tab:blue",
        alpha=0.8,
    )
    plt.plot(
        sales_grid_usd,
        upper_opex_grid_usd,
        "--",
        label=f"Posterior predictive {upper_q:.0f}%",
        color="tab:blue",
        alpha=0.8,
    )
    plt.xlim(x_left, x_right)
    plt.title("OpEx vs Sales with Learned Probabilistic Linear Regression")
    plt.xlabel("Sales (USD)")
    plt.ylabel("OpEx (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "gaussian_ci" if use_gaussian_ci else "monte_carlo"
    plt.savefig(f"opex_vs_sales_fit_{timestamp}_{tag}.png", dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_historical_and_forecast(
    historical_years,
    forecast_years,
    historical_data,
    forecast_trajectories,
    amount_scale,
    sales_hist_usd=None,
    sales_forecast_usd=None,
    historical_fit=None,
    historical_fit_years=None,
    show_plot=False,
):
    """
    Plots all financial elements from historical period through forecast period,
    optionally overlaying the model's one-step-ahead fitted values on historical data.

    Args:
        historical_years: array of year labels for historical data (e.g., [2018, ..., 2025])
        forecast_years: array of year labels for forecast data (e.g., [2025, ..., 2033])
        historical_data: dict of {name: array_in_usd} for historical data
        forecast_trajectories: dict of {name: array[n_samples, n_years] in scaled units}
        amount_scale: scaling factor to convert scaled units back to USD
        sales_hist_usd: optional array of historical sales in USD
        sales_forecast_usd: optional array of deterministic sales forecast in USD
        historical_fit: optional dict of {name: array_in_usd} for model-fitted historical values
        historical_fit_years: optional array of year labels for fitted values
        show_plot: whether to call plt.show()
    """
    elements = list(forecast_trajectories.keys())
    n_elements = len(elements)

    # Compute mean, 2.5%, 97.5% for each element
    forecast_stats = {}
    for name in elements:
        trajs = forecast_trajectories[name]
        forecast_stats[name] = {
            "mean": np.mean(trajs, axis=0) * amount_scale,
            "lower": np.percentile(trajs, 2.5, axis=0) * amount_scale,
            "upper": np.percentile(trajs, 97.5, axis=0) * amount_scale,
        }

    # Layout: add 1 for sales if provided
    total_plots = n_elements + (1 if sales_forecast_usd is not None else 0)
    ncols = 3
    nrows = (total_plots + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    axs = axs.flatten()

    # Readable display names
    display_names = {
        "net_income": "Net Income",
        "total_assets": "Total Assets",
        "nca": "Non-Current Assets",
        "advance_payments_purchases": "Advance Payments (Purchases)",
        "accounts_receivable": "Accounts Receivable",
        "inventory": "Inventory",
        "cash": "Cash",
        "investment_in_market_securities": "Investment in Market Securities",
        "accounts_payable": "Accounts Payable",
        "advance_payments_sales": "Advance Payments (Sales)",
        "current_liabilities": "Current Liabilities",
        "non_current_liabilities": "Non-Current Liabilities",
        "equity": "Stockholders' Equity",
        "depreciation": "Depreciation",
        "dividends": "Dividends",
        "stock_buyback": "Stock Buyback",
        "new_short_term_loan": "New Short-Term Loan",
        "new_long_term_loan": "New Long-Term Loan",
        "equity_financing": "Equity Financing",
        "liquidity_deficit_st": "Liquidity Deficit (Short-Term)",
    }

    ax_idx = 0

    # Plot Sales (deterministic — exogenous input, no model fit)
    if sales_forecast_usd is not None:
        ax = axs[ax_idx]
        if sales_hist_usd is not None:
            ax.plot(
                historical_years,
                sales_hist_usd,
                "ko-",
                label="Historical",
                markersize=5,
                linewidth=1.5,
            )
        ax.plot(
            forecast_years,
            sales_forecast_usd,
            "s-",
            color="tab:blue",
            label="Forecast",
            markersize=5,
            linewidth=1.5,
        )
        ax.set_title(
            "Sales (Revenue) [Exogenous Input]", fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("USD")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.tick_params(axis="x", rotation=45)
        ax_idx += 1

    # Plot each forecasted element
    for name in elements:
        ax = axs[ax_idx]
        stats = forecast_stats[name]
        label = display_names.get(name, name)

        # Historical actual data
        if name in historical_data:
            ax.plot(
                historical_years,
                historical_data[name],
                "ko-",
                label="Historical",
                markersize=5,
                linewidth=1.5,
            )

        # Model fit on historical data (one-step-ahead predictions)
        if (
            historical_fit is not None
            and historical_fit_years is not None
            and name in historical_fit
        ):
            ax.plot(
                historical_fit_years,
                historical_fit[name],
                "^--",
                color="tab:red",
                label="Model Fit (1-step)",
                markersize=5,
                linewidth=1.2,
                alpha=0.85,
            )

        # Forecast mean + 95% CI
        ax.plot(
            forecast_years,
            stats["mean"],
            "s-",
            color="tab:blue",
            label="Forecast Mean",
            markersize=5,
            linewidth=1.5,
        )
        ax.fill_between(
            forecast_years,
            stats["lower"],
            stats["upper"],
            color="tab:blue",
            alpha=0.2,
            label="95% CI",
        )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("USD")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.tick_params(axis="x", rotation=45)
        ax_idx += 1

    # Hide unused axes
    for i in range(ax_idx, len(axs)):
        axs[i].set_visible(False)

    fig.suptitle(
        "Financial Model: Historical Fit & Monte Carlo Forecast",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"all_elements_forecast_{timestamp}.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: all_elements_forecast_{timestamp}.png")
    if show_plot:
        plt.show()
    else:
        plt.close()


def run_training_and_forecast(
    use_trained_parameters=False,
    parameters_path="trained_parameters.npz",
    use_inflation=True,
):
    model = TrainableFinancialModel(base_year=2018)

    # --- 1. HISTORICAL DATA FROM APPLE (2018-2025)---
    # Revenues from Income Statement
    sales_hist = np.array(
        [
            2.65595e11,
            2.60174e11,
            2.74515e11,
            3.65817e11,
            3.94328e11,
            3.83285e11,
            3.91035e11,
            4.16161e11,
        ],
        dtype=np.float64,
    )
    # Inventory from Balance Sheet, includes one additional year at the beginning to determine purchases history
    inventory_hist_plus_one = np.array(
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
        ],
        dtype=np.float64,
    )
    # Depreciation from Reconciled Depreciation in Income Statement
    depr_hist = np.array(
        [
            10903000000,
            12547000000,
            11056000000,
            11284000000,
            11104000000,
            11519000000,
            11445000000,
            11698000000,
        ],
        dtype=np.float64,
    )
    # COGS from Cost of Revenue - Depreciation in Income Statement
    cogs_hist = np.array(
        [
            1.52853e11,
            1.49235e11,
            1.58503e11,
            2.01697e11,
            2.12442e11,
            2.02618e11,
            1.98907e11,
            2.09262e11,
        ],
        dtype=np.float64,
    )
    # Purchases from COGS + Inventory_t - Inventory_t-1
    purchases_hist = (
        cogs_hist + inventory_hist_plus_one[1:] - inventory_hist_plus_one[:-1]
    )
    # Inventory with matched length
    inventory_hist = inventory_hist_plus_one[1:]
    # Non-current Assets from Balance Sheet
    nca_hist = np.array(
        [
            2.34386e11,
            1.75697e11,
            1.80175e11,
            2.16166e11,
            2.1735e11,
            2.09017e11,
            2.11993e11,
            2.11284e11,
        ],
        dtype=np.float64,
    )
    # Advance Payments for Purchases from Other Current Assets in Balance Sheet
    advance_payments_purchases_hist = np.array(
        [
            12087000000,
            12352000000,
            11264000000,
            14111000000,
            21223000000,
            14695000000,
            14287000000,
            14585000000,
        ],
        dtype=np.float64,
    )
    # Accounts Receivable from Receivables in Balance Sheet
    accounts_receivable_hist = np.array(
        [
            48995000000,
            45804000000,
            37445000000,
            51506000000,
            60932000000,
            60985000000,
            66243000000,
            72957000000,
        ],
        dtype=np.float64,
    )
    # Cash from Cash and Cash Equivalents in Balance Sheet
    cash_hist = np.array(
        [
            25913000000,
            48844000000,
            38016000000,
            34940000000,
            23646000000,
            29965000000,
            29943000000,
            35934000000,
        ],
        dtype=np.float64,
    )
    # Investment in Market Securities from Other Short Term Investments in Balance Sheet
    investment_in_market_securities_hist = np.array(
        [
            40388000000,
            51713000000,
            52927000000,
            27699000000,
            24658000000,
            31590000000,
            35228000000,
            18763000000,
        ],
        dtype=np.float64,
    )
    # Accounts Payable from Balance Sheet
    accounts_payable_hist = np.array(
        [
            55888000000,
            46236000000,
            42296000000,
            54763000000,
            64115000000,
            62611000000,
            68960000000,
            69860000000,
        ],
        dtype=np.float64,
    )
    # Advance Payments Sales from Current Deferred Revenue in Balance Sheet
    advance_payments_sales_hist = np.array(
        [
            5966000000,
            5522000000,
            6643000000,
            7612000000,
            7912000000,
            8061000000,
            8249000000,
            9055000000,
        ],
        dtype=np.float64,
    )
    # Current Liabilities from Total Current Liabilities - Accounts Payable - Current Deferred Revenue in Balance Sheet
    current_liabilities_hist = np.array(
        [
            55012000000,
            53960000000,
            56453000000,
            63106000000,
            81955000000,
            74636000000,
            99183000000,
            86716000000,
        ],
        dtype=np.float64,
    )
    # Non-current Liabilities from Total Non Current Liabilities Net Minority Interest in Balance Sheet
    non_current_liabilities_hist = np.array(
        [
            1.41712e11,
            1.4231e11,
            1.53157e11,
            1.62431e11,
            1.48101e11,
            1.45129e11,
            1.31638e11,
            1.19877e11,
        ],
        dtype=np.float64,
    )
    # Equity from Stockholders Equity in Balance Sheet
    equity_hist = np.array(
        [
            1.07147e11,
            90488000000,
            65339000000,
            63090000000,
            50672000000,
            62146000000,
            56950000000,
            73733000000,
        ],
        dtype=np.float64,
    )
    # Net Income from Income Statement
    net_income_hist = np.array(
        [
            59531000000,
            55256000000,
            57411000000,
            94680000000,
            99803000000,
            96995000000,
            93736000000,
            1.1201e11,
        ],
        dtype=np.float64,
    )
    # Dividends paid this year from Common Stock Dividends Paid in Cash Flow
    dividends_hist = np.array(
        [
            13712000000,
            14119000000,
            14081000000,
            14467000000,
            14841000000,
            15025000000,
            15234000000,
            15421000000,
        ],
        dtype=np.float64,
    )
    # Stock Buyback from Repurchase of Capital Stock in Cash Flow
    stock_buyback_hist = np.array(
        [
            72738000000,
            66897000000,
            72358000000,
            85971000000,
            89402000000,
            77550000000,
            94949000000,
            90711000000,
        ],
        dtype=np.float64,
    )
    # OpEx from Operating Expenses in Income Statement
    opex_hist = np.array(
        [
            30941000000,
            34462000000,
            38668000000,
            43887000000,
            51345000000,
            54847000000,
            57467000000,
            62151000000,
        ],
        dtype=np.float64,
    )
    # Tax Provision from Income Statement
    tax_hist = np.array(
        [
            13372000000,
            10481000000,
            9680000000,
            14527000000,
            19300000000,
            16741000000,
            29749000000,
            20719000000,
        ],
        dtype=np.float64,
    )
    # Short-Term Debt (Commercial Paper / Revolving Credit) from Balance Sheet
    st_debt_hist = np.array(
        [
            45291000000,
            43700000000,
            47680000000,
            53493000000,
            70827000000,
            64814000000,
            88271000000,
            74366000000,
        ],
        dtype=np.float64,
    )
    # Inflation History
    inflation_hist = np.array(
        [0.024, 0.018, 0.012, 0.047, 0.08, 0.041, 0.029, 0.027],
        dtype=np.float64,
    )
    if not use_inflation:
        inflation_hist = np.zeros_like(inflation_hist)

    # --- 2. SCALE INPUTS AND TARGETS TO BILLIONS FOR TRAINING STABILITY ---
    amount_scale = model.amount_scale
    sales_hist_bil = sales_hist / amount_scale
    purchases_hist_bil = purchases_hist / amount_scale
    cogs_hist_bil = cogs_hist / amount_scale
    nca_hist_bil = nca_hist / amount_scale
    depr_hist_bil = depr_hist / amount_scale
    advance_payments_sales_hist_bil = advance_payments_sales_hist / amount_scale
    advance_payments_purchases_hist_bil = advance_payments_purchases_hist / amount_scale
    accounts_receivable_hist_bil = accounts_receivable_hist / amount_scale
    accounts_payable_hist_bil = accounts_payable_hist / amount_scale
    inventory_hist_bil = inventory_hist / amount_scale
    cash_hist_bil = cash_hist / amount_scale
    investment_in_market_securities_hist_bil = (
        investment_in_market_securities_hist / amount_scale
    )
    net_income_hist_bil = net_income_hist / amount_scale
    dividends_hist_bil = dividends_hist / amount_scale
    stock_buyback_hist_bil = stock_buyback_hist / amount_scale
    opex_hist_bil = opex_hist / amount_scale
    tax_hist_bil = tax_hist / amount_scale
    st_debt_hist_bil = st_debt_hist / amount_scale
    current_liabilities_hist_bil = current_liabilities_hist / amount_scale
    non_current_liabilities_hist_bil = non_current_liabilities_hist / amount_scale
    equity_hist_bil = equity_hist / amount_scale

    if use_trained_parameters:
        model.load_parameters(parameters_path)
    else:
        # --- 2. TRAIN THE MODEL ---
        # We feed in the historical arrays from 2018-2024, and leave 2025 for forecast testing.
        # Historical years: FY2018..FY2024 (training), FY2025 held out for testing
        n_train = len(sales_hist_bil[:-1])
        train_years = np.arange(
            model.base_year, model.base_year + n_train, dtype=np.float64
        )
        model.train_simple_policies(
            sales_hist_bil[:-1],
            purchases_hist_bil[:-1],
            cogs_hist_bil[:-1],
            nca_hist_bil[:-1],
            depr_hist_bil[:-1],
            advance_payments_sales_hist_bil[:-1],
            advance_payments_purchases_hist_bil[:-1],
            accounts_receivable_hist_bil[:-1],
            accounts_payable_hist_bil[:-1],
            inventory_hist_bil[:-1],
            cash_hist_bil[:-1],
            investment_in_market_securities_hist_bil[:-1],
            net_income_hist_bil[:-1],
            dividends_hist_bil[:-1],
            stock_buyback_hist_bil[:-1],
            opex_hist_bil[:-1],
            tax_hist_bil[:-1],
            historical_st_debt=st_debt_hist_bil[:-1],
            historical_inflation=inflation_hist[:-1],
            historical_years=train_years,
            show_plot=False,
        )

        # --- 3. TRAIN STRUCTURAL PARAMETERS ---
        # We still only feed in the historical arrays from FY2018-FY2024, and leave FY2025 for forecast testing.
        model.train_structural_parameters(
            sales_hist_bil[:-1],
            nca_hist_bil[:-1],
            advance_payments_sales_hist_bil[:-1],
            advance_payments_purchases_hist_bil[:-1],
            accounts_receivable_hist_bil[:-1],
            accounts_payable_hist_bil[:-1],
            inventory_hist_bil[:-1],
            cash_hist_bil[:-1],
            investment_in_market_securities_hist_bil[:-1],
            net_income_hist_bil[:-1],
            dividends_hist_bil[:-1],
            stock_buyback_hist_bil[:-1],
            opex_hist_bil[:-1],
            tax_hist_bil[:-1],
            current_liabilities_hist_bil[:-1],
            non_current_liabilities_hist_bil[:-1],
            equity_hist_bil[:-1],
            inflation_hist[:-1],
            train_years,
        )
        model.save_parameters(parameters_path)

    # --- 4. PLOT OPEX FIT (Mean + Aleatoric Sigma) ---
    historical_years = np.arange(1, len(opex_hist_bil) + 1)

    # Posterior prediction by Gaussian Confidence Interval
    plot_opex_fit_with_aleatoric_noise(
        model,
        historical_years,
        sales_hist_bil,
        opex_hist_bil,
        inflation_hist,
        show_plot=False,
        use_gaussian_ci=True,
    )

    # Posterior prediction by sampling (Monte Carlo)
    plot_opex_fit_with_aleatoric_noise(
        model,
        historical_years,
        sales_hist_bil,
        opex_hist_bil,
        inflation_hist,
        show_plot=False,
        use_gaussian_ci=False,
    )

    # --- 5. RUN FORECAST (Using new parameters) ---
    # Initial State (t=0) 2024 Apple Balance Sheet
    state = {
        "nca": tf.constant(nca_hist_bil[-2], dtype=tf.float64),
        "advance_payments_purchases": tf.constant(
            advance_payments_purchases_hist_bil[-2], dtype=tf.float64
        ),
        "accounts_receivable": tf.constant(
            accounts_receivable_hist_bil[-2], dtype=tf.float64
        ),
        "inventory": tf.constant(inventory_hist_bil[-2], dtype=tf.float64),
        "cash": tf.constant(cash_hist_bil[-2], dtype=tf.float64),
        "investment_in_market_securities": tf.constant(
            investment_in_market_securities_hist_bil[-2], dtype=tf.float64
        ),
        "accounts_payable": tf.constant(
            accounts_payable_hist_bil[-2], dtype=tf.float64
        ),
        "advance_payments_sales": tf.constant(
            advance_payments_sales_hist_bil[-2], dtype=tf.float64
        ),
        "current_liabilities": tf.constant(
            current_liabilities_hist_bil[-2], dtype=tf.float64
        ),
        "non_current_liabilities": tf.constant(
            non_current_liabilities_hist_bil[-2], dtype=tf.float64
        ),
        "equity": tf.constant(equity_hist_bil[-2], dtype=tf.float64),
        "net_income": tf.constant(net_income_hist_bil[-2], dtype=tf.float64),
        "dividends": tf.constant(dividends_hist_bil[-2], dtype=tf.float64),
    }

    # Forecast Drivers: Sales is the sole exogenous driver.
    # Purchases are derived inside forecast_step from the learned cost ratio.
    n_hist = len(sales_hist)  # 8 (FY2018-FY2025)
    n_forecast_years = 10
    # Average linear growth per year from full historical data
    yearly_deltas = np.diff(sales_hist_bil)
    avg_linear_growth = np.mean(yearly_deltas)
    sales_forecast = np.array(
        [sales_hist_bil[-1] + avg_linear_growth * i for i in range(n_forecast_years)],
        dtype=np.float64,
    )
    # Forecast starts at FY2025 (last historical year) and continues forward
    last_hist_year = model.base_year + n_hist - 1  # FY2025
    forecast_years = np.arange(
        last_hist_year, last_hist_year + n_forecast_years, dtype=np.float64
    )

    # Year 1 to 4 inflation rate (2025-2028)
    inflation_forecast = np.array(
        [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        dtype=np.float64,
    )
    if not use_inflation:
        inflation_forecast = np.zeros_like(inflation_forecast)
    cum_inf_forecast = np.cumprod(1 + inflation_forecast)

    # --- Execute Monte Carlo Forecast ---
    forecast_trajectories = run_monte_carlo_forecast(
        model,
        state,
        sales_forecast,
        cum_inf_forecast,
        forecast_years,
        n_samples=1000,
    )

    # --- 6. COMPUTE ONE-STEP-AHEAD HISTORICAL FIT ---
    # For each year t+1, use actual state at t and predict state at t+1
    # This shows how well the model's learned parameters fit the historical data.
    n_hist_points = len(sales_hist)
    cum_inf_hist = np.cumprod(1 + inflation_hist)

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
        "current_liabilities",
        "non_current_liabilities",
        "equity",
        "depreciation",
        "dividends",
        "stock_buyback",
        "new_short_term_loan",
        "new_long_term_loan",
        "equity_financing",
        "liquidity_deficit_st",
    ]
    historical_fit = {k: [] for k in hist_fit_keys}
    historical_fit_years = []

    for t in range(n_hist_points - 1):  # t = 0..6, predicting index t+1
        # Actual state at year t
        state_t = {
            "nca": tf.constant(nca_hist_bil[t], dtype=tf.float64),
            "advance_payments_purchases": tf.constant(
                advance_payments_purchases_hist_bil[t], dtype=tf.float64
            ),
            "accounts_receivable": tf.constant(
                accounts_receivable_hist_bil[t], dtype=tf.float64
            ),
            "inventory": tf.constant(inventory_hist_bil[t], dtype=tf.float64),
            "cash": tf.constant(cash_hist_bil[t], dtype=tf.float64),
            "investment_in_market_securities": tf.constant(
                investment_in_market_securities_hist_bil[t], dtype=tf.float64
            ),
            "accounts_payable": tf.constant(
                accounts_payable_hist_bil[t], dtype=tf.float64
            ),
            "advance_payments_sales": tf.constant(
                advance_payments_sales_hist_bil[t], dtype=tf.float64
            ),
            "current_liabilities": tf.constant(
                current_liabilities_hist_bil[t], dtype=tf.float64
            ),
            "non_current_liabilities": tf.constant(
                non_current_liabilities_hist_bil[t], dtype=tf.float64
            ),
            "equity": tf.constant(equity_hist_bil[t], dtype=tf.float64),
            "net_income": tf.constant(net_income_hist_bil[t], dtype=tf.float64),
            "dividends": tf.constant(dividends_hist_bil[t], dtype=tf.float64),
        }

        # Sales at t+1 (current) and t+2 (lookahead for advance payments)
        sales_t1 = sales_hist_bil[t + 1]
        if t + 2 < n_hist_points:
            sales_t2 = sales_hist_bil[t + 2]
        else:
            # For the last historical transition, use projected FY2026 sales
            sales_t2 = sales_forecast[1]

        inputs_t = {
            "sales_t": tf.constant(sales_t1, dtype=tf.float64),
            "sales_t_plus_1": tf.constant(sales_t2, dtype=tf.float64),
            "year": tf.constant(float(model.base_year + t + 1), dtype=tf.float64),
            "cum_inflation": tf.constant(cum_inf_hist[t + 1], dtype=tf.float64),
        }

        pred = model.forecast_step(state_t, inputs_t, use_mean_opex=True)

        # Collect predicted values (convert back to USD)
        historical_fit["net_income"].append(
            float(pred["net_income"].numpy()) * amount_scale
        )
        historical_fit["nca"].append(float(pred["nca"].numpy()) * amount_scale)
        historical_fit["advance_payments_purchases"].append(
            float(pred["advance_payments_purchases"].numpy()) * amount_scale
        )
        historical_fit["accounts_receivable"].append(
            float(pred["accounts_receivable"].numpy()) * amount_scale
        )
        historical_fit["inventory"].append(
            float(pred["inventory"].numpy()) * amount_scale
        )
        historical_fit["cash"].append(float(pred["cash"].numpy()) * amount_scale)
        historical_fit["investment_in_market_securities"].append(
            float(pred["investment_in_market_securities"].numpy()) * amount_scale
        )
        historical_fit["accounts_payable"].append(
            float(pred["accounts_payable"].numpy()) * amount_scale
        )
        historical_fit["advance_payments_sales"].append(
            float(pred["advance_payments_sales"].numpy()) * amount_scale
        )
        historical_fit["current_liabilities"].append(
            float(pred["current_liabilities"].numpy()) * amount_scale
        )
        historical_fit["non_current_liabilities"].append(
            float(pred["non_current_liabilities"].numpy()) * amount_scale
        )
        historical_fit["equity"].append(float(pred["equity"].numpy()) * amount_scale)
        historical_fit["depreciation"].append(
            float(pred["depreciation"].numpy()) * amount_scale
        )
        historical_fit["dividends"].append(
            float(pred["dividends"].numpy()) * amount_scale
        )
        historical_fit["stock_buyback"].append(
            float(pred["stock_buyback"].numpy()) * amount_scale
        )
        historical_fit["new_short_term_loan"].append(
            float(pred["new_short_term_loan"].numpy()) * amount_scale
        )
        historical_fit["new_long_term_loan"].append(
            float(pred["new_long_term_loan"].numpy()) * amount_scale
        )
        historical_fit["equity_financing"].append(
            float(pred["equity_financing"].numpy()) * amount_scale
        )
        historical_fit["liquidity_deficit_st"].append(
            float(pred["liquidity_deficit_st"].numpy()) * amount_scale
        )

        total_assets_pred = (
            pred["nca"]
            + pred["advance_payments_purchases"]
            + pred["accounts_receivable"]
            + pred["inventory"]
            + pred["cash"]
            + pred["investment_in_market_securities"]
        )
        historical_fit["total_assets"].append(
            float(total_assets_pred.numpy()) * amount_scale
        )

        historical_fit_years.append(model.base_year + t + 1)

    # Convert to numpy arrays
    for k in historical_fit:
        historical_fit[k] = np.array(historical_fit[k])
    historical_fit_years = np.array(historical_fit_years)

    # --- 7. PLOT ALL ELEMENTS: HISTORICAL + FIT + FORECAST ---
    historical_years = np.arange(model.base_year, model.base_year + n_hist_points)

    # Forecast years for plotting: FY2025, FY2026, ..., FY2033 (9 years)
    n_forecast_steps = len(sales_forecast) - 1
    forecast_year_start = model.base_year + n_hist_points - 1  # FY2025
    plot_forecast_years = np.arange(
        forecast_year_start, forecast_year_start + n_forecast_steps
    )

    # Build historical data dict (in USD, not scaled)
    total_assets_hist = (
        nca_hist
        + advance_payments_purchases_hist
        + accounts_receivable_hist
        + inventory_hist
        + cash_hist
        + investment_in_market_securities_hist
    )
    historical_data = {
        "net_income": net_income_hist,
        "total_assets": total_assets_hist,
        "nca": nca_hist,
        "advance_payments_purchases": advance_payments_purchases_hist,
        "accounts_receivable": accounts_receivable_hist,
        "inventory": inventory_hist,
        "cash": cash_hist,
        "investment_in_market_securities": investment_in_market_securities_hist,
        "accounts_payable": accounts_payable_hist,
        "advance_payments_sales": advance_payments_sales_hist,
        "current_liabilities": current_liabilities_hist,
        "non_current_liabilities": non_current_liabilities_hist,
        "equity": equity_hist,
        "depreciation": depr_hist,
        "dividends": dividends_hist,
        "stock_buyback": stock_buyback_hist,
        "new_short_term_loan": st_debt_hist,
    }

    # Sales forecast in USD for the forecasted years
    sales_forecast_usd = sales_forecast[:n_forecast_steps] * amount_scale

    plot_historical_and_forecast(
        historical_years=historical_years,
        forecast_years=plot_forecast_years,
        historical_data=historical_data,
        forecast_trajectories=forecast_trajectories,
        amount_scale=amount_scale,
        sales_hist_usd=sales_hist,
        sales_forecast_usd=sales_forecast_usd,
        historical_fit=historical_fit,
        historical_fit_years=historical_fit_years,
        show_plot=False,
    )


if __name__ == "__main__":
    run_training_and_forecast(
        use_trained_parameters=False,
        parameters_path="trained_parameters.npz",
        use_inflation=True,  # Set to False to disable inflation (all rates → 0%)
    )
