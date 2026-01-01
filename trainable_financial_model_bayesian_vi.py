import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors


# --- 1. Define the Trainable Model ---
class TrainableFinancialModel(tf.Module):
    def __init__(self):
        # --- Policy Parameters (Deterministic) ---
        ## These are trainable with simple linear regression
        self.asset_growth = tf.Variable(
            0.0076, name="asset_growth", dtype=tf.float64
        )  # %AG
        self.depreciation_rate = tf.Variable(
            0.055, name="depr_rate", dtype=tf.float64
        )  # %Depr
        self.advance_payments_sales_pct = tf.Variable(
            0.0206, name="adv_ps", dtype=tf.float64
        )  # %AdvPS
        self.advance_payments_purchases_pct = tf.Variable(
            0.0735, name="adv_pp", dtype=tf.float64
        )  # %AdvPP
        self.account_receivables_pct = tf.Variable(
            0.1591, name="ar_pct", dtype=tf.float64
        )  # %AR
        self.account_payables_pct = tf.Variable(
            0.3501, name="ap_pct", dtype=tf.float64
        )  # %AP
        self.inventory_pct = tf.Variable(
            0.0165, name="inv_pct", dtype=tf.float64
        )  # %Inv
        self.total_liquidity_pct = tf.Variable(
            0.16, name="tl_pct", dtype=tf.float64
        )  # %TL
        self.cash_pct_of_liquidity = tf.Variable(
            0.487, name="cash_pct", dtype=tf.float64
        )  # %Cash
        self.income_tax_pct = tf.Variable(
            0.147, name="tax_pct", dtype=tf.float64
        )  # %IT
        self.dividend_payout_ratio_pct = tf.Variable(
            0.15, name="div_pct", dtype=tf.float64
        )  # %PR
        self.stock_buyback_pct = tf.Variable(
            7.5, name="bb_pct", dtype=tf.float64
        )  # %BB

        # --- BAYESIAN OPEX PARAMETERS (Variational Inference) ---
        # We learn a distribution (Normal) defined by a Mean (loc) and StdDev (scale)

        # 1. Variable OpEx %
        self.q_var_opex_loc = tf.Variable(0.22, dtype=tf.float64, name="q_var_opex_loc")
        self.q_var_opex_scale = tfp.util.TransformedVariable(
            initial_value=0.01,
            bijector=tfb.Softplus(),  # Ensures scale is always positive
            dtype=tf.float64,
            name="q_var_opex_scale",
        )

        # 2. Baseline OpEx (Large negative number)
        self.q_base_opex_loc = tf.Variable(
            -3.0e10, dtype=tf.float64, name="q_base_opex_loc"
        )
        self.q_base_opex_scale = tfp.util.TransformedVariable(
            initial_value=1.0e9,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="q_base_opex_scale",
        )

        # 3. Aleatoric Uncertainty (The inherent noise in the OpEx data)
        self.noise_sigma = tfp.util.TransformedVariable(
            initial_value=1.0e9,
            bijector=tfb.Softplus(),
            dtype=tf.float64,
            name="noise_sigma",
        )

        # --- Structural Parameters ---
        ## These are trained with gradient descent with the trained variables from above and other data (sales, purchases, equity, liabilities, etc.) as inputs
        self.avg_short_term_interest_pct = tf.Variable(
            0.6, name="avg_short_term_interest_pct", dtype=tf.float64
        )  # %AvgSTInt
        self.avg_long_term_interest_pct = tf.Variable(
            0.06, name="avg_long_term_interest_pct", dtype=tf.float64
        )  # %AvgLTInt
        self.avg_maturity_years = tf.Variable(
            3.0, name="avg_maturity_years", dtype=tf.float64
        )  # AvgM
        self.market_securities_return_pct = tf.Variable(
            0.05, name="market_securities_return_pct", dtype=tf.float64
        )  # %MSReturn
        self.equity_financing_pct = tf.Variable(
            0.15, name="equity_financing_pct", dtype=tf.float64
        )  # %EF

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
        prior_var = tfd.Normal(loc=tf.constant(0.20, dtype=tf.float64), scale=0.1)
        # Prior: Baseline OpEx is around -30B with large wiggle room
        prior_base = tfd.Normal(
            loc=tf.constant(-3.0e10, dtype=tf.float64), scale=1.0e10
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
        historical_inflation=None,
        learning_rate=0.0001,
        epochs=5000,
    ):
        """
        Trains simple policy parameters using historical data.
        """

        # Convert inputs to tensors and ensure float64
        sales_tensor = tf.convert_to_tensor(historical_sales, dtype=tf.float64)
        purchases_tensor = tf.convert_to_tensor(historical_purchases, dtype=tf.float64)
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

        if historical_inflation is None:
            historical_inflation = tf.zeros_like(sales_tensor)
        inf_tensor = tf.convert_to_tensor(historical_inflation, dtype=tf.float64)
        cum_inf_tensor = tf.math.cumprod(1 + inf_tensor)

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

        # 4. Advance Payments Purchases: adv_pp_t = purchases_{t+1} * adv_pp_pct
        adv_pp_true = adv_pay_purch_tensor[:-1]
        purchases_next_aligned = purchases_tensor[1:]
        # 5. Dividends: div_t = ni_{t-1} * div_pct
        div_true = div_tensor[1:]
        ni_prev_aligned = ni_tensor[:-1]

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        print(f"Training on {len(historical_sales)} years of historical data...")

        # --- Training Loop ---
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
            # Bayesian Params
            self.q_var_opex_loc,
            self.q_var_opex_scale.trainable_variables[0],
            self.q_base_opex_loc,
            self.q_base_opex_scale.trainable_variables[0],
            self.noise_sigma.trainable_variables[0],
        ]

        for i in range(epochs):
            with tf.GradientTape() as tape:
                # --- Deterministic Losses (MSE) ---
                loss_growth = tf.reduce_mean(
                    tf.square(delta_nca_true - sales_aligned_growth * self.asset_growth)
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
                        - purchases_next_aligned * self.advance_payments_purchases_pct
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
                loss_tl = tf.reduce_mean(
                    tf.square(
                        (cash_tensor + ims_tensor)
                        - sales_tensor * self.total_liquidity_pct
                    )
                )
                loss_cash = tf.reduce_mean(
                    tf.square(
                        cash_tensor
                        - (cash_tensor + ims_tensor) * self.cash_pct_of_liquidity
                    )
                )
                loss_tax = tf.reduce_mean(
                    tf.square(tax_tensor - ni_tensor * self.income_tax_pct)
                )
                loss_div = tf.reduce_mean(
                    tf.square(
                        div_true - ni_prev_aligned * self.dividend_payout_ratio_pct
                    )
                )
                loss_bb = tf.reduce_mean(
                    tf.square(bb_tensor - depr_tensor * self.stock_buyback_pct)
                )

                # --- Bayesian OpEx Loss (Negative ELBO) ---
                # 1. Sample from Posterior
                var_opex_sample, base_opex_sample = self.sample_opex_params()

                # 2. Predict OpEx using samples
                # opex = baseline * product(1+inf) + var * sales
                pred_opex_mean = (base_opex_sample * cum_inf_tensor) + (
                    var_opex_sample * sales_tensor
                )

                # 3. Calculate Negative Log Likelihood
                # We assume the Observed OpEx comes from N(pred_mean, noise_sigma)
                likelihood_dist = tfd.Normal(loc=pred_opex_mean, scale=self.noise_sigma)
                neg_log_likelihood = -tf.reduce_sum(
                    likelihood_dist.log_prob(opex_tensor)
                )

                # 4. Calculate KL Divergence
                kl = self.get_opex_kl_divergence()

                # Weighting: Scale down likelihood or up KL?
                # Since we have few data points and large numbers, simple summation is risky.
                # Heuristic: Scale KL by 1.0 (standard) and treat NegLL as usual.
                # Note: Because financial numbers are ~1e10, NegLL will be huge.
                # We normalize the NegLL by a factor to make gradients stable relative to MSE losses.
                scale_factor = 1e-20  # Empirically helps with large numbers
                loss_opex_bayes = (neg_log_likelihood + kl) * scale_factor

                # Combined Loss (Heuristic: Normalize by scale to help Adam)
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
                    + loss_opex_bayes
                )

            # Compute Gradients
            grads = tape.gradient(total_loss, vars_to_train)

            # Apply Gradients
            optimizer.apply_gradients(zip(grads, vars_to_train))

            # --- Constraints (Clipping)
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

            if i % 1000 == 0:
                print(
                    f"Epoch {i}: Loss={total_loss.numpy():.4e} | OpEx Noise={self.noise_sigma.numpy():.2e}"
                )

        print("-" * 50)
        print("Training Complete.")
        print(f"Final %AG: {self.asset_growth.numpy():.5f}")
        print(f"Final %Depr: {self.depreciation_rate.numpy():.5f}")
        print(f"Final %AdvPS: {self.advance_payments_sales_pct.numpy():.5f}")
        print(f"Final %AdvPP: {self.advance_payments_purchases_pct.numpy():.5f}")
        print(f"Final %AR: {self.account_receivables_pct.numpy():.5f}")
        print(f"Final %AP: {self.account_payables_pct.numpy():.5f}")
        print(f"Final %Inv: {self.inventory_pct.numpy():.5f}")
        print(f"Final %TL: {self.total_liquidity_pct.numpy():.5f}")
        print(f"Final %Cash: {self.cash_pct_of_liquidity.numpy():.5f}")
        print(f"Final %IT: {self.income_tax_pct.numpy():.5f}")
        print(f"Final %PR: {self.dividend_payout_ratio_pct.numpy():.5f}")
        print(f"Final %BB: {self.stock_buyback_pct.numpy():.5f}")
        print(
            f"Bayesian OpEx Variable %: Mean={self.q_var_opex_loc.numpy():.4f}, Std={self.q_var_opex_scale.numpy():.4f}"
        )
        print(
            f"Bayesian OpEx Baseline:   Mean={self.q_base_opex_loc.numpy():.2e}, Std={self.q_base_opex_scale.numpy():.2e}"
        )

        print("-" * 50)

    def train_structural_parameters(
        self,
        historical_sales,
        historical_purchases,
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
        learning_rate=0.0001,
        epochs=5000,
    ):
        """
        Trains structural parameters (interest rates, maturity, financing)
        using historical state transitions.
        """
        # NOTE: When calling forecast_step inside here, we need to pass use_mean_opex=True
        # because we want to learn structural parameters based on the "most likely" OpEx, not noisy samples.
        sales_t = tf.convert_to_tensor(historical_sales, dtype=tf.float64)
        purch_t = tf.convert_to_tensor(historical_purchases, dtype=tf.float64)
        nca_t = tf.convert_to_tensor(historical_nca, dtype=tf.float64)
        adv_ps_t = tf.convert_to_tensor(historical_adv_pay_sales, dtype=tf.float64)
        adv_pp_t = tf.convert_to_tensor(historical_adv_pay_purch, dtype=tf.float64)
        ar_t = tf.convert_to_tensor(historical_ar, dtype=tf.float64)
        ap_t = tf.convert_to_tensor(historical_ap, dtype=tf.float64)
        inv_t = tf.convert_to_tensor(historical_inventory, dtype=tf.float64)
        cash_t = tf.convert_to_tensor(historical_cash, dtype=tf.float64)
        ims_t = tf.convert_to_tensor(historical_ims, dtype=tf.float64)
        ni_t = tf.convert_to_tensor(historical_net_income, dtype=tf.float64)
        cl_t = tf.convert_to_tensor(historical_current_liabilities, dtype=tf.float64)
        ncl_t = tf.convert_to_tensor(
            historical_non_current_liabilities, dtype=tf.float64
        )
        equity_t = tf.convert_to_tensor(historical_equity, dtype=tf.float64)

        if historical_inflation is None:
            historical_inflation = tf.zeros_like(sales_t)
        inf_t = tf.convert_to_tensor(historical_inflation, dtype=tf.float64)
        cum_inf_t = tf.math.cumprod(1 + inf_t)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        vars_to_train = [
            self.avg_short_term_interest_pct,
            self.avg_long_term_interest_pct,
            self.avg_maturity_years,
            self.market_securities_return_pct,
            self.equity_financing_pct,
        ]

        print(f"Training structural parameters...")
        for i in range(epochs):
            with tf.GradientTape() as tape:
                total_loss = 0.0
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
                    }

                    # Inputs for predicting state at t+1
                    inputs_curr = {
                        "sales_t": sales_t[t + 1],
                        "purchases_t": purch_t[t + 1],
                        "sales_t_plus_1": sales_t[t + 2],
                        "purchases_t_plus_1": purch_t[t + 2],
                        "cum_inflation": cum_inf_t[t + 1],
                    }

                    # IMPORTANT: Use mean (deterministic) OpEx for structural training
                    state_pred = self.forecast_step(
                        state_prev, inputs_curr, use_mean_opex=True
                    )

                    # Targets are values at t+1
                    loss_ni = tf.square(state_pred["net_income"] - ni_t[t + 1])
                    loss_cl = tf.square(state_pred["current_liabilities"] - cl_t[t + 1])
                    loss_ncl = tf.square(
                        state_pred["non_current_liabilities"] - ncl_t[t + 1]
                    )
                    loss_equity = tf.square(state_pred["equity"] - equity_t[t + 1])

                    # Heuristic normalization
                    total_loss += (loss_ni + loss_cl + loss_ncl + loss_equity) / 1e18

            grads = tape.gradient(total_loss, vars_to_train)
            optimizer.apply_gradients(zip(grads, vars_to_train))

            # --- Constraints ---
            self.avg_short_term_interest_pct.assign(
                tf.maximum(0.0, self.avg_short_term_interest_pct)
            )
            self.avg_long_term_interest_pct.assign(
                tf.maximum(0.0, self.avg_long_term_interest_pct)
            )
            # Maturity must be > 1 to avoid division by zero in (AvgM - 1)
            self.avg_maturity_years.assign(tf.maximum(1.001, self.avg_maturity_years))
            self.market_securities_return_pct.assign(
                tf.maximum(0.0, self.market_securities_return_pct)
            )
            # Financing percentage should be between 0 and 1
            self.equity_financing_pct.assign(
                tf.clip_by_value(self.equity_financing_pct, 0.0, 1.0)
            )

            if i % 1000 == 0:
                print(f"Epoch {i}: Structural Loss={total_loss.numpy():.4e}")

        print("Structural Training Complete.")
        print(f"Final %AvgSTInt: {self.avg_short_term_interest_pct.numpy():.5f}")
        print(f"Final %AvgLTInt: {self.avg_long_term_interest_pct.numpy():.5f}")
        print(f"Final AvgM: {self.avg_maturity_years.numpy():.5f}")
        print(f"Final %MSReturn: {self.market_securities_return_pct.numpy():.5f}")
        print(f"Final %EF: {self.equity_financing_pct.numpy():.5f}")
        print("-" * 50)

    def forecast_step(self, state, inputs, use_mean_opex=False):
        """
        Calculate t based on t-1 state and t inputs.
        Implements the logic of Pareja (09) Cash Budget construction
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

        depreciation = nca_prev * self.depreciation_rate
        stock_buyback = depreciation * self.stock_buyback_pct
        dividends_prev = net_income_prev * self.dividend_payout_ratio_pct

        # Unpack current inputs (t)
        sales_t = inputs["sales_t"]
        purchases_t = inputs["purchases_t"]
        sales_t_plus_1 = inputs["sales_t_plus_1"]
        purchases_t_plus_1 = inputs["purchases_t_plus_1"]
        cum_inflation = inputs["cum_inflation"]

        # --- 1. Assets Evolution ---
        # 1.1. Non-current Assets (NCA)
        # Policy: Maintain NCA + Growth (Simplified for this model)
        # Investment required to replace depreciation + grow
        capex = depreciation + (sales_t * self.asset_growth)
        nca_curr = nca_prev - depreciation + capex

        # 1.2. Advance Payments (AdvPP)
        advance_payments_purchases_curr = (
            purchases_t_plus_1 * self.advance_payments_purchases_pct
        )

        # 1.3. Accounts Receivable (AR)
        accounts_receivable_curr = sales_t * self.account_receivables_pct

        # 1.4. Inventory (Inv)
        inventory_curr = sales_t * self.inventory_pct

        # 1.5. Total Liquidity Target (TL)
        total_liquidity_curr = sales_t * self.total_liquidity_pct

        # 1.6. Cash Target (Cash)
        cash_curr = total_liquidity_curr * self.cash_pct_of_liquidity

        # 1.7. Investment in Market Securities Target (IMS)
        investment_in_market_securities_curr = total_liquidity_curr * (
            1 - self.cash_pct_of_liquidity
        )

        # --- 2. Income Statement (IS) ---
        # Before we move on to Liabilities, we need to calculate Income Statement quantities and Liquidity Budget quantities, as they connect the assets to liabilities and equity.
        # Net income (NI) is calculated by first calculating EBITDA = Sales - COGS - OpEx
        # Then, EBT is calculated by EBITDA - Depreciation - loan interest payments + return from market securities.
        # Finally, NI is calculated by EBT - Tax.
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

        opex = (base_opex * cum_inflation + sales_t * var_opex) + noise

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
        ## New short-term loan is found by:
        liquidity_deficit_st = (
            total_liquidity_curr
            - (cash_prev + investment_in_market_securities_prev)
            - external_investment_nlb
            - operating_nlb
            + principal_st
            + interest_st
        )
        new_short_term_loan = tf.maximum(0.0, liquidity_deficit_st)

        ## New long-term loan is found by:
        liquidity_deficit_lt = (
            liquidity_deficit_st
            - new_short_term_loan
            - capex_nlb
            + principal_lt
            + interest_lt
            + dividends_prev
            + stock_buyback
        )
        long_term_financing = tf.maximum(0.0, liquidity_deficit_lt)
        new_long_term_loan = long_term_financing * (1 - self.equity_financing_pct)
        equity_financing = long_term_financing * self.equity_financing_pct

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
        liquidity_check = total_liquidity_prev + total_nlb - total_liquidity_curr

        # --- 4. Liabilities Evolution ---
        # 4.1. Accounts Payable (AP)
        accounts_payable_curr = purchases_t * self.account_payables_pct

        # 4.2. Advance Payments Sales (AdvPS)
        # Already calculated in Liquidity Budget
        # advance_payments_sales_curr = sales_t_plus_1 * self.advance_payments_sales_pct

        # 4.3. Non-current Liabilities (NLiab)
        ## This is equal to the total long-term liabilities minus the effective principal due next year
        non_current_liabilities_curr = (
            new_long_term_loan + non_current_liabilities_prev
        ) * ((self.avg_maturity_years - 1) / self.avg_maturity_years)

        # 4.4. Current Liabilities (CLiab)
        # This is equal to the new short-term plus the long-term liabilities' effective principal due next year
        current_liabilities_curr = new_short_term_loan + (
            non_current_liabilities_curr / (self.avg_maturity_years - 1)
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
            "liquidity_check": liquidity_check,
            "check": check,
        }


# --- Monte Carlo Forecast Execution ---
def run_monte_carlo_forecast(
    model,
    initial_state,
    sales_forecast,
    purchases_forecast,
    cum_inf_forecast,
    n_samples=1000,
):
    print(f"\n--- Running Monte Carlo Forecast ({n_samples} samples) ---")

    # Store trajectories
    # Shape: [Samples, Years]
    ni_trajectories = []
    equity_trajectories = []

    for i in range(n_samples):
        current_state = initial_state.copy()
        sample_ni = []
        sample_equity = []

        for t in range(len(sales_forecast) - 1):
            inputs = {
                "sales_t": tf.constant(sales_forecast[t]),
                "purchases_t": tf.constant(purchases_forecast[t]),
                "sales_t_plus_1": tf.constant(sales_forecast[t + 1]),
                "purchases_t_plus_1": tf.constant(purchases_forecast[t + 1]),
                "cum_inflation": tf.constant(cum_inf_forecast[t]),
            }
            # use_mean_opex=False triggers sampling
            current_state = model.forecast_step(
                current_state, inputs, use_mean_opex=False
            )

            sample_ni.append(current_state["net_income"].numpy())
            sample_equity.append(current_state["equity"].numpy())

        ni_trajectories.append(sample_ni)
        equity_trajectories.append(sample_equity)

    ni_trajectories = np.array(ni_trajectories)

    # Calculate Statistics
    mean_ni = np.mean(ni_trajectories, axis=0)
    lower_bound = np.percentile(ni_trajectories, 2.5, axis=0)
    upper_bound = np.percentile(ni_trajectories, 97.5, axis=0)

    print(f"{'Year':<5} | {'Mean NI':<15} | {'2.5% CI':<15} | {'97.5% CI':<15}")
    print("-" * 60)
    for t in range(len(mean_ni)):
        print(
            f"{t+1:<5} | {mean_ni[t]:<15.2e} | {lower_bound[t]:<15.2e} | {upper_bound[t]:<15.2e}"
        )


def run_training_and_forecast():
    model = TrainableFinancialModel()

    # --- 1. HISTORICAL DATA FROM APPLE (2022-2025)---
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
    # Inflation History
    inflation_hist = np.array(
        [0.024, 0.018, 0.012, 0.047, 0.08, 0.041, 0.029, 0.027], dtype=np.float64
    )

    # --- 2. TRAIN THE MODEL ---
    # We feed in the historical arrays from 2022-2024, and leave 2025 for forecast testing.
    model.train_simple_policies(
        sales_hist[:-1],
        purchases_hist[:-1],
        nca_hist[:-1],
        depr_hist[:-1],
        advance_payments_sales_hist[:-1],
        advance_payments_purchases_hist[:-1],
        accounts_receivable_hist[:-1],
        accounts_payable_hist[:-1],
        inventory_hist[:-1],
        cash_hist[:-1],
        investment_in_market_securities_hist[:-1],
        net_income_hist[:-1],
        dividends_hist[:-1],
        stock_buyback_hist[:-1],
        opex_hist[:-1],
        tax_hist[:-1],
        inflation_hist[:-1],
    )

    # --- 3. TRAIN STRUCTURAL PARAMETERS ---
    # We still only feed in the historical arrays from 2022-2024, and leave 2025 for forecast testing.
    model.train_structural_parameters(
        sales_hist[:-1],
        purchases_hist[:-1],
        nca_hist[:-1],
        advance_payments_sales_hist[:-1],
        advance_payments_purchases_hist[:-1],
        accounts_receivable_hist[:-1],
        accounts_payable_hist[:-1],
        inventory_hist[:-1],
        cash_hist[:-1],
        investment_in_market_securities_hist[:-1],
        net_income_hist[:-1],
        dividends_hist[:-1],
        stock_buyback_hist[:-1],
        opex_hist[:-1],
        tax_hist[:-1],
        current_liabilities_hist[:-1],
        non_current_liabilities_hist[:-1],
        equity_hist[:-1],
        inflation_hist[:-1],
    )

    # --- 4. RUN FORECAST (Using new parameters) ---
    # Initial State (t=0) 2024 Apple Balance Sheet
    state = {
        "nca": tf.constant(nca_hist[-2], dtype=tf.float64),
        "advance_payments_purchases": tf.constant(
            advance_payments_purchases_hist[-2], dtype=tf.float64
        ),
        "accounts_receivable": tf.constant(
            accounts_receivable_hist[-2], dtype=tf.float64
        ),
        "inventory": tf.constant(inventory_hist[-2], dtype=tf.float64),
        "cash": tf.constant(cash_hist[-2], dtype=tf.float64),
        "investment_in_market_securities": tf.constant(
            investment_in_market_securities_hist[-2], dtype=tf.float64
        ),
        "accounts_payable": tf.constant(accounts_payable_hist[-2], dtype=tf.float64),
        "advance_payments_sales": tf.constant(
            advance_payments_sales_hist[-2], dtype=tf.float64
        ),
        "current_liabilities": tf.constant(
            current_liabilities_hist[-2], dtype=tf.float64
        ),
        "non_current_liabilities": tf.constant(
            non_current_liabilities_hist[-2], dtype=tf.float64
        ),
        "equity": tf.constant(equity_hist[-2], dtype=tf.float64),
        "net_income": tf.constant(net_income_hist[-2], dtype=tf.float64),
    }

    # Forecast Drivers (Sales/Purchases) for 2025-2028
    # Year 1 to 4. We are only interested in Year 1 to 3. The padding is needed for forecasting. Use float64
    sales_forecast = np.array(
        [3.94328e11, 3.83285e11, 3.91035e11, 4.16161e11], dtype=np.float64
    )
    purchases_forecast = np.array(
        [2.07694e11, 1.99862e11, 2.04003e11, 2.10808e11], dtype=np.float64
    )

    # Year 1 to 4 inflation rate (2025-2028)
    inflation_forecast = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cum_inf_forecast = np.cumprod(1 + inflation_forecast)

    # --- Execute Monte Carlo Forecast ---
    run_monte_carlo_forecast(
        model,
        state,
        sales_forecast,
        purchases_forecast,
        cum_inf_forecast,
        n_samples=1000,
    )


if __name__ == "__main__":
    run_training_and_forecast()
