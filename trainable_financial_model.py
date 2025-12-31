import tensorflow as tf
import numpy as np


# --- 1. Define the Trainable Model ---
class TrainableFinancialModel(tf.Module):
    def __init__(self):
        # --- Policy Parameters ---
        ## These are trainable with simple linear regression
        self.asset_growth = tf.Variable(
            0.0076, name="asset_growth", dtype=tf.float64
        )  # %AG
        self.depreciation_rate = tf.Variable(
            0.055, name="depr_rate", dtype=tf.float64
        )  # %Depr
        self.advance_payments_sales_pct = tf.Variable(
            0.020614523, name="advance_payments_sales_pct", dtype=tf.float64
        )  # %AdvPS
        self.advance_payments_purchases_pct = tf.Variable(
            0.073525733, name="advance_payments_purchases_pct", dtype=tf.float64
        )  # %AdvPP
        self.account_receivables_pct = tf.Variable(
            0.159111366, name="account_receivables_pct", dtype=tf.float64
        )  # %AR
        self.account_payables_pct = tf.Variable(
            0.35014191, name="account_payables_pct", dtype=tf.float64
        )  # %AP
        self.inventory_pct = tf.Variable(
            0.0165, name="inventory_pct", dtype=tf.float64
        )  # %Inv
        self.total_liquidity_pct = tf.Variable(
            0.16, name="total_liquidity_pct", dtype=tf.float64
        )  # %TL
        self.cash_pct_of_liquidity = tf.Variable(
            0.487, name="cash_pct_of_liquidity", dtype=tf.float64
        )  # %Cash
        self.income_tax_pct = tf.Variable(
            0.147, name="income_tax_pct", dtype=tf.float64
        )  # %IT
        self.variable_opex_pct = tf.Variable(
            0.222168147, name="variable_opex_pct", dtype=tf.float64
        )  # %OR
        self.baseline_opex = tf.Variable(
            -30306718214, name="baseline_opex", dtype=tf.float64
        )  # OBT_start
        self.dividend_payout_ratio_pct = tf.Variable(
            0.15, name="dividend_payout_ratio_pct", dtype=tf.float64
        )  # %PR
        self.stock_buyback_pct = tf.Variable(
            7.5, name="stock_buyback_pct", dtype=tf.float64
        )  # %BB

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

    def __call__(self, initial_state, sales_series, purchases_series):
        # Run the full forecast loop
        state = initial_state
        outputs = []
        for t in range(len(sales_series)):
            inputs = {
                "sales_t": sales_series[t],
                "purchases_t": purchases_series[t],
            }
        return outputs

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
        learning_rate=0.0001,  # Lower LR for stability with large numbers
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

        # 5. Accounts Receivable: ar_t = sales_t * ar_pct
        ar_true = ar_tensor
        sales_aligned_ar = sales_tensor

        # 6. Accounts Payable: ap_t = purchases_t * ap_pct
        ap_true = ap_tensor
        purchases_aligned_ap = purchases_tensor

        # 7. Inventory: inv_t = sales_t * inv_pct
        inv_true = inv_tensor
        sales_aligned_inv = sales_tensor

        # 8. Total Liquidity: (cash_t + ims_t) = sales_t * tl_pct
        tl_true = cash_tensor + ims_tensor
        sales_aligned_tl = sales_tensor

        # 9. Cash as % of Liquidity: cash_t = (cash_t + ims_t) * cash_pct
        cash_true = cash_tensor
        tl_aligned_cash = cash_tensor + ims_tensor

        # 10. Income Tax: tax_t = ni_t * it_pct (as requested by user)
        tax_true = tax_tensor
        ni_aligned_tax = ni_tensor

        # 11. OpEx: opex_t = baseline_opex * (1+inf)**t + variable_opex_pct * sales_t
        opex_true = opex_tensor
        sales_aligned_opex = sales_tensor
        # time index for opex (t=1, 2, 3...)
        t_indices = tf.range(1, len(historical_sales) + 1, dtype=tf.float64)

        # 12. Dividends: div_t = ni_{t-1} * div_pct
        div_true = div_tensor[1:]
        ni_prev_aligned = ni_tensor[:-1]

        # 13. Stock Buyback: bb_t = depr_t * bb_pct
        bb_true = bb_tensor
        depr_aligned_bb = depr_tensor

        # Optimizer
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
            self.variable_opex_pct,
            self.baseline_opex,
            self.dividend_payout_ratio_pct,
            self.stock_buyback_pct,
        ]

        for i in range(epochs):
            with tf.GradientTape() as tape:
                # Losses
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
                    tf.square(ar_true - sales_aligned_ar * self.account_receivables_pct)
                )
                loss_ap = tf.reduce_mean(
                    tf.square(
                        ap_true - purchases_aligned_ap * self.account_payables_pct
                    )
                )
                loss_inv = tf.reduce_mean(
                    tf.square(inv_true - sales_aligned_inv * self.inventory_pct)
                )
                loss_tl = tf.reduce_mean(
                    tf.square(tl_true - sales_aligned_tl * self.total_liquidity_pct)
                )
                loss_cash = tf.reduce_mean(
                    tf.square(cash_true - tl_aligned_cash * self.cash_pct_of_liquidity)
                )
                loss_tax = tf.reduce_mean(
                    tf.square(tax_true - ni_aligned_tax * self.income_tax_pct)
                )
                # opex = baseline * (1+inf)**t + var * sales
                pred_opex = (
                    self.baseline_opex * (1 + inf_tensor) ** t_indices
                    + self.variable_opex_pct * sales_aligned_opex
                )
                loss_opex = tf.reduce_mean(tf.square(opex_true - pred_opex))

                loss_div = tf.reduce_mean(
                    tf.square(
                        div_true - ni_prev_aligned * self.dividend_payout_ratio_pct
                    )
                )
                loss_bb = tf.reduce_mean(
                    tf.square(bb_true - depr_aligned_bb * self.stock_buyback_pct)
                )

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
                    + loss_opex
                    + loss_div
                    + loss_bb
                )

            # Compute Gradients
            grads = tape.gradient(total_loss, vars_to_train)

            # Apply Gradients
            optimizer.apply_gradients(zip(grads, vars_to_train))

            if i % 1000 == 0:
                print(f"Epoch {i}: Loss={total_loss.numpy():.4e}")

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
        print(f"Final %OR: {self.variable_opex_pct.numpy():.5f}")
        print(f"Final Baseline OpEx: {self.baseline_opex.numpy():.2f}")
        print(f"Final %PR: {self.dividend_payout_ratio_pct.numpy():.5f}")
        print(f"Final %BB: {self.stock_buyback_pct.numpy():.5f}")
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
        # Convert inputs to tensors and ensure float64
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
        div_t = tf.convert_to_tensor(historical_dividends, dtype=tf.float64)
        bb_t = tf.convert_to_tensor(historical_stock_buyback, dtype=tf.float64)
        opex_t = tf.convert_to_tensor(historical_opex, dtype=tf.float64)
        tax_t = tf.convert_to_tensor(historical_tax, dtype=tf.float64)
        cl_t = tf.convert_to_tensor(historical_current_liabilities, dtype=tf.float64)
        ncl_t = tf.convert_to_tensor(
            historical_non_current_liabilities, dtype=tf.float64
        )
        equity_t = tf.convert_to_tensor(historical_equity, dtype=tf.float64)

        if historical_inflation is None:
            historical_inflation = tf.zeros_like(sales_t)
        inf_t = tf.convert_to_tensor(historical_inflation, dtype=tf.float64)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        vars_to_train = [
            self.avg_short_term_interest_pct,
            self.avg_long_term_interest_pct,
            self.avg_maturity_years,
            self.market_securities_return_pct,
            self.equity_financing_pct,
        ]

        print(
            f"Training structural parameters on {len(historical_sales)-2} transitions..."
        )

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
                        "inflation": inf_t[t + 1],
                        "t": tf.cast(t + 2, dtype=tf.float64),  # OpEx time index
                    }

                    state_pred = self.forecast_step(state_prev, inputs_curr)

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

            if i % 1000 == 0:
                print(f"Epoch {i}: Structural Loss={total_loss.numpy():.4e}")

        print("Structural Training Complete.")
        print(f"Final %AvgSTInt: {self.avg_short_term_interest_pct.numpy():.5f}")
        print(f"Final %AvgLTInt: {self.avg_long_term_interest_pct.numpy():.5f}")
        print(f"Final AvgM: {self.avg_maturity_years.numpy():.5f}")
        print(f"Final %MSReturn: {self.market_securities_return_pct.numpy():.5f}")
        print(f"Final %EF: {self.equity_financing_pct.numpy():.5f}")
        print("-" * 50)

    def forecast_step(self, state, inputs):
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
        total_liquidity_prev = cash_prev + investment_in_market_securities_prev
        depreciation = nca_prev * self.depreciation_rate

        ## Liabilities and Equity
        accounts_payable_prev = state["accounts_payable"]
        advance_payments_sales_prev = state["advance_payments_sales"]
        current_liabilities_prev = state["current_liabilities"]
        non_current_liabilities_prev = state["non_current_liabilities"]

        ## Equity
        equity_prev = state["equity"]  # Initial equity assumed to remain constant
        stock_buyback = depreciation * self.stock_buyback_pct

        ## Net Income
        net_income_prev = state["net_income"]
        dividends_prev = net_income_prev * self.dividend_payout_ratio_pct

        # Unpack current inputs (t)
        sales_t = inputs["sales_t"]
        purchases_t = inputs["purchases_t"]
        sales_t_plus_1 = inputs["sales_t_plus_1"]
        purchases_t_plus_1 = inputs["purchases_t_plus_1"]
        inflation = inputs["inflation"]
        t = inputs["t"]

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
        opex = (
            self.baseline_opex * (1 + inflation) ** t + sales_t * self.variable_opex_pct
        )
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

        interest_payment = interest_st + interest_lt
        ms_return = (
            investment_in_market_securities_prev * self.market_securities_return_pct
        )

        ebt = ebitda - depreciation - interest_payment + ms_return
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
            - total_liquidity_prev
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
            (new_long_term_loan + non_current_liabilities_prev)
            * (self.avg_maturity_years - 1)
            / self.avg_maturity_years
        )

        # 4.4. Current Liabilities (CLiab)
        # This is equal to the new short-term plus the long-term liabilities' effective principal due next year
        current_liabilities_curr = (
            new_short_term_loan
            + non_current_liabilities_curr / (self.avg_maturity_years - 1)
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

        new_state = {
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
            "stloan": new_short_term_loan,
            "ltloan": new_long_term_loan,
            "st_principal_paid": principal_st,
            "lt_principal_paid": principal_lt,
        }

        return new_state


# --- Training and Forecast Execution ---
def run_training_and_forecast():
    model = TrainableFinancialModel()

    # --- 1. HISTORICAL DATA FROM APPLE (2022-2025)---
    # Revenues from Income Statement
    sales_hist = np.array(
        [3.94328e11, 3.83285e11, 3.91035e11, 4.16161e11], dtype=np.float64
    )
    # Inventory from Balance Sheet, includes one additional year at the beginning to determine purchases history
    inventory_hist_plus_one = np.array(
        [6580000000, 4946000000, 6331000000, 7286000000, 5718000000], dtype=np.float64
    )
    # Depreciation from Reconciled Depreciation in Income Statement
    depr_hist = np.array(
        [11698000000, 11445000000, 11519000000, 11104000000], dtype=np.float64
    )
    # COGS from Cost of Revenue - Depreciation in Income Statement
    cogs_hist = np.array(
        [2.12442e11, 2.02618e11, 1.98907e11, 2.09262e11], dtype=np.float64
    )
    # Purchases from COGS + Inventory_t - Inventory_t-1
    purchases_hist = (
        cogs_hist + inventory_hist_plus_one[1:] - inventory_hist_plus_one[:-1]
    )
    # Inventory with matched length
    inventory_hist = inventory_hist_plus_one[1:]
    # Non-current Assets from Balance Sheet
    nca_hist = np.array(
        [2.1735e11, 2.09017e11, 2.11993e11, 2.11284e11], dtype=np.float64
    )
    # Advance Payments for Purchases from Other Current Assets in Balance Sheet
    advance_payments_purchases_hist = np.array(
        [21223000000, 14695000000, 14287000000, 14585000000], dtype=np.float64
    )
    # Accounts Receivable from Receivables in Balance Sheet
    accounts_receivable_hist = np.array(
        [60932000000, 60985000000, 66243000000, 72957000000], dtype=np.float64
    )
    # Cash from Cash and Cash Equivalents in Balance Sheet
    cash_hist = np.array(
        [23646000000, 29965000000, 29943000000, 35934000000], dtype=np.float64
    )
    # Investment in Market Securities from Other Short Term Investments in Balance Sheet
    investment_in_market_securities_hist = np.array(
        [24658000000, 31590000000, 35228000000, 18763000000], dtype=np.float64
    )
    # Accounts Payable from Balance Sheet
    accounts_payable_hist = np.array(
        [64115000000, 62611000000, 68960000000, 69860000000], dtype=np.float64
    )
    # Advance Payments Sales from Current Deferred Revenue in Balance Sheet
    advance_payments_sales_hist = np.array(
        [7912000000, 8061000000, 8249000000, 9055000000], dtype=np.float64
    )
    # Current Liabilities from Total Current Liabilities - Accounts Payable - Current Deferred Revenue in Balance Sheet
    current_liabilities_hist = np.array(
        [81955000000, 74636000000, 99183000000, 86716000000], dtype=np.float64
    )
    # Non-current Liabilities from Total Non Current Liabilities Net Minority Interest in Balance Sheet
    non_current_liabilities_hist = np.array(
        [1.48101e11, 1.45129e11, 1.31638e11, 1.19877e11], dtype=np.float64
    )
    # Equity from Stockholders Equity in Balance Sheet
    equity_hist = np.array(
        [50672000000, 62146000000, 56950000000, 73733000000], dtype=np.float64
    )
    # Net Income from Income Statement
    net_income_hist = np.array(
        [99803000000, 96995000000, 93736000000, 112012000000], dtype=np.float64
    )
    # Dividends paid this year from Common Stock Dividends Paid in Cash Flow
    dividends_hist = np.array(
        [14841000000, 15025000000, 15234000000, 15421000000], dtype=np.float64
    )
    # Stock Buyback from Repurchase of Capital Stock in Cash Flow
    stock_buyback_hist = np.array(
        [89402000000, 77550000000, 94949000000, 90711000000], dtype=np.float64
    )
    # OpEx from Operating Expenses in Income Statement
    opex_hist = np.array(
        [51345000000, 54847000000, 57467000000, 62151000000], dtype=np.float64
    )
    # Tax Provision from Income Statement
    tax_hist = np.array(
        [19300000000, 16741000000, 29749000000, 20719000000], dtype=np.float64
    )
    # Inflation hist
    inflation_hist = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # --- 2. TRAIN THE MODEL ---
    # We feed in the historical arrays from 2022-2024, and leave 2025 for forecast testing.
    model.train_simple_policies(
        historical_sales=sales_hist[:-1],
        historical_purchases=purchases_hist[:-1],
        historical_nca=nca_hist[:-1],
        historical_depreciation=depr_hist[:-1],
        historical_adv_pay_sales=advance_payments_sales_hist[:-1],
        historical_adv_pay_purch=advance_payments_purchases_hist[:-1],
        historical_ar=accounts_receivable_hist[:-1],
        historical_ap=accounts_payable_hist[:-1],
        historical_inventory=inventory_hist[:-1],
        historical_cash=cash_hist[:-1],
        historical_ims=investment_in_market_securities_hist[:-1],
        historical_net_income=net_income_hist[:-1],
        historical_dividends=dividends_hist[:-1],
        historical_stock_buyback=stock_buyback_hist[:-1],
        historical_opex=opex_hist[:-1],
        historical_tax=tax_hist[:-1],
        historical_inflation=inflation_hist[:-1],
    )

    # --- 3. TRAIN STRUCTURAL PARAMETERS ---
    # We still only feed in the historical arrays from 2022-2024, and leave 2025 for forecast testing.
    model.train_structural_parameters(
        historical_sales=sales_hist[:-1],
        historical_purchases=purchases_hist[:-1],
        historical_nca=nca_hist[:-1],
        historical_adv_pay_sales=advance_payments_sales_hist[:-1],
        historical_adv_pay_purch=advance_payments_purchases_hist[:-1],
        historical_ar=accounts_receivable_hist[:-1],
        historical_ap=accounts_payable_hist[:-1],
        historical_inventory=inventory_hist[:-1],
        historical_cash=cash_hist[:-1],
        historical_ims=investment_in_market_securities_hist[:-1],
        historical_net_income=net_income_hist[:-1],
        historical_dividends=dividends_hist[:-1],
        historical_stock_buyback=stock_buyback_hist[:-1],
        historical_opex=opex_hist[:-1],
        historical_tax=tax_hist[:-1],
        historical_current_liabilities=current_liabilities_hist[:-1],
        historical_non_current_liabilities=non_current_liabilities_hist[:-1],
        historical_equity=equity_hist[:-1],
        historical_inflation=inflation_hist[:-1],
    )

    # --- 4. RUN FORECAST (Using new parameters) ---
    # Initial State (t=0) 2024 Apple Balance Sheet
    state = {
        "nca": tf.constant(nca_hist[-2], dtype=tf.float64),  # float number
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
        "liquidity_check": tf.constant(0.0, dtype=tf.float64),
        "check": tf.constant(0.0, dtype=tf.float64),
        "stloan": tf.constant(0.0, dtype=tf.float64),
        "ltloan": tf.constant(0.0, dtype=tf.float64),
        "st_principal_paid": tf.constant(0.0, dtype=tf.float64),
        "lt_principal_paid": tf.constant(0.0, dtype=tf.float64),
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
    inflation = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    print("\n--- Running Forecast with TRAINED parameters ---")
    print(
        f"{'Year':<5} | {'Assets':<15} | {'Liabilities':<15} | {'Equity':<15} | {'Check (Plug)':<15}"
    )
    print("-" * 70)

    # Loop explicitly to handle the recursive dependency.
    for t in range(len(sales_forecast) - 1):
        inputs = {
            "sales_t": tf.constant(sales_forecast[t]),
            "purchases_t": tf.constant(purchases_forecast[t]),
            "sales_t_plus_1": tf.constant(sales_forecast[t + 1]),
            "purchases_t_plus_1": tf.constant(purchases_forecast[t + 1]),
            "inflation": tf.constant(inflation[t]),
            "t": t + 1,
        }

        state = model.forecast_step(state, inputs)

        # Calculate totals for display
        assets = (
            state["nca"]
            + state["advance_payments_purchases"]
            + state["accounts_receivable"]
            + state["inventory"]
            + state["cash"]
            + state["investment_in_market_securities"]
        )

        liabilities = (
            state["accounts_payable"]
            + state["advance_payments_sales"]
            + state["current_liabilities"]
            + state["non_current_liabilities"]
        )

        print(
            f"{t+1:<5} | {assets.numpy():<15.2f} | {liabilities.numpy():<15.2f} | {state['equity'].numpy():<15.2f} | {state['check'].numpy():<15.2f}"
        )


if __name__ == "__main__":
    run_training_and_forecast()
