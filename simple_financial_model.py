import tensorflow as tf


class SimpleFinancialModel(tf.Module):
    def __init__(self):
        # --- Policy Parameters ---
        self.asset_growth = tf.constant(0.05, dtype=tf.float32)  # %AG
        self.tax_rate = tf.constant(0.35, dtype=tf.float32)
        self.interest_rate_st = tf.constant(0.06, dtype=tf.float32)
        self.depreciation_rate = tf.constant(0.10, dtype=tf.float32)  # %Depr
        self.advance_payments_sales_pct = tf.constant(
            0.020614523, dtype=tf.float32
        )  # %AdvPS
        self.advance_payments_purchases_pct = tf.constant(
            0.073525733, dtype=tf.float32
        )  # %AdvPP
        self.account_receivables_pct = tf.constant(0.159111366, dtype=tf.float32)  # %AR
        self.account_payables_pct = tf.constant(0.35014191, dtype=tf.float32)  # %AP
        self.inventory_pct = tf.constant(0.15, dtype=tf.float32)  # %Inv
        self.total_liquidity_pct = tf.constant(0.16, dtype=tf.float32)  # %TL
        self.cash_pct_of_liquidity = tf.constant(0.25, dtype=tf.float32)  # %Cash
        self.income_tax_pct = tf.constant(0.25, dtype=tf.float32)  # %IT
        self.variable_opex_pct = tf.constant(0.60, dtype=tf.float32)  # %OR
        self.baseline_opex = tf.constant(54847000000, dtype=tf.float32)  # OBT_start
        self.avg_short_term_interest_pct = tf.constant(
            0.04, dtype=tf.float32
        )  # %AvgSTInt
        self.avg_long_term_interest_pct = tf.constant(
            0.06, dtype=tf.float32
        )  # %AvgLTInt
        self.avg_maturity_years = tf.constant(5.0, dtype=tf.float32)  # AvgM
        self.market_securities_return_pct = tf.constant(
            0.05, dtype=tf.float32
        )  # %MSReturn
        self.equity_financing_pct = tf.constant(0.30, dtype=tf.float32)  # %EF
        self.dividend_payout_ratio_pct = tf.constant(0.40, dtype=tf.float32)  # %PR
        self.stock_buyback_pct = tf.constant(0.10, dtype=tf.float32)  # %BB

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
        sales_t_minus_1 = inputs["sales_t_minus_1"]
        purchases_t_minus_1 = inputs["purchases_t_minus_1"]
        inflation = inputs["inflation"]
        t = inputs["t"]

        # --- 1. Assets Evolution ---
        # 1.1. Non-current Assets (NCA)
        # Policy: Maintain NCA + Growth (Simplified for this model)
        # Investment required to replace depreciation + grow
        capex = depreciation + (sales_t * self.asset_growth)
        nca_curr = nca_prev + capex - depreciation

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
        interest_st = self.interest_rate_st * principal_st

        interest_payment = interest_st + interest_lt
        ms_return = (
            investment_in_market_securities_prev * self.market_securities_return_pct
        )

        ebt = ebitda - depreciation - interest_payment + ms_return
        tax = ebt * self.tax_rate
        net_income_curr = ebt - tax

        # --- 3. Liquidity Budget (LB) ---
        # We need quantities from Liquidity Budget calculations before proceeding to liabilities.

        # 3.1. Operating Net Liquidity Balance (Operating NLB)
        # Inflows: Sales | Outflows: Purchases, OpEx, Tax, Interest

        # Sales: cash flow from current year's sales + accounts receivable from previous year + advance payment for next year's sales
        sales_curr = sales_t * (
            1 - self.advance_payments_sales_pct - self.account_receivables_pct
        )
        advance_payments_sales_curr = sales_t_plus_1 * self.advance_payments_sales_pct
        inflows = sales_curr + accounts_receivable_prev + advance_payments_sales_curr

        # Purchases: cash flow from current year's purchases + cash flow from previous year's purchases + cash flow from next year's purchases
        purchases_curr = purchases_t * (
            1 - self.advance_payments_purchases_pct - self.account_payables_pct
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

        # 3.3. Financing Net Liquidity Balance (Financing NLB)
        ## First, we need to figure out how much new short-term loan and long-term loan to issue this year.
        ## New short-term loan is found by:
        cash_deficit_st = (
            cash_curr
            - cash_prev
            - operating_nlb
            + principal_st
            + interest_st
            - ms_return
        )
        new_short_term_loan = tf.maximum(0.0, cash_deficit_st)

        ## New long-term loan is found by:
        cash_deficit_lt = (
            cash_deficit_st
            - new_short_term_loan
            - capex_nlb
            + principal_lt
            + interest_lt
            + dividends_prev
            + stock_buyback
        )

        long_term_financing = tf.maximum(0.0, cash_deficit_lt)
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

        # 3.4. External Investment Net Liquidity Balance (External Investment NLB)
        external_investment_nlb = (
            investment_in_market_securities_prev
            + ms_return
            - investment_in_market_securities_curr
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
        liquidity_check = cash_prev + total_nlb - total_liquidity_curr

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
        }

        return new_state

    # --- Simulation Execution ---


def run_forecast():
    model = SimpleFinancialModel()

    # Initial State (t=0) 2023 Apple Balance Sheet
    state = {
        "nca": tf.constant(2.09017e11, dtype=tf.float32),  # float number
        "advance_payments_purchases": tf.constant(14695000000, dtype=tf.float32),
        "accounts_receivable": tf.constant(60985000000, dtype=tf.float32),
        "inventory": tf.constant(6331000000, dtype=tf.float32),
        "cash": tf.constant(29965000000, dtype=tf.float32),
        "investment_in_market_securities": tf.constant(31590000000, dtype=tf.float32),
        "accounts_payable": tf.constant(71430000000, dtype=tf.float32),
        "advance_payments_sales": tf.constant(8061000000, dtype=tf.float32),
        "current_liabilities": tf.constant(65817000000, dtype=tf.float32),
        "non_current_liabilities": tf.constant(1.45129e11, dtype=tf.float32),
        "equity": tf.constant(62146000000, dtype=tf.float32),
        "net_income": tf.constant(96995000000, dtype=tf.float32),
        "liquidity_check": tf.constant(0.0, dtype=tf.float32),
        "check": tf.constant(0.0, dtype=tf.float32),
    }

    # time Series Inputs (Forecasted Sales/Costs)
    # Year 0 to 5. We are only interested in Year 1 to 4. The padding is needed for forecasting.
    sales_forecast = [3.94328e11, 3.83285e11, 3.91035e11, 4.16161e11]
    purchases_forecast = [2.07694e11, 1.99862e11, 2.04003e11, 2.10808e11]

    # Year 1 to 4 inflation rate
    inflation = [0.02, 0.02, 0.02, 0.02]

    print(
        f"{'Year':<5} | {'Assets':<10} | {'CLiab':<10} | {'NLiab':<10} | {'Equity':<10} | {'Check (Plug)':<12} | {'Liquidity Check (Plug)':<12}"
    )
    print("-" * 70)

    # Loop explicitly to handle the recursive dependency.
    for t in range(len(sales_forecast) - 2):
        inputs = {
            "sales_t_minus_1": tf.constant(sales_forecast[t]),
            "purchases_t_minus_1": tf.constant(purchases_forecast[t]),
            "sales_t": tf.constant(sales_forecast[t + 1]),
            "purchases_t": tf.constant(purchases_forecast[t + 1]),
            "sales_t_plus_1": tf.constant(sales_forecast[t + 2]),
            "purchases_t_plus_1": tf.constant(purchases_forecast[t + 2]),
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

        current_liabilities = state["current_liabilities"]
        non_current_liabilities = state["non_current_liabilities"]
        equity = state["equity"]

        # L+E = Liabilities + Equity
        le = (
            state["accounts_payable"]
            + state["advance_payments_sales"]
            + state["current_liabilities"]
            + state["non_current_liabilities"]
            + state["equity"]
        )

        print(
            f"{t+1:<5} | {assets.numpy():<10.2f} | {current_liabilities.numpy():<10.2f} | {non_current_liabilities.numpy():<10.2f} | {equity.numpy():<10.2f} |  {state['check'].numpy():<12.2f} | {state['liquidity_check'].numpy():<12.2f} "
        )


if __name__ == "__main__":
    run_forecast()
