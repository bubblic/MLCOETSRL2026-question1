import tensorflow as tf


class SimpleFinancialModel(tf.Module):
    def __init__(self):
        # --- Policy Parameters ---
        self.asset_growth = tf.constant(0.05, dtype=tf.float32)  # %AG
        self.tax_rate = tf.constant(0.35, dtype=tf.float32)
        self.min_cash = tf.constant(10.0, dtype=tf.float32)
        self.interest_rate_st = tf.constant(0.06, dtype=tf.float32)
        self.depreciation_rate = tf.constant(0.10, dtype=tf.float32)  # %Depr
        self.advance_payments_sales_pct = tf.constant(0.05, dtype=tf.float32)  # %AdvPS
        self.advance_payments_purchases_pct = tf.constant(
            0.03, dtype=tf.float32
        )  # %AdvPP
        self.account_receivables_pct = tf.constant(0.10, dtype=tf.float32)  # %AR
        self.account_payables_pct = tf.constant(0.08, dtype=tf.float32)  # %AP
        self.inventory_pct = tf.constant(0.15, dtype=tf.float32)  # %Inv
        self.total_liquidity_pct = tf.constant(0.20, dtype=tf.float32)  # %TL
        self.cash_pct_of_liquidity = tf.constant(0.25, dtype=tf.float32)  # %Cash
        self.income_tax_pct = tf.constant(0.25, dtype=tf.float32)  # %IT
        self.variable_opex_pct = tf.constant(0.60, dtype=tf.float32)  # %OR
        self.baseline_opex = tf.constant(50.0, dtype=tf.float32)  # OBT_start
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
        advance_payments_prev = state["advance_payments"]
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
        equity_prev = state["equity"]  # Initial equity assumed to remain constant

        # Unpack current inputs (t)
        sales_t = inputs["sales_t"]
        purchases_t = inputs["purchases_t"]
        sales_t_plus_1 = inputs["sales_t_plus_1"]
        purchases_t_plus_1 = inputs["purchases_t_plus_1"]
        sales_t_minus_1 = inputs["sales_t_minus_1"]
        purchases_t_minus_1 = inputs["purchases_t_minus_1"]
        inflation = inputs["inflation"]

        # --- 1. Non-current Assets (NCA) Evolution ---
        # Policy: Maintain NCA + Growth (Simplified for this model)
        # Investment required to replace depreciation + grow
        depreciation = nca_prev * self.depreciation_rate
        capex = depreciation + (sales_t * self.asset_growth)
        nca_curr = nca_prev + capex - depreciation

        # --- 2. Advance Payments (AdvPP) ---
        advance_payments_curr = sales * self.advance_payments_sales_pct

        # --- 3. Accounts Receivable (AR) ---
        accounts_receivable_curr = sales * self.account_receivables_pct

        # --- 4. Inventory (Inv) ---
        inventory_curr = sales * self.inventory_pct

        # --- 5. Total Liquidity Target (TL) ---
        total_liquidity_curr = sales * self.total_liquidity_pct

        # --- 6. Cash Target (Cash) ---
        cash_curr = total_liquidity_curr * self.cash_pct_of_liquidity

        # --- 7. Investment in Market Securities Target (IMS) ---
        investment_in_market_securities_curr = total_liquidity_curr * (
            1 - self.cash_pct_of_liquidity
        )

        # --- 8. Liquidity Budget (LB) ---
        # Operating Cash Flow (Proxy)
        # Inflows: Sales | Outflows: Purchases, OpEx, Tax, Interest

        # Sales: cash flow from current year's sales + cash flow from previous year's sales + cash flow from next year's sales
        sales_curr = sales * (
            1 - self.advance_payments_sales_pct - self.account_receivables_pct
        )
        sales_ar = sales_t_minus_1 * self.account_receivables_pct
        sales_adv = sales_t_plus_1 * self.advance_payments_sales_pct
        inflows = sales_curr + sales_ar + sales_adv

        # Purchases: cash flow from current year's purchases + cash flow from previous year's purchases + cash flow from next year's purchases
        purchases_curr = purchases_t * (
            1 - self.advance_payments_purchases_pct - self.account_payables_pct
        )
        purchases_ap = purchases_t_minus_1 * self.account_payables_pct
        purchases_adv = purchases_t_plus_1 * self.advance_payments_purchases_pct

        opex = self.baseline_opex + purchases_curr * self.variable_opex_pct
        outflows = purchases_curr + purchases_ap + purchases_adv

        # Calculate raw cash flow before financing
        operational_cf = sales - costs - tax - interest_payment
        investing_cf = -capex

        # Cash available before new debt decisions
        cash_available = cash_prev + operational_cf + investing_cf

        # Determine deficit
        # If Cash < Min_Cash, we need to borrow.
        # If Cash > Min_Cash, we pay down debt or hold excess.
        deficit = self.min_cash - cash_available

        # Logic: If deficit > 0, Borrow. If deficit < 0, Pay Debt/Save.
        # This acts as the "No Plug" logic by explicitly calculating financing needs.
        new_borrowing = tf.maximum(0.0, deficit)

        # Update Debt State
        debt_curr = debt_prev + new_borrowing

        # Update Cash State (Cash should equal Min Cash if we borrowed efficiently)
        # Or be higher if we generated surplus
        cash_curr = cash_available + new_borrowing

        # Update Retained Earnings
        # Assuming 0 dividends for simplicity in this step
        re_curr = re_prev + net_income

        # Assume same equity maintained
        equity_curr = equity_prev

        # --- ????1. Income Statement (IS) Logic ---
        # Interest is based on PREVIOUS debt (No Circularity)
        interest_payment = debt_prev * self.interest_rate_st

        # Depreciation
        depreciation = nca_prev * self.depreciation_rate

        # Taxable Income
        ebit = sales - cogs - depreciation
        ebt = ebit - interest_payment
        tax = tf.maximum(0.0, ebt * self.tax_rate)
        net_income = ebt - tax

        # --- ????4. Balance Sheet Identity Check ---
        # Assets = Cash + NCA
        total_assets = cash_curr + nca_curr
        # Liabilities + Equity = Debt + (Initial Equity + RE)
        # Note: Assuming Initial Equity is constant (e.g., 50.0)
        total_liab_equity = debt_curr + (equity_prev + re_curr)

        # Check mismatch (Should be near zero if logic is consistent)
        check = total_assets - total_liab_equity

        new_state = {
            "nca": nca_curr,
            "advance_payments": advance_payments_curr,
            "accounts_receivable": accounts_receivable_curr,
            "inventory": inventory_curr,
            "cash": cash_curr,
            "investment_in_market_securities": investment_in_market_securities_curr,
            "accounts_payable": accounts_payable_curr,
            "advance_payments_sales": advance_payments_sales_curr,
            "current_liabilities": current_liabilities_curr,
            "non_current_liabilities": non_current_liabilities_curr,
            "equity": equity_curr,
            "check": check,
        }

        return new_state

    # --- Simulation Execution ---


def run_forecast():
    model = SimpleFinancialModel()

    # Initial State (t=0)
    state = {
        "nca": tf.constant(100.0),
        "advance_payments": tf.constant(0.0),
        "accounts_receivable": tf.constant(0.0),
        "inventory": tf.constant(0.0),
        "cash": tf.constant(10.0),
        "investment_in_market_securities": tf.constant(0.0),
        "accounts_payable": tf.constant(0.0),
        "advance_payments_sales": tf.constant(0.0),
        "current_liabilities": tf.constant(0.0),
        "non_current_liabilities": tf.constant(0.0),
        "equity": tf.constant(50.0),
        "net_income": tf.constant(0.0),
        "check": tf.constant(0.0),
    }

    # time Series Inputs (Forecasted Sales/Costs)
    # Year 0 to 5. We are only interested in Year 1 to 4. The padding is needed for forecasting.
    sales_forecast = [200.0, 220.0, 240.0, 260.0, 280.0, 300.0]
    purchases_forecast = [140.0, 150.0, 165.0, 180.0, 195.0, 210.0]

    # Year 1 to 4 inflation rate
    inflation = [0.02, 0.02, 0.02, 0.02]

    print(
        f"{'Year':<5} | {'Assets':<10} | {'L+E':<10} | {'Check (Plug)':<12} | {'Debt':<10} | {'Cash':<10}"
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
        }

        state = model.forecast_step(state, inputs)

        # Calculate totals for display
        assets = state["cash"] + state["nca"]
        # L+E = Debt + Initial Equity (50) + RE
        le = state["debt"] + state["equity"] + state["re_earnings"]

        print(
            f"{t+1:<5} | {assets.numpy():<10.2f} | {le.numpy():<10.2f} | {state['check'].numpy():<12.2f} | {state['debt'].numpy():<10.2f} | {state['cash'].numpy():<10.2f}"
        )


if __name__ == "__main__":
    run_forecast()
