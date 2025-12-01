import tensorflow as tf


class SimpleFinancialModel(tf.Module):
    def __init__(self):
        # --- Policy Parameters ---
        self.tax_rate = tf.constant(0.35, dtype=tf.float32)
        self.min_cash = tf.constant(10.0, dtype=tf.float32)
        self.interest_rate_st = tf.constant(0.06, dtype=tf.float32)
        self.depreciation_rate = tf.constant(0.10, dtype=tf.float32)

    def forecast_step(self, state, inputs):
        """
        Calculate t based on t-1 state and t inputs.
        Implements the logic of Pareja (09) Cash Budget construction
        """
        # Unpack previous state (t-1)
        cash_prev = state["cash"]
        nfa_prev = state["nfa"]  # Net fixed assets
        debt_prev = state["debt"]
        re_prev = state["re_earnings"]  # Retained Earnings
        equity_prev = state["equity"]  # Initial equity assumed to remain constant

        # Unpack current inputs (t)
        sales = inputs["sales"]
        costs = inputs["costs"]  # OpEx excluding depreciation/interest

        # --- 1. Income Statement (IS) Logic ---
        # Interest is based on PREVIOUS debt (No Circularity)
        interest_payment = debt_prev * self.interest_rate_st

        # Depreciation
        depreciation = nfa_prev * self.depreciation_rate

        # Taxable Income
        ebit = sales - costs - depreciation
        tax = tf.maximum(0.0, ebit * self.tax_rate)
        net_income = ebit - interest_payment - tax

        # --- 2. Asset Evolution (Tanks) ---
        # Policy: Maintain NFA + Growth (Simplified for this model)
        # Investment required to replace depreciation + grow
        capex = depreciation + (sales * 0.05)
        nfa_curr = nfa_prev + capex - depreciation

        # --- 3. Cash Budget (CB) & Debt Logic ---
        # Operating Cash Flow (Proxy)
        # Inflows: Sales | Outflows: Costs, Tax, Interest, Capex
        # Note: Pareja uses detailed CB modules. Simplified here for TF

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

        # --- 4. Balance Sheet Identity Check ---
        # Assets = Cash + NFA
        total_assets = cash_curr + nfa_curr
        # Liabilities + Equity = Debt + (Initial Equity + RE)
        # Note: Assuming Initial Equity is constant (e.g., 50.0)
        total_liab_equity = debt_curr + (equity_prev + re_curr)

        # Check mismatch (Should be near zero if logic is consistent)
        check = total_assets - total_liab_equity

        new_state = {
            "cash": cash_curr,
            "nfa": nfa_curr,
            "debt": debt_curr,
            "equity": equity_curr,
            "re_earnings": re_curr,
            "net_income": net_income,
            "check": check,
        }

        return new_state

    # --- Simulation Execution ---


def run_forecast():
    model = SimpleFinancialModel()

    # Initial State (t=0)
    state = {
        "cash": tf.constant(10.0),
        "nfa": tf.constant(100.0),
        "debt": tf.constant(60.0),  # Initial Debt
        "equity": tf.constant(50.0),  # Initial equity. Kept the same in this example
        "re_earnings": tf.constant(0.0),  # Initial RE
        "net_income": tf.constant(0.0),
        "check": tf.constant(0.0),
    }

    # time Series Inputs (Forecasted Sales/Costs)
    # Year 1 to 5
    sales_forecast = [200.0, 220.0, 240.0, 260.0, 280.0]
    costs_forecast = [140.0, 150.0, 165.0, 180.0, 195.0]

    print(
        f"{'Year':<5} | {'Assets':<10} | {'L+E':<10} | {'Check (Plug)':<12} | {'Debt':<10} | {'Cash':<10}"
    )
    print("-" * 70)

    # Loop explicitly to handle the recursive dependency
    for t in range(len(sales_forecast)):
        inputs = {
            "sales": tf.constant(sales_forecast[t]),
            "costs": tf.constant(costs_forecast[t]),
        }

        state = model.forecast_step(state, inputs)

        # Calculate totals for display
        assets = state["cash"] + state["nfa"]
        # L+E = Debt + Initial Equity (50) + RE
        le = state["debt"] + state["equity"] + state["re_earnings"]

        print(
            f"{t+1:<5} | {assets.numpy():<10.2f} | {le.numpy():<10.2f} | {state['check'].numpy():<12.2f} | {state['debt'].numpy():<10.2f} | {state['cash'].numpy():<10.2f}"
        )


if __name__ == "__main__":
    run_forecast()
