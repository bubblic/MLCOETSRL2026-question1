import tensorflow as tf
import numpy as np


# --- 1. Define the Trainable Model ---
class TrainableFinancialModel(tf.Module):
    def __init__(self):
        # We make these variables trainable to "learn" the company's behavior
        self.tax_rate = tf.Variable(0.25, name="tax_rate")
        self.depreciation_rate = tf.Variable(0.10, name="depr_rate")  # %Depr
        self.min_cash_ratio = tf.Variable(
            0.05, name="min_cash_ratio"
        )  # %Cash (as % of total liquidity)

        # Additional policy parameters
        self.inventory_pct = tf.Variable(0.15, name="inventory_pct")  # %Inv
        self.total_liquidity_pct = tf.Variable(0.20, name="total_liquidity_pct")  # %TL

        # These are set constant for now (keeping original ones for backward compatibility)
        self.debt_rate = tf.constant(0.05, name="debt_rate")
        self.growth_rate = tf.constant(0.02, name="growth_rate")

    def __call__(self, initial_state, sales_series, cost_series):
        # Run the full forecast loop (same logic ans previous answer)
        state = initial_state
        outputs = []

        for t in range(len(sales_series)):
            # Dynamic inputs
            sales = sales_series[t]
            costs = cost_series[t]

            # --- Simplified Core Logic (Pareja 09) ---
            # 1. Income Statement
            interest = state["debt"] * self.debt_rate
            depreciation = state["nfa"] * self.depreciation_rate
            ebit = sales - costs - depreciation
            tax = tf.maximum(0.0, ebit * self.tax_rate)
            net_income = ebit - interest - tax

            # 2. Asset Flows
            target_cash = sales * self.min_cash_ratio
            capex = depreciation + (sales * self.growth_rate)
            nfa_next = state["nfa"] + capex - depreciation

            # 3. Cash Budget & Debt (No Plug)
            # Cash Flow = Sales - Costs - Tax - Interest - Capex
            op_cf = sales - costs - tax - interest
            cash_flow = op_cf - capex

            # Update Cash/Debt based on deficit
            # If cash flow adds to cash, do we have enough?
            # Current Cash + Cash Flow
            raw_cash_end = state["cash"] + cash_flow

            deficit = target_cash - raw_cash_end
            new_borrowing = tf.maximum(0.0, deficit)
            debt_next = state["debt"] + new_borrowing

            # If we borrowed, cash = target. If surplus, cash > Target.
            cash_next = raw_cash_end + new_borrowing

            # 4. Equity
            re_next = state["re_earnings"] + net_income

            # Store state for next iteration
            state = {
                "cash": cash_next,
                "nfa": nfa_next,
                "debt": debt_next,
                "re_earnings": re_next,
            }

            # Output the Total Assets for loss calculation
            total_assets = cash_next + nfa_next
            outputs.append(total_assets)

        return tf.stack(outputs)


# --- 2. The Training Loop ---
def train_model(model, actual_assets, sales_data, cost_data, epochs=100):
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    # Initial State (Historical Year T-5)
    initial_state = {
        "cash": tf.constant(10.0),
        "nfa": tf.constant(100.0),
        "debt": tf.constant(50.0),
        "re_earnings": tf.constant(60.0),
    }

    print(
        f"{'Epoch':<10} | {'Loss':<10} | {'Cash Ratio':<10} | {'Tax Rate':<10} | {'Depr Rate':<10} | {'Asset GR':<10} | {'Inv %':<10} | {'TL %':<10}"
    )

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Forward pass: Forecast the balance sheet
            predicted_assets = model(initial_state, sales_data, cost_data)

            # Loss: Difference between Forecasted Total Assets and Actual Historical Assets
            loss = tf.reduce_mean(tf.square(predicted_assets - actual_assets))

        # Compute gradients for parameters (tax_rate, dep_rate, min_cash_ratio, etc.)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 20 == 0:
            print(
                f"{epoch:<10} | {loss.numpy():<10.2f} | {model.min_cash_ratio.numpy():<10.2f} | {model.tax_rate.numpy():<10.2f} | {model.depreciation_rate.numpy():<10.2f} | {model.growth_rate.numpy():<10.2f} | {model.inventory_pct.numpy():<10.2f} | {model.total_liquidity_pct.numpy():<10.2f}"
            )


# Example Dummy Data (5 Years of history)
sales_data = tf.constant([200.0, 220.0, 240.0, 260.0, 280.0])
cost_data = tf.constant([140.0, 150.0, 165.0, 180.0, 195.0])
# "Actual" Assets we are trying to fit
actual_assets = tf.constant([115.0, 128.0, 142.0, 158.0, 175.0])

model = TrainableFinancialModel()
train_model(model, actual_assets, sales_data, cost_data, 101)
