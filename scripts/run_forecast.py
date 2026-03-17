"""Run the deterministic financial forecast simulation.

Uses the SimpleFinancialModel with constant parameters to forecast
Apple's balance sheet from a 2023 initial state.

Example:
    python -m scripts.run_forecast
"""

import tensorflow as tf

from financial_forecast.models.simple_model import SimpleFinancialModel
from financial_forecast.types import EconomicInputs, FinancialState


def main() -> None:
    """Execute the deterministic forecast simulation."""
    model = SimpleFinancialModel()

    initial_data = {
        "nca": 2.1735e11,
        "advance_payments_purchases": 21223000000,
        "accounts_receivable": 60932000000,
        "inventory": 4946000000,
        "cash": 23646000000,
        "investment_in_market_securities": 24658000000,
        "accounts_payable": 64115000000,
        "advance_payments_sales": 7912000000,
        "current_liabilities": 81955000000,
        "non_current_liabilities": 1.48101e11,
        "equity": 50672000000,
        "net_income": 99803000000,
    }
    state = FinancialState.from_dict(initial_data)

    sales_forecast = [3.94328e11, 3.83285e11, 3.91035e11, 4.16161e11]
    purch_forecast = [2.07694e11, 1.99862e11, 2.04003e11, 2.10808e11]
    cum_inflation = [1.0] * 4

    print(
        f"\n{'Year':<5} | {'Assets':<15} | {'Liabilities':<15} | {'Equity':<15} | {'Check':<15}"
    )
    print("-" * 75)

    total_assets = (
        state.nca + state.advance_payments_purchases + state.accounts_receivable
        + state.inventory + state.cash + state.investment_in_market_securities
    )
    total_liabilities = (
        state.accounts_payable + state.advance_payments_sales
        + state.current_liabilities + state.non_current_liabilities
    )
    print(
        f"{0:<5} | {total_assets/1e9:>14.2f}B | {total_liabilities/1e9:>14.2f}B | {state.equity/1e9:>14.2f}B | {state.balance_sheet_check:>14.2f}"
    )

    for t in range(len(sales_forecast) - 1):
        inputs = EconomicInputs(
            sales_t=tf.constant(sales_forecast[t], dtype=tf.float64),
            purchases_t=tf.constant(purch_forecast[t], dtype=tf.float64),
            sales_t_plus_1=tf.constant(sales_forecast[t + 1], dtype=tf.float64),
            purchases_t_plus_1=tf.constant(purch_forecast[t + 1], dtype=tf.float64),
            cum_inflation=tf.constant(cum_inflation[t], dtype=tf.float64),
            t=t + 1,
        )
        state = model.forecast_step(state, inputs)

        curr_assets = (
            state.nca + state.advance_payments_purchases + state.accounts_receivable
            + state.inventory + state.cash + state.investment_in_market_securities
        )
        curr_liabilities = (
            state.accounts_payable + state.advance_payments_sales
            + state.current_liabilities + state.non_current_liabilities
        )
        print(
            f"{t+1:<5} | {curr_assets.numpy()/1e9:>14.2f}B | {curr_liabilities.numpy()/1e9:>14.2f}B | {state.equity.numpy()/1e9:>14.2f}B | {state.balance_sheet_check.numpy():>14.2f}"
        )


if __name__ == "__main__":
    main()
