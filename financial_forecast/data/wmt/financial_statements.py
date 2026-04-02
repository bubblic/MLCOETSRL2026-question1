"""Walmart Inc. historical financial data (FY2018-FY2025).

Data sourced from SEC EDGAR XBRL company facts API.
Generated on 2026-04-02.
All monetary values are in USD.
"""

import tensorflow as tf


def get_financial_statements():
    """Return Walmart Inc. historical financial data as a dictionary of TensorFlow tensors.

    Returns:
        dict with the following keys (all tf.float64 tensors):

        Metadata:
            years             - fiscal year labels [2018..2025]

        Income Statement:
            sales                - total revenues
            cogs                 - cost of goods sold (excl. depreciation)
            depreciation         - reconciled depreciation
            cost_of_revenue      - cogs + depreciation
            opex                 - operating expenses
            net_income           - net income
            tax                  - income tax provision
            interest_payment     - interest expense (non-operating)
            ms_return            - non-interest investment returns

        Balance Sheet:
            inventory            - inventory
            change_in_inventory  - year-over-year inventory change
            nca                  - non-current assets
            accounts_receivable  - accounts receivable
            accounts_payable     - accounts payable
            advance_payments_purchases - other current assets
            advance_payments_sales     - current deferred revenue
            cash                       - cash and cash equivalents
            ims                        - short-term investments
            current_liabilities        - derived to enforce Assets = L + E
            current_liabilities_source - raw value from source (for comparison)
            current_lt_debt            - current portion of long-term debt
            non_current_liabilities    - non-current liabilities
            equity                     - stockholders' equity

        Cash Flow:
            dividends       - common stock dividends paid
            stock_buyback   - repurchase of capital stock

        Derived:
            purchases       - cogs + change_in_inventory
            cost_of_revenue - cogs + depreciation
    """
    years = tf.range(2018, 2026, dtype=tf.float64)

    # --- Income Statement ---
    sales = tf.constant(
        [
        5.144e+11,
        5.240e+11,
        5.592e+11,
        5.728e+11,
        6.113e+11,
        6.481e+11,
        6.810e+11,
        7.132e+11,
        ],
        dtype=tf.float64,
    )
    cogs = tf.constant(
        [
        3.746e+11,
        3.836e+11,
        4.092e+11,
        4.183e+11,
        4.528e+11,
        4.783e+11,
        4.988e+11,
        5.212e+11,
        ],
        dtype=tf.float64,
    )
    depreciation = tf.constant(
        [
        1.068e+10,
        1.099e+10,
        1.115e+10,
        1.066e+10,
        1.094e+10,
        1.185e+10,
        1.297e+10,
        1.420e+10,
        ],
        dtype=tf.float64,
    )
    opex = tf.constant(
        [
        1.071e+11,
        1.088e+11,
        1.163e+11,
        1.178e+11,
        1.271e+11,
        1.310e+11,
        1.399e+11,
        1.479e+11,
        ],
        dtype=tf.float64,
    )
    net_income = tf.constant(
        [
        6.670e+09,
        1.488e+10,
        1.351e+10,
        1.367e+10,
        1.168e+10,
        1.551e+10,
        1.944e+10,
        2.189e+10,
        ],
        dtype=tf.float64,
    )
    tax = tf.constant(
        [
        4.281e+09,
        4.915e+09,
        6.858e+09,
        4.756e+09,
        5.724e+09,
        5.578e+09,
        6.152e+09,
        7.199e+09,
        ],
        dtype=tf.float64,
    )
    interest_expense = tf.constant(
        [
        1.975e+09,
        2.262e+09,
        1.976e+09,
        1.674e+09,
        1.787e+09,
        2.259e+09,
        2.249e+09,
        2.318e+09,
        ],
        dtype=tf.float64,
    )
    ms_investment_return = tf.constant(
        [
        -8.522e+09,
        1.810e+09,
        -8000000,
        -5.572e+09,
        -1.625e+09,
        -2.905e+09,
        -790000000,
        1.962e+09,
        ],
        dtype=tf.float64,
    )

    # --- Balance Sheet ---
    inventory = tf.constant(
        [
        4.427e+10,
        4.444e+10,
        4.495e+10,
        5.651e+10,
        5.658e+10,
        5.489e+10,
        5.644e+10,
        5.885e+10,
        ],
        dtype=tf.float64,
    )
    change_in_inventory = tf.constant(
        [
        486000000,
        166000000,
        514000000,
        1.156e+10,
        65000000,
        -1.684e+09,
        1.543e+09,
        2.416e+09,
        ],
        dtype=tf.float64,
    )
    nca = tf.constant(
        [
        1.574e+11,
        1.747e+11,
        1.624e+11,
        1.638e+11,
        1.675e+11,
        1.755e+11,
        1.814e+11,
        1.998e+11,
        ],
        dtype=tf.float64,
    )
    accounts_receivable = tf.constant(
        [
        6.283e+09,
        6.284e+09,
        6.516e+09,
        8.280e+09,
        7.933e+09,
        8.796e+09,
        9.975e+09,
        1.117e+10,
        ],
        dtype=tf.float64,
    )
    accounts_payable = tf.constant(
        [
        4.706e+10,
        4.697e+10,
        4.914e+10,
        5.526e+10,
        5.374e+10,
        5.681e+10,
        5.867e+10,
        6.306e+10,
        ],
        dtype=tf.float64,
    )
    advance_payments_purchases = tf.constant(
        [
        3.623e+09,
        1.622e+09,
        2.086e+10,
        1.519e+09,
        2.521e+09,
        3.322e+09,
        4.011e+09,
        4.124e+09,
        ],
        dtype=tf.float64,
    )
    advance_payments_sales = tf.constant(
        [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ],
        dtype=tf.float64,
    )
    cash = tf.constant(
        [
        7.722e+09,
        9.465e+09,
        1.774e+10,
        1.476e+10,
        8.625e+09,
        9.867e+09,
        9.037e+09,
        1.073e+10,
        ],
        dtype=tf.float64,
    )
    ims = tf.constant(
        [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ],
        dtype=tf.float64,
    )
    current_lt_debt = tf.constant(
        [
        1.876e+09,
        5.362e+09,
        3.115e+09,
        2.803e+09,
        4.191e+09,
        3.447e+09,
        2.598e+09,
        3.542e+09,
        ],
        dtype=tf.float64,
    )
    non_current_liabilities = tf.constant(
        [
        1.418e+11,
        1.587e+11,
        1.599e+11,
        1.575e+11,
        1.510e+11,
        1.600e+11,
        1.642e+11,
        1.772e+11,
        ],
        dtype=tf.float64,
    )
    equity = tf.constant(
        [
        7.250e+10,
        7.467e+10,
        8.092e+10,
        8.325e+10,
        7.669e+10,
        8.386e+10,
        9.101e+10,
        9.962e+10,
        ],
        dtype=tf.float64,
    )
    current_liabilities_source = tf.constant(
        [
        7.748e+10,
        7.779e+10,
        9.264e+10,
        8.738e+10,
        9.220e+10,
        9.242e+10,
        9.658e+10,
        1.075e+11,
        ],
        dtype=tf.float64,
    )

    # --- Cash Flow ---
    dividends = tf.constant(
        [
        6.102e+09,
        6.048e+09,
        6.116e+09,
        6.152e+09,
        6.114e+09,
        6.140e+09,
        6.688e+09,
        7.507e+09,
        ],
        dtype=tf.float64,
    )
    stock_buyback = tf.constant(
        [
        7.410e+09,
        5.717e+09,
        2.625e+09,
        9.787e+09,
        9.920e+09,
        2.779e+09,
        4.494e+09,
        8.088e+09,
        ],
        dtype=tf.float64,
    )

    # --- Derived ---
    # Enforce balance sheet identity: Assets = Liabilities + Equity
    current_liabilities = (
        nca + advance_payments_purchases + accounts_receivable
        + inventory + cash + ims
        - non_current_liabilities - equity
    )
    purchases = cogs + change_in_inventory
    cost_of_revenue = cogs + depreciation

    return {
        "years": years,
        # Income Statement
        "sales": sales,
        "cogs": cogs,
        "depreciation": depreciation,
        "cost_of_revenue": cost_of_revenue,
        "opex": opex,
        "net_income": net_income,
        "tax": tax,
        "interest_payment": interest_expense,
        "ms_return": ms_investment_return,
        # Balance Sheet
        "inventory": inventory,
        "change_in_inventory": change_in_inventory,
        "nca": nca,
        "accounts_receivable": accounts_receivable,
        "accounts_payable": accounts_payable,
        "advance_payments_purchases": advance_payments_purchases,
        "advance_payments_sales": advance_payments_sales,
        "cash": cash,
        "ims": ims,
        "current_liabilities": current_liabilities,
        "current_liabilities_source": current_liabilities_source,
        "current_lt_debt": current_lt_debt,
        "non_current_liabilities": non_current_liabilities,
        "equity": equity,
        # Cash Flow
        "dividends": dividends,
        "stock_buyback": stock_buyback,
        # Derived
        "purchases": purchases,
    }
