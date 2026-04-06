"""MICROSOFT CORP historical financial data (FY2018-FY2025).

Data sourced from SEC EDGAR XBRL company facts API.
Generated on 2026-04-06.
All monetary values are in USD.
"""

import tensorflow as tf


def get_financial_statements():
    """Return MICROSOFT CORP historical financial data as a dictionary of TensorFlow tensors.

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
        1.104e+11,
        1.258e+11,
        1.430e+11,
        1.681e+11,
        1.983e+11,
        2.119e+11,
        2.451e+11,
        2.817e+11,
        ],
        dtype=tf.float64,
    )
    cogs = tf.constant(
        [
        3.065e+10,
        3.321e+10,
        3.538e+10,
        4.293e+10,
        5.005e+10,
        5.486e+10,
        5.891e+10,
        6.583e+10,
        ],
        dtype=tf.float64,
    )
    depreciation = tf.constant(
        [
        7.700e+09,
        9.700e+09,
        1.070e+10,
        9.300e+09,
        1.260e+10,
        1.100e+10,
        1.520e+10,
        2.200e+10,
        ],
        dtype=tf.float64,
    )
    opex = tf.constant(
        [
        3.695e+10,
        3.997e+10,
        4.398e+10,
        4.594e+10,
        5.224e+10,
        5.753e+10,
        6.158e+10,
        6.536e+10,
        ],
        dtype=tf.float64,
    )
    net_income = tf.constant(
        [
        1.657e+10,
        3.924e+10,
        4.428e+10,
        6.127e+10,
        7.274e+10,
        7.236e+10,
        8.814e+10,
        1.018e+11,
        ],
        dtype=tf.float64,
    )
    tax = tf.constant(
        [
        1.990e+10,
        4.448e+09,
        8.755e+09,
        9.831e+09,
        1.098e+10,
        1.695e+10,
        1.965e+10,
        2.180e+10,
        ],
        dtype=tf.float64,
    )
    interest_expense = tf.constant(
        [
        2.733e+09,
        2.686e+09,
        2.591e+09,
        2.346e+09,
        2.063e+09,
        1.968e+09,
        2.935e+09,
        2.385e+09,
        ],
        dtype=tf.float64,
    )
    ms_investment_return = tf.constant(
        [
        4.149e+09,
        3.415e+09,
        2.668e+09,
        3.532e+09,
        2.396e+09,
        2.756e+09,
        1.289e+09,
        -2.516e+09,
        ],
        dtype=tf.float64,
    )

    # --- Balance Sheet ---
    inventory = tf.constant(
        [
        2.662e+09,
        2.063e+09,
        1.895e+09,
        2.636e+09,
        3.742e+09,
        2.500e+09,
        1.246e+09,
        938000000,
        ],
        dtype=tf.float64,
    )
    change_in_inventory = tf.constant(
        [
        481000000,
        -599000000,
        -168000000,
        741000000,
        1.106e+09,
        -1.242e+09,
        -1.254e+09,
        -308000000,
        ],
        dtype=tf.float64,
    )
    nca = tf.constant(
        [
        8.919e+10,
        1.110e+11,
        1.194e+11,
        1.494e+11,
        1.952e+11,
        2.277e+11,
        3.524e+11,
        4.279e+11,
        ],
        dtype=tf.float64,
    )
    accounts_receivable = tf.constant(
        [
        2.648e+10,
        2.952e+10,
        3.201e+10,
        3.804e+10,
        4.426e+10,
        4.869e+10,
        5.692e+10,
        6.990e+10,
        ],
        dtype=tf.float64,
    )
    accounts_payable = tf.constant(
        [
        8.617e+09,
        9.382e+09,
        1.253e+10,
        1.516e+10,
        1.900e+10,
        1.810e+10,
        2.200e+10,
        2.772e+10,
        ],
        dtype=tf.float64,
    )
    advance_payments_purchases = tf.constant(
        [
        6.751e+09,
        1.015e+10,
        1.148e+10,
        1.339e+10,
        1.692e+10,
        2.181e+10,
        2.602e+10,
        2.572e+10,
        ],
        dtype=tf.float64,
    )
    advance_payments_sales = tf.constant(
        [
        2.890e+10,
        3.268e+10,
        3.600e+10,
        4.152e+10,
        4.554e+10,
        5.090e+10,
        5.758e+10,
        6.456e+10,
        ],
        dtype=tf.float64,
    )
    cash = tf.constant(
        [
        1.195e+10,
        1.136e+10,
        1.358e+10,
        1.422e+10,
        1.393e+10,
        3.470e+10,
        1.832e+10,
        3.024e+10,
        ],
        dtype=tf.float64,
    )
    ims = tf.constant(
        [
        1.218e+11,
        1.225e+11,
        1.230e+11,
        1.161e+11,
        9.083e+10,
        7.656e+10,
        5.723e+10,
        6.432e+10,
        ],
        dtype=tf.float64,
    )
    current_lt_debt = tf.constant(
        [
        3.998e+09,
        5.516e+09,
        3.749e+09,
        8.072e+09,
        2.749e+09,
        5.247e+09,
        2.249e+09,
        2.999e+09,
        ],
        dtype=tf.float64,
    )
    non_current_liabilities = tf.constant(
        [
        1.176e+11,
        1.148e+11,
        1.107e+11,
        1.031e+11,
        1.032e+11,
        1.016e+11,
        1.184e+11,
        1.343e+11,
        ],
        dtype=tf.float64,
    )
    equity = tf.constant(
        [
        8.272e+10,
        1.023e+11,
        1.183e+11,
        1.420e+11,
        1.665e+11,
        2.062e+11,
        2.685e+11,
        3.435e+11,
        ],
        dtype=tf.float64,
    )
    current_liabilities_source = tf.constant(
        [
        5.849e+10,
        6.942e+10,
        7.231e+10,
        8.866e+10,
        9.508e+10,
        1.041e+11,
        1.253e+11,
        1.412e+11,
        ],
        dtype=tf.float64,
    )

    # --- Cash Flow ---
    dividends = tf.constant(
        [
        1.270e+10,
        1.381e+10,
        1.514e+10,
        1.652e+10,
        1.814e+10,
        1.980e+10,
        2.177e+10,
        2.408e+10,
        ],
        dtype=tf.float64,
    )
    stock_buyback = tf.constant(
        [
        1.072e+10,
        1.954e+10,
        2.297e+10,
        2.738e+10,
        3.270e+10,
        2.224e+10,
        1.725e+10,
        1.842e+10,
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
