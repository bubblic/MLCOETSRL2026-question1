"""PFIZER INC historical financial data (FY2018-FY2025).

Data sourced from SEC EDGAR XBRL company facts API.
Generated on 2026-04-06.
All monetary values are in USD.
"""

import tensorflow as tf


def get_financial_statements():
    """Return PFIZER INC historical financial data as a dictionary of TensorFlow tensors.

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
        4.082e+10,
        4.090e+10,
        4.165e+10,
        8.129e+10,
        1.012e+11,
        5.955e+10,
        6.363e+10,
        6.258e+10,
        ],
        dtype=tf.float64,
    )
    cogs = tf.constant(
        [
        2.603e+09,
        2.299e+09,
        3.803e+09,
        2.563e+10,
        2.928e+10,
        1.866e+10,
        1.084e+10,
        9.475e+09,
        ],
        dtype=tf.float64,
    )
    depreciation = tf.constant(
        [
        6.384e+09,
        5.755e+09,
        4.681e+09,
        5.191e+09,
        5.064e+09,
        6.290e+09,
        7.013e+09,
        6.592e+09,
        ],
        dtype=tf.float64,
    )
    opex = tf.constant(
        [
        1.261e+10,
        1.273e+10,
        1.160e+10,
        1.270e+10,
        1.368e+10,
        1.477e+10,
        1.473e+10,
        1.379e+10,
        ],
        dtype=tf.float64,
    )
    net_income = tf.constant(
        [
        1.115e+10,
        1.603e+10,
        9.159e+09,
        2.198e+10,
        3.137e+10,
        2.119e+09,
        8.031e+09,
        7.771e+09,
        ],
        dtype=tf.float64,
    )
    tax = tf.constant(
        [
        -266000000,
        583000000,
        370000000,
        1.852e+09,
        3.328e+09,
        -1.115e+09,
        -28000000,
        -266000000,
        ],
        dtype=tf.float64,
    )
    interest_expense = tf.constant(
        [
        float('nan'),
        float('nan'),
        float('nan'),
        float('nan'),
        float('nan'),
        float('nan'),
        float('nan'),
        float('nan'),
        ],
        dtype=tf.float64,
    )
    ms_investment_return = tf.constant(
        [
        float('nan'),
        float('nan'),
        float('nan'),
        float('nan'),
        float('nan'),
        float('nan'),
        float('nan'),
        float('nan'),
        ],
        dtype=tf.float64,
    )

    # --- Balance Sheet ---
    inventory = tf.constant(
        [
        7.508e+09,
        7.068e+09,
        8.020e+09,
        9.059e+09,
        8.981e+09,
        1.019e+10,
        1.085e+10,
        1.065e+10,
        ],
        dtype=tf.float64,
    )
    change_in_inventory = tf.constant(
        [
        -70000000,
        -440000000,
        952000000,
        1.039e+09,
        -78000000,
        1.208e+09,
        662000000,
        -197000000,
        ],
        dtype=tf.float64,
    )
    nca = tf.constant(
        [
        1.095e+11,
        1.348e+11,
        1.192e+11,
        1.218e+11,
        1.459e+11,
        1.832e+11,
        1.630e+11,
        1.653e+11,
        ],
        dtype=tf.float64,
    )
    accounts_receivable = tf.constant(
        [
        8.025e+09,
        6.772e+09,
        7.913e+09,
        1.148e+10,
        1.095e+10,
        1.157e+10,
        1.146e+10,
        1.187e+10,
        ],
        dtype=tf.float64,
    )
    accounts_payable = tf.constant(
        [
        4.674e+09,
        3.887e+09,
        4.283e+09,
        5.578e+09,
        6.809e+09,
        6.710e+09,
        5.633e+09,
        5.240e+09,
        ],
        dtype=tf.float64,
    )
    advance_payments_purchases = tf.constant(
        [
        2.461e+09,
        2.357e+09,
        3.646e+09,
        3.820e+09,
        5.017e+09,
        4.911e+09,
        4.253e+09,
        2.808e+09,
        ],
        dtype=tf.float64,
    )
    advance_payments_sales = tf.constant(
        [
        0,
        0,
        1.113e+09,
        3.067e+09,
        2.520e+09,
        2.700e+09,
        1.511e+09,
        784000000,
        ],
        dtype=tf.float64,
    )
    cash = tf.constant(
        [
        1.139e+09,
        1.121e+09,
        1.786e+09,
        1.944e+09,
        416000000,
        2.853e+09,
        1.043e+09,
        1.142e+09,
        ],
        dtype=tf.float64,
    )
    ims = tf.constant(
        [
        1.769e+10,
        8.525e+09,
        9.709e+09,
        2.201e+10,
        1.874e+10,
        4.400e+09,
        1.088e+10,
        9.183e+09,
        ],
        dtype=tf.float64,
    )
    current_lt_debt = tf.constant(
        [
        4.776e+09,
        1.462e+09,
        2.002e+09,
        1.636e+09,
        2.560e+09,
        2.254e+09,
        3.747e+09,
        2.997e+09,
        ],
        dtype=tf.float64,
    )
    non_current_liabilities = tf.constant(
        [
        6.381e+10,
        6.684e+10,
        6.484e+10,
        6.134e+10,
        5.915e+10,
        8.942e+10,
        8.190e+10,
        8.440e+10,
        ],
        dtype=tf.float64,
    )
    equity = tf.constant(
        [
        6.341e+10,
        6.314e+10,
        6.324e+10,
        7.720e+10,
        9.566e+10,
        8.901e+10,
        8.820e+10,
        8.648e+10,
        ],
        dtype=tf.float64,
    )
    current_liabilities_source = tf.constant(
        [
        3.186e+10,
        3.730e+10,
        2.592e+10,
        4.267e+10,
        4.214e+10,
        4.779e+10,
        4.300e+10,
        3.698e+10,
        ],
        dtype=tf.float64,
    )

    # --- Cash Flow ---
    dividends = tf.constant(
        [
        7.978e+09,
        8.043e+09,
        8.440e+09,
        8.729e+09,
        8.983e+09,
        9.247e+09,
        9.512e+09,
        9.771e+09,
        ],
        dtype=tf.float64,
    )
    stock_buyback = tf.constant(
        [
        1.220e+10,
        8.865e+09,
        0,
        0,
        2.000e+09,
        0,
        0,
        0,
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
