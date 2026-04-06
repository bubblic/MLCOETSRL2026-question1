"""Apple Inc. historical financial data (FY2018-FY2025).

Data sourced from SEC EDGAR XBRL company facts API.
Generated on 2026-04-06.
All monetary values are in USD.
"""

import tensorflow as tf


def get_financial_statements():
    """Return Apple Inc. historical financial data as a dictionary of TensorFlow tensors.

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
            2.656e11,
            2.602e11,
            2.745e11,
            3.658e11,
            3.943e11,
            3.833e11,
            3.910e11,
            4.162e11,
        ],
        dtype=tf.float64,
    )
    cogs = tf.constant(
        [
            1.529e11,
            1.492e11,
            1.585e11,
            2.017e11,
            2.124e11,
            2.026e11,
            1.989e11,
            2.093e11,
        ],
        dtype=tf.float64,
    )
    depreciation = tf.constant(
        [
            1.090e10,
            1.255e10,
            1.106e10,
            1.128e10,
            1.110e10,
            1.152e10,
            1.144e10,
            1.170e10,
        ],
        dtype=tf.float64,
    )
    opex = tf.constant(
        [
            3.094e10,
            3.446e10,
            3.867e10,
            4.389e10,
            5.134e10,
            5.485e10,
            5.747e10,
            6.215e10,
        ],
        dtype=tf.float64,
    )
    net_income = tf.constant(
        [
            5.953e10,
            5.526e10,
            5.741e10,
            9.468e10,
            9.980e10,
            9.700e10,
            9.374e10,
            1.120e11,
        ],
        dtype=tf.float64,
    )
    tax = tf.constant(
        [
            1.337e10,
            1.048e10,
            9.680e09,
            1.453e10,
            1.930e10,
            1.674e10,
            2.975e10,
            2.072e10,
        ],
        dtype=tf.float64,
    )
    interest_expense = tf.constant(
        [
            3.240e09,
            3.576e09,
            2.873e09,
            2.645e09,
            2.931e09,
            3.933e09,
            0,
            0,
        ],
        dtype=tf.float64,
    )
    ms_investment_return = tf.constant(
        [
            5.245e09,
            5.383e09,
            3.676e09,
            2.903e09,
            2.597e09,
            3.368e09,
            269000000,
            -321000000,
        ],
        dtype=tf.float64,
    )

    # --- Balance Sheet ---
    inventory = tf.constant(
        [
            3.956e09,
            4.106e09,
            4.061e09,
            6.580e09,
            4.946e09,
            6.331e09,
            7.286e09,
            5.718e09,
        ],
        dtype=tf.float64,
    )
    change_in_inventory = tf.constant(
        [
            -899000000,
            150000000,
            -45000000,
            2.519e09,
            -1.634e09,
            1.385e09,
            955000000,
            -1.568e09,
        ],
        dtype=tf.float64,
    )
    nca = tf.constant(
        [
            2.344e11,
            1.757e11,
            1.802e11,
            2.162e11,
            2.174e11,
            2.090e11,
            2.120e11,
            2.113e11,
        ],
        dtype=tf.float64,
    )
    accounts_receivable = tf.constant(
        [
            4.900e10,
            4.580e10,
            3.744e10,
            5.151e10,
            6.093e10,
            6.098e10,
            6.624e10,
            7.296e10,
        ],
        dtype=tf.float64,
    )
    accounts_payable = tf.constant(
        [
            5.589e10,
            4.624e10,
            4.230e10,
            5.476e10,
            6.412e10,
            6.261e10,
            6.896e10,
            6.986e10,
        ],
        dtype=tf.float64,
    )
    advance_payments_purchases = tf.constant(
        [
            1.209e10,
            1.235e10,
            1.126e10,
            1.411e10,
            2.122e10,
            1.470e10,
            1.429e10,
            1.458e10,
        ],
        dtype=tf.float64,
    )
    advance_payments_sales = tf.constant(
        [
            5.966e09,
            5.522e09,
            6.643e09,
            7.612e09,
            7.912e09,
            8.061e09,
            8.249e09,
            9.055e09,
        ],
        dtype=tf.float64,
    )
    cash = tf.constant(
        [
            2.591e10,
            4.884e10,
            3.802e10,
            3.494e10,
            2.365e10,
            2.996e10,
            2.994e10,
            3.593e10,
        ],
        dtype=tf.float64,
    )
    ims = tf.constant(
        [
            4.039e10,
            5.171e10,
            5.293e10,
            2.770e10,
            2.466e10,
            3.159e10,
            3.523e10,
            1.876e10,
        ],
        dtype=tf.float64,
    )
    current_lt_debt = tf.constant(
        [
            8.784e09,
            1.026e10,
            8.773e09,
            9.613e09,
            1.113e10,
            9.822e09,
            1.091e10,
            1.235e10,
        ],
        dtype=tf.float64,
    )
    non_current_liabilities = tf.constant(
        [
            1.426e11,
            1.423e11,
            1.532e11,
            1.624e11,
            1.481e11,
            1.451e11,
            1.316e11,
            1.199e11,
        ],
        dtype=tf.float64,
    )
    equity = tf.constant(
        [
            1.071e11,
            9.049e10,
            6.534e10,
            6.309e10,
            5.067e10,
            6.215e10,
            5.695e10,
            7.373e10,
        ],
        dtype=tf.float64,
    )
    current_liabilities_source = tf.constant(
        [
            1.159e11,
            1.057e11,
            1.054e11,
            1.255e11,
            1.540e11,
            1.453e11,
            1.764e11,
            1.656e11,
        ],
        dtype=tf.float64,
    )

    # --- Cash Flow ---
    dividends = tf.constant(
        [
            1.371e10,
            1.412e10,
            1.408e10,
            1.447e10,
            1.484e10,
            1.502e10,
            1.523e10,
            1.542e10,
        ],
        dtype=tf.float64,
    )
    stock_buyback = tf.constant(
        [
            7.274e10,
            6.690e10,
            7.236e10,
            8.597e10,
            8.940e10,
            7.755e10,
            9.495e10,
            9.071e10,
        ],
        dtype=tf.float64,
    )

    # --- Derived ---
    # Enforce balance sheet identity: Assets = Liabilities + Equity
    current_liabilities = (
        nca
        + advance_payments_purchases
        + accounts_receivable
        + inventory
        + cash
        + ims
        - non_current_liabilities
        - equity
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
