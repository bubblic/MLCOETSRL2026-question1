"""Apple Inc. historical financial data (FY2018-FY2025).

Data sourced from Apple's SEC filings (10-K annual reports).
All monetary values are in USD. Fiscal years are Apple's fiscal years
(ending in late September).
"""

import numpy as np


def get_apple_historical_data():
    """Return Apple's historical financial data as a dictionary of numpy arrays.

    Returns:
        dict with the following keys (all np.float64 arrays):

        Metadata:
            years             – fiscal year labels [2018..2025]

        Income Statement:
            sales                – total revenues
            cogs                 – cost of goods sold (excl. depreciation)
            depreciation         – reconciled depreciation
            cost_of_revenue      – cogs + depreciation
            opex                 – operating expenses
            net_income           – net income
            tax                  – income tax provision
            tax_onetime_payments – one-time tax anomaly amounts in dollars (null -> 0)

        Balance Sheet:
            inventory_plus_one – inventory with one extra leading year (9 values)
            inventory          – inventory aligned to fiscal years (8 values)
            nca                – non-current assets
            accounts_receivable
            accounts_payable
            advance_payments_purchases – other current assets
            advance_payments_sales     – current deferred revenue
            cash                       – cash and cash equivalents
            ims                        – investment in market securities
            current_liabilities        – total current liabilities minus AP minus deferred rev
            non_current_liabilities
            equity                     – stockholders' equity
            st_debt                    – short-term debt (commercial paper / revolving credit)

        Cash Flow:
            dividends       – common stock dividends paid
            stock_buyback   – repurchase of capital stock

        Derived:
            purchases       – cogs + delta(inventory)

        Macro:
            inflation       – annual CPI inflation rates

        Note:
            Fields may contain np.nan for years where a value is unavailable.
    """
    years = np.arange(2018, 2026)

    # --- Income Statement ---
    sales = np.array(
        [
            2.65595e11,
            2.60174e11,
            2.74515e11,
            3.65817e11,
            3.94328e11,
            3.83285e11,
            3.91035e11,
            4.16161e11,
        ],
        dtype=np.float64,
    )
    cogs = np.array(
        [
            1.52853e11,
            1.49235e11,
            1.58503e11,
            2.01697e11,
            2.12442e11,
            2.02618e11,
            1.98907e11,
            2.09262e11,
        ],
        dtype=np.float64,
    )
    depreciation = np.array(
        [
            10903000000,
            12547000000,
            11056000000,
            11284000000,
            11104000000,
            11519000000,
            11445000000,
            11698000000,
        ],
        dtype=np.float64,
    )
    opex = np.array(
        [
            30941000000,
            34462000000,
            38668000000,
            43887000000,
            51345000000,
            54847000000,
            57467000000,
            62151000000,
        ],
        dtype=np.float64,
    )
    net_income = np.array(
        [
            59531000000,
            55256000000,
            57411000000,
            94680000000,
            99803000000,
            96995000000,
            93736000000,
            1.1201e11,
        ],
        dtype=np.float64,
    )
    tax = np.array(
        [
            13372000000,
            10481000000,
            9680000000,
            14527000000,
            19300000000,
            16741000000,
            29749000000,
            20719000000,
        ],
        dtype=np.float64,
    )
    # Extracted from extracted_text/apple_YYYY.tax-anomalies-contingencies.llm.json.
    # Source values are in billions; null values are mapped to 0.0.
    tax_onetime_payments = np.array(
        [
            1.5e9,  # 2018
            0.0,  # 2019 (null)
            -0.582e9,  # 2020
            0.0,  # 2021 (null)
            0.0,  # 2022 (null)
            0.0,  # 2023 (null)
            10.2e9,  # 2024
            0.0,  # 2025 (null)
        ],
        dtype=np.float64,
    )
    interest_expense = np.array(
        [
            3240e6,
            3576e6,
            2873e6,
            2645e6,
            2931e6,
            3933e6,
            np.nan,  # Placeholder removed: data unavailable for this fiscal year
            np.nan,  # Placeholder removed: data unavailable for this fiscal year
        ],
        dtype=np.float64,
    )
    ms_investment_return = np.array(
        [-3406e6, 3827e6, 1139e6, -967e6, -11899e6, 1816e6, 6054e6, 1231e6],
        dtype=np.float64,
    )

    # --- Balance Sheet ---
    # Includes one additional year at the beginning to derive purchases
    inventory_plus_one = np.array(
        [
            4855000000,
            3956000000,
            4106000000,
            4061000000,
            6580000000,
            4946000000,
            6331000000,
            7286000000,
            5718000000,
        ],
        dtype=np.float64,
    )
    nca = np.array(
        [
            2.34386e11,
            1.75697e11,
            1.80175e11,
            2.16166e11,
            2.1735e11,
            2.09017e11,
            2.11993e11,
            2.11284e11,
        ],
        dtype=np.float64,
    )
    accounts_receivable = np.array(
        [
            48995000000,
            45804000000,
            37445000000,
            51506000000,
            60932000000,
            60985000000,
            66243000000,
            72957000000,
        ],
        dtype=np.float64,
    )
    accounts_payable = np.array(
        [
            55888000000,
            46236000000,
            42296000000,
            54763000000,
            64115000000,
            62611000000,
            68960000000,
            69860000000,
        ],
        dtype=np.float64,
    )
    advance_payments_purchases = np.array(
        [
            12087000000,
            12352000000,
            11264000000,
            14111000000,
            21223000000,
            14695000000,
            14287000000,
            14585000000,
        ],
        dtype=np.float64,
    )
    advance_payments_sales = np.array(
        [
            5966000000,
            5522000000,
            6643000000,
            7612000000,
            7912000000,
            8061000000,
            8249000000,
            9055000000,
        ],
        dtype=np.float64,
    )
    cash = np.array(
        [
            25913000000,
            48844000000,
            38016000000,
            34940000000,
            23646000000,
            29965000000,
            29943000000,
            35934000000,
        ],
        dtype=np.float64,
    )
    ims = np.array(
        [
            40388000000,
            51713000000,
            52927000000,
            27699000000,
            24658000000,
            31590000000,
            35228000000,
            18763000000,
        ],
        dtype=np.float64,
    )
    current_liabilities = np.array(
        [
            115929e6,
            105718e6,
            105392e6,
            125481e6,
            153982e6,
            145308e6,
            176392e6,
            165631e6,
        ],
        dtype=np.float64,
    )
    current_lt_debt = np.array(
        [8784e6, 10260e6, 8773e6, 9613e6, 11128e6, 9822e6, 10912e6, 12350e6],
        dtype=np.float64,
    )
    non_current_liabilities = np.array(
        [
            1.41712e11,
            1.4231e11,
            1.53157e11,
            1.62431e11,
            1.48101e11,
            1.45129e11,
            1.31638e11,
            1.19877e11,
        ],
        dtype=np.float64,
    )
    equity = np.array(
        [
            1.07147e11,
            90488000000,
            65339000000,
            63090000000,
            50672000000,
            62146000000,
            56950000000,
            73733000000,
        ],
        dtype=np.float64,
    )

    # --- Cash Flow ---
    dividends = np.array(
        [
            13712000000,
            14119000000,
            14081000000,
            14467000000,
            14841000000,
            15025000000,
            15234000000,
            15421000000,
        ],
        dtype=np.float64,
    )
    stock_buyback = np.array(
        [
            72738000000,
            66897000000,
            72358000000,
            85971000000,
            89402000000,
            77550000000,
            94949000000,
            90711000000,
        ],
        dtype=np.float64,
    )

    # --- Derived ---
    inventory = inventory_plus_one[1:]
    purchases = cogs + inventory_plus_one[1:] - inventory_plus_one[:-1]
    cost_of_revenue = cogs + depreciation

    # --- Macro ---
    inflation = np.array(
        [0.024, 0.018, 0.012, 0.047, 0.08, 0.041, 0.029, 0.027],
        dtype=np.float64,
    )

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
        "tax_onetime_payments": tax_onetime_payments,
        "interest_payment": interest_expense,
        "ms_return": ms_investment_return,
        # Balance Sheet
        "inventory_plus_one": inventory_plus_one,
        "inventory": inventory,
        "nca": nca,
        "accounts_receivable": accounts_receivable,
        "accounts_payable": accounts_payable,
        "advance_payments_purchases": advance_payments_purchases,
        "advance_payments_sales": advance_payments_sales,
        "cash": cash,
        "ims": ims,
        "current_liabilities": current_liabilities,
        "non_current_liabilities": non_current_liabilities,
        "equity": equity,
        "current_lt_debt": current_lt_debt,
        # Cash Flow
        "dividends": dividends,
        "stock_buyback": stock_buyback,
        # Derived
        "purchases": purchases,
        # Macro
        "inflation": inflation,
    }
