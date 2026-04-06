"""Build a financial_statements.py module from Yahoo Finance data.

Pulls income statement, balance sheet, and cash flow data from Yahoo
Finance and writes a ``financial_statements.py`` module to
``financial_forecast/data/<ticker>/``.  The generated module follows
the same structure as the hand-curated Apple data, producing a dict
of TensorFlow tensors consumable by :class:`HistoricalDataLoader`.

Field Mapping (Yahoo Finance -> Model):
    Income Statement:
        Total Revenue                           -> sales
        Cost Of Revenue - Reconciled Depreciation -> cogs
        Reconciled Depreciation                 -> depreciation
        Operating Expense                       -> opex
        Net Income                              -> net_income
        Tax Provision                           -> tax
        Interest Expense Non Operating          -> interest_payment
        Pretax Income - EBIT + Interest Expense -> ms_return

    ms_return Derivation (fallback chain):
        The model computes ``EBT = EBIT - interest_payment + ms_return``.
        1. If EBIT, EBT, and Interest Expense are all available:
           ``ms_return = EBT - EBIT + Interest Expense``
        2. If Interest Expense is NaN but EBIT and EBT are available:
           ``interest_payment = 0``, ``ms_return = EBT - EBIT``
           (lumps interest into ms_return, but preserves EBT identity)
        3. Otherwise: NaN

    Balance Sheet:
        Inventory                               -> inventory
        Total Non Current Assets                -> nca
        Accounts Receivable                     -> accounts_receivable
        Accounts Payable                        -> accounts_payable
        Other Current Assets                    -> advance_payments_purchases
        Current Deferred Revenue                -> advance_payments_sales
        Cash And Cash Equivalents               -> cash
        Other Short Term Investments            -> ims
        (Derived from identity)                 -> current_liabilities
        Current Debt                            -> current_lt_debt
        Total Non Current Liabilities ...       -> non_current_liabilities
        Stockholders Equity                     -> equity

    Cash Flow:
        Change In Inventory (negated)           -> change_in_inventory
        Common Stock Dividend Paid (negated)    -> dividends
        Repurchase Of Capital Stock (negated)   -> stock_buyback

Notes:
    - Yahoo Finance typically provides 4-5 years of annual data.
    - Fields not available for a company are filled with 0.0.
    - Dividends and buybacks use 0.0 if the company does not pay/repurchase.
    - The script is not suitable for financial institutions (banks,
      insurance) whose balance sheet structure differs fundamentally.

Usage::

    python run_build_financial_statements.py
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_yahoo_data(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch annual financials from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. ``"TSLA"``).

    Returns:
        Tuple of (income_statement, balance_sheet, cash_flow) DataFrames,
        each with columns sorted chronologically.
    """
    t = yf.Ticker(ticker)
    inc = t.financials
    bs = t.balance_sheet
    cf = t.cashflow

    # Sort columns chronologically
    inc = inc[sorted(inc.columns)]
    bs = bs[sorted(bs.columns)]
    cf = cf[sorted(cf.columns)]

    return inc, bs, cf


def safe_get(df: pd.DataFrame, field: str, years: List[int]) -> List[float]:
    """Extract a field from a Yahoo Finance DataFrame, aligned to years.

    Args:
        df: Yahoo Finance DataFrame with Timestamp columns.
        field: Row label to extract.
        years: List of fiscal years to align to.

    Returns:
        List of float values, one per year.  Missing values are ``nan``.
    """
    if field not in df.index:
        return [float("nan")] * len(years)

    row = df.loc[field]
    result = []
    for yr in years:
        val = float("nan")
        for col in row.index:
            if col.year == yr:
                v = row[col]
                if pd.notna(v):
                    val = float(v)
                break
        result.append(val)
    return result


def build_financial_data(
    ticker: str,
    inc: pd.DataFrame,
    bs: pd.DataFrame,
    cf: pd.DataFrame,
) -> Dict[str, List[float]]:
    """Map Yahoo Finance fields to the model's expected format.

    Args:
        ticker: Stock ticker symbol.
        inc: Income statement DataFrame.
        bs: Balance sheet DataFrame.
        cf: Cash flow DataFrame.

    Returns:
        Dict mapping model field names to lists of float values.
    """
    # Determine fiscal years from income statement columns
    years = sorted(set(c.year for c in inc.columns))

    # --- Income Statement ---
    total_revenue = safe_get(inc, "Total Revenue", years)
    cost_of_revenue_raw = safe_get(inc, "Cost Of Revenue", years)
    depreciation = safe_get(inc, "Reconciled Depreciation", years)
    opex = safe_get(inc, "Operating Expense", years)
    net_income = safe_get(inc, "Net Income", years)
    tax = safe_get(inc, "Tax Provision", years)
    ebit = safe_get(inc, "EBIT", years)
    ebt = safe_get(inc, "Pretax Income", years)
    interest_expense_raw = safe_get(inc, "Interest Expense Non Operating", years)

    # cogs = cost_of_revenue - depreciation
    cogs = [
        cr - d if not (np.isnan(cr) or np.isnan(d)) else float("nan")
        for cr, d in zip(cost_of_revenue_raw, depreciation)
    ]

    # The model computes: EBT = EBIT - interest_payment + ms_return
    # So: ms_return = EBT - EBIT + interest_payment
    #
    # Fallback chain:
    #   1. If all three available: ms_return = EBT - EBIT + interest_expense
    #   2. If interest is NaN but EBT and EBIT available:
    #      set interest_payment = 0, ms_return = EBT - EBIT
    #      (lumps interest into ms_return, but keeps EBT correct)
    #   3. Otherwise: NaN
    interest_expense = []
    ms_return = []
    for i in range(len(years)):
        ie = interest_expense_raw[i]
        eb = ebt[i]
        ei = ebit[i]

        if not np.isnan(ie) and not np.isnan(eb) and not np.isnan(ei):
            # Case 1: all available
            interest_expense.append(ie)
            ms_return.append(eb - ei + ie)
        elif not np.isnan(eb) and not np.isnan(ei):
            # Case 2: interest missing, lump into ms_return
            interest_expense.append(0.0)
            ms_return.append(eb - ei)
        else:
            # Case 3: insufficient data
            interest_expense.append(float("nan"))
            ms_return.append(float("nan"))

    # --- Balance Sheet ---
    inventory = safe_get(bs, "Inventory", years)
    nca = safe_get(bs, "Total Non Current Assets", years)
    # Try Accounts Receivable first, fall back to Receivables
    accounts_receivable = safe_get(bs, "Accounts Receivable", years)
    if all(np.isnan(v) for v in accounts_receivable):
        accounts_receivable = safe_get(bs, "Receivables", years)
    accounts_payable = safe_get(bs, "Accounts Payable", years)
    advance_payments_purchases = safe_get(bs, "Other Current Assets", years)
    advance_payments_sales = safe_get(bs, "Current Deferred Revenue", years)
    cash = safe_get(bs, "Cash And Cash Equivalents", years)
    ims = safe_get(bs, "Other Short Term Investments", years)
    current_lt_debt = safe_get(bs, "Current Debt", years)
    non_current_liabilities = safe_get(
        bs, "Total Non Current Liabilities Net Minority Interest", years
    )
    equity = safe_get(bs, "Stockholders Equity", years)

    # Save raw source value for comparison
    current_liabilities_source = safe_get(bs, "Current Liabilities", years)

    # Derive current_liabilities to enforce the balance sheet identity:
    # Assets = Liabilities + Equity
    # CL = (NCA + AdvPP + AR + Inv + Cash + IMS) - NCL - Equity
    current_liabilities = [
        (
            (n + ap_ + ar + inv + c + im) - ncl_ - eq
            if not any(np.isnan(v) for v in [n, ap_, ar, inv, c, im, ncl_, eq])
            else float("nan")
        )
        for n, ap_, ar, inv, c, im, ncl_, eq in zip(
            nca,
            advance_payments_purchases,
            accounts_receivable,
            inventory,
            cash,
            ims,
            non_current_liabilities,
            equity,
        )
    ]

    # --- Cash Flow (negate: Yahoo uses cash-outflow-negative convention) ---
    change_in_inventory_raw = safe_get(cf, "Change In Inventory", years)
    change_in_inventory = [
        -v if not np.isnan(v) else float("nan") for v in change_in_inventory_raw
    ]

    dividends_raw = safe_get(cf, "Common Stock Dividend Paid", years)
    dividends = [-v if not np.isnan(v) else 0.0 for v in dividends_raw]

    buyback_raw = safe_get(cf, "Repurchase Of Capital Stock", years)
    stock_buyback = [-v if not np.isnan(v) else 0.0 for v in buyback_raw]

    # Replace NaN with 0.0 for fields where absence means zero
    zero_if_nan_fields = {
        "inventory": inventory,
        "advance_payments_purchases": advance_payments_purchases,
        "advance_payments_sales": advance_payments_sales,
        "ims": ims,
        "change_in_inventory": change_in_inventory,
    }
    for key, vals in zero_if_nan_fields.items():
        zero_if_nan_fields[key] = [0.0 if np.isnan(v) else v for v in vals]

    return {
        "years": years,
        "sales": total_revenue,
        "cogs": cogs,
        "depreciation": depreciation,
        "opex": opex,
        "net_income": net_income,
        "tax": tax,
        "interest_payment": interest_expense,
        "ms_return": ms_return,
        "inventory": zero_if_nan_fields["inventory"],
        "change_in_inventory": zero_if_nan_fields["change_in_inventory"],
        "nca": nca,
        "accounts_receivable": accounts_receivable,
        "accounts_payable": accounts_payable,
        "advance_payments_purchases": zero_if_nan_fields["advance_payments_purchases"],
        "advance_payments_sales": zero_if_nan_fields["advance_payments_sales"],
        "cash": cash,
        "ims": zero_if_nan_fields["ims"],
        "current_liabilities": current_liabilities,
        "current_liabilities_source": current_liabilities_source,
        "current_lt_debt": current_lt_debt,
        "non_current_liabilities": non_current_liabilities,
        "equity": equity,
        "dividends": dividends,
        "stock_buyback": stock_buyback,
    }


def format_value(v: float) -> str:
    """Format a float for the generated Python source.

    Uses scientific notation for large values and ``float('nan')`` for NaN.
    """
    if np.isnan(v):
        return "float('nan')"
    if abs(v) >= 1e9:
        return f"{v:.3e}"
    return f"{v:.0f}"


def generate_module_source(
    ticker: str,
    company_name: str,
    data: Dict[str, list],
) -> str:
    """Generate the Python source code for a financial_statements.py module.

    Args:
        ticker: Stock ticker symbol.
        company_name: Full company name for the docstring.
        data: Dict from :func:`build_financial_data`.

    Returns:
        Complete Python source as a string.
    """
    years = data["years"]
    year_range = f"FY{years[0]}-FY{years[-1]}"

    def tensor_block(name: str, values: list, indent: int = 4) -> str:
        pad = " " * indent
        inner_pad = " " * (indent + 4)
        formatted = [f"{inner_pad}{format_value(v)}," for v in values]
        lines = "\n".join(formatted)
        return (
            f"{pad}{name} = tf.constant(\n"
            f"{pad}    [\n"
            f"{lines}\n"
            f"{pad}    ],\n"
            f"{pad}    dtype=tf.float64,\n"
            f"{pad})"
        )

    # Build all tensor blocks
    sections = []

    # Income Statement
    sections.append("    # --- Income Statement ---")
    for name, key in [
        ("sales", "sales"),
        ("cogs", "cogs"),
        ("depreciation", "depreciation"),
        ("opex", "opex"),
        ("net_income", "net_income"),
        ("tax", "tax"),
        ("interest_expense", "interest_payment"),
        ("ms_investment_return", "ms_return"),
    ]:
        sections.append(tensor_block(name, data[key]))

    # Balance Sheet
    sections.append("")
    sections.append("    # --- Balance Sheet ---")
    for name, key in [
        ("inventory", "inventory"),
        ("change_in_inventory", "change_in_inventory"),
        ("nca", "nca"),
        ("accounts_receivable", "accounts_receivable"),
        ("accounts_payable", "accounts_payable"),
        ("advance_payments_purchases", "advance_payments_purchases"),
        ("advance_payments_sales", "advance_payments_sales"),
        ("cash", "cash"),
        ("ims", "ims"),
        ("current_lt_debt", "current_lt_debt"),
        ("non_current_liabilities", "non_current_liabilities"),
        ("equity", "equity"),
        ("current_liabilities_source", "current_liabilities_source"),
    ]:
        sections.append(tensor_block(name, data[key]))

    # Cash Flow
    sections.append("")
    sections.append("    # --- Cash Flow ---")
    for name, key in [
        ("dividends", "dividends"),
        ("stock_buyback", "stock_buyback"),
    ]:
        sections.append(tensor_block(name, data[key]))

    tensor_lines = "\n".join(sections)

    return f'''"""{company_name} historical financial data ({year_range}).

Data sourced from Yahoo Finance via yfinance.
Generated on {datetime.now().strftime("%Y-%m-%d")}.
All monetary values are in USD.
"""

import tensorflow as tf


def get_financial_statements():
    """Return {company_name} historical financial data as a dictionary of TensorFlow tensors.

    Returns:
        dict with the following keys (all tf.float64 tensors):

        Metadata:
            years             - fiscal year labels [{years[0]}..{years[-1]}]

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
            equity                     - stockholders\' equity

        Cash Flow:
            dividends       - common stock dividends paid
            stock_buyback   - repurchase of capital stock

        Derived:
            purchases       - cogs + change_in_inventory
            cost_of_revenue - cogs + depreciation
    """
    years = tf.range({years[0]}, {years[-1] + 1}, dtype=tf.float64)

{tensor_lines}

    # --- Derived ---
    # Enforce balance sheet identity: Assets = Liabilities + Equity
    current_liabilities = (
        nca + advance_payments_purchases + accounts_receivable
        + inventory + cash + ims
        - non_current_liabilities - equity
    )
    purchases = cogs + change_in_inventory
    cost_of_revenue = cogs + depreciation

    return {{
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
    }}
'''


def write_module(ticker: str, source: str) -> str:
    """Write the generated source to the data package directory.

    Args:
        ticker: Stock ticker (used as directory name, lowercased).
        source: Python source code to write.

    Returns:
        Path to the written file.
    """
    dir_path = os.path.join("financial_forecast", "data", ticker.lower())
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "financial_statements.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(source)
    return file_path


if __name__ == "__main__":

    ticker = "aapl"

    # -- Step 1: Fetch data from Yahoo Finance --
    print(f"Fetching Yahoo Finance data for {ticker}...")
    inc, bs, cf = fetch_yahoo_data(ticker)

    years = sorted(c.year for c in inc.columns)
    print(f"Available fiscal years: {years}")

    # -- Step 2: Map fields to model format --
    company_name = yf.Ticker(ticker).info.get("longName", ticker)
    data = build_financial_data(ticker, inc, bs, cf)

    # -- Step 3: Report ms_return derivation and field coverage --
    ebit_vals = safe_get(inc, "EBIT", years)
    ebt_vals = safe_get(inc, "Pretax Income", years)
    ie_vals = safe_get(inc, "Interest Expense Non Operating", years)

    print(f"\nms_return derivation for {ticker}:")
    for i, yr in enumerate(years):
        ie = ie_vals[i]
        eb = ebt_vals[i]
        ei = ebit_vals[i]
        if not np.isnan(ie) and not np.isnan(eb) and not np.isnan(ei):
            print(f"  FY{yr}: EBT - EBIT + IntExp = {data['ms_return'][i]/1e6:>8,.0f}M")
        elif not np.isnan(eb) and not np.isnan(ei):
            print(
                f"  FY{yr}: EBT - EBIT = {data['ms_return'][i]/1e6:>8,.0f}M"
                f"  (interest_payment set to 0 — fallback)"
            )
        else:
            print(f"  FY{yr}: insufficient data (NaN)")

    print(f"\nField coverage for {ticker}:")
    for key, vals in data.items():
        if key == "years":
            continue
        nan_count = sum(1 for v in vals if isinstance(v, float) and np.isnan(v))
        if nan_count > 0:
            print(f"  WARNING: {key} has {nan_count}/{len(vals)} NaN values")

    # -- Step 4: Generate and write the module --
    source = generate_module_source(ticker, company_name, data)
    tmpticker = ticker + "_test"
    path = write_module(ticker, source)
    print(f"\nGenerated: {path}")
    print(f"Company:   {company_name}")
    print(f"Years:     FY{years[0]}-FY{years[-1]}")
