"""Build a financial_statements.py module from SEC EDGAR XBRL company facts.

Uses the SEC EDGAR company facts API to extract annual financial data from
10-K filings.  This typically provides 10-15+ years of data, far more than
Yahoo Finance's ~4-5 years.

The SEC mandated XBRL in 10-K filings from 2009 for large accelerated filers.
Multiple XBRL tags are tried for each model field to handle GAAP taxonomy
changes over the years (e.g. ASC 606 revenue recognition adopted ~2018).

Field Mapping (SEC XBRL -> Model):
    Income Statement:
        Revenues / RevenueFromContract...     -> sales
        CostOfRevenue - DepreciationD&A       -> cogs
        DepreciationDepletionAndAmort.         -> depreciation
        GrossProfit - OperatingIncomeLoss      -> opex
        NetIncomeLoss                          -> net_income
        IncomeTaxExpenseBenefit                -> tax
        InterestExpense                        -> interest_payment
        EBT - EBIT + InterestExpense           -> ms_return

    Balance Sheet:
        InventoryNet                           -> inventory
        AssetsNoncurrent                       -> nca
        AccountsReceivableNetCurrent           -> accounts_receivable
        AccountsPayableCurrent                 -> accounts_payable
        OtherAssetsCurrent                     -> advance_payments_purchases
        ContractWithCustomerLiabilityCurrent   -> advance_payments_sales
        CashAndCashEquivalentsAtCarrying...    -> cash
        MarketableSecuritiesCurrent            -> ims
        (Derived from identity)                 -> current_liabilities
        LongTermDebtCurrent                    -> current_lt_debt
        LiabilitiesNoncurrent                  -> non_current_liabilities
        StockholdersEquity                     -> equity

    Cash Flow / Derived:
        Inventory[t] - Inventory[t-1]          -> change_in_inventory
        PaymentsOfDividends                    -> dividends
        PaymentsForRepurchaseOfCommonStock      -> stock_buyback

Usage::

    python run_build_financial_statements_sec.py
"""

import gzip
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np


# ===== Configuration ==========================================================
# Change these for each company.
TICKER = "AAPL"

# Optional year range filter.  Set to None to use all available years.
# The script automatically fetches one extra prior year for inventory
# change_in_inventory derivation.
START_YEAR = 2018  # e.g. 2018
END_YEAR = 2025  # e.g. 2025

# REQUIRED BY SEC: The SEC blocks requests without a valid User-Agent header.
# Update with your actual details before running.
ORG_NAME = "Personal Research"
EMAIL = "your.email@example.com"

# ===== XBRL Tag Mapping =======================================================
# Each list is tried in priority order. Multiple alternatives handle GAAP
# taxonomy version changes (e.g. pre/post ASC 606 revenue tags).

# Income Statement (duration facts)
REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]
COST_OF_REVENUE_TAGS = [
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
]
DEPRECIATION_TAGS = [
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "Depreciation",
]
GROSS_PROFIT_TAGS = ["GrossProfit"]
OPERATING_INCOME_TAGS = ["OperatingIncomeLoss"]
NET_INCOME_TAGS = ["NetIncomeLoss"]
TAX_TAGS = ["IncomeTaxExpenseBenefit"]
EBT_TAGS = [
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
]
INTEREST_EXPENSE_TAGS = [
    "InterestExpense",
    "InterestExpenseDebt",
    "InterestExpenseNonoperating",
]
RD_EXPENSE_TAGS = ["ResearchAndDevelopmentExpense"]
SGA_EXPENSE_TAGS = ["SellingGeneralAndAdministrativeExpense"]
# Google/Alphabet reports S&M and G&A separately (no combined SGA tag)
SM_EXPENSE_TAGS = ["SellingAndMarketingExpense"]
GA_EXPENSE_TAGS = ["GeneralAndAdministrativeExpense"]

# Balance Sheet (instant facts)
INVENTORY_TAGS = ["InventoryNet"]
TOTAL_ASSETS_TAGS = ["Assets"]
CURRENT_ASSETS_TAGS = ["AssetsCurrent"]
NCA_TAGS = ["AssetsNoncurrent"]
# AR: ReceivablesNetCurrent captures total receivables (trade + nontrade).
# When absent, we sum AccountsReceivableNetCurrent + NontradeReceivablesCurrent
# (Apple's vendor non-trade receivables are nearly as large as trade AR).
AR_TAGS = ["ReceivablesNetCurrent", "AccountsReceivableNetCurrent"]
NONTRADE_AR_TAGS = ["NontradeReceivablesCurrent"]
AP_TAGS = ["AccountsPayableCurrent"]
OTHER_CURRENT_ASSETS_TAGS = ["OtherAssetsCurrent"]
DEFERRED_REV_CURRENT_TAGS = [
    "ContractWithCustomerLiabilityCurrent",
    "DeferredRevenueCurrent",
]
TOTAL_LIABILITIES_TAGS = ["Liabilities"]
CASH_TAGS = ["CashAndCashEquivalentsAtCarryingValue"]
SHORT_TERM_INVEST_TAGS = [
    "MarketableSecuritiesCurrent",
    "ShortTermInvestments",
    "AvailableForSaleSecuritiesCurrent",
    "OtherShortTermInvestments",
]
CURRENT_LIABILITIES_TAGS = ["LiabilitiesCurrent"]
CURRENT_DEBT_TAGS = [
    "LongTermDebtCurrent",
    "DebtCurrent",
    "LongTermDebtAndCapitalLeaseObligationsCurrent",
]
NON_CURRENT_LIABILITIES_TAGS = ["LiabilitiesNoncurrent"]
EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]

# Cash Flow (duration facts)
DIVIDENDS_TAGS = [
    "PaymentsOfDividends",
    "PaymentsOfDividendsCommonStock",
    "PaymentsOfOrdinaryDividends",
]
BUYBACK_TAGS = [
    "PaymentsForRepurchaseOfCommonStock",
    "PaymentsForRepurchaseOfEquity",
]


# ===== SEC EDGAR API ===========================================================


def _sec_request(url: str) -> bytes:
    """Make an HTTP request to SEC EDGAR with the required User-Agent header.

    Handles HTTP 429 (rate limit) with a single retry after 2 seconds.
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"{ORG_NAME} {EMAIL}",
            "Accept-Encoding": "gzip, deflate",
        },
    )

    def _read_response(resp):
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        return raw

    try:
        with urllib.request.urlopen(req) as resp:
            return _read_response(resp)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("  Rate limited by SEC EDGAR. Retrying in 2s...")
            time.sleep(2)
            with urllib.request.urlopen(req) as resp:
                return _read_response(resp)
        raise


def resolve_cik(ticker: str) -> Tuple[int, str]:
    """Look up CIK and company name for a stock ticker via SEC EDGAR.

    Args:
        ticker: Stock ticker symbol (e.g. ``"GOOGL"``).

    Returns:
        ``(cik, company_name)`` tuple.

    Raises:
        ValueError: If ticker is not found.
    """
    data = json.loads(_sec_request("https://www.sec.gov/files/company_tickers.json"))
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            return int(entry["cik_str"]), entry["title"]
    raise ValueError(
        f"Ticker {ticker!r} not found in SEC EDGAR. "
        f"Try the CIK number directly (e.g. 1652044 for Alphabet)."
    )


def fetch_company_facts(cik: int) -> dict:
    """Fetch all XBRL facts for a company from the SEC company facts API.

    Args:
        cik: SEC Central Index Key number.

    Returns:
        Parsed JSON response containing all XBRL facts across all filings.
    """
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    print(f"  URL: {url}")
    return json.loads(_sec_request(url))


# ===== Data Extraction =========================================================


def _parse_frame_year(frame: str) -> int:
    """Extract calendar year from SEC XBRL frame reference.

    Frame formats: ``"CY2024"`` (annual), ``"CY2024Q4I"`` (instant).
    Returns the year as an int, or ``None`` if the frame is not parseable.
    """
    if not frame or not frame.startswith("CY") or len(frame) < 6:
        return None
    digits = frame[2:6]
    return int(digits) if digits.isdigit() else None


def determine_fiscal_years(us_gaap: dict) -> List[int]:
    """Determine available fiscal years by scanning all revenue tag variants.

    Collects years from both the ``fy`` field (filing fiscal year) and the
    ``frame`` field (standardised calendar year) to capture the full range.
    """
    years = set()
    for tag in REVENUE_TAGS:
        if tag not in us_gaap:
            continue
        for fact in us_gaap[tag].get("units", {}).get("USD", []):
            if fact.get("form") in ("10-K", "10-K/A") and fact.get("fp") == "FY":
                fy = fact.get("fy")
                if fy is not None:
                    years.add(fy)
                fy_frame = _parse_frame_year(fact.get("frame", ""))
                if fy_frame is not None:
                    years.add(fy_frame)
    return sorted(years)


def get_annual_values(
    us_gaap: dict,
    tags: List[str],
    fiscal_years: List[int],
    unit: str = "USD",
) -> Dict[int, float]:
    """Extract annual values from 10-K filings for given XBRL tags.

    Tries each tag in priority order, merging results across tags to
    maximise year coverage (earlier tags take priority for overlapping
    years).

    Year identification uses a two-phase approach:

    1. **Frame-based** (preferred): facts with ``frame=CY{year}`` are
       keyed by the frame year, which reliably identifies the data year
       regardless of which filing it appears in.
    2. **fy-based** (fallback): facts without a usable frame are keyed
       by the ``fy`` field.  Comparative facts (where the frame year
       differs from ``fy``) are excluded to avoid mis-keying.

    Within each year, the most recently filed value is preferred
    (handles amendments).  Among same-date facts, the largest absolute
    value is chosen (handles consolidated-vs-segment duplicates).

    Args:
        us_gaap: The ``us-gaap`` section of the company facts JSON.
        tags: XBRL concept names to try, highest priority first.
        fiscal_years: Fiscal years to extract values for.
        unit: Measurement unit (``"USD"``, ``"shares"``, or ``"pure"``).

    Returns:
        Dict mapping fiscal year to float value.
    """
    result: Dict[int, float] = {}
    fy_set = set(fiscal_years)

    for tag in tags:
        if tag not in us_gaap:
            continue

        facts = us_gaap[tag].get("units", {}).get(unit, [])
        if not facts:
            continue

        # Filter for 10-K annual data
        annual = [
            f
            for f in facts
            if f.get("form") in ("10-K", "10-K/A") and f.get("fp") == "FY"
        ]

        # --- Phase 1: frame-based year identification (most reliable) ---
        by_frame: Dict[int, dict] = {}
        for f in annual:
            frame_year = _parse_frame_year(f.get("frame", ""))
            if frame_year is None or frame_year not in fy_set:
                continue
            filed = f.get("filed", "")
            existing = by_frame.get(frame_year)
            if existing is None:
                by_frame[frame_year] = f
            elif filed > existing["filed"]:
                by_frame[frame_year] = f
            elif filed == existing["filed"]:
                if abs(float(f["val"])) > abs(float(existing["val"])):
                    by_frame[frame_year] = f

        # --- Phase 2: fy-based fallback (for years without framed facts) ---
        # For same-filing duplicates, prefer the latest ``end`` date: in a
        # 10-K the current-year balance sheet date is always later than
        # comparatives (e.g. Sep-2024 > Sep-2023).  This is critical for
        # non-December fiscal-year companies like Apple where the current
        # year's fact often lacks a standardised frame.
        by_fy: Dict[int, dict] = {}
        for f in annual:
            fy = f.get("fy")
            if fy is None or fy not in fy_set or fy in by_frame:
                continue
            # Skip comparative facts: frame year differs from fy
            frame_year = _parse_frame_year(f.get("frame", ""))
            if frame_year is not None and frame_year != fy:
                continue
            filed = f.get("filed", "")
            end = f.get("end", "")
            existing = by_fy.get(fy)
            if existing is None:
                by_fy[fy] = f
            elif filed > existing["filed"]:
                by_fy[fy] = f
            elif filed == existing["filed"]:
                if end >= existing.get("end", ""):
                    by_fy[fy] = f

        # Merge: frame-based takes priority over fy-based
        by_year = {yr: f for yr, f in by_frame.items()}
        for yr, f in by_fy.items():
            if yr not in by_year:
                by_year[yr] = f

        # Fill gaps in result (earlier tags have higher priority)
        for yr, fact in by_year.items():
            if yr not in result:
                result[yr] = float(fact["val"])

    return result


def build_financial_data(
    company_facts: dict,
    start_year: int = None,
    end_year: int = None,
) -> Dict[str, list]:
    """Map SEC EDGAR XBRL facts to the model's expected financial data format.

    Produces the same dict structure as ``run_build_financial_statements.py``
    (Yahoo Finance version), compatible with :class:`HistoricalDataLoader`.

    Key derivations:
        cogs              = cost_of_revenue - depreciation
        opex              = gross_profit - operating_income   (fb: R&D + SGA)
        ms_return         = EBT - EBIT + interest_expense     (fb chain)
        change_in_inventory = inventory[t] - inventory[t-1]

    Args:
        company_facts: Full JSON response from the company facts API.
        start_year: First fiscal year to include (default: earliest available).
        end_year: Last fiscal year to include (default: latest available).

    Returns:
        Dict mapping field names to lists of float values, one per year.
    """
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    raw_years = determine_fiscal_years(us_gaap)

    if not raw_years:
        raise ValueError("No fiscal years found in SEC company facts")

    # Apply year range filter
    lo = start_year if start_year is not None else raw_years[0]
    hi = end_year if end_year is not None else raw_years[-1]
    raw_years = [y for y in raw_years if lo <= y <= hi]

    if not raw_years:
        raise ValueError(f"No fiscal years in range {lo}-{hi}")

    # Ensure contiguous year range (tf.range requires it)
    years = list(range(raw_years[0], raw_years[-1] + 1))
    if years != raw_years:
        gaps = sorted(set(years) - set(raw_years))
        print(f"  Note: filling year gaps {gaps} with NaN")

    print(f"  Fiscal years: {years[0]}-{years[-1]} ({len(years)} years)")

    # Include one prior year for change_in_inventory computation
    extended_years = [years[0] - 1] + years

    def get(tags, yrs=None):
        """Get annual values aligned to fiscal years (NaN for missing)."""
        target = yrs if yrs is not None else years
        vals = get_annual_values(us_gaap, tags, target)
        return [vals.get(yr, float("nan")) for yr in target]

    # ── Income Statement ──────────────────────────────────────────
    revenue = get(REVENUE_TAGS)
    cost_of_revenue_raw = get(COST_OF_REVENUE_TAGS)
    depreciation_raw = get(DEPRECIATION_TAGS)
    net_income = get(NET_INCOME_TAGS)
    tax = get(TAX_TAGS)
    ebit = get(OPERATING_INCOME_TAGS)
    ebt = get(EBT_TAGS)
    interest_raw = get(INTEREST_EXPENSE_TAGS)
    gross_profit_tag = get(GROSS_PROFIT_TAGS)
    rd = get(RD_EXPENSE_TAGS)
    sga = get(SGA_EXPENSE_TAGS)
    sm = get(SM_EXPENSE_TAGS)
    ga = get(GA_EXPENSE_TAGS)

    # cogs and depreciation: if depreciation is available, split CoR;
    # otherwise treat all CoR as cogs with depreciation=0
    cogs = []
    depreciation = []
    for i in range(len(years)):
        d = depreciation_raw[i]
        cr = cost_of_revenue_raw[i]
        if not np.isnan(d) and not np.isnan(cr):
            cogs.append(cr - d)
            depreciation.append(d)
        elif not np.isnan(cr):
            # No depreciation data: treat all CoR as cogs
            cogs.append(cr)
            depreciation.append(0.0)
        else:
            cogs.append(float("nan"))
            depreciation.append(float("nan"))

    # Derive gross profit: use tag if available, else Revenue - CoR
    gross_profit = []
    for i in range(len(years)):
        if not np.isnan(gross_profit_tag[i]):
            gross_profit.append(gross_profit_tag[i])
        elif not np.isnan(revenue[i]) and not np.isnan(cost_of_revenue_raw[i]):
            gross_profit.append(revenue[i] - cost_of_revenue_raw[i])
        else:
            gross_profit.append(float("nan"))

    # opex: prefer gross_profit - EBIT; fallback to sum of R&D + SGA
    # (Google uses S&M + G&A instead of combined SGA)
    opex = []
    for i in range(len(years)):
        gp, oi = gross_profit[i], ebit[i]
        if not np.isnan(gp) and not np.isnan(oi):
            opex.append(gp - oi)
        else:
            # Sum available operating expense components
            components = [rd[i], sga[i], sm[i], ga[i]]
            present = [v for v in components if not np.isnan(v)]
            if present:
                opex.append(sum(present))
            else:
                opex.append(float("nan"))

    # interest_payment & ms_return (same derivation as Yahoo Finance script):
    #   1. All three available: ms_return = EBT - EBIT + Interest
    #   2. Interest missing:    interest=0, ms_return = EBT - EBIT
    #   3. Otherwise:           NaN
    interest_payment = []
    ms_return = []
    for i in range(len(years)):
        ie, eb, ei = interest_raw[i], ebt[i], ebit[i]
        if not np.isnan(ie) and not np.isnan(eb) and not np.isnan(ei):
            interest_payment.append(ie)
            ms_return.append(eb - ei + ie)
        elif not np.isnan(eb) and not np.isnan(ei):
            interest_payment.append(0.0)
            ms_return.append(eb - ei)
        else:
            interest_payment.append(float("nan"))
            ms_return.append(float("nan"))

    # ── Balance Sheet ─────────────────────────────────────────────
    # Fetch inventory for extended years (prior year needed for delta)
    inventory_ext = get(INVENTORY_TAGS, extended_years)
    inventory = inventory_ext[1:]  # Drop the extra prior-year entry

    # Derive change_in_inventory from consecutive balance-sheet values
    change_in_inventory = []
    for i in range(len(years)):
        curr = inventory_ext[i + 1]
        prev = inventory_ext[i]
        if not np.isnan(curr) and not np.isnan(prev):
            change_in_inventory.append(curr - prev)
        else:
            change_in_inventory.append(0.0)

    # nca with fallback: total_assets - current_assets
    nca_direct = get(NCA_TAGS)
    total_assets = get(TOTAL_ASSETS_TAGS)
    current_assets = get(CURRENT_ASSETS_TAGS)
    nca = []
    for i in range(len(years)):
        if not np.isnan(nca_direct[i]):
            nca.append(nca_direct[i])
        elif not np.isnan(total_assets[i]) and not np.isnan(current_assets[i]):
            nca.append(total_assets[i] - current_assets[i])
        else:
            nca.append(float("nan"))

    # AR: ReceivablesNetCurrent already includes nontrade receivables.
    # When only trade AR is available, add NontradeReceivablesCurrent
    # (important for Apple where vendor nontrade AR ~ trade AR in size).
    ar_total = get(["ReceivablesNetCurrent"])
    ar_trade = get(["AccountsReceivableNetCurrent"])
    nontrade_ar = get(NONTRADE_AR_TAGS)
    accounts_receivable = []
    for i in range(len(years)):
        if not np.isnan(ar_total[i]):
            accounts_receivable.append(ar_total[i])
        elif not np.isnan(ar_trade[i]):
            n = nontrade_ar[i] if not np.isnan(nontrade_ar[i]) else 0.0
            accounts_receivable.append(ar_trade[i] + n)
        else:
            accounts_receivable.append(float("nan"))
    accounts_payable = get(AP_TAGS)
    advance_payments_purchases = get(OTHER_CURRENT_ASSETS_TAGS)
    advance_payments_sales = get(DEFERRED_REV_CURRENT_TAGS)
    cash = get(CASH_TAGS)
    ims = get(SHORT_TERM_INVEST_TAGS)
    current_liabilities_raw = get(CURRENT_LIABILITIES_TAGS)
    current_lt_debt = get(CURRENT_DEBT_TAGS)
    equity = get(EQUITY_TAGS)

    # non_current_liabilities with fallback: total_liabilities - current_liabilities
    ncl_direct = get(NON_CURRENT_LIABILITIES_TAGS)
    total_liabilities = get(TOTAL_LIABILITIES_TAGS)
    non_current_liabilities = []
    for i in range(len(years)):
        if not np.isnan(ncl_direct[i]):
            non_current_liabilities.append(ncl_direct[i])
        elif not np.isnan(total_liabilities[i]) and not np.isnan(
            current_liabilities_raw[i]
        ):
            non_current_liabilities.append(
                total_liabilities[i] - current_liabilities_raw[i]
            )
        else:
            non_current_liabilities.append(float("nan"))

    # Derive current_liabilities to enforce the balance sheet identity:
    # Assets = Liabilities + Equity
    # CL = (NCA + AdvPP + AR + Inv + Cash + IMS) - NCL - Equity
    current_liabilities = []
    for i in range(len(years)):
        vals = [
            nca[i],
            advance_payments_purchases[i],
            accounts_receivable[i],
            inventory[i],
            cash[i],
            ims[i],
            non_current_liabilities[i],
            equity[i],
        ]
        if any(np.isnan(v) for v in vals):
            current_liabilities.append(float("nan"))
        else:
            assets = (
                nca[i]
                + advance_payments_purchases[i]
                + accounts_receivable[i]
                + inventory[i]
                + cash[i]
                + ims[i]
            )
            current_liabilities.append(assets - non_current_liabilities[i] - equity[i])

    # ── Cash Flow ─────────────────────────────────────────────────
    dividends_raw = get(DIVIDENDS_TAGS)
    dividends = [abs(v) if not np.isnan(v) else 0.0 for v in dividends_raw]

    buyback_raw = get(BUYBACK_TAGS)
    stock_buyback = [abs(v) if not np.isnan(v) else 0.0 for v in buyback_raw]

    # NaN -> 0.0 for fields where absence means zero
    def zero_nan(vals):
        return [0.0 if (isinstance(v, float) and np.isnan(v)) else v for v in vals]

    return {
        "years": years,
        "sales": revenue,
        "cogs": cogs,
        "depreciation": depreciation,
        "opex": opex,
        "net_income": net_income,
        "tax": tax,
        "interest_payment": interest_payment,
        "ms_return": ms_return,
        "inventory": zero_nan(inventory),
        "change_in_inventory": change_in_inventory,
        "nca": nca,
        "accounts_receivable": accounts_receivable,
        "accounts_payable": accounts_payable,
        "advance_payments_purchases": zero_nan(advance_payments_purchases),
        "advance_payments_sales": zero_nan(advance_payments_sales),
        "cash": cash,
        "ims": zero_nan(ims),
        "current_liabilities": current_liabilities,
        "current_lt_debt": current_lt_debt,
        "non_current_liabilities": non_current_liabilities,
        "equity": equity,
        "dividends": dividends,
        "stock_buyback": stock_buyback,
    }


# ===== Code Generation =========================================================


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
        ("current_liabilities", "current_liabilities"),
        ("current_lt_debt", "current_lt_debt"),
        ("non_current_liabilities", "non_current_liabilities"),
        ("equity", "equity"),
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

Data sourced from SEC EDGAR XBRL company facts API.
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


# ===== Main ====================================================================

if __name__ == "__main__":
    ticker = TICKER

    # -- Step 1: Resolve CIK --
    print(f"Resolving CIK for {ticker}...")
    cik, company_name = resolve_cik(ticker)
    print(f"  {company_name} (CIK: {cik})")

    # -- Step 2: Fetch company facts --
    print(f"\nFetching SEC EDGAR company facts...")
    company_facts = fetch_company_facts(cik)

    # -- Step 3: Build financial data --
    print(f"\nMapping XBRL facts to model fields...")
    data = build_financial_data(company_facts, start_year=START_YEAR, end_year=END_YEAR)
    years = data["years"]

    # -- Step 4: Diagnostics --
    print(f"\nms_return derivation for {ticker}:")
    for i, yr in enumerate(years):
        ms = data["ms_return"][i]
        ip = data["interest_payment"][i]
        if np.isnan(ms):
            print(f"  FY{yr}: insufficient data (NaN)")
        elif ip == 0.0:
            print(
                f"  FY{yr}: EBT - EBIT = {ms / 1e6:>10,.0f}M"
                f"  (interest_payment=0, fallback)"
            )
        else:
            print(
                f"  FY{yr}: EBT - EBIT + IntExp = {ms / 1e6:>10,.0f}M"
                f"  (interest={ip / 1e6:,.0f}M)"
            )

    print(f"\nField coverage for {ticker}:")
    all_good = True
    for key, vals in data.items():
        if key == "years":
            continue
        nan_count = sum(1 for v in vals if isinstance(v, float) and np.isnan(v))
        if nan_count > 0:
            print(f"  WARNING: {key:30s} has {nan_count}/{len(vals)} NaN values")
            all_good = False
    if all_good:
        print("  All fields have complete data (no NaN values)")

    # -- Step 5: Generate and write --
    print(f"\nGenerating module...")
    source = generate_module_source(ticker, company_name, data)
    path = write_module(ticker, source)

    print(f"\nDone!")
    print(f"  Generated: {path}")
    print(f"  Company:   {company_name}")
    print(f"  Years:     FY{years[0]}-FY{years[-1]} ({len(years)} years)")
    print(f"\nUsage:")
    print(f'  data = HistoricalDataLoader("{ticker.lower()}")')
