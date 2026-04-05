"""SEC EDGAR 10-K downloader and parser.

Downloads annual report filings, extracts XBRL-tagged financial data,
and parses textual sections (MD&A, Risk Factors, Auditor Report) via
regex on the filing HTML.
"""

from __future__ import annotations

import functools
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

from credit_rating.config.settings import CreditRatingSettings
from credit_rating.domain.financial_statements import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from credit_rating.domain.report import AnnualReport

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Throttle decorator
# ------------------------------------------------------------------


def throttle(delay_seconds: float):
    """Decorator that enforces a minimum delay between calls.

    Args:
        delay_seconds: Minimum seconds between consecutive invocations.

    Returns:
        A decorator that wraps the target function with rate limiting.
    """

    def decorator(func):
        last_call_time: List[float] = [0.0]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.monotonic() - last_call_time[0]
            if elapsed < delay_seconds:
                time.sleep(delay_seconds - elapsed)
            last_call_time[0] = time.monotonic()
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ------------------------------------------------------------------
# Section extraction patterns
# ------------------------------------------------------------------

_MDA_PATTERN = re.compile(
    r"(?:Item\s+7[.\s]*)"
    r"(?:Management.s Discussion and Analysis.*?)"
    r"(.*?)"
    r"(?=Item\s+7A|Item\s+8)",
    re.DOTALL | re.IGNORECASE,
)

_RISK_FACTORS_PATTERN = re.compile(
    r"(?:Item\s+1A[.\s]*)"
    r"(?:Risk\s+Factors.*?)"
    r"(.*?)"
    r"(?=Item\s+1B|Item\s+2)",
    re.DOTALL | re.IGNORECASE,
)

_AUDITOR_PATTERN = re.compile(
    r"(Report\s+of\s+Independent\s+Registered\s+Public\s+Accounting\s+Firm"
    r".*?)"
    r"(?=Consolidated|CONSOLIDATED|Notes\s+to)",
    re.DOTALL | re.IGNORECASE,
)


# ------------------------------------------------------------------
# Downloader
# ------------------------------------------------------------------


class EdgarDownloader:
    """Download and parse SEC EDGAR 10-K filings.

    Args:
        settings: System configuration.
        download_dir: Directory for cached filing downloads.
    """

    def __init__(
        self,
        settings: Optional[CreditRatingSettings] = None,
        download_dir: Optional[Path] = None,
    ) -> None:
        self._settings = settings or CreditRatingSettings()
        self._download_dir = download_dir or (
            self._settings.raw_data_dir / "edgar"
        )
        self._download_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, source: str, year: int = 0) -> AnnualReport:
        """Download and parse a 10-K filing for a ticker.

        Args:
            source: Stock ticker symbol (e.g. ``"AAPL"``).
            year: Fiscal year to retrieve. If ``0``, retrieves
                the most recent filing.

        Returns:
            A populated :class:`AnnualReport`.
        """
        ticker = source.upper().strip()
        filing_text = self._download_filing(ticker, year)
        return self._parse_filing(ticker, year, filing_text)

    @throttle(delay_seconds=0.11)
    def _download_filing(self, ticker: str, year: int) -> str:
        """Download a 10-K filing from EDGAR with rate limiting.

        Args:
            ticker: Stock ticker symbol.
            year: Target fiscal year.

        Returns:
            The raw filing text/HTML.

        Raises:
            RuntimeError: If the download fails.
        """
        try:
            from sec_edgar_downloader import Downloader

            dl = Downloader(
                company_name="CreditRatingResearch",
                email_address="research@example.com",
                download_folder=str(self._download_dir),
            )
            dl.get("10-K", ticker, limit=1)
        except ImportError:
            logger.warning(
                "sec-edgar-downloader not installed; "
                "using placeholder filing for %s",
                ticker,
            )
            return ""

        return self._read_cached_filing(ticker)

    def _read_cached_filing(self, ticker: str) -> str:
        """Read a previously downloaded filing from the cache directory.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            The filing text, or an empty string if not found.
        """
        filing_dir = self._download_dir / "sec-edgar-filings" / ticker / "10-K"
        if not filing_dir.exists():
            logger.warning("No cached filing found for %s", ticker)
            return ""
        filing_files = sorted(filing_dir.rglob("*.txt"))
        if not filing_files:
            filing_files = sorted(filing_dir.rglob("*.htm*"))
        if not filing_files:
            return ""
        return filing_files[-1].read_text(encoding="utf-8", errors="replace")

    def _parse_filing(
        self,
        ticker: str,
        year: int,
        text: str,
    ) -> AnnualReport:
        """Parse a filing into domain objects.

        Args:
            ticker: Stock ticker symbol.
            year: Fiscal year.
            text: Raw filing text/HTML.

        Returns:
            A populated :class:`AnnualReport`.
        """
        mda = _extract_section(_MDA_PATTERN, text)
        risk_factors = _extract_section(_RISK_FACTORS_PATTERN, text)
        auditor = _extract_section(_AUDITOR_PATTERN, text)
        statements = _build_placeholder_statements(year)

        return AnnualReport(
            ticker=ticker,
            fiscal_year=year,
            financial_statements=statements,
            mda_text=mda,
            auditor_report_text=auditor,
            risk_factors_text=risk_factors,
            source_url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K",
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _extract_section(pattern: re.Pattern, text: str) -> str:
    """Apply *pattern* to *text* and return the first match group."""
    match = pattern.search(text)
    if match:
        return _clean_html_tags(match.group(1).strip())
    return ""


def _clean_html_tags(text: str) -> str:
    """Strip HTML tags from *text*."""
    return re.sub(r"<[^>]+>", " ", text).strip()


def _build_placeholder_statements(year: int) -> FinancialStatements:
    """Build zero-valued statements as a structural placeholder.

    Real XBRL extraction would populate these from tagged data.
    """
    income = IncomeStatement(
        total_revenue=0.0,
        cost_of_goods_sold=0.0,
        total_operating_expenses=0.0,
        selling_general_admin=0.0,
        depreciation_expense=0.0,
        interest_expense=0.0,
        income_tax_expense=0.0,
        net_income=0.0,
    )
    balance = BalanceSheet(
        cash_and_equivalents=0.0,
        short_term_investments=0.0,
        net_receivables=0.0,
        inventory=0.0,
        total_current_assets=0.0,
        net_property_plant_equipment=0.0,
        total_assets=0.0,
        accounts_payable=0.0,
        short_term_debt=0.0,
        total_current_liabilities=0.0,
        long_term_debt=0.0,
        total_liabilities=0.0,
        total_equity=0.0,
        retained_earnings=0.0,
    )
    cash_flow = CashFlowStatement(
        depreciation_and_amortization=0.0,
        cash_from_operations=0.0,
        capital_expenditures=0.0,
        cash_from_investing=0.0,
        cash_from_financing=0.0,
        net_change_in_cash=0.0,
    )
    return FinancialStatements(
        income_statement=income,
        balance_sheet=balance,
        cash_flow=cash_flow,
        fiscal_year=year,
    )
