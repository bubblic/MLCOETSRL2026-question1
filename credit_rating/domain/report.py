"""Annual report data container.

:class:`AnnualReport` bundles a company's financial statements with
the raw textual sections (MD&A, auditor report, risk factors) needed
for the text tower and shenanigans detectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from credit_rating.domain.financial_statements import FinancialStatements


@dataclass(frozen=True)
class AnnualReport:
    """All data extracted from a single annual report filing.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        fiscal_year: Fiscal year the report covers.
        financial_statements: Structured financial data.
        mda_text: Raw Management Discussion & Analysis text.
        auditor_report_text: Raw independent auditor report text.
        risk_factors_text: Raw risk factors section text.
        source_url: URL or file path the report was loaded from.
        filing_date: Date the filing was submitted.
    """

    ticker: str
    fiscal_year: int
    financial_statements: FinancialStatements
    mda_text: str = ""
    auditor_report_text: str = ""
    risk_factors_text: str = ""
    source_url: str = ""
    filing_date: Optional[date] = None
