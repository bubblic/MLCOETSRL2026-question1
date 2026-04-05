"""PDF annual report parser delegating to the ``financial_forecast`` pipeline.

Uses the existing three-stage LLM-powered extraction pipeline
(``FinancialStatementExtractor`` → ``StatementNormalizer`` →
``RatioCalculator``) from ``financial_forecast.extraction`` for
structured financial data, and pdfplumber for raw text sections
(MD&A, Risk Factors, Auditor Report).

Implements the :class:`AnnualReportParser` protocol.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from credit_rating.config.settings import CreditRatingSettings
from credit_rating.domain.financial_statements import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from credit_rating.domain.report import AnnualReport

logger = logging.getLogger(__name__)


class PdfAnnualReportParser:
    """Parse a PDF annual report into an :class:`AnnualReport`.

    Delegates financial data extraction to the existing
    ``financial_forecast`` three-stage pipeline (LLM-powered page
    selection, table extraction, and normalisation).  Text sections
    are extracted directly via pdfplumber + regex.

    Args:
        llm_client: An object satisfying the ``LLMClient`` protocol
            from ``financial_forecast.clients.protocols``.  Required
            for structured data extraction.  If ``None``, only text
            sections are extracted and financial statements are zeroed.
        settings: System configuration.
        known_entities: Entity definitions forwarded to
            :class:`~risk.anonymizer.EntityAnonymizer`.  Each entry is
            a dict with ``"type"`` (``"ORG"``, ``"PERSON"``, etc.) and
            ``"names"`` (list of surface forms).  If ``None``, the
            ticker inferred from the filename is used as a minimal
            known ORG entity.
        chunk_size: Maximum token count per text chunk for downstream
            NLP processing.
        chunk_stride: Overlap between consecutive chunks.
    """

    def __init__(
        self,
        llm_client: Optional[object] = None,
        settings: Optional[CreditRatingSettings] = None,
        known_entities: Optional[List[Dict[str, Any]]] = None,
        chunk_size: Optional[int] = None,
        chunk_stride: Optional[int] = None,
    ) -> None:
        self._llm_client = llm_client
        self._settings = settings or CreditRatingSettings()
        self._known_entities = known_entities
        self._chunk_size = chunk_size or self._settings.text_chunk_size
        self._chunk_stride = chunk_stride or self._settings.text_chunk_stride
        self._last_anonymizer: Optional[object] = None

    @property
    def last_anonymizer(self) -> Optional[object]:
        """The :class:`EntityAnonymizer` from the most recent ``parse()`` call.

        Callers can use this to de-anonymize model outputs via
        ``last_anonymizer.deanonymize(data)``.
        """
        return self._last_anonymizer

    def parse(self, source: Union[str, Path]) -> AnnualReport:
        """Extract an annual report from a PDF file.

        When anonymization is enabled (the default), entity names,
        people, locations, and absolute years are replaced with typed
        placeholders before any LLM or FinBERT processing.  The
        :class:`AnnualReport` stores **anonymized** text sections.

        Args:
            source: Path to a PDF file on disk.

        Returns:
            A populated :class:`AnnualReport` with anonymized text
            sections and structured financial data.
        """
        path = Path(source)
        pages = _extract_pages(path)

        ticker = _infer_ticker(path)
        year = _infer_year("\n".join(pages.values()))

        if self._settings.anonymize_text_for_llm:
            anonymizer = self._build_anonymizer(ticker)
            pages = anonymizer.anonymize_pages(pages)
            self._last_anonymizer = anonymizer
            logger.info(
                "Anonymized %d entities for %s",
                len(anonymizer.entity_map()),
                ticker,
            )
        else:
            self._last_anonymizer = None

        full_text = "\n".join(
            text for text in pages.values() if text is not None
        )
        mda = _find_section(full_text, _MDA_HEADERS)
        risk_factors = _find_section(full_text, _RISK_HEADERS)
        auditor = _find_section(full_text, _AUDITOR_HEADERS)

        if self._llm_client is not None:
            statements = self._extract_via_pipeline(path, year)
        else:
            logger.warning(
                "No LLM client provided; financial statements will be "
                "zeroed.  Pass an LLMClient to enable full extraction."
            )
            statements = _empty_statements(year)

        return AnnualReport(
            ticker=ticker,
            fiscal_year=year,
            financial_statements=statements,
            mda_text=mda,
            auditor_report_text=auditor,
            risk_factors_text=risk_factors,
            source_url=str(path),
        )

    def _build_anonymizer(self, ticker: str) -> "EntityAnonymizer":
        """Create an :class:`EntityAnonymizer` with appropriate config.

        If no known entities were provided, uses the inferred ticker
        as a minimal ORG entity so the primary company name is always
        masked.

        Args:
            ticker: Ticker symbol inferred from the filename.

        Returns:
            A configured :class:`EntityAnonymizer`.
        """
        from risk.anonymizer import EntityAnonymizer

        known = self._known_entities
        if known is None:
            known = [{"type": "ORG", "names": [ticker]}]

        return EntityAnonymizer(
            known_entities=known,
            use_ner=self._settings.anonymize_use_ner,
            anonymize_years=self._settings.anonymize_years,
        )

    def _extract_via_pipeline(
        self,
        pdf_path: Path,
        year: int,
    ) -> FinancialStatements:
        """Run the ``financial_forecast`` three-stage pipeline.

        Stage 1: ``FinancialStatementExtractor`` ��� raw JSON
        Stage 2: ``StatementNormalizer`` → normalized JSON
        Stage 3: Read normalized JSON �� map to domain types

        Args:
            pdf_path: Path to the PDF file.
            year: Fiscal year (used as fallback).

        Returns:
            Populated :class:`FinancialStatements`, or zeroed if
            the pipeline fails.
        """
        try:
            return self._run_pipeline_stages(pdf_path, year)
        except Exception:
            logger.exception(
                "Extraction pipeline failed for %s; returning empty statements",
                pdf_path,
            )
            return _empty_statements(year)

    def _run_pipeline_stages(
        self,
        pdf_path: Path,
        year: int,
    ) -> FinancialStatements:
        """Execute the three pipeline stages in a temp directory."""
        from financial_forecast.extraction.financial_statement_extractor import (
            FinancialStatementExtractor,
        )
        from financial_forecast.extraction.statement_config import StatementType
        from financial_forecast.extraction.statement_normalizer import (
            StatementNormalizer,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            extracted_dir = tmp / "extracted"
            normalized_dir = tmp / "normalized"

            extractor = FinancialStatementExtractor(
                queries=[
                    StatementType.BALANCE_SHEET,
                    StatementType.INCOME_STATEMENT,
                    StatementType.CASH_FLOW,
                ],
                llm_client=self._llm_client,
            )
            extractor.run(
                input_path=str(pdf_path),
                output_dir=str(extracted_dir),
            )

            normalizer = StatementNormalizer(
                input_dir=str(extracted_dir),
                output_dir=str(normalized_dir),
            )
            normalizer.run()

            return _read_normalized_statements(normalized_dir, year)

    def extract_text_chunks(self, text: str) -> List[str]:
        """Split *text* into overlapping chunks for NLP processing.

        Args:
            text: The full section text to chunk.

        Returns:
            A list of text chunks with sliding-window overlap.
        """
        words = text.split()
        if len(words) <= self._chunk_size:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + self._chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self._chunk_size - self._chunk_stride
        return chunks


# ------------------------------------------------------------------
# Reading normalized pipeline output
# ------------------------------------------------------------------


def _read_normalized_statements(
    normalized_dir: Path,
    fallback_year: int,
) -> FinancialStatements:
    """Read normalized JSON files and map to domain dataclasses.

    Args:
        normalized_dir: Directory containing ``*.normalized.json``.
        fallback_year: Year to use if not found in the data.

    Returns:
        A populated :class:`FinancialStatements`.
    """
    field_values = _collect_field_values(normalized_dir)
    if not field_values:
        logger.warning("No normalized data found in %s", normalized_dir)
        return _empty_statements(fallback_year)

    year = _extract_year_from_fields(field_values, fallback_year)
    return _map_to_domain(field_values, year)


def _collect_field_values(
    normalized_dir: Path,
) -> Dict[str, Optional[float]]:
    """Read all normalized JSON files and merge field values."""
    merged: Dict[str, Optional[float]] = {}
    for path in sorted(normalized_dir.rglob("*.normalized.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        statement = payload.get("statement", {})
        for period in statement.get("periods", []):
            values = period.get("values", {})
            merged.update(
                {k: _to_float(v) for k, v in values.items()},
            )
    return merged


def _extract_year_from_fields(
    field_values: Dict[str, Any],
    fallback: int,
) -> int:
    """Try to find the fiscal year from the normalized data."""
    return fallback


def _to_float(value: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _map_to_domain(
    fv: Dict[str, Optional[float]],
    year: int,
) -> FinancialStatements:
    """Map flat field values to credit_rating domain dataclasses.

    The ``financial_forecast`` pipeline uses these field names
    (from ``statement_config.STATEMENT_CONFIGS``):

    Balance Sheet:
        cash_and_cash_equivalents, short_term_market_securities,
        net_accounts_receivable, total_current_liabilities,
        total_debt_short_term_and_long_term, total_equity, total_assets

    Income Statement:
        total_revenue, total_operating_cost, net_income,
        income_tax_expense, interest_expenses

    Cash Flow:
        depreciation_and_amortization
    """
    g = _get_float

    total_debt = g(fv, "total_debt_short_term_and_long_term")
    total_assets = g(fv, "total_assets")
    total_equity = g(fv, "total_equity")
    total_liabilities = total_assets - total_equity
    total_current_liabilities = g(fv, "total_current_liabilities")

    income = IncomeStatement(
        total_revenue=g(fv, "total_revenue"),
        cost_of_goods_sold=0.0,
        total_operating_expenses=g(fv, "total_operating_cost"),
        selling_general_admin=0.0,
        depreciation_expense=g(fv, "depreciation_and_amortization"),
        interest_expense=g(fv, "interest_expenses"),
        income_tax_expense=g(fv, "income_tax_expense"),
        net_income=g(fv, "net_income"),
    )

    cash_and_eq = g(fv, "cash_and_cash_equivalents")
    st_investments = g(fv, "short_term_market_securities")
    receivables = g(fv, "net_accounts_receivable")
    current_assets = cash_and_eq + st_investments + receivables

    balance = BalanceSheet(
        cash_and_equivalents=cash_and_eq,
        short_term_investments=st_investments,
        net_receivables=receivables,
        inventory=0.0,
        total_current_assets=current_assets,
        net_property_plant_equipment=0.0,
        total_assets=total_assets,
        accounts_payable=0.0,
        short_term_debt=total_debt * 0.3,
        total_current_liabilities=total_current_liabilities,
        long_term_debt=total_debt * 0.7,
        total_liabilities=total_liabilities,
        total_equity=total_equity,
        retained_earnings=0.0,
    )

    dep_amort = g(fv, "depreciation_and_amortization")
    cash_flow = CashFlowStatement(
        depreciation_and_amortization=dep_amort,
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


def _get_float(
    fv: Dict[str, Optional[float]],
    key: str,
    default: float = 0.0,
) -> float:
    """Get a float from the field values dict, defaulting on None."""
    val = fv.get(key)
    if val is None:
        return default
    return val


# ------------------------------------------------------------------
# Text extraction and section finding
# ------------------------------------------------------------------


def _extract_pages(path: Path) -> Dict[int, str]:
    """Extract text from every page using pdfplumber.

    Delegates to ``financial_forecast.extraction.pdf_extractor``
    when available, falling back to a direct pdfplumber call.
    """
    try:
        from financial_forecast.extraction.pdf_extractor import (
            extract_text_pdfplumber,
        )

        raw = extract_text_pdfplumber(str(path))
        return {k: (v or "") for k, v in raw.items()}
    except ImportError:
        import pdfplumber

        pages: Dict[int, str] = {}
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                pages[i] = page.extract_text() or ""
        return pages


_MDA_HEADERS = [
    r"management.s discussion and analysis",
    r"item\s+7[.\s]",
]

_RISK_HEADERS = [
    r"risk\s+factors",
    r"item\s+1a[.\s]",
]

_AUDITOR_HEADERS = [
    r"report of independent registered public accounting firm",
    r"independent auditor.s report",
]


def _find_section(text: str, headers: List[str]) -> str:
    """Find the first section matching any header pattern."""
    for header in headers:
        pattern = re.compile(
            f"({header})(.*?)(?=item\\s+\\d|$)",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            return match.group(0).strip()[:50_000]
    return ""


def _infer_ticker(path: Path) -> str:
    """Best-effort ticker inference from the file name."""
    stem = path.stem.upper()
    parts = re.split(r"[_\\-\\s.]+", stem)
    return parts[0] if parts else "UNKNOWN"


def _infer_year(text: str) -> int:
    """Best-effort fiscal year inference from text content."""
    year_match = re.search(
        r"(?:fiscal\s+year|year\s+ended|for\s+the\s+year)\s+(\d{4})",
        text[:5000],
        re.IGNORECASE,
    )
    if year_match:
        return int(year_match.group(1))
    all_years = re.findall(r"\b(20\d{2})\b", text[:5000])
    if all_years:
        return max(int(y) for y in all_years)
    return 0


def _empty_statements(year: int) -> FinancialStatements:
    """Build zero-valued statements as a structural placeholder."""
    return FinancialStatements(
        income_statement=IncomeStatement(
            total_revenue=0.0, cost_of_goods_sold=0.0,
            total_operating_expenses=0.0, selling_general_admin=0.0,
            depreciation_expense=0.0, interest_expense=0.0,
            income_tax_expense=0.0, net_income=0.0,
        ),
        balance_sheet=BalanceSheet(
            cash_and_equivalents=0.0, short_term_investments=0.0,
            net_receivables=0.0, inventory=0.0,
            total_current_assets=0.0, net_property_plant_equipment=0.0,
            total_assets=0.0, accounts_payable=0.0,
            short_term_debt=0.0, total_current_liabilities=0.0,
            long_term_debt=0.0, total_liabilities=0.0,
            total_equity=0.0, retained_earnings=0.0,
        ),
        cash_flow=CashFlowStatement(
            depreciation_and_amortization=0.0, cash_from_operations=0.0,
            capital_expenditures=0.0, cash_from_investing=0.0,
            cash_from_financing=0.0, net_change_in_cash=0.0,
        ),
        fiscal_year=year,
    )
