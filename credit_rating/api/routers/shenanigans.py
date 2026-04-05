"""Shenanigans analysis router.

Exposes ``POST /analyze`` which runs all shenanigan detectors on a
company's financial statements and returns a composite risk report.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

from credit_rating.api.dependencies import (
    get_settings,
    get_shenanigans_detectors,
)
from credit_rating.api.schemas import ShenanigansRequest, ShenanigansResponse
from credit_rating.config.settings import CreditRatingSettings
from credit_rating.ingestion.sec_edgar import EdgarDownloader
from credit_rating.shenanigans.beneish import BeneishMScoreDetector
from credit_rating.shenanigans.cash_flow import CashFlowShenanigansDetector
from credit_rating.shenanigans.earnings_manipulation import (
    EarningsManipulationDetector,
)
from credit_rating.shenanigans.report_builder import ShenanigansReportBuilder

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=ShenanigansResponse)
async def analyze_shenanigans(
    request: ShenanigansRequest,
    settings: CreditRatingSettings = Depends(get_settings),
    detectors: Dict[str, object] = Depends(get_shenanigans_detectors),
) -> ShenanigansResponse:
    """Run all shenanigan detectors on a company's financials.

    Downloads current and prior-year 10-K filings from SEC EDGAR,
    then runs the Beneish M-Score detector, Schilit earnings
    manipulation signals, and cash flow shenanigan signals.  Results
    are assembled via the :class:`ShenanigansReportBuilder`.

    Args:
        request: The shenanigans analysis request payload.
        settings: Injected application settings.
        detectors: Injected detector instances.

    Returns:
        A :class:`ShenanigansResponse` with the composite report.

    Raises:
        HTTPException: 404 if financial data cannot be found, or
            500 on unexpected errors.
    """
    try:
        downloader = EdgarDownloader(settings=settings)
        report = downloader.parse(request.ticker, year=request.year)
        current = report.financial_statements
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Financial data not found for {request.ticker} "
                f"({request.year}): {exc}"
            ),
        ) from exc

    # Prior year is optional -- many signals still work without it.
    prior = None
    try:
        prior_report = downloader.parse(
            request.ticker, year=request.year - 1,
        )
        prior = prior_report.financial_statements
    except (FileNotFoundError, RuntimeError):
        logger.info(
            "Prior-year data unavailable for %s (%d); "
            "Beneish M-Score will be skipped",
            request.ticker,
            request.year - 1,
        )

    try:
        builder = ShenanigansReportBuilder()

        # Beneish M-Score (requires prior-year data).
        if prior is not None:
            beneish_detector: BeneishMScoreDetector = detectors["beneish"]
            beneish_score = beneish_detector.calculate(current, prior)
            builder = builder.with_beneish(beneish_score)

        # Schilit earnings manipulation signals.
        ems_detector = EarningsManipulationDetector(current, prior)
        builder = builder.with_earnings_signals(ems_detector.detect_all())

        # Cash flow shenanigan signals.
        cfs_detector = CashFlowShenanigansDetector(current, prior)
        builder = builder.with_cash_flow_signals(cfs_detector.detect_all())

        report = builder.build()

        # Serialise individual signals into dicts for the response.
        signals: List[Dict[str, object]] = []
        for sig in report.earnings_signals:
            signals.append({
                "signal_id": sig.signal_id,
                "name": sig.name,
                "is_flagged": sig.is_flagged,
                "observed_value": sig.observed_value,
                "threshold": sig.threshold,
                "explanation": sig.explanation,
            })
        for sig in report.cash_flow_signals:
            signals.append({
                "signal_id": sig.signal_id,
                "name": sig.name,
                "is_flagged": sig.is_flagged,
                "observed_value": sig.observed_value,
                "threshold": sig.threshold,
                "explanation": sig.explanation,
            })

        return ShenanigansResponse(
            overall_risk=report.overall_risk.name,
            total_flags=report.total_flags_raised,
            beneish_m_score=(
                report.beneish.m_score if report.beneish is not None else None
            ),
            signals=signals,
        )
    except Exception as exc:
        logger.exception(
            "Shenanigans analysis failed for %s", request.ticker,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Shenanigans analysis failed: {exc}",
        ) from exc
