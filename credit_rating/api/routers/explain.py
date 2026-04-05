"""Explainability router.

Exposes ``POST /explain`` which computes SHAP feature attributions
for the structured tower's prediction on a given company and year.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException

from credit_rating.api.dependencies import get_rating_model, get_settings
from credit_rating.api.schemas import ExplainRequest, ExplainResponse
from credit_rating.config.settings import CreditRatingSettings
from credit_rating.explainability.shap_explainer import StructuredTowerExplainer
from credit_rating.features.ratio_calculator import FinancialRatioCalculator
from credit_rating.ingestion.sec_edgar import EdgarDownloader
from credit_rating.models.hybrid import HybridRatingModel

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(
    request: ExplainRequest,
    model: HybridRatingModel = Depends(get_rating_model),
    settings: CreditRatingSettings = Depends(get_settings),
) -> ExplainResponse:
    """Compute SHAP feature attributions for a rating prediction.

    Downloads the 10-K filing, computes financial ratios, and uses
    :class:`StructuredTowerExplainer` to attribute the model output
    to each of the 25 financial ratio features.

    Args:
        request: The explain request payload.
        model: Injected hybrid rating model.
        settings: Injected application settings.

    Returns:
        A :class:`ExplainResponse` with SHAP values and ranked
        top features.

    Raises:
        HTTPException: 404 if financial data cannot be found, 503
            if SHAP is not available, or 500 on unexpected errors.
    """
    try:
        downloader = EdgarDownloader(settings=settings)
        report = downloader.parse(request.ticker, year=request.year)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Financial data not found for {request.ticker} "
                f"({request.year}): {exc}"
            ),
        ) from exc

    try:
        import numpy as np

        calculator = FinancialRatioCalculator()
        ratios = calculator.calculate(report.financial_statements)

        explainer = StructuredTowerExplainer(model=model.structured_tower)

        # Build a minimal background dataset from the current sample.
        # In production this would use a larger reference set loaded
        # from the training data.
        background = np.array([ratios.to_flat_list()], dtype=np.float32)
        explainer.set_background(background)

        shap_values: Dict[str, float] = explainer.explain(ratios)

        # Rank features by absolute SHAP magnitude.
        sorted_features = sorted(
            shap_values.items(),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        top_features: List[Dict[str, object]] = [
            {
                "feature": name,
                "shap_value": value,
                "rank": rank + 1,
            }
            for rank, (name, value) in enumerate(sorted_features)
        ]

        return ExplainResponse(
            shap_values=shap_values,
            top_features=top_features,
        )
    except RuntimeError as exc:
        logger.warning("SHAP explanation unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"SHAP explanation unavailable: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception(
            "Explanation failed for %s (%d)",
            request.ticker,
            request.year,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {exc}",
        ) from exc
