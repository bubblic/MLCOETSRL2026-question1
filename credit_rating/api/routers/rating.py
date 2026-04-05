"""Rating prediction router.

Exposes ``POST /rate`` which accepts a :class:`RatingRequest` and
returns a :class:`RatingResponse` by running the hybrid model's
structured tower.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from credit_rating.api.dependencies import get_rating_model, get_settings
from credit_rating.api.schemas import RatingRequest, RatingResponse
from credit_rating.config.settings import CreditRatingSettings, RatingClass
from credit_rating.features.altman import AltmanZScoreCalculator
from credit_rating.features.ratio_calculator import FinancialRatioCalculator
from credit_rating.ingestion.sec_edgar import EdgarDownloader
from credit_rating.models.hybrid import HybridRatingModel

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/rate", response_model=RatingResponse)
async def rate_company(
    request: RatingRequest,
    model: HybridRatingModel = Depends(get_rating_model),
    settings: CreditRatingSettings = Depends(get_settings),
) -> RatingResponse:
    """Predict the credit rating for a company in a given fiscal year.

    Downloads the 10-K filing from SEC EDGAR, extracts financial
    statements, computes the 25-feature ratio vector, runs the
    structured tower, and returns the predicted rating with class
    probabilities and the Altman Z-Score.

    Args:
        request: The rating request payload.
        model: Injected hybrid rating model.
        settings: Injected application settings.

    Returns:
        A :class:`RatingResponse` with the prediction results.

    Raises:
        HTTPException: 404 if financial data cannot be found, or
            500 on unexpected model errors.
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
        import tensorflow as tf

        statements = report.financial_statements
        calculator = FinancialRatioCalculator()
        ratios = calculator.calculate(statements)

        x_struct = tf.expand_dims(ratios.to_tensor(), axis=0)
        logits = model.predict_from_structured(x_struct)
        probs = tf.nn.softmax(logits[0]).numpy()

        predicted_idx = int(tf.argmax(logits[0]).numpy())
        predicted_rating = RatingClass(predicted_idx)

        probabilities = {
            RatingClass(i).name: float(probs[i])
            for i in range(settings.num_rating_classes)
        }

        altman_calc = AltmanZScoreCalculator(settings=settings)
        z_score = altman_calc.calculate_z(statements)

        return RatingResponse(
            rating=predicted_rating.name,
            probabilities=probabilities,
            altman_z_score=z_score,
            is_investment_grade=predicted_rating.is_investment_grade,
        )
    except Exception as exc:
        logger.exception("Rating prediction failed for %s", request.ticker)
        raise HTTPException(
            status_code=500,
            detail=f"Rating prediction failed: {exc}",
        ) from exc
