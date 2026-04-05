"""Pydantic v2 request/response models for the credit rating API.

These schemas live at the API boundary and are intentionally separate
from the domain dataclasses so that internal representations can evolve
independently of the wire format.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Rating
# ------------------------------------------------------------------


class RatingRequest(BaseModel):
    """Request body for the ``POST /rate`` endpoint.

    Args:
        ticker: Company ticker symbol (e.g. ``"AAPL"``).
        year: Fiscal year for the rating prediction.
        pdf_url: Optional URL to a 10-K PDF for text-tower input.
    """

    ticker: str = Field(..., description="Company ticker symbol.")
    year: int = Field(..., description="Fiscal year to rate.")
    pdf_url: Optional[str] = Field(
        default=None,
        description="Optional URL to a 10-K PDF for text-tower input.",
    )


class RatingResponse(BaseModel):
    """Response body returned by the ``POST /rate`` endpoint.

    Args:
        rating: Predicted credit rating bucket (e.g. ``"BBB"``).
        probabilities: Per-class probability distribution.
        altman_z_score: Altman Z (or Z'') score.
        is_investment_grade: Whether the predicted rating is IG.
    """

    rating: str = Field(..., description="Predicted credit rating bucket.")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Per-class probability distribution.",
    )
    altman_z_score: float = Field(
        ...,
        description="Altman Z-Score for the company.",
    )
    is_investment_grade: bool = Field(
        ...,
        description="Whether the predicted rating is investment grade.",
    )


# ------------------------------------------------------------------
# Shenanigans
# ------------------------------------------------------------------


class ShenanigansRequest(BaseModel):
    """Request body for the ``POST /analyze`` endpoint.

    Args:
        ticker: Company ticker symbol.
        year: Fiscal year to analyse for manipulation signals.
    """

    ticker: str = Field(..., description="Company ticker symbol.")
    year: int = Field(..., description="Fiscal year to analyse.")


class ShenanigansResponse(BaseModel):
    """Response body returned by the ``POST /analyze`` endpoint.

    Args:
        overall_risk: Composite risk level (LOW / MEDIUM / HIGH / CRITICAL).
        total_flags: Count of triggered shenanigan signals.
        beneish_m_score: Beneish M-Score if computable.
        signals: List of individual signal results.
    """

    overall_risk: str = Field(
        ...,
        description="Composite risk level.",
    )
    total_flags: int = Field(
        ...,
        description="Number of triggered shenanigan signals.",
    )
    beneish_m_score: Optional[float] = Field(
        default=None,
        description="Beneish M-Score (None if prior-year data unavailable).",
    )
    signals: List[Dict[str, object]] = Field(
        default_factory=list,
        description="Individual signal results.",
    )


# ------------------------------------------------------------------
# Explainability
# ------------------------------------------------------------------


class ExplainRequest(BaseModel):
    """Request body for the ``POST /explain`` endpoint.

    Args:
        ticker: Company ticker symbol.
        year: Fiscal year whose prediction to explain.
    """

    ticker: str = Field(..., description="Company ticker symbol.")
    year: int = Field(..., description="Fiscal year to explain.")


class ExplainResponse(BaseModel):
    """Response body returned by the ``POST /explain`` endpoint.

    Args:
        shap_values: Feature-name-to-SHAP-value mapping.
        top_features: Ranked list of the most influential features.
    """

    shap_values: Dict[str, float] = Field(
        ...,
        description="Feature-name-to-SHAP-value mapping.",
    )
    top_features: List[Dict[str, object]] = Field(
        default_factory=list,
        description="Top features ranked by absolute SHAP magnitude.",
    )
