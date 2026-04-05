"""FastAPI application for the credit rating system.

Exposes endpoints for rating prediction, shenanigans analysis, and
SHAP-based explainability.  The trained model is loaded once during
application startup via the lifespan context manager.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from credit_rating.api.dependencies import get_rating_model, get_settings
from credit_rating.api.routers import explain, rating, shenanigans

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the rating model on startup and clean up on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the ASGI server while the app is running.
    """
    logger.info("Starting credit rating API")
    settings = get_settings()
    get_rating_model(settings)
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down credit rating API")


app = FastAPI(
    title="Credit Rating API",
    description=(
        "Two-tower hybrid credit rating model with shenanigans "
        "detection and SHAP explainability."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(rating.router, tags=["rating"])
app.include_router(shenanigans.router, tags=["shenanigans"])
app.include_router(explain.router, tags=["explainability"])


@app.get("/health")
async def health_check() -> dict:
    """Return a simple liveness probe.

    Returns:
        A JSON object with ``{"status": "ok"}``.
    """
    return {"status": "ok"}
