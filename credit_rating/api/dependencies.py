"""FastAPI dependency injection for the credit rating API.

Provides singleton-like access to expensive resources (settings,
trained model, shenanigans detectors) so they are constructed once
and shared across request handlers.
"""

from __future__ import annotations

import functools
import logging
from typing import Dict

from credit_rating.config.settings import CreditRatingSettings
from credit_rating.models.hybrid import HybridRatingModel
from credit_rating.shenanigans.beneish import BeneishMScoreDetector
from credit_rating.shenanigans.text_signals import TextSignalDetector

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_settings() -> CreditRatingSettings:
    """Return the application-wide settings singleton.

    The result is cached so that every ``Depends(get_settings)`` call
    in FastAPI receives the same instance without re-parsing the
    environment.

    Returns:
        The :class:`CreditRatingSettings` instance.
    """
    logger.info("Loading credit rating settings")
    return CreditRatingSettings()


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

# Module-level cache for the trained model.
_model_singleton: Dict[str, HybridRatingModel] = {}


def get_rating_model(
    settings: CreditRatingSettings = None,
) -> HybridRatingModel:
    """Return the trained :class:`HybridRatingModel` singleton.

    On first call the model is constructed from the latest checkpoint
    found under ``settings.checkpoint_dir``.  Subsequent calls return
    the cached instance.

    Args:
        settings: Configuration for model construction and checkpoint
            location.  Falls back to :func:`get_settings` if ``None``.

    Returns:
        The loaded :class:`HybridRatingModel`.
    """
    if "model" in _model_singleton:
        return _model_singleton["model"]

    if settings is None:
        settings = get_settings()

    checkpoint_path = settings.checkpoint_dir
    if checkpoint_path.exists():
        logger.info("Loading model from checkpoint: %s", checkpoint_path)
        model = HybridRatingModel.load_checkpoint(
            checkpoint_path, settings=settings,
        )
    else:
        logger.warning(
            "No checkpoint found at %s; creating uninitialised model",
            checkpoint_path,
        )
        model = HybridRatingModel(settings=settings)

    _model_singleton["model"] = model
    return model


# ------------------------------------------------------------------
# Shenanigans detectors
# ------------------------------------------------------------------


def get_shenanigans_detectors(
    settings: CreditRatingSettings = None,
) -> Dict[str, object]:
    """Return a dictionary of all shenanigan detector instances.

    Args:
        settings: Configuration supplying coefficients and thresholds.
            Falls back to :func:`get_settings` if ``None``.

    Returns:
        A mapping with keys ``"beneish"`` and ``"text_signals"``,
        each pointing to the corresponding detector instance.
    """
    if settings is None:
        settings = get_settings()

    return {
        "beneish": BeneishMScoreDetector(settings=settings),
        "text_signals": TextSignalDetector(settings=settings),
    }
