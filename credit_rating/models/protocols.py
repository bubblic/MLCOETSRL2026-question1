"""Structural interfaces for credit rating models.

``CreditRatingModel`` and ``Trainable`` are ``Protocol`` classes —
implementations satisfy them via structural subtyping without
inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from credit_rating.domain.rating import RatingPrediction
from credit_rating.domain.report import AnnualReport


@runtime_checkable
class CreditRatingModel(Protocol):
    """Any object that predicts a credit rating from an annual report."""

    def predict(self, report: AnnualReport) -> RatingPrediction:
        """Produce a rating prediction for *report*.

        Args:
            report: A fully populated annual report.

        Returns:
            A :class:`RatingPrediction` with rating, probabilities,
            and optional SHAP values.
        """
        ...


@runtime_checkable
class Trainable(Protocol):
    """Any model that can be fitted on labelled data."""

    def fit(self, features: object, labels: object) -> object:
        """Train the model on *features* and *labels*.

        Args:
            features: Training feature matrix or dataset.
            labels: Training labels.

        Returns:
            Training metrics or history object.
        """
        ...
