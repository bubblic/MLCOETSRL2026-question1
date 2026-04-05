"""Credit rating prediction data containers.

Re-exports :class:`~credit_rating.config.settings.RatingClass` for
convenience and defines :class:`RatingPrediction`, the structured
output of every rating model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from credit_rating.config.settings import RatingClass


@dataclass(frozen=True)
class RatingPrediction:
    """Result of a credit rating model prediction.

    Args:
        rating: The predicted ordinal rating bucket.
        probabilities: Per-class probability distribution.
        altman_z_score: Altman Z (or Z'') score if computed.
        shap_values: Feature-name-to-SHAP-value mapping for
            the structured tower (empty if not computed).
    """

    rating: RatingClass
    probabilities: dict[RatingClass, float]
    altman_z_score: float = 0.0
    shap_values: dict[str, float] = field(default_factory=dict)

    @property
    def is_investment_grade(self) -> bool:
        """Whether the predicted rating is investment grade."""
        return self.rating.is_investment_grade

    @property
    def confidence(self) -> float:
        """Probability assigned to the predicted class."""
        return self.probabilities.get(self.rating, 0.0)

    def top_n_classes(self, n: int = 3) -> list[tuple[RatingClass, float]]:
        """Return the *n* most probable rating classes.

        Args:
            n: Number of top classes to return.

        Returns:
            List of ``(RatingClass, probability)`` tuples sorted
            by descending probability.
        """
        sorted_pairs = sorted(
            self.probabilities.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return sorted_pairs[:n]
