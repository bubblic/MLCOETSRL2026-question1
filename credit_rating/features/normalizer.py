"""Feature normalization with persistence for consistent inference.

Wraps ``sklearn.preprocessing.RobustScaler`` (robust to outliers
common in distressed issuers) with fit/transform/persist semantics.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from credit_rating.domain.features import FinancialRatios

logger = logging.getLogger(__name__)


class RatioNormalizer:
    """Normalize :class:`FinancialRatios` vectors using ``RobustScaler``.

    The scaler state can be persisted to disk so that inference-time
    normalization is consistent with training-time normalization.

    Args:
        scaler_path: Optional path to load a pre-fitted scaler from.
    """

    def __init__(self, scaler_path: Optional[Path] = None) -> None:
        from sklearn.preprocessing import RobustScaler

        if scaler_path and scaler_path.exists():
            self._scaler = self._load(scaler_path)
            self._is_fitted = True
        else:
            self._scaler = RobustScaler()
            self._is_fitted = False

    def fit(self, ratios_list: List[FinancialRatios]) -> "RatioNormalizer":
        """Fit the scaler on a collection of ratio vectors.

        Args:
            ratios_list: Training-set financial ratios.

        Returns:
            ``self`` for method chaining.
        """
        matrix = _ratios_to_matrix(ratios_list)
        self._scaler.fit(matrix)
        self._is_fitted = True
        return self

    def transform(self, ratios_list: List[FinancialRatios]) -> np.ndarray:
        """Transform ratio vectors using the fitted scaler.

        Args:
            ratios_list: Financial ratios to normalize.

        Returns:
            A 2-D numpy array of shape ``(n_samples, 25)``.

        Raises:
            RuntimeError: If the scaler has not been fitted.
        """
        self._check_fitted()
        matrix = _ratios_to_matrix(ratios_list)
        return self._scaler.transform(matrix)

    def fit_transform(
        self,
        ratios_list: List[FinancialRatios],
    ) -> np.ndarray:
        """Fit and transform in a single step.

        Args:
            ratios_list: Training-set financial ratios.

        Returns:
            A 2-D numpy array of shape ``(n_samples, 25)``.
        """
        matrix = _ratios_to_matrix(ratios_list)
        result = self._scaler.fit_transform(matrix)
        self._is_fitted = True
        return result

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """Reverse the scaling transformation.

        Args:
            scaled: Previously scaled data.

        Returns:
            Data in the original feature space.
        """
        self._check_fitted()
        return self._scaler.inverse_transform(scaled)

    def save(self, path: Path) -> None:
        """Persist the fitted scaler to disk.

        Args:
            path: Destination file path.
        """
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file_handle:
            pickle.dump(self._scaler, file_handle)
        logger.info("Saved scaler to %s", path)

    @property
    def is_fitted(self) -> bool:
        """Whether the scaler has been fitted."""
        return self._is_fitted

    def _check_fitted(self) -> None:
        """Raise if the scaler is not yet fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "RatioNormalizer has not been fitted. "
                "Call .fit() or .fit_transform() first."
            )

    @staticmethod
    def _load(path: Path) -> object:
        """Load a scaler from a pickle file."""
        with path.open("rb") as file_handle:
            return pickle.load(file_handle)


def _ratios_to_matrix(ratios_list: List[FinancialRatios]) -> np.ndarray:
    """Stack a list of :class:`FinancialRatios` into a 2-D array."""
    return np.array(
        [r.to_flat_list() for r in ratios_list],
        dtype=np.float64,
    )
