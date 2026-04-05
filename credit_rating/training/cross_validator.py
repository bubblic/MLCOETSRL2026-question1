"""Temporal cross-validation for financial data.

Implements walk-forward validation where the training set always
precedes the validation set, which precedes the test set.  This
prevents future data leakage common in standard k-fold CV applied
to time-series financial data.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TemporalCrossValidator:
    """Walk-forward temporal cross-validator.

    Given ``n`` time periods, produces folds where:
    - Train: periods ``[0, split_point)``
    - Validation: periods ``[split_point, split_point + val_size)``
    - Test: periods ``[split_point + val_size, split_point + val_size + test_size)``

    The split point advances by ``step`` each fold.

    Args:
        n_splits: Number of walk-forward folds.
        val_size: Number of periods in the validation window.
        test_size: Number of periods in the test window.
        min_train_size: Minimum periods required for training.
    """

    def __init__(
        self,
        n_splits: int = 5,
        val_size: int = 1,
        test_size: int = 1,
        min_train_size: int = 3,
    ) -> None:
        self._n_splits = n_splits
        self._val_size = val_size
        self._test_size = test_size
        self._min_train_size = min_train_size

    def split(
        self,
        years: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Iterator[Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]]:
        """Generate train/val/test splits by year.

        Args:
            years: 1-D array of fiscal years for each sample.
            features: Feature matrix, shape ``(n, d)``.
            labels: Label vector, shape ``(n,)``.

        Yields:
            Tuples of ``(train, val, test)`` where each element is
            a ``(features, labels)`` pair.
        """
        unique_years = sorted(set(years))
        window = self._val_size + self._test_size
        folds_generated = 0

        for split_end in range(
            self._min_train_size + window,
            len(unique_years) + 1,
        ):
            if folds_generated >= self._n_splits:
                break

            train_years = set(unique_years[: split_end - window])
            val_years = set(
                unique_years[split_end - window : split_end - self._test_size],
            )
            test_years = set(unique_years[split_end - self._test_size : split_end])

            train_mask = np.isin(years, list(train_years))
            val_mask = np.isin(years, list(val_years))
            test_mask = np.isin(years, list(test_years))

            if not (train_mask.any() and val_mask.any() and test_mask.any()):
                continue

            logger.info(
                "Fold %d: train=%s, val=%s, test=%s",
                folds_generated + 1,
                sorted(train_years),
                sorted(val_years),
                sorted(test_years),
            )

            yield (
                (features[train_mask], labels[train_mask]),
                (features[val_mask], labels[val_mask]),
                (features[test_mask], labels[test_mask]),
            )
            folds_generated += 1
