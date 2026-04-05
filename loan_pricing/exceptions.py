"""Custom exception hierarchy for the term loan pricing model.

All exceptions inherit from :class:`LoanPricingError` so callers can
catch the full family with a single ``except`` clause when appropriate.
"""

from __future__ import annotations


class LoanPricingError(Exception):
    """Base exception for all loan-pricing errors."""


class DataFetchError(LoanPricingError):
    """Raised when an external data source cannot be reached.

    Typical triggers include FRED API failures, SEC EDGAR timeouts,
    and exhausted retry budgets.
    """


class InsufficientDataError(LoanPricingError):
    """Raised when a dataset is too small for the requested operation.

    For example, the Ornstein-Uhlenbeck calibrator requires at least
    ``ProjectConfig.min_ou_series_length`` observations.
    """


class DataLeakageError(LoanPricingError):
    """Raised when an operation would introduce data leakage.

    The most common trigger is calling ``FeatureEngineer.fit()`` on
    data other than the designated training set.
    """
