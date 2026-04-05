"""Structural interfaces for data-fetching components.

High-level pipeline stages depend on these protocols rather than on
concrete fetcher implementations, following the Dependency Inversion
Principle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataFetcher(Protocol):
    """Structural interface satisfied by any data-source fetcher.

    Implementations include :class:`~loan_pricing.data.fetch_fred.FredDataFetcher`
    and :class:`~loan_pricing.data.fetch_sec.SecEdgarFetcher`.
    """

    def fetch(
        self,
        series_ids: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Retrieve one or more data series for a date range.

        Args:
            series_ids: Identifiers understood by the backing data source
                (e.g. FRED series codes or SEC CIK numbers).
            start_date: Inclusive start in ``YYYY-MM-DD`` format.
            end_date: Inclusive end in ``YYYY-MM-DD`` format.

        Returns:
            A tidy (long-format) DataFrame whose exact columns depend on
            the data source but always include a date or time dimension.

        Raises:
            DataFetchError: If the backing service is unreachable after
                the configured number of retries.
        """
        ...
