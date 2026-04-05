"""Structural interfaces for data ingestion components.

Uses ``typing.Protocol`` for duck-typed interfaces — parsers and data
sources satisfy these protocols without inheriting from them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, Tuple, Union, runtime_checkable

from credit_rating.config.settings import RatingClass
from credit_rating.domain.features import FinancialRatios
from credit_rating.domain.report import AnnualReport


@runtime_checkable
class AnnualReportParser(Protocol):
    """Any object that can produce an :class:`AnnualReport` from a source."""

    def parse(self, source: Union[str, Path]) -> AnnualReport:
        """Parse an annual report from *source*.

        Args:
            source: A file path, URL, or ticker string depending on
                the implementation.

        Returns:
            A fully populated :class:`AnnualReport`.
        """
        ...


@runtime_checkable
class FinancialDataSource(Protocol):
    """Any object that yields labelled financial ratio vectors."""

    def load(self) -> Iterator[Tuple[FinancialRatios, RatingClass]]:
        """Yield ``(features, label)`` pairs.

        Returns:
            An iterator of tuples suitable for training or evaluation.
        """
        ...
