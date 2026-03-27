"""Shared utility functions for the extraction pipeline."""

from typing import Optional


def safe_divide(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    """Safely divide two values, returning None when not possible."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def sum_if_all_present(*values: Optional[float]) -> Optional[float]:
    """Return sum only when every input is present."""
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)
