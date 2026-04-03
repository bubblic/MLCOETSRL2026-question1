"""Risk warning categories and configuration constants.

Defines :class:`RiskCategory` enum and shared constants used across
the risk extraction pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import List


class RiskCategory(str, Enum):
    """Risk warning categories detected in annual reports."""

    AUDITOR_OPINION = "auditor_opinion"
    GOING_CONCERN = "going_concern"
    CONTINGENT_LIABILITIES = "contingent_liabilities"
    DEBT_COVENANTS = "debt_covenants"
    RELATED_PARTY = "related_party"
    ACCOUNTING_POLICY = "accounting_policy"
    DIRECTOR_CHANGES = "director_changes"
    CASH_FLOW_WARNINGS = "cash_flow_warnings"
    MD_AND_A_RED_FLAGS = "md_and_a_red_flags"


ALL_CATEGORIES: List[str] = [cat.value for cat in RiskCategory]

DEFAULT_CHUNK_SIZE = 30
