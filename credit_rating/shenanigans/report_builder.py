"""Builder pattern for assembling a :class:`ShenanigansReport`.

Callers chain detector outputs in any order and produce a composite
report with an overall :class:`RiskLevel` computed from the weighted
combination of signals.
"""

from __future__ import annotations

from typing import List, Optional

from credit_rating.config.settings import RiskLevel
from credit_rating.domain.shenanigans import (
    BeneishScore,
    CashFlowSignal,
    EarningsManipulationSignal,
    ShenanigansReport,
    TextSignals,
)


class ShenanigansReportBuilder:
    """Builder that assembles a :class:`ShenanigansReport`.

    Usage::

        report = (
            ShenanigansReportBuilder()
            .with_beneish(beneish_score)
            .with_earnings_signals(ems_signals)
            .with_cash_flow_signals(cfs_signals)
            .with_text_signals(text_signals)
            .build()
        )
    """

    def __init__(self) -> None:
        self._beneish: Optional[BeneishScore] = None
        self._earnings: List[EarningsManipulationSignal] = []
        self._cash_flow: List[CashFlowSignal] = []
        self._text: Optional[TextSignals] = None

    def with_beneish(
        self,
        score: BeneishScore,
    ) -> "ShenanigansReportBuilder":
        """Add a Beneish M-Score result.

        Args:
            score: The computed Beneish score.

        Returns:
            ``self`` for method chaining.
        """
        self._beneish = score
        return self

    def with_earnings_signals(
        self,
        signals: List[EarningsManipulationSignal],
    ) -> "ShenanigansReportBuilder":
        """Add earnings manipulation signals (EMS-1..EMS-8).

        Args:
            signals: List of EMS detection results.

        Returns:
            ``self`` for method chaining.
        """
        self._earnings = signals
        return self

    def with_cash_flow_signals(
        self,
        signals: List[CashFlowSignal],
    ) -> "ShenanigansReportBuilder":
        """Add cash flow shenanigan signals (CFS-1..CFS-4).

        Args:
            signals: List of CFS detection results.

        Returns:
            ``self`` for method chaining.
        """
        self._cash_flow = signals
        return self

    def with_text_signals(
        self,
        signals: TextSignals,
    ) -> "ShenanigansReportBuilder":
        """Add text-based signals (Fog, tone shift, boilerplate).

        Args:
            signals: The computed text signals.

        Returns:
            ``self`` for method chaining.
        """
        self._text = signals
        return self

    def build(self) -> ShenanigansReport:
        """Assemble the final report with an overall risk level.

        Returns:
            A frozen :class:`ShenanigansReport`.
        """
        risk_level = self._compute_risk_level()
        return ShenanigansReport(
            beneish=self._beneish,
            earnings_signals=self._earnings,
            cash_flow_signals=self._cash_flow,
            text_signals=self._text,
            overall_risk=risk_level,
        )

    def _compute_risk_level(self) -> RiskLevel:
        """Derive overall risk from the weighted signal counts."""
        score = 0.0

        if self._beneish is not None and self._beneish.is_likely_manipulator:
            score += 3.0

        ems_flags = sum(1 for s in self._earnings if s.is_flagged)
        score += ems_flags * 1.0

        cfs_flags = sum(1 for s in self._cash_flow if s.is_flagged)
        score += cfs_flags * 1.5

        if self._text is not None:
            if self._text.fog_is_flagged:
                score += 0.5
            if self._text.tone_shift and self._text.tone_shift.is_flagged:
                score += 1.0
            if self._text.boilerplate_is_flagged:
                score += 0.5

        return _score_to_risk_level(score)


def _score_to_risk_level(score: float) -> RiskLevel:
    """Map a numeric risk score to a :class:`RiskLevel`.

    Args:
        score: Weighted sum of signal flags.

    Returns:
        The corresponding :class:`RiskLevel`.
    """
    if score >= 6.0:
        return RiskLevel.CRITICAL
    if score >= 3.0:
        return RiskLevel.HIGH
    if score >= 1.5:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
