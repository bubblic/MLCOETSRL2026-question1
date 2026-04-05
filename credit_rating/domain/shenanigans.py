"""Data containers for the financial shenanigans detection suite.

Covers Beneish M-Score, Schilit earnings manipulation signals (EMS),
cash flow shenanigan signals (CFS), text-based signals, and the
composite :class:`ShenanigansReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from credit_rating.config.settings import RiskLevel


# ------------------------------------------------------------------
# Beneish M-Score
# ------------------------------------------------------------------


@dataclass(frozen=True)
class BeneishScore:
    """Result of the eight-variable Beneish M-Score computation.

    Args:
        dsri: Days Sales in Receivables Index.
        gmi: Gross Margin Index.
        aqi: Asset Quality Index.
        sgi: Sales Growth Index.
        depi: Depreciation Index.
        sgai: SG&A Index.
        lvgi: Leverage Index.
        tata: Total Accruals to Total Assets.
        m_score: Weighted combination of the eight variables.
        is_likely_manipulator: ``True`` when *m_score* exceeds the
            manipulation threshold (default -2.22).
    """

    dsri: float
    gmi: float
    aqi: float
    sgi: float
    depi: float
    sgai: float
    lvgi: float
    tata: float
    m_score: float
    is_likely_manipulator: bool


# ------------------------------------------------------------------
# Schilit signals
# ------------------------------------------------------------------


@dataclass(frozen=True)
class EarningsManipulationSignal:
    """Single earnings manipulation shenanigan signal.

    Args:
        signal_id: Identifier such as ``"EMS-1"``.
        name: Human-readable name of the shenanigan.
        is_flagged: Whether this signal was triggered.
        observed_value: The computed metric value.
        threshold: The threshold that triggers a flag.
        explanation: One-sentence human-readable explanation.
    """

    signal_id: str
    name: str
    is_flagged: bool
    observed_value: float
    threshold: float
    explanation: str


@dataclass(frozen=True)
class CashFlowSignal:
    """Single cash flow shenanigan signal.

    Args:
        signal_id: Identifier such as ``"CFS-1"``.
        name: Human-readable name of the shenanigan.
        is_flagged: Whether this signal was triggered.
        observed_value: The computed metric value.
        threshold: The threshold that triggers a flag.
        explanation: One-sentence human-readable explanation.
    """

    signal_id: str
    name: str
    is_flagged: bool
    observed_value: float
    threshold: float
    explanation: str


# ------------------------------------------------------------------
# Text-based signals
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ToneShiftResult:
    """FinBERT tone comparison between consecutive MD&A sections.

    Args:
        current_positive_score: Positive sentiment in current period.
        prior_positive_score: Positive sentiment in prior period.
        delta: Absolute change in positive sentiment.
        is_flagged: Whether *delta* exceeds the cutoff.
    """

    current_positive_score: float
    prior_positive_score: float
    delta: float
    is_flagged: bool


@dataclass(frozen=True)
class TextSignals:
    """Aggregated text-based shenanigan signals.

    Args:
        fog_index: Gunning Fog readability index of risk factors.
        fog_is_flagged: ``True`` if fog index exceeds cutoff.
        tone_shift: FinBERT tone shift result (``None`` if no
            prior-year comparison is available).
        boilerplate_similarity: TF-IDF cosine similarity score
            (``None`` if no prior-year comparison is available).
        boilerplate_is_flagged: ``True`` if similarity exceeds cutoff.
    """

    fog_index: float
    fog_is_flagged: bool
    tone_shift: Optional[ToneShiftResult] = None
    boilerplate_similarity: Optional[float] = None
    boilerplate_is_flagged: bool = False


# ------------------------------------------------------------------
# Composite report
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ShenanigansReport:
    """Complete shenanigans analysis for a single annual report.

    Args:
        beneish: Beneish M-Score result (``None`` if prior-year data
            unavailable).
        earnings_signals: List of EMS-1 through EMS-8 results.
        cash_flow_signals: List of CFS-1 through CFS-4 results.
        text_signals: Text-based signal results.
        overall_risk: Weighted overall risk assessment.
    """

    beneish: Optional[BeneishScore]
    earnings_signals: List[EarningsManipulationSignal] = field(
        default_factory=list,
    )
    cash_flow_signals: List[CashFlowSignal] = field(default_factory=list)
    text_signals: Optional[TextSignals] = None
    overall_risk: RiskLevel = RiskLevel.LOW

    @property
    def total_flags_raised(self) -> int:
        """Count of all individual signals that were triggered."""
        count = 0
        if self.beneish is not None and self.beneish.is_likely_manipulator:
            count += 1
        count += sum(1 for s in self.earnings_signals if s.is_flagged)
        count += sum(1 for s in self.cash_flow_signals if s.is_flagged)
        if self.text_signals is not None:
            if self.text_signals.fog_is_flagged:
                count += 1
            if self.text_signals.tone_shift and self.text_signals.tone_shift.is_flagged:
                count += 1
            if self.text_signals.boilerplate_is_flagged:
                count += 1
        return count
