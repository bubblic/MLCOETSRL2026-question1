"""Cash flow shenanigan detectors (CFS-1..CFS-4).

Each detector identifies a specific pattern of cash flow
manipulation or misrepresentation.
"""

from __future__ import annotations

from typing import List, Optional

from credit_rating.domain.financial_statements import FinancialStatements
from credit_rating.domain.shenanigans import CashFlowSignal


class CashFlowShenanigansDetector:
    """Detect Schilit cash flow shenanigans.

    Args:
        current: Current period financial statements.
        prior: Prior period financial statements (optional).
    """

    def __init__(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements] = None,
    ) -> None:
        self._current = current
        self._prior = prior

    def detect_all(self) -> List[CashFlowSignal]:
        """Run all four CFS detectors.

        Returns:
            A list of four :class:`CashFlowSignal` results.
        """
        return [
            self.detect_cfs_1_operating_to_investing(),
            self.detect_cfs_2_financing_as_operating(),
            self.detect_cfs_3_asset_sales_boosting_ocf(),
            self.detect_cfs_4_receivables_factoring(),
        ]

    def detect_cfs_1_operating_to_investing(self) -> CashFlowSignal:
        """CFS-1: Shifting operating outflows to investing.

        Flag when capital expenditures exceed 80% of operating cash
        flow, suggesting operating costs are disguised as investment.
        """
        threshold = 0.80
        ratio = _safe_divide(
            self._current.cash_flow.capital_expenditures,
            max(self._current.cash_flow.cash_from_operations, 1.0),
        )
        return CashFlowSignal(
            signal_id="CFS-1",
            name="Shifting Operating Cash Outflows to Investing",
            is_flagged=ratio > threshold,
            observed_value=ratio,
            threshold=threshold,
            explanation=(
                f"CapEx / Operating CF = {ratio:.1%} "
                f"(threshold: {threshold:.0%})"
            ),
        )

    def detect_cfs_2_financing_as_operating(self) -> CashFlowSignal:
        """CFS-2: Financing inflows inflating operating cash flow.

        Flag when financing cash inflows (positive) exceed 50% of
        operating cash flow, suggesting debt proceeds are mis-classified.
        """
        threshold = 0.50
        financing = self._current.cash_flow.cash_from_financing
        ocf = max(self._current.cash_flow.cash_from_operations, 1.0)
        ratio = _safe_divide(max(financing, 0.0), ocf)
        return CashFlowSignal(
            signal_id="CFS-2",
            name="Financing Inflows Inflating Operating CF",
            is_flagged=financing > 0 and ratio > threshold,
            observed_value=ratio,
            threshold=threshold,
            explanation=(
                f"Financing inflows / Operating CF = {ratio:.1%} "
                f"(threshold: {threshold:.0%})"
            ),
        )

    def detect_cfs_3_asset_sales_boosting_ocf(self) -> CashFlowSignal:
        """CFS-3: Boosting operating CF via asset sales.

        Flag when investing inflows (positive) exceed 30% of
        operating cash flow — suggests one-off asset disposals
        are sustaining apparent cash generation.
        """
        threshold = 0.30
        investing = self._current.cash_flow.cash_from_investing
        ocf = max(self._current.cash_flow.cash_from_operations, 1.0)
        ratio = _safe_divide(max(investing, 0.0), ocf)
        return CashFlowSignal(
            signal_id="CFS-3",
            name="Boosting Operating CF with Asset Sales",
            is_flagged=investing > 0 and ratio > threshold,
            observed_value=ratio,
            threshold=threshold,
            explanation=(
                f"Investing inflows / Operating CF = {ratio:.1%} "
                f"(threshold: {threshold:.0%})"
            ),
        )

    def detect_cfs_4_receivables_factoring(self) -> CashFlowSignal:
        """CFS-4: Receivables factoring inflating operating CF.

        Flag when receivables drop > 20% while revenue is flat or
        growing — suggests receivables are being sold (factored)
        to generate short-term cash.
        """
        threshold = -0.20
        if self._prior is None:
            return _no_prior_signal("CFS-4", "Receivables Factoring")

        cur_recv = self._current.balance_sheet.net_receivables
        pri_recv = self._prior.balance_sheet.net_receivables
        recv_change = _safe_divide(cur_recv - pri_recv, abs(pri_recv))

        cur_rev = self._current.income_statement.total_revenue
        pri_rev = self._prior.income_statement.total_revenue
        rev_growth = _safe_divide(cur_rev - pri_rev, abs(pri_rev))

        flagged = recv_change < threshold and rev_growth >= 0
        return CashFlowSignal(
            signal_id="CFS-4",
            name="Receivables Factoring",
            is_flagged=flagged,
            observed_value=recv_change,
            threshold=threshold,
            explanation=(
                f"Receivables change = {recv_change:+.1%} with "
                f"revenue growth = {rev_growth:+.1%} "
                f"(flag if receivables < {threshold:+.0%} and revenue flat/up)"
            ),
        )


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide with zero-protection."""
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _no_prior_signal(signal_id: str, name: str) -> CashFlowSignal:
    """Return a non-flagged signal when prior data is unavailable."""
    return CashFlowSignal(
        signal_id=signal_id,
        name=name,
        is_flagged=False,
        observed_value=0.0,
        threshold=0.0,
        explanation="Prior-period data unavailable; signal not evaluated.",
    )
