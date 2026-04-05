"""Schilit earnings manipulation shenanigan detectors (EMS-1..EMS-8).

Each shenanigan has a dedicated detection method returning an
:class:`EarningsManipulationSignal`.  No generic ``detect(n)``
dispatch — each signal has distinct logic and thresholds.
"""

from __future__ import annotations

from typing import List, Optional

from credit_rating.domain.financial_statements import FinancialStatements
from credit_rating.domain.shenanigans import EarningsManipulationSignal


class EarningsManipulationDetector:
    """Detect Schilit earnings manipulation shenanigans.

    Each method analyses one specific manipulation pattern.

    Args:
        current: Current period financial statements.
        prior: Prior period financial statements (``None`` if
            unavailable; signals requiring comparison will not flag).
    """

    def __init__(
        self,
        current: FinancialStatements,
        prior: Optional[FinancialStatements] = None,
    ) -> None:
        self._current = current
        self._prior = prior

    def detect_all(self) -> List[EarningsManipulationSignal]:
        """Run all eight EMS detectors.

        Returns:
            A list of eight :class:`EarningsManipulationSignal` results.
        """
        return [
            self.detect_ems_1_premature_revenue(),
            self.detect_ems_2_bogus_revenue(),
            self.detect_ems_3_one_time_gains(),
            self.detect_ems_4_shifting_expenses(),
            self.detect_ems_5_improper_capitalization(),
            self.detect_ems_6_extended_depreciation(),
            self.detect_ems_7_understated_liabilities(),
            self.detect_ems_8_revenue_channel_stuffing(),
        ]

    def detect_ems_1_premature_revenue(self) -> EarningsManipulationSignal:
        """EMS-1: Receivables growing faster than revenue.

        Premature revenue recognition inflates receivables relative
        to sales.  Flag when the receivables/revenue ratio rises
        more than 10% year-over-year.
        """
        threshold = 0.10
        cur_ratio = _safe_divide(
            self._current.balance_sheet.net_receivables,
            self._current.income_statement.total_revenue,
        )
        growth = self._ratio_growth(cur_ratio, "receivables_to_revenue")
        return EarningsManipulationSignal(
            signal_id="EMS-1",
            name="Premature Revenue Recognition",
            is_flagged=growth > threshold,
            observed_value=growth,
            threshold=threshold,
            explanation=(
                f"Receivables/revenue ratio grew {growth:.1%} "
                f"(threshold: {threshold:.0%})"
            ),
        )

    def detect_ems_2_bogus_revenue(self) -> EarningsManipulationSignal:
        """EMS-2: Cash collections lagging revenue.

        Revenue recognised but not collected shows up as a gap
        between net income and operating cash flow.  Flag when
        accruals exceed 10% of revenue.
        """
        threshold = 0.10
        accruals = (
            self._current.income_statement.net_income
            - self._current.cash_flow.cash_from_operations
        )
        ratio = _safe_divide(
            accruals,
            self._current.income_statement.total_revenue,
        )
        return EarningsManipulationSignal(
            signal_id="EMS-2",
            name="Bogus Revenue",
            is_flagged=ratio > threshold,
            observed_value=ratio,
            threshold=threshold,
            explanation=(
                f"Accruals/revenue = {ratio:.1%} "
                f"(threshold: {threshold:.0%})"
            ),
        )

    def detect_ems_3_one_time_gains(self) -> EarningsManipulationSignal:
        """EMS-3: Non-operating income boosting earnings.

        Flag when investing cash inflows exceed 20% of operating
        cash flow (suggests asset sales inflating income).
        """
        threshold = 0.20
        investing_ratio = _safe_divide(
            abs(self._current.cash_flow.cash_from_investing),
            max(self._current.cash_flow.cash_from_operations, 1.0),
        )
        flagged = (
            self._current.cash_flow.cash_from_investing > 0
            and investing_ratio > threshold
        )
        return EarningsManipulationSignal(
            signal_id="EMS-3",
            name="One-Time Gains Boosting Income",
            is_flagged=flagged,
            observed_value=investing_ratio,
            threshold=threshold,
            explanation=(
                f"Investing inflows / operating CF = {investing_ratio:.1%} "
                f"(threshold: {threshold:.0%})"
            ),
        )

    def detect_ems_4_shifting_expenses(self) -> EarningsManipulationSignal:
        """EMS-4: Operating expenses declining as a share of revenue.

        Flag when cost-to-income drops more than 5pp vs. prior year,
        suggesting expenses are being deferred.
        """
        threshold = -0.05
        cur_ratio = _safe_divide(
            self._current.income_statement.total_operating_expenses,
            self._current.income_statement.total_revenue,
        )
        change = self._ratio_change(cur_ratio, "opex_to_revenue")
        return EarningsManipulationSignal(
            signal_id="EMS-4",
            name="Shifting Current Expenses to Later Period",
            is_flagged=change < threshold,
            observed_value=change,
            threshold=threshold,
            explanation=(
                f"OpEx/revenue change = {change:+.1%} "
                f"(flag if < {threshold:+.0%})"
            ),
        )

    def detect_ems_5_improper_capitalization(self) -> EarningsManipulationSignal:
        """EMS-5: CapEx growing faster than depreciation.

        When capital expenditures vastly exceed depreciation, the
        firm may be improperly capitalising operating expenses.
        Flag when CapEx/D&A > 2.5.
        """
        threshold = 2.5
        ratio = _safe_divide(
            self._current.cash_flow.capital_expenditures,
            self._current.cash_flow.depreciation_and_amortization,
        )
        return EarningsManipulationSignal(
            signal_id="EMS-5",
            name="Improper Capitalisation of Expenses",
            is_flagged=ratio > threshold,
            observed_value=ratio,
            threshold=threshold,
            explanation=(
                f"CapEx / D&A = {ratio:.2f} (threshold: {threshold})"
            ),
        )

    def detect_ems_6_extended_depreciation(self) -> EarningsManipulationSignal:
        """EMS-6: Depreciation rate declining year-over-year.

        Extending useful lives of assets reduces depreciation and
        inflates earnings.  Flag when depreciation/PPE drops > 15%.
        """
        threshold = -0.15
        dep = self._current.income_statement.depreciation_expense
        ppe = self._current.balance_sheet.net_property_plant_equipment
        cur_rate = _safe_divide(dep, dep + ppe)
        change = self._ratio_change(cur_rate, "depreciation_rate")
        return EarningsManipulationSignal(
            signal_id="EMS-6",
            name="Extended Depreciation / Amortisation",
            is_flagged=change < threshold,
            observed_value=change,
            threshold=threshold,
            explanation=(
                f"Depreciation rate change = {change:+.1%} "
                f"(flag if < {threshold:+.0%})"
            ),
        )

    def detect_ems_7_understated_liabilities(self) -> EarningsManipulationSignal:
        """EMS-7: Payables declining relative to COGS.

        Understating liabilities inflates equity and earnings.
        Flag when payables/COGS drops > 15% vs. prior year.
        """
        threshold = -0.15
        cur_ratio = _safe_divide(
            self._current.balance_sheet.accounts_payable,
            self._current.income_statement.cost_of_goods_sold,
        )
        change = self._ratio_change(cur_ratio, "payables_to_cogs")
        return EarningsManipulationSignal(
            signal_id="EMS-7",
            name="Understated Liabilities",
            is_flagged=change < threshold,
            observed_value=change,
            threshold=threshold,
            explanation=(
                f"Payables/COGS change = {change:+.1%} "
                f"(flag if < {threshold:+.0%})"
            ),
        )

    def detect_ems_8_revenue_channel_stuffing(self) -> EarningsManipulationSignal:
        """EMS-8: Inventory growing faster than revenue.

        Channel stuffing (shipping product to distributors to inflate
        sales) manifests as inventory building faster than revenue.
        Flag when inventory/revenue grows > 10%.
        """
        threshold = 0.10
        cur_ratio = _safe_divide(
            self._current.balance_sheet.inventory,
            self._current.income_statement.total_revenue,
        )
        growth = self._ratio_growth(cur_ratio, "inventory_to_revenue")
        return EarningsManipulationSignal(
            signal_id="EMS-8",
            name="Revenue Channel Stuffing",
            is_flagged=growth > threshold,
            observed_value=growth,
            threshold=threshold,
            explanation=(
                f"Inventory/revenue growth = {growth:.1%} "
                f"(threshold: {threshold:.0%})"
            ),
        )

    # -- Helpers ---------------------------------------------------

    def _ratio_growth(self, current_ratio: float, name: str) -> float:
        """Compute growth of a ratio vs. the prior period."""
        if self._prior is None:
            return 0.0
        prior_ratio = self._prior_ratio(name)
        return _safe_divide(current_ratio - prior_ratio, abs(prior_ratio))

    def _ratio_change(self, current_ratio: float, name: str) -> float:
        """Compute absolute change of a ratio vs. the prior period."""
        if self._prior is None:
            return 0.0
        prior_ratio = self._prior_ratio(name)
        return current_ratio - prior_ratio

    def _prior_ratio(self, name: str) -> float:
        """Look up a prior-period ratio by name."""
        if self._prior is None:
            return 0.0
        p = self._prior
        ratios = {
            "receivables_to_revenue": _safe_divide(
                p.balance_sheet.net_receivables,
                p.income_statement.total_revenue,
            ),
            "opex_to_revenue": _safe_divide(
                p.income_statement.total_operating_expenses,
                p.income_statement.total_revenue,
            ),
            "depreciation_rate": _safe_divide(
                p.income_statement.depreciation_expense,
                p.income_statement.depreciation_expense
                + p.balance_sheet.net_property_plant_equipment,
            ),
            "payables_to_cogs": _safe_divide(
                p.balance_sheet.accounts_payable,
                p.income_statement.cost_of_goods_sold,
            ),
            "inventory_to_revenue": _safe_divide(
                p.balance_sheet.inventory,
                p.income_statement.total_revenue,
            ),
        }
        return ratios.get(name, 0.0)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide with zero-protection."""
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator
