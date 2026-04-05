"""Compute the 25-dimensional financial ratio feature vector.

Each of the five ratio groups (leverage, liquidity, profitability,
efficiency, size/growth) is computed by a dedicated private method.
All computations are pure functions: given the same
:class:`FinancialStatements`, the result is always identical.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from credit_rating.domain.features import (
    EfficiencyRatios,
    FinancialRatios,
    LeverageRatios,
    LiquidityRatios,
    ProfitabilityRatios,
    SizeRatios,
)
from credit_rating.domain.financial_statements import FinancialStatements

logger = logging.getLogger(__name__)


class FinancialRatioCalculator:
    """Calculate all 25 financial ratios from a :class:`FinancialStatements`.

    Usage::

        calculator = FinancialRatioCalculator()
        ratios = calculator.calculate(statements)
    """

    def calculate(
        self,
        statements: FinancialStatements,
        prior_statements: Optional[FinancialStatements] = None,
    ) -> FinancialRatios:
        """Compute the full 25-ratio feature vector.

        Args:
            statements: Current period financial statements.
            prior_statements: Prior period statements for growth
                ratios. If ``None``, growth ratios default to ``0.0``.

        Returns:
            A frozen :class:`FinancialRatios` instance.
        """
        return FinancialRatios(
            leverage=self._leverage(statements),
            liquidity=self._liquidity(statements),
            profitability=self._profitability(statements),
            efficiency=self._efficiency(statements),
            size=self._size(statements, prior_statements),
        )

    @staticmethod
    def _leverage(stmts: FinancialStatements) -> LeverageRatios:
        """Compute the five leverage ratios."""
        bs = stmts.balance_sheet
        inc = stmts.income_statement
        total_debt = bs.total_debt()
        ebitda = inc.computed_ebit() + stmts.cash_flow.depreciation_and_amortization

        return LeverageRatios(
            debt_to_equity=_safe_divide(total_debt, bs.total_equity),
            debt_to_assets=_safe_divide(total_debt, bs.total_assets),
            debt_to_capital=_safe_divide(
                total_debt, total_debt + bs.total_equity,
            ),
            debt_to_ebitda=_safe_divide(total_debt, ebitda),
            interest_coverage=_safe_divide(
                inc.computed_ebit(), inc.interest_expense,
            ),
        )

    @staticmethod
    def _liquidity(stmts: FinancialStatements) -> LiquidityRatios:
        """Compute the five liquidity ratios."""
        bs = stmts.balance_sheet
        quick_assets = (
            bs.cash_and_equivalents
            + bs.short_term_investments
            + bs.net_receivables
        )

        return LiquidityRatios(
            current_ratio=_safe_divide(
                bs.total_current_assets, bs.total_current_liabilities,
            ),
            quick_ratio=_safe_divide(
                quick_assets, bs.total_current_liabilities,
            ),
            cash_ratio=_safe_divide(
                bs.cash_and_equivalents, bs.total_current_liabilities,
            ),
            working_capital_to_assets=_safe_divide(
                bs.computed_working_capital(), bs.total_assets,
            ),
            operating_cash_flow_ratio=_safe_divide(
                stmts.cash_flow.cash_from_operations,
                bs.total_current_liabilities,
            ),
        )

    @staticmethod
    def _profitability(stmts: FinancialStatements) -> ProfitabilityRatios:
        """Compute the five profitability ratios."""
        inc = stmts.income_statement
        bs = stmts.balance_sheet

        return ProfitabilityRatios(
            gross_margin=_safe_divide(
                inc.computed_gross_profit(), inc.total_revenue,
            ),
            operating_margin=_safe_divide(
                inc.computed_ebit(), inc.total_revenue,
            ),
            net_margin=_safe_divide(inc.net_income, inc.total_revenue),
            return_on_assets=_safe_divide(inc.net_income, bs.total_assets),
            return_on_equity=_safe_divide(inc.net_income, bs.total_equity),
        )

    @staticmethod
    def _efficiency(stmts: FinancialStatements) -> EfficiencyRatios:
        """Compute the five efficiency ratios."""
        inc = stmts.income_statement
        bs = stmts.balance_sheet

        return EfficiencyRatios(
            asset_turnover=_safe_divide(inc.total_revenue, bs.total_assets),
            receivables_turnover=_safe_divide(
                inc.total_revenue, bs.net_receivables,
            ),
            inventory_turnover=_safe_divide(
                inc.cost_of_goods_sold, bs.inventory,
            ),
            payables_turnover=_safe_divide(
                inc.cost_of_goods_sold, bs.accounts_payable,
            ),
            cost_to_income=_safe_divide(
                inc.total_operating_expenses, inc.total_revenue,
            ),
        )

    @staticmethod
    def _size(
        stmts: FinancialStatements,
        prior: Optional[FinancialStatements],
    ) -> SizeRatios:
        """Compute the five size/growth ratios."""
        bs = stmts.balance_sheet
        inc = stmts.income_statement

        revenue_growth = 0.0
        asset_growth = 0.0
        if prior is not None:
            revenue_growth = _safe_growth(
                inc.total_revenue,
                prior.income_statement.total_revenue,
            )
            asset_growth = _safe_growth(
                bs.total_assets,
                prior.balance_sheet.total_assets,
            )

        return SizeRatios(
            log_total_assets=_safe_log(bs.total_assets),
            log_total_revenue=_safe_log(inc.total_revenue),
            revenue_growth=revenue_growth,
            asset_growth=asset_growth,
            retained_earnings_to_assets=_safe_divide(
                bs.retained_earnings, bs.total_assets,
            ),
        )


# ------------------------------------------------------------------
# Arithmetic helpers
# ------------------------------------------------------------------


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide *numerator* by *denominator*, returning 0.0 on zero division."""
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _safe_log(value: float) -> float:
    """Return ``ln(value)`` or ``0.0`` for non-positive inputs."""
    if value <= 0:
        return 0.0
    return math.log(value)


def _safe_growth(current: float, prior: float) -> float:
    """Compute ``(current - prior) / |prior|``, returning 0 if prior is 0."""
    if abs(prior) < 1e-12:
        return 0.0
    return (current - prior) / abs(prior)
