"""Financial ratio dataclasses that serve as the structured feature vector.

Five ratio groups — leverage, liquidity, profitability, efficiency, and
size/growth — compose into :class:`FinancialRatios`, the 25-dimensional
feature vector consumed by the structured tower.

``FinancialRatios`` implements ``__iter__``, ``__len__``, and
``to_tensor()`` so it behaves as a native Python sequence and converts
directly to a TensorFlow tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterator


@dataclass(frozen=True)
class LeverageRatios:
    """Debt burden and capital structure indicators.

    Args:
        debt_to_equity: Total debt / total equity.
        debt_to_assets: Total debt / total assets.
        debt_to_capital: Total debt / (total debt + total equity).
        debt_to_ebitda: Total debt / EBITDA.
        interest_coverage: EBIT / interest expense.
    """

    debt_to_equity: float
    debt_to_assets: float
    debt_to_capital: float
    debt_to_ebitda: float
    interest_coverage: float


@dataclass(frozen=True)
class LiquidityRatios:
    """Short-term solvency indicators.

    Args:
        current_ratio: Current assets / current liabilities.
        quick_ratio: (Cash + ST investments + receivables) / current liabilities.
        cash_ratio: Cash and equivalents / current liabilities.
        working_capital_to_assets: Working capital / total assets.
        operating_cash_flow_ratio: Operating cash flow / current liabilities.
    """

    current_ratio: float
    quick_ratio: float
    cash_ratio: float
    working_capital_to_assets: float
    operating_cash_flow_ratio: float


@dataclass(frozen=True)
class ProfitabilityRatios:
    """Earnings quality and margin indicators.

    Args:
        gross_margin: Gross profit / revenue.
        operating_margin: EBIT / revenue.
        net_margin: Net income / revenue.
        return_on_assets: Net income / total assets.
        return_on_equity: Net income / total equity.
    """

    gross_margin: float
    operating_margin: float
    net_margin: float
    return_on_assets: float
    return_on_equity: float


@dataclass(frozen=True)
class EfficiencyRatios:
    """Asset utilisation and turnover indicators.

    Args:
        asset_turnover: Revenue / total assets.
        receivables_turnover: Revenue / net receivables.
        inventory_turnover: COGS / inventory.
        payables_turnover: COGS / accounts payable.
        cost_to_income: Total operating expenses / revenue.
    """

    asset_turnover: float
    receivables_turnover: float
    inventory_turnover: float
    payables_turnover: float
    cost_to_income: float


@dataclass(frozen=True)
class SizeRatios:
    """Scale and growth indicators.

    Args:
        log_total_assets: Natural log of total assets.
        log_total_revenue: Natural log of total revenue.
        revenue_growth: Year-over-year revenue growth rate.
        asset_growth: Year-over-year asset growth rate.
        retained_earnings_to_assets: Retained earnings / total assets.
    """

    log_total_assets: float
    log_total_revenue: float
    revenue_growth: float
    asset_growth: float
    retained_earnings_to_assets: float


RATIO_GROUP_ORDER = (
    "leverage",
    "liquidity",
    "profitability",
    "efficiency",
    "size",
)

FEATURE_NAMES: tuple[str, ...] = (
    # Leverage
    "debt_to_equity",
    "debt_to_assets",
    "debt_to_capital",
    "debt_to_ebitda",
    "interest_coverage",
    # Liquidity
    "current_ratio",
    "quick_ratio",
    "cash_ratio",
    "working_capital_to_assets",
    "operating_cash_flow_ratio",
    # Profitability
    "gross_margin",
    "operating_margin",
    "net_margin",
    "return_on_assets",
    "return_on_equity",
    # Efficiency
    "asset_turnover",
    "receivables_turnover",
    "inventory_turnover",
    "payables_turnover",
    "cost_to_income",
    # Size
    "log_total_assets",
    "log_total_revenue",
    "revenue_growth",
    "asset_growth",
    "retained_earnings_to_assets",
)

NUM_FEATURES: int = len(FEATURE_NAMES)


@dataclass(frozen=True)
class FinancialRatios:
    """Composite of all five ratio groups — the 25-feature vector.

    Implements the sequence protocol (``__iter__``, ``__len__``,
    ``__getitem__``, ``__contains__``) so callers can iterate over
    ratio values directly.  ``to_tensor()`` converts to a rank-1
    ``tf.Tensor`` for model consumption.

    Args:
        leverage: Leverage ratio group.
        liquidity: Liquidity ratio group.
        profitability: Profitability ratio group.
        efficiency: Efficiency ratio group.
        size: Size/growth ratio group.
    """

    leverage: LeverageRatios
    liquidity: LiquidityRatios
    profitability: ProfitabilityRatios
    efficiency: EfficiencyRatios
    size: SizeRatios

    def to_flat_list(self) -> list[float]:
        """Return all 25 ratio values in canonical order.

        Returns:
            A list of floats in the order defined by
            :data:`FEATURE_NAMES`.
        """
        values: list[float] = []
        for group in (
            self.leverage,
            self.liquidity,
            self.profitability,
            self.efficiency,
            self.size,
        ):
            for field in fields(group):
                values.append(getattr(group, field.name))
        return values

    def to_tensor(self) -> "tf.Tensor":
        """Convert to a rank-1 ``tf.Tensor`` of dtype ``float32``.

        Returns:
            A 1-D tensor with 25 elements.
        """
        import tensorflow as tf

        return tf.constant(self.to_flat_list(), dtype=tf.float32)

    def to_dict(self) -> dict[str, float]:
        """Return a ``{feature_name: value}`` mapping.

        Returns:
            Ordered dictionary matching :data:`FEATURE_NAMES`.
        """
        return dict(zip(FEATURE_NAMES, self.to_flat_list()))

    def __iter__(self) -> Iterator[float]:
        """Iterate over the 25 ratio values in canonical order."""
        return iter(self.to_flat_list())

    def __len__(self) -> int:
        """Return the number of features (always 25)."""
        return NUM_FEATURES

    def __getitem__(self, index: int) -> float:
        """Return the ratio value at *index* in canonical order."""
        return self.to_flat_list()[index]

    def __contains__(self, value: object) -> bool:
        """Check whether *value* appears in the ratio values."""
        return value in self.to_flat_list()
