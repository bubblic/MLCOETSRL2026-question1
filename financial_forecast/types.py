"""Shared dataclasses for financial state and economic inputs.

This module consolidates the ``FinancialState`` and ``EconomicInputs``
dataclasses that were previously duplicated across multiple modules.  All
model variants import from here to ensure a single source of truth.

Why TensorFlow types?
    Field values are typed as ``Any`` so they can hold either plain floats
    (for initial data) or ``tf.Tensor`` objects (for gradient-tracked
    computations).  TensorFlow tensors enable automatic differentiation
    via ``tf.GradientTape`` and GPU-accelerated arithmetic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class FinancialState:
    """Represents the financial state of a company at a specific point in time.

    All monetary values are typically ``tf.Tensor`` (float64) or plain
    ``float``.  The frozen dataclass ensures immutability, which is
    important when states are passed through multi-step forecast chains.

    Attributes:
        nca: Non-current assets.
        advance_payments_purchases: Prepayments for future purchases.
        accounts_receivable: Outstanding customer receivables.
        inventory: Current inventory holdings.
        cash: Cash and cash equivalents.
        investment_in_market_securities: Short-term marketable securities.
        accounts_payable: Outstanding supplier payables.
        advance_payments_sales: Deferred revenue from customer prepayments.
        current_liabilities: Current liabilities (excl. AP and deferred rev).
        non_current_liabilities: Long-term liabilities.
        equity: Stockholders' equity.
        net_income: Net income for the period.
        liquidity_check: Diagnostic — should be near zero if budget closes.
        balance_sheet_check: Diagnostic — Assets minus (Liabilities + Equity).
        st_loan_issued: New short-term debt issued this period.
        lt_loan_issued: New long-term debt issued this period.
        st_principal_paid: Short-term principal repaid this period.
        lt_principal_paid: Long-term principal repaid this period.
    """

    nca: Any
    advance_payments_purchases: Any
    accounts_receivable: Any
    inventory: Any
    cash: Any
    investment_in_market_securities: Any
    accounts_payable: Any
    advance_payments_sales: Any
    current_liabilities: Any
    non_current_liabilities: Any
    equity: Any
    net_income: Any

    # Diagnostic fields
    liquidity_check: Any = 0.0
    balance_sheet_check: Any = 0.0
    st_loan_issued: Any = 0.0
    lt_loan_issued: Any = 0.0
    st_principal_paid: Any = 0.0
    lt_principal_paid: Any = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FinancialState:
        """Create a ``FinancialState`` from a dictionary.

        Supports legacy short-form keys (e.g. ``"ar"`` for
        ``"accounts_receivable"``) and silently ignores unknown keys.

        Args:
            data: Dictionary of field values, possibly with legacy keys.

        Returns:
            A new ``FinancialState`` instance.
        """
        mapping = {
            "adv_pp": "advance_payments_purchases",
            "adv_ps": "advance_payments_sales",
            "ar": "accounts_receivable",
            "inv": "inventory",
            "ap": "accounts_payable",
            "cl": "current_liabilities",
            "ncl": "non_current_liabilities",
            "ni": "net_income",
            "ims": "investment_in_market_securities",
            # Legacy field name from SimpleFinancialModel
            "check": "balance_sheet_check",
            "stloan": "st_loan_issued",
            "ltloan": "lt_loan_issued",
        }
        mapped_data = {mapping.get(k, k): v for k, v in data.items()}
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in mapped_data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass(frozen=True)
class EconomicInputs:
    """External economic drivers for a single forecast period.

    Attributes:
        sales_t: Revenue for the current period.
        purchases_t: Purchases for the current period.
        sales_t_plus_1: Revenue forecast for the next period (used for
            advance-payment calculations).
        purchases_t_plus_1: Purchases forecast for the next period.
        cum_inflation: Cumulative inflation factor from base year to *t*.
        t: Period index (integer year offset from base).
    """

    sales_t: Any
    purchases_t: Any
    sales_t_plus_1: Any
    purchases_t_plus_1: Any
    cum_inflation: Any
    t: int = 0
