"""Immutable data containers for the three core financial statements.

Each statement is a frozen dataclass holding the fields required for
ratio calculation, Altman Z-Score, and Beneish M-Score analysis.
``FinancialStatements`` composes all three into a single object that
flows through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IncomeStatement:
    """Key line items from a consolidated income statement.

    Args:
        total_revenue: Net revenue / net sales.
        cost_of_goods_sold: Direct costs attributable to goods sold.
        total_operating_expenses: Total operating costs including COGS.
        selling_general_admin: SG&A expenses.
        depreciation_expense: Depreciation charged to the income statement.
        interest_expense: Interest on outstanding debt.
        income_tax_expense: Provision for income taxes.
        net_income: Bottom-line profit after all charges.
        gross_profit: Revenue minus COGS (derived if not reported).
        ebit: Earnings before interest and taxes.
    """

    total_revenue: float
    cost_of_goods_sold: float
    total_operating_expenses: float
    selling_general_admin: float
    depreciation_expense: float
    interest_expense: float
    income_tax_expense: float
    net_income: float
    gross_profit: Optional[float] = None
    ebit: Optional[float] = None

    def computed_gross_profit(self) -> float:
        """Return explicit gross profit or derive from revenue - COGS."""
        if self.gross_profit is not None:
            return self.gross_profit
        return self.total_revenue - self.cost_of_goods_sold

    def computed_ebit(self) -> float:
        """Return explicit EBIT or derive from net income + interest + tax."""
        if self.ebit is not None:
            return self.ebit
        return self.net_income + self.interest_expense + self.income_tax_expense


@dataclass(frozen=True)
class BalanceSheet:
    """Key line items from a consolidated balance sheet.

    Args:
        cash_and_equivalents: Cash and cash equivalents.
        short_term_investments: Short-term marketable securities.
        net_receivables: Net accounts receivable.
        inventory: Total inventory.
        total_current_assets: Sum of all current assets.
        net_property_plant_equipment: PP&E net of depreciation.
        total_assets: Total assets.
        accounts_payable: Accounts payable.
        short_term_debt: Current portion of debt.
        total_current_liabilities: Sum of all current liabilities.
        long_term_debt: Non-current debt obligations.
        total_liabilities: Total liabilities.
        total_equity: Total shareholders' equity.
        retained_earnings: Accumulated retained earnings.
        working_capital: Current assets minus current liabilities.
    """

    cash_and_equivalents: float
    short_term_investments: float
    net_receivables: float
    inventory: float
    total_current_assets: float
    net_property_plant_equipment: float
    total_assets: float
    accounts_payable: float
    short_term_debt: float
    total_current_liabilities: float
    long_term_debt: float
    total_liabilities: float
    total_equity: float
    retained_earnings: float
    working_capital: Optional[float] = None

    def computed_working_capital(self) -> float:
        """Return explicit working capital or derive from CA - CL."""
        if self.working_capital is not None:
            return self.working_capital
        return self.total_current_assets - self.total_current_liabilities

    def total_debt(self) -> float:
        """Return combined short-term and long-term debt."""
        return self.short_term_debt + self.long_term_debt

    def market_value_equity_proxy(self) -> float:
        """Use book equity as a proxy when market cap is unavailable."""
        return self.total_equity


@dataclass(frozen=True)
class CashFlowStatement:
    """Key line items from a consolidated cash flow statement.

    Args:
        depreciation_and_amortization: D&A from operating activities.
        cash_from_operations: Net cash provided by operating activities.
        capital_expenditures: Purchases of PP&E (reported as positive).
        cash_from_investing: Net cash used in investing activities.
        cash_from_financing: Net cash from financing activities.
        net_change_in_cash: Total change in cash for the period.
    """

    depreciation_and_amortization: float
    cash_from_operations: float
    capital_expenditures: float
    cash_from_investing: float
    cash_from_financing: float
    net_change_in_cash: float

    def free_cash_flow(self) -> float:
        """Operating cash flow minus capital expenditures."""
        return self.cash_from_operations - self.capital_expenditures


@dataclass(frozen=True)
class FinancialStatements:
    """Composite of all three statements for a single fiscal period.

    Args:
        income_statement: The income statement for the period.
        balance_sheet: The balance sheet at period end.
        cash_flow: The cash flow statement for the period.
        fiscal_year: The fiscal year these statements cover.
        currency: The reporting currency (e.g. ``"USD"``).
    """

    income_statement: IncomeStatement
    balance_sheet: BalanceSheet
    cash_flow: CashFlowStatement
    fiscal_year: int
    currency: str = "USD"
