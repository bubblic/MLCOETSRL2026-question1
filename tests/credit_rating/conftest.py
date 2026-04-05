"""Shared pytest fixtures for the credit rating test suite."""

from __future__ import annotations

import pytest

from credit_rating.config.settings import CreditRatingSettings
from credit_rating.domain.financial_statements import (
    BalanceSheet,
    CashFlowStatement,
    FinancialStatements,
    IncomeStatement,
)
from credit_rating.domain.features import (
    EfficiencyRatios,
    FinancialRatios,
    LeverageRatios,
    LiquidityRatios,
    ProfitabilityRatios,
    SizeRatios,
)


@pytest.fixture
def settings() -> CreditRatingSettings:
    """Default test settings."""
    return CreditRatingSettings()


@pytest.fixture
def sample_income_statement() -> IncomeStatement:
    """Apple-inspired income statement (simplified, FY2023)."""
    return IncomeStatement(
        total_revenue=383_285_000_000,
        cost_of_goods_sold=214_137_000_000,
        total_operating_expenses=268_984_000_000,
        selling_general_admin=24_932_000_000,
        depreciation_expense=11_519_000_000,
        interest_expense=3_933_000_000,
        income_tax_expense=16_741_000_000,
        net_income=96_995_000_000,
    )


@pytest.fixture
def sample_balance_sheet() -> BalanceSheet:
    """Apple-inspired balance sheet (simplified, FY2023)."""
    return BalanceSheet(
        cash_and_equivalents=29_965_000_000,
        short_term_investments=31_590_000_000,
        net_receivables=60_985_000_000,
        inventory=6_331_000_000,
        total_current_assets=143_566_000_000,
        net_property_plant_equipment=43_715_000_000,
        total_assets=352_583_000_000,
        accounts_payable=62_611_000_000,
        short_term_debt=15_807_000_000,
        total_current_liabilities=145_308_000_000,
        long_term_debt=95_281_000_000,
        total_liabilities=290_437_000_000,
        total_equity=62_146_000_000,
        retained_earnings=-214_000_000,
    )


@pytest.fixture
def sample_cash_flow() -> CashFlowStatement:
    """Apple-inspired cash flow statement (simplified, FY2023)."""
    return CashFlowStatement(
        depreciation_and_amortization=11_519_000_000,
        cash_from_operations=110_543_000_000,
        capital_expenditures=10_959_000_000,
        cash_from_investing=-7_077_000_000,
        cash_from_financing=-108_488_000_000,
        net_change_in_cash=-5_022_000_000,
    )


@pytest.fixture
def sample_statements(
    sample_income_statement,
    sample_balance_sheet,
    sample_cash_flow,
) -> FinancialStatements:
    """Composite financial statements fixture."""
    return FinancialStatements(
        income_statement=sample_income_statement,
        balance_sheet=sample_balance_sheet,
        cash_flow=sample_cash_flow,
        fiscal_year=2023,
        currency="USD",
    )


@pytest.fixture
def sample_prior_statements() -> FinancialStatements:
    """Prior-year statements for growth / Beneish calculations."""
    return FinancialStatements(
        income_statement=IncomeStatement(
            total_revenue=394_328_000_000,
            cost_of_goods_sold=223_546_000_000,
            total_operating_expenses=274_891_000_000,
            selling_general_admin=25_094_000_000,
            depreciation_expense=11_104_000_000,
            interest_expense=2_931_000_000,
            income_tax_expense=19_300_000_000,
            net_income=99_803_000_000,
        ),
        balance_sheet=BalanceSheet(
            cash_and_equivalents=23_646_000_000,
            short_term_investments=24_658_000_000,
            net_receivables=60_932_000_000,
            inventory=4_946_000_000,
            total_current_assets=135_405_000_000,
            net_property_plant_equipment=42_117_000_000,
            total_assets=352_755_000_000,
            accounts_payable=64_115_000_000,
            short_term_debt=11_128_000_000,
            total_current_liabilities=153_982_000_000,
            long_term_debt=98_959_000_000,
            total_liabilities=302_083_000_000,
            total_equity=50_672_000_000,
            retained_earnings=-3_068_000_000,
        ),
        cash_flow=CashFlowStatement(
            depreciation_and_amortization=11_104_000_000,
            cash_from_operations=122_151_000_000,
            capital_expenditures=10_708_000_000,
            cash_from_investing=-22_354_000_000,
            cash_from_financing=-110_749_000_000,
            net_change_in_cash=-10_952_000_000,
        ),
        fiscal_year=2022,
        currency="USD",
    )


@pytest.fixture
def sample_ratios() -> FinancialRatios:
    """Pre-computed sample ratios for testing."""
    return FinancialRatios(
        leverage=LeverageRatios(
            debt_to_equity=1.79,
            debt_to_assets=0.31,
            debt_to_capital=0.64,
            debt_to_ebitda=0.91,
            interest_coverage=29.88,
        ),
        liquidity=LiquidityRatios(
            current_ratio=0.99,
            quick_ratio=0.84,
            cash_ratio=0.21,
            working_capital_to_assets=-0.005,
            operating_cash_flow_ratio=0.76,
        ),
        profitability=ProfitabilityRatios(
            gross_margin=0.44,
            operating_margin=0.30,
            net_margin=0.25,
            return_on_assets=0.28,
            return_on_equity=1.56,
        ),
        efficiency=EfficiencyRatios(
            asset_turnover=1.09,
            receivables_turnover=6.28,
            inventory_turnover=33.83,
            payables_turnover=3.42,
            cost_to_income=0.70,
        ),
        size=SizeRatios(
            log_total_assets=26.59,
            log_total_revenue=26.67,
            revenue_growth=-0.028,
            asset_growth=-0.0005,
            retained_earnings_to_assets=-0.0006,
        ),
    )
