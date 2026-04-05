"""Tests for the FinancialRatioCalculator — every ratio formula."""

from __future__ import annotations

import math

import pytest

from credit_rating.features.ratio_calculator import FinancialRatioCalculator


class TestRatioCalculator:
    """Verify each ratio against hand-calculated expected values."""

    @pytest.fixture
    def calculator(self):
        return FinancialRatioCalculator()

    def test_calculate_returns_25_features(
        self, calculator, sample_statements,
    ):
        ratios = calculator.calculate(sample_statements)
        assert len(ratios) == 25

    def test_debt_to_equity(self, calculator, sample_statements):
        ratios = calculator.calculate(sample_statements)
        bs = sample_statements.balance_sheet
        expected = bs.total_debt() / bs.total_equity
        assert ratios.leverage.debt_to_equity == pytest.approx(expected)

    def test_debt_to_assets(self, calculator, sample_statements):
        ratios = calculator.calculate(sample_statements)
        bs = sample_statements.balance_sheet
        expected = bs.total_debt() / bs.total_assets
        assert ratios.leverage.debt_to_assets == pytest.approx(expected)

    def test_interest_coverage(self, calculator, sample_statements):
        ratios = calculator.calculate(sample_statements)
        inc = sample_statements.income_statement
        expected = inc.computed_ebit() / inc.interest_expense
        assert ratios.leverage.interest_coverage == pytest.approx(expected)

    def test_current_ratio(self, calculator, sample_statements):
        ratios = calculator.calculate(sample_statements)
        bs = sample_statements.balance_sheet
        expected = bs.total_current_assets / bs.total_current_liabilities
        assert ratios.liquidity.current_ratio == pytest.approx(expected)

    def test_quick_ratio(self, calculator, sample_statements):
        ratios = calculator.calculate(sample_statements)
        bs = sample_statements.balance_sheet
        quick = bs.cash_and_equivalents + bs.short_term_investments + bs.net_receivables
        expected = quick / bs.total_current_liabilities
        assert ratios.liquidity.quick_ratio == pytest.approx(expected)

    def test_gross_margin(self, calculator, sample_statements):
        ratios = calculator.calculate(sample_statements)
        inc = sample_statements.income_statement
        expected = inc.computed_gross_profit() / inc.total_revenue
        assert ratios.profitability.gross_margin == pytest.approx(expected)

    def test_net_margin(self, calculator, sample_statements):
        ratios = calculator.calculate(sample_statements)
        inc = sample_statements.income_statement
        expected = inc.net_income / inc.total_revenue
        assert ratios.profitability.net_margin == pytest.approx(expected)

    def test_asset_turnover(self, calculator, sample_statements):
        ratios = calculator.calculate(sample_statements)
        expected = (
            sample_statements.income_statement.total_revenue
            / sample_statements.balance_sheet.total_assets
        )
        assert ratios.efficiency.asset_turnover == pytest.approx(expected)

    def test_log_total_assets(self, calculator, sample_statements):
        ratios = calculator.calculate(sample_statements)
        expected = math.log(sample_statements.balance_sheet.total_assets)
        assert ratios.size.log_total_assets == pytest.approx(expected)

    def test_revenue_growth_with_prior(
        self, calculator, sample_statements, sample_prior_statements,
    ):
        ratios = calculator.calculate(sample_statements, sample_prior_statements)
        cur = sample_statements.income_statement.total_revenue
        pri = sample_prior_statements.income_statement.total_revenue
        expected = (cur - pri) / abs(pri)
        assert ratios.size.revenue_growth == pytest.approx(expected)

    def test_revenue_growth_without_prior(
        self, calculator, sample_statements,
    ):
        ratios = calculator.calculate(sample_statements)
        assert ratios.size.revenue_growth == 0.0

    def test_zero_denominator_safety(self, calculator):
        """Ratios should be 0 when denominators are zero, not NaN."""
        from credit_rating.domain.financial_statements import (
            BalanceSheet,
            CashFlowStatement,
            FinancialStatements,
            IncomeStatement,
        )

        zero_stmts = FinancialStatements(
            income_statement=IncomeStatement(
                total_revenue=0, cost_of_goods_sold=0,
                total_operating_expenses=0, selling_general_admin=0,
                depreciation_expense=0, interest_expense=0,
                income_tax_expense=0, net_income=0,
            ),
            balance_sheet=BalanceSheet(
                cash_and_equivalents=0, short_term_investments=0,
                net_receivables=0, inventory=0, total_current_assets=0,
                net_property_plant_equipment=0, total_assets=0,
                accounts_payable=0, short_term_debt=0,
                total_current_liabilities=0, long_term_debt=0,
                total_liabilities=0, total_equity=0, retained_earnings=0,
            ),
            cash_flow=CashFlowStatement(
                depreciation_and_amortization=0, cash_from_operations=0,
                capital_expenditures=0, cash_from_investing=0,
                cash_from_financing=0, net_change_in_cash=0,
            ),
            fiscal_year=2023,
        )
        ratios = calculator.calculate(zero_stmts)
        values = list(ratios)
        assert all(not math.isnan(v) for v in values)
        assert all(not math.isinf(v) for v in values)
