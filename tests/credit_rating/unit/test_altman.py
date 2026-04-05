"""Tests for Altman Z-Score and Z''-Score calculators."""

from __future__ import annotations

import pytest

from credit_rating.config.settings import AltmanZone
from credit_rating.features.altman import AltmanZScoreCalculator


class TestAltmanZScore:
    """Verify Z-Score computation and zone classification."""

    @pytest.fixture
    def calculator(self):
        return AltmanZScoreCalculator()

    def test_z_score_positive_for_healthy_company(
        self, calculator, sample_statements,
    ):
        z = calculator.calculate_z(sample_statements)
        assert z > 0

    def test_z_double_prime_positive(
        self, calculator, sample_statements,
    ):
        zpp = calculator.calculate_z_prime_prime(sample_statements)
        assert zpp > 0

    def test_classify_safe_zone(self, calculator):
        assert calculator.classify_z(4.0) == AltmanZone.SAFE

    def test_classify_grey_zone(self, calculator):
        assert calculator.classify_z(2.5) == AltmanZone.GREY

    def test_classify_distress_zone(self, calculator):
        assert calculator.classify_z(1.0) == AltmanZone.DISTRESS

    def test_classify_z_prime_prime_safe(self, calculator):
        assert calculator.classify_z_prime_prime(3.0) == AltmanZone.SAFE

    def test_classify_z_prime_prime_distress(self, calculator):
        assert calculator.classify_z_prime_prime(0.5) == AltmanZone.DISTRESS

    def test_zero_total_assets_returns_zero(self, calculator):
        from credit_rating.domain.financial_statements import (
            BalanceSheet, CashFlowStatement, FinancialStatements,
            IncomeStatement,
        )

        zero = FinancialStatements(
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
        assert calculator.calculate_z(zero) == 0.0
        assert calculator.calculate_z_prime_prime(zero) == 0.0
