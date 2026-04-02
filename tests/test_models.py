"""Tests for Pydantic models in the risk extraction pipeline."""

import pytest
from pydantic import ValidationError

from financial_forecast.extraction.risk.models import (
    AuditorOpinionResult,
    CategoryResult,
    CashFlowWarning,
    ContingentLiability,
    FlaggedPage,
    GoingConcernResult,
    MdaRedFlag,
    MdaRedFlagsResult,
    RiskMemoOutput,
)


class TestFlaggedPage:
    def test_valid_data(self):
        fp = FlaggedPage(page=12, category="going_concern", reason="material uncertainty")
        assert fp.page == 12
        assert fp.category == "going_concern"
        assert fp.reason == "material uncertainty"

    def test_missing_reason_uses_default(self):
        fp = FlaggedPage(page=5, category="debt_covenants")
        assert fp.reason == ""

    def test_invalid_page_type_raises(self):
        with pytest.raises(ValidationError):
            FlaggedPage(page="not_a_number", category="going_concern")

    def test_coerce_string_page_number(self):
        fp = FlaggedPage(page="42", category="going_concern")
        assert fp.page == 42


class TestCategoryModels:
    def test_auditor_opinion_all_none(self):
        result = AuditorOpinionResult()
        assert result.opinion_type is None
        assert result.qualification_basis is None

    def test_going_concern_with_data(self):
        result = GoingConcernResult(
            disclosure_text="Material uncertainty exists",
            time_period="12 months",
            mitigating_actions="Asset disposal programme",
        )
        assert result.disclosure_text == "Material uncertainty exists"

    def test_contingent_liability_optional_fields(self):
        cl = ContingentLiability(nature="Lawsuit")
        assert cl.nature == "Lawsuit"
        assert cl.exposure is None
        assert cl.likelihood is None

    def test_cash_flow_warning_divergence_flag(self):
        cfw = CashFlowWarning(divergence_flag=True)
        assert cfw.divergence_flag is True

    def test_mda_red_flag(self):
        flag = MdaRedFlag(
            description="Vague language",
            severity="high",
            quoted_text="results were impacted by macro conditions",
        )
        assert flag.severity == "high"

    def test_mda_red_flags_result_empty(self):
        result = MdaRedFlagsResult()
        assert result.red_flags == []


class TestCategoryResult:
    def test_valid_dict_extraction(self):
        cr = CategoryResult(
            category="going_concern",
            extraction={"disclosure_text": "test"},
        )
        assert cr.category == "going_concern"
        assert cr.extraction["disclosure_text"] == "test"

    def test_valid_list_extraction(self):
        cr = CategoryResult(
            category="contingent_liabilities",
            extraction=[{"nature": "Lawsuit"}, {"nature": "Regulatory"}],
        )
        assert len(cr.extraction) == 2

    def test_none_extraction(self):
        cr = CategoryResult(category="debt_covenants", extraction=None)
        assert cr.extraction is None

    def test_rejects_non_dict_non_list_extraction(self):
        """extraction: Any would have accepted this; the Union type does not."""
        with pytest.raises(ValidationError):
            CategoryResult(category="test", extraction="a plain string")


class TestRiskMemoOutput:
    def test_default_empty_memo(self):
        memo = RiskMemoOutput()
        assert memo.risk_memo == ""

    def test_memo_with_text(self):
        memo = RiskMemoOutput(risk_memo="HIGH RISK: Going concern doubt.")
        assert "HIGH RISK" in memo.risk_memo
