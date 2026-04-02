"""Tests for category_extraction module."""

from typing import Any, Dict
from unittest.mock import Mock

from financial_forecast.clients.protocols import LLMClient
from financial_forecast.extraction.risk.category_extraction import (
    extract_all_categories,
    extract_category,
)
from financial_forecast.extraction.risk.risk_categories import RiskCategory
from financial_forecast.extraction.risk.usage_tracker import UsageTracker


class FakeLLMClient:
    """Typed test double that satisfies the LLMClient Protocol.

    Unlike ``Mock()``, a type checker can verify that this class
    conforms to the ``LLMClient`` protocol at definition time.
    """

    def __init__(self, json_response: Any = None) -> None:
        self.json_response = json_response
        self.call_count = 0

    def ask_json(
        self,
        message: str,
        prompt: str,
        parameters: Dict[str, Any],
        reasoning: bool = True,
    ) -> Dict[str, Any]:
        self.call_count += 1
        return self.json_response

    def ask_text(
        self,
        message: str,
        prompt: str,
        parameters: Dict[str, Any],
        reasoning: bool = True,
    ) -> str:
        self.call_count += 1
        return str(self.json_response)


# Verify at import time that FakeLLMClient satisfies the Protocol.
assert isinstance(FakeLLMClient(), LLMClient)


SAMPLE_PAGES = {
    10: "Page 10: The auditor issued a qualified opinion...",
    11: "Page 11: Management acknowledges material uncertainty...",
    20: "Page 20: Contingent liabilities include a pending lawsuit...",
    30: "Page 30: The company has breached its debt covenants...",
}


class TestExtractCategory:
    def test_extract_auditor_opinion(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = {
            "opinion_type": "qualified",
            "qualification_basis": "going concern",
            "emphasis_of_matter": None,
            "going_concern_language": "material uncertainty",
        }
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.AUDITOR_OPINION, [10, 11], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["category"] == "auditor_opinion"
        assert result["extraction"]["opinion_type"] == "qualified"
        mock_client.ask_json.assert_called_once()

    def test_extract_going_concern(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = {
            "disclosure_text": "Material uncertainty exists",
            "time_period": "12 months",
            "mitigating_actions": "Asset disposal",
        }
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.GOING_CONCERN, [11], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["category"] == "going_concern"
        assert result["extraction"]["time_period"] == "12 months"

    def test_extract_contingent_liabilities(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = [
            {"nature": "Lawsuit", "exposure": "$500M", "likelihood": "probable", "timeline": "2024"},
        ]
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.CONTINGENT_LIABILITIES, [20], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["category"] == "contingent_liabilities"
        assert result["extraction"][0]["nature"] == "Lawsuit"

    def test_extract_debt_covenants(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = [
            {"covenant_type": "leverage", "condition": "< 4x", "compliance_status": "breached", "headroom": None},
        ]
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.DEBT_COVENANTS, [30], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["extraction"][0]["compliance_status"] == "breached"

    def test_extract_related_party(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = [
            {"counterparty": "CEO", "relationship": "director", "transaction_nature": "loan",
             "value": "$10M", "independent_approval": "no"},
        ]
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.RELATED_PARTY, [10], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["extraction"][0]["counterparty"] == "CEO"

    def test_extract_accounting_policy(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = [
            {"policy_changed": "revenue recognition", "reason": "IFRS 15",
             "financial_impact": "$50M", "restatement": "yes"},
        ]
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.ACCOUNTING_POLICY, [10], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["extraction"][0]["policy_changed"] == "revenue recognition"

    def test_extract_director_changes(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = [
            {"name": "John Doe", "role": "CFO", "change_type": "resignation",
             "timing": "March 2022", "reason": "personal"},
        ]
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.DIRECTOR_CHANGES, [10], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["extraction"][0]["change_type"] == "resignation"

    def test_extract_cash_flow_warnings(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = {
            "operating_cf": "-$2B",
            "net_income": "$500M",
            "divergence_flag": True,
            "key_observations": "Significant divergence",
        }
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.CASH_FLOW_WARNINGS, [10], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["extraction"]["divergence_flag"] is True

    def test_extract_md_and_a_red_flags(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = {
            "red_flags": [
                {"description": "Vague language", "severity": "high",
                 "quoted_text": "macro headwinds impacted results"},
            ]
        }
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.MD_AND_A_RED_FLAGS, [10], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["extraction"]["red_flags"][0]["severity"] == "high"

    def test_extract_category_empty_pages(self):
        mock_client = Mock()
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.GOING_CONCERN, [], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["extraction"] is None
        mock_client.ask_json.assert_not_called()

    def test_extract_category_raw_response_fallback(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = {
            "raw_response": '{"opinion_type": "qualified", "qualification_basis": null}'
        }
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.AUDITOR_OPINION, [10], SAMPLE_PAGES,
            mock_client, {"temperature": 0}, tracker,
        )

        assert result["extraction"]["opinion_type"] == "qualified"


class TestPydanticValidation:
    """Uses FakeLLMClient (typed test double) instead of Mock to
    demonstrate Protocol-based testability."""

    def test_valid_dict_response_passes_through_model(self):
        """AuditorOpinionResult validates and coerces the LLM response."""
        fake_client = FakeLLMClient(json_response={
            "opinion_type": "qualified",
            "qualification_basis": "going concern",
            "emphasis_of_matter": None,
            "going_concern_language": "material uncertainty",
            "extra_field_from_llm": "should be dropped",
        })
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.AUDITOR_OPINION, [10], SAMPLE_PAGES,
            fake_client, {"temperature": 0}, tracker,
        )

        ext = result["extraction"]
        assert ext["opinion_type"] == "qualified"
        # Pydantic drops extra fields not in the model
        assert "extra_field_from_llm" not in ext
        assert fake_client.call_count == 1

    def test_valid_list_response_validates_each_item(self):
        """List-returning categories validate each item."""
        fake_client = FakeLLMClient(json_response=[
            {"nature": "Lawsuit", "exposure": "$500M",
             "likelihood": "probable", "timeline": "2024"},
            {"nature": "Regulatory", "exposure": None,
             "likelihood": None, "timeline": None},
        ])
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.CONTINGENT_LIABILITIES, [10], SAMPLE_PAGES,
            fake_client, {"temperature": 0}, tracker,
        )

        ext = result["extraction"]
        assert len(ext) == 2
        assert ext[0]["nature"] == "Lawsuit"
        assert ext[1]["exposure"] is None

    def test_invalid_response_falls_back_to_raw(self, capsys):
        """When validation fails, raw response is kept and warning printed."""
        fake_client = FakeLLMClient(json_response="not a dict at all")
        tracker = UsageTracker()

        result = extract_category(
            RiskCategory.CASH_FLOW_WARNINGS, [10], SAMPLE_PAGES,
            fake_client, {"temperature": 0}, tracker,
        )

        # Falls back to the raw (normalized) response
        assert result["extraction"] == "not a dict at all"
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "cash_flow_warnings" in captured.out


class TestExtractAllCategories:
    def test_all_categories_processed(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = {"extraction": "test"}
        tracker = UsageTracker()

        flagged = {
            RiskCategory.GOING_CONCERN: [10],
            RiskCategory.DEBT_COVENANTS: [30],
            RiskCategory.AUDITOR_OPINION: [10, 11],
        }

        results = extract_all_categories(
            flagged, SAMPLE_PAGES, mock_client,
            {"temperature": 0}, tracker, max_workers=2,
        )

        assert len(results) == 3
        assert "going_concern" in results
        assert "debt_covenants" in results
        assert "auditor_opinion" in results

    def test_partial_flags_only_extracts_flagged(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = {"extraction": "test"}
        tracker = UsageTracker()

        flagged = {RiskCategory.GOING_CONCERN: [10]}

        results = extract_all_categories(
            flagged, SAMPLE_PAGES, mock_client,
            {"temperature": 0}, tracker, max_workers=1,
        )

        assert len(results) == 1
        assert "going_concern" in results

    def test_empty_flagged_pages(self):
        mock_client = Mock()
        tracker = UsageTracker()

        results = extract_all_categories(
            {}, SAMPLE_PAGES, mock_client,
            {"temperature": 0}, tracker,
        )

        assert results == {}
        mock_client.ask_json.assert_not_called()
