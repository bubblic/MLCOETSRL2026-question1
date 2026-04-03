"""Tests for synthesiser module."""

from unittest.mock import Mock

from risk.synthesiser import synthesise_risk_memo
from risk.usage_tracker import UsageTracker


SAMPLE_FINDINGS = {
    "going_concern": {
        "category": "going_concern",
        "extraction": {
            "disclosure_text": "Material uncertainty",
            "time_period": "12 months",
            "mitigating_actions": "Asset disposal",
        },
    },
    "debt_covenants": {
        "category": "debt_covenants",
        "extraction": [
            {"covenant_type": "leverage", "compliance_status": "breached"},
        ],
    },
}


class TestSynthesiseRiskMemo:
    def test_returns_memo_structure(self):
        mock_client = Mock()
        mock_client.ask_text.return_value = (
            "RISK RATING: HIGH\n\n"
            "1. Going concern material uncertainty\n"
            "2. Debt covenant breach\n"
            "3. Qualified auditor opinion\n\n"
            "Recommended: Immediate review."
        )
        tracker = UsageTracker()

        result = synthesise_risk_memo(
            SAMPLE_FINDINGS, mock_client, {"temperature": 0}, tracker,
        )

        assert "risk_memo" in result
        assert "HIGH" in result["risk_memo"]
        mock_client.ask_text.assert_called_once()

    def test_prompt_includes_all_findings(self):
        mock_client = Mock()
        mock_client.ask_text.return_value = "Low risk."
        tracker = UsageTracker()

        synthesise_risk_memo(
            SAMPLE_FINDINGS, mock_client, {"temperature": 0}, tracker,
        )

        call_args = mock_client.ask_text.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt") or call_args[0][1]
        assert "going_concern" in prompt
        assert "debt_covenants" in prompt
        assert "Material uncertainty" in prompt

    def test_with_empty_findings(self):
        mock_client = Mock()
        mock_client.ask_text.return_value = "No significant risks identified."
        tracker = UsageTracker()

        result = synthesise_risk_memo(
            {}, mock_client, {"temperature": 0}, tracker,
        )

        assert "risk_memo" in result
        assert result["risk_memo"] == "No significant risks identified."

    def test_with_partial_findings(self):
        mock_client = Mock()
        mock_client.ask_text.return_value = "Medium risk."
        tracker = UsageTracker()

        partial = {"going_concern": SAMPLE_FINDINGS["going_concern"]}
        result = synthesise_risk_memo(
            partial, mock_client, {"temperature": 0}, tracker,
        )

        assert result["risk_memo"] == "Medium risk."

    def test_tracks_usage(self):
        mock_client = Mock()
        mock_client.ask_text.return_value = "Memo text."
        tracker = UsageTracker()

        synthesise_risk_memo(
            SAMPLE_FINDINGS, mock_client, {"temperature": 0}, tracker,
        )

        summary = tracker.summary()
        assert summary["stages"]["synthesis"]["call_count"] == 1
        assert summary["stages"]["synthesis"]["chars_in"] > 0
        assert summary["stages"]["synthesis"]["chars_out"] > 0
