"""Tests for page_flagger module."""

from unittest.mock import Mock

from risk.page_flagger import (
    PageChunk,
    chunk_pages,
    flag_all_pages,
    flag_pages_in_chunk,
    _parse_flag_response,
)
from risk.risk_categories import (
    ALL_CATEGORIES,
    RiskCategory,
)
from risk.usage_tracker import UsageTracker


def _make_pages(n: int):
    """Create a dict of {page_num: text} for testing."""
    return {i + 1: f"Text of page {i + 1}" for i in range(n)}


class TestChunkPages:
    def test_exact_division(self):
        pages = _make_pages(90)
        chunks = chunk_pages(pages, chunk_size=30)
        assert len(chunks) == 3
        assert chunks[0].start == 1
        assert chunks[0].end == 30
        assert chunks[2].start == 61
        assert chunks[2].end == 90

    def test_remainder(self):
        pages = _make_pages(50)
        chunks = chunk_pages(pages, chunk_size=30)
        assert len(chunks) == 2
        assert chunks[0].end == 30
        assert chunks[1].start == 31
        assert chunks[1].end == 50

    def test_single_page(self):
        pages = {1: "Only page"}
        chunks = chunk_pages(pages, chunk_size=30)
        assert len(chunks) == 1
        assert chunks[0].start == 1
        assert chunks[0].end == 1

    def test_returns_pagechunk_instances(self):
        pages = _make_pages(10)
        chunks = chunk_pages(pages, chunk_size=10)
        assert isinstance(chunks[0], PageChunk)

    def test_pagechunk_is_immutable(self):
        pages = _make_pages(10)
        chunks = chunk_pages(pages, chunk_size=10)
        import pytest
        with pytest.raises(AttributeError):
            chunks[0].start = 999

    def test_empty_pages(self):
        chunks = chunk_pages({}, chunk_size=30)
        assert chunks == []

    def test_none_pages_skipped(self):
        pages = {1: "text", 2: None, 3: "text"}
        chunks = chunk_pages(pages, chunk_size=30)
        assert len(chunks) == 1
        # Only pages 1 and 3 included (2 has None text)
        assert "[PAGE 1]" in chunks[0].text
        assert "[PAGE 3]" in chunks[0].text
        assert "[PAGE 2]" not in chunks[0].text

    def test_page_markers_present(self):
        pages = _make_pages(5)
        chunks = chunk_pages(pages, chunk_size=10)
        assert len(chunks) == 1
        for i in range(1, 6):
            assert f"[PAGE {i}]" in chunks[0].text


class TestParseFlagResponse:
    def test_parses_list_response(self):
        response = [
            {"page": 12, "category": "going_concern", "reason": "doubt"},
            {"page": 15, "category": "debt_covenants", "reason": "breach"},
        ]
        result = _parse_flag_response(response, ALL_CATEGORIES)
        assert len(result) == 2
        assert result[0].page == 12
        assert result[0].category == "going_concern"

    def test_parses_wrapped_response(self):
        response = {
            "pages": [
                {"page": 5, "category": "auditor_opinion", "reason": "qualified"},
            ]
        }
        result = _parse_flag_response(response, ALL_CATEGORIES)
        assert len(result) == 1
        assert result[0].page == 5

    def test_parses_results_key(self):
        response = {
            "results": [
                {"page": 8, "category": "related_party", "reason": "transaction"},
            ]
        }
        result = _parse_flag_response(response, ALL_CATEGORIES)
        assert len(result) == 1

    def test_handles_empty_list(self):
        result = _parse_flag_response([], ALL_CATEGORIES)
        assert result == []

    def test_handles_empty_dict(self):
        result = _parse_flag_response({}, ALL_CATEGORIES)
        assert result == []

    def test_filters_invalid_categories(self):
        response = [
            {"page": 1, "category": "going_concern", "reason": "valid"},
            {"page": 2, "category": "made_up_category", "reason": "invalid"},
        ]
        result = _parse_flag_response(response, ALL_CATEGORIES)
        assert len(result) == 1
        assert result[0].category == "going_concern"

    def test_handles_raw_response_fallback(self):
        response = {
            "raw_response": '[{"page": 3, "category": "going_concern", "reason": "test"}]'
        }
        # raw_response containing a JSON array — extract_json_from_text
        # looks for {} not [], so this should return empty
        result = _parse_flag_response(response, ALL_CATEGORIES)
        assert result == []

    def test_handles_raw_response_with_dict(self):
        response = {
            "raw_response": '{"pages": [{"page": 3, "category": "going_concern", "reason": "test"}]}'
        }
        result = _parse_flag_response(response, ALL_CATEGORIES)
        assert len(result) == 1

    def test_skips_malformed_items(self):
        response = [
            {"page": 1, "category": "going_concern", "reason": "ok"},
            {"no_page_key": True},
            "not a dict",
            {"page": "not_int", "category": "going_concern"},
        ]
        result = _parse_flag_response(response, ALL_CATEGORIES)
        assert len(result) == 1


class TestFlagPagesInChunk:
    def test_calls_llm_and_parses(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = [
            {"page": 10, "category": "going_concern", "reason": "doubt"},
        ]
        tracker = UsageTracker()
        chunk = PageChunk(start=1, end=30, text="some text")

        result = flag_pages_in_chunk(
            chunk, mock_client, {"temperature": 0}, ALL_CATEGORIES, tracker
        )

        assert len(result) == 1
        assert result[0].page == 10
        mock_client.ask_json.assert_called_once()
        assert tracker.summary()["total_calls"] == 1


class TestFlagAllPages:
    def test_groups_by_category(self):
        mock_client = Mock()
        mock_client.ask_json.return_value = [
            {"page": 5, "category": "going_concern", "reason": "r1"},
            {"page": 10, "category": "debt_covenants", "reason": "r2"},
        ]
        tracker = UsageTracker()
        pages = _make_pages(30)

        result = flag_all_pages(
            pages, list(RiskCategory), mock_client,
            {"temperature": 0}, 30, tracker,
        )

        assert RiskCategory.GOING_CONCERN in result
        assert RiskCategory.DEBT_COVENANTS in result
        assert 5 in result[RiskCategory.GOING_CONCERN]
        assert 10 in result[RiskCategory.DEBT_COVENANTS]

    def test_deduplicates_pages(self):
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Both chunks flag page 15
            if call_count == 1:
                return [{"page": 15, "category": "going_concern", "reason": "r"}]
            return [{"page": 15, "category": "going_concern", "reason": "r"}]

        mock_client = Mock()
        mock_client.ask_json.side_effect = side_effect
        tracker = UsageTracker()
        pages = _make_pages(60)  # 2 chunks of 30

        result = flag_all_pages(
            pages, list(RiskCategory), mock_client,
            {"temperature": 0}, 30, tracker,
        )

        # Page 15 should appear only once
        assert result[RiskCategory.GOING_CONCERN] == [15]

    def test_empty_pages(self):
        mock_client = Mock()
        tracker = UsageTracker()

        result = flag_all_pages(
            {}, list(RiskCategory), mock_client,
            {"temperature": 0}, 30, tracker,
        )

        assert result == {}
        mock_client.ask_json.assert_not_called()
