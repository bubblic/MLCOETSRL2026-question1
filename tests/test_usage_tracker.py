"""Tests for usage_tracker module."""

from concurrent.futures import ThreadPoolExecutor, as_completed

from financial_forecast.extraction.risk.usage_tracker import UsageTracker


class TestUsageTracker:
    def test_record_increments_counts(self):
        tracker = UsageTracker()
        tracker.record("flagging", "prompt1", "response1")
        tracker.record("flagging", "prompt2", "response2")
        tracker.record("flagging", "prompt3", "response3")
        summary = tracker.summary()
        assert summary["stages"]["flagging"]["call_count"] == 3

    def test_record_tracks_characters(self):
        tracker = UsageTracker()
        tracker.record("extraction", "hello", "world!")
        summary = tracker.summary()
        assert summary["stages"]["extraction"]["chars_in"] == 5
        assert summary["stages"]["extraction"]["chars_out"] == 6

    def test_summary_returns_all_stages(self):
        tracker = UsageTracker()
        tracker.record("flagging", "p", "r")
        tracker.record("extraction", "p", "r")
        tracker.record("synthesis", "p", "r")
        summary = tracker.summary()
        assert len(summary["stages"]) == 3
        assert "flagging" in summary["stages"]
        assert "extraction" in summary["stages"]
        assert "synthesis" in summary["stages"]

    def test_summary_totals(self):
        tracker = UsageTracker()
        tracker.record("flagging", "aa", "bbb")
        tracker.record("extraction", "cccc", "ddddd")
        summary = tracker.summary()
        assert summary["total_calls"] == 2
        assert summary["total_chars_in"] == 6  # 2 + 4
        assert summary["total_chars_out"] == 8  # 3 + 5

    def test_summary_empty_tracker(self):
        tracker = UsageTracker()
        summary = tracker.summary()
        assert summary["stages"] == {}
        assert summary["total_calls"] == 0
        assert summary["total_chars_in"] == 0
        assert summary["total_chars_out"] == 0

    def test_concurrent_record_is_threadsafe(self):
        """Without the Lock, this test would intermittently fail because
        concurrent += operations would lose increments."""
        tracker = UsageTracker()
        n_workers = 8
        calls_per_worker = 200

        def worker(_: int) -> None:
            for _ in range(calls_per_worker):
                tracker.record("extraction", "p", "r")

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(n_workers)]
            for f in as_completed(futures):
                f.result()

        summary = tracker.summary()
        expected = n_workers * calls_per_worker
        assert summary["stages"]["extraction"]["call_count"] == expected
        assert summary["stages"]["extraction"]["chars_in"] == expected  # len("p") == 1
        assert summary["stages"]["extraction"]["chars_out"] == expected  # len("r") == 1

    def test_print_summary_does_not_raise(self, capsys):
        tracker = UsageTracker()
        tracker.record("flagging", "prompt", "response")
        tracker.print_summary()
        captured = capsys.readouterr()
        assert "flagging" in captured.out
        assert "1 calls" in captured.out
