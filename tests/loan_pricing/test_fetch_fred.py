"""Tests for the FRED data fetcher — caching, retries, and error handling."""

from __future__ import annotations

import datetime
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from loan_pricing.config import ProjectConfig
from loan_pricing.data.fetch_fred import (
    FredDataFetcher,
    _cache_is_fresh,
    _cache_path_for_series,
    _read_cache,
    _write_cache,
)
from loan_pricing.exceptions import DataFetchError


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------


class TestCachePathForSeries:
    def test_returns_csv_path(self, tmp_path: Path) -> None:
        result = _cache_path_for_series(tmp_path, "DGS10")
        assert result == tmp_path / "DGS10.csv"


class TestCacheIsFresh:
    def test_missing_file_is_not_fresh(self, tmp_path: Path) -> None:
        assert not _cache_is_fresh(tmp_path / "missing.csv", ttl_seconds=3600)

    def test_recent_file_is_fresh(self, tmp_path: Path) -> None:
        path = tmp_path / "recent.csv"
        path.write_text("data")
        assert _cache_is_fresh(path, ttl_seconds=3600)

    def test_old_file_is_stale(self, tmp_path: Path) -> None:
        path = tmp_path / "old.csv"
        path.write_text("data")
        # Backdate modification time.
        ancient = time.time() - 48 * 3600
        import os

        os.utime(path, (ancient, ancient))
        assert not _cache_is_fresh(path, ttl_seconds=24 * 3600)


class TestWriteAndReadCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "date": [datetime.date(2024, 1, 2)],
                "series_id": ["DGS10"],
                "value": [4.25],
            }
        )
        path = tmp_path / "DGS10.csv"
        _write_cache(df, path)
        loaded = _read_cache(path)
        assert loaded.shape == (1, 3)
        assert loaded["date"].iloc[0] == datetime.date(2024, 1, 2)
        assert loaded["value"].iloc[0] == pytest.approx(4.25)


# ------------------------------------------------------------------
# FredDataFetcher integration
# ------------------------------------------------------------------


class TestFredDataFetcherCacheHit:
    """When a fresh cache exists the fetcher must NOT call the API."""

    def test_reads_from_cache(self, project_config: ProjectConfig) -> None:
        # Seed the cache with a single-row CSV.
        cache = project_config.raw_fred_dir / "DGS10.csv"
        df = pd.DataFrame(
            {
                "date": [datetime.date(2024, 6, 1)],
                "series_id": ["DGS10"],
                "value": [4.5],
            }
        )
        df.to_csv(cache, index=False)

        fetcher = FredDataFetcher(config=project_config)
        result = fetcher.fetch(["DGS10"], "2024-01-01", "2024-12-31")

        assert len(result) == 1
        assert result["series_id"].iloc[0] == "DGS10"


class TestFredDataFetcherMissingApiKey:
    """When the API key env var is unset, a DataFetchError must be raised."""

    def test_raises_without_api_key(self, project_config: ProjectConfig) -> None:
        fetcher = FredDataFetcher(config=project_config)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(DataFetchError, match="not set"):
                fetcher.fetch(["DGS10"], "2024-01-01", "2024-12-31")


class TestFredDataFetcherRetries:
    """After 3 failed API calls the fetcher must raise DataFetchError."""

    @patch("loan_pricing.data.fetch_fred.time.sleep")
    def test_raises_after_max_retries(
        self,
        mock_sleep: MagicMock,
        project_config: ProjectConfig,
    ) -> None:
        fetcher = FredDataFetcher(config=project_config)

        # Create a fake fredapi module so the deferred import succeeds
        # even when fredapi is not installed.
        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.side_effect = ConnectionError("offline")

        fake_fredapi = MagicMock()
        fake_fredapi.Fred.return_value = mock_fred_instance

        with patch.dict("os.environ", {"FRED_API_KEY": "fake-key"}):
            with patch.dict("sys.modules", {"fredapi": fake_fredapi}):
                with pytest.raises(DataFetchError, match="3 retries"):
                    fetcher.fetch(["DGS10"], "2024-01-01", "2024-12-31")

                assert mock_fred_instance.get_series.call_count == 3


class TestFredDataFetcherEmptyInput:
    def test_empty_series_list(self, project_config: ProjectConfig) -> None:
        fetcher = FredDataFetcher(config=project_config)
        result = fetcher.fetch([], "2024-01-01", "2024-12-31")
        assert result.empty
        assert list(result.columns) == ["date", "series_id", "value"]
