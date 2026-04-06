"""Fetch macroeconomic time-series from the FRED API.

The :class:`FredDataFetcher` satisfies the :class:`DataFetcher` protocol
and adds file-system caching with a configurable time-to-live so that
repeated pipeline runs do not hammer the FRED servers.
"""

from __future__ import annotations

import datetime
import os
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from loan_pricing.config import ProjectConfig
from loan_pricing.exceptions import DataFetchError
from loan_pricing.logging_config import get_logger

# Load .env so FRED_API_KEY is available even when not exported in the shell.
load_dotenv()

logger = get_logger(__name__)

#: FRED series used by the pricing model.
DEFAULT_FRED_SERIES: tuple[str, ...] = (
    "DGS1",
    "DGS2",
    "DGS5",
    "DGS10",
    "DGS30",
    "BAMLC0A0CM",
    "BAMLH0A0HYM2",
    "VIXCLS",
    "T10Y2Y",
    "UNRATE",
    "A191RL1Q225SBEA",
)

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0
_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours


def _cache_path_for_series(cache_dir: Path, series_id: str) -> Path:
    """Return the CSV cache path for a single FRED series.

    Args:
        cache_dir: Directory where cached CSV files are stored.
        series_id: FRED series identifier (e.g. ``"DGS10"``).

    Returns:
        Absolute path to the cache file.
    """
    return cache_dir / f"{series_id}.csv"


def _cache_is_fresh(path: Path, ttl_seconds: float) -> bool:
    """Check whether a cached file exists and is younger than *ttl_seconds*.

    Args:
        path: Path to the cached file.
        ttl_seconds: Maximum age in seconds before the cache is stale.

    Returns:
        ``True`` if the file exists and its modification time is within
        the TTL window.
    """
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_seconds


def _read_cache(path: Path) -> pd.DataFrame:
    """Read a cached FRED series from a CSV file.

    Args:
        path: Path written by :func:`_write_cache`.

    Returns:
        DataFrame with columns ``[date, series_id, value]`` and
        ``date`` parsed as :class:`datetime.date`.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a FRED series DataFrame to CSV.

    Args:
        df: DataFrame with columns ``[date, series_id, value]``.
        path: Target cache file path.
    """
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _fetch_single_series_from_api(
    series_id: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> pd.DataFrame:
    """Hit the FRED API for one series with retry logic.

    Args:
        series_id: FRED series identifier.
        start_date: Inclusive start date (``YYYY-MM-DD``).
        end_date: Inclusive end date (``YYYY-MM-DD``).
        api_key: FRED API key.

    Returns:
        Tidy DataFrame with columns ``[date, series_id, value]``.

    Raises:
        DataFetchError: After *_MAX_RETRIES* consecutive failures.
    """
    from fredapi import Fred  # type: ignore[import-untyped]

    fred = Fred(api_key=api_key)
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            raw: pd.Series = fred.get_series(  # type: ignore[assignment]
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            df = pd.DataFrame(
                {
                    "date": [d.date() for d in raw.index],
                    "series_id": series_id,
                    "value": raw.values,
                }
            )
            return df.dropna(subset=["value"]).reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = _BACKOFF_BASE_SECONDS * (2**attempt)
            logger.warning(
                "FRED fetch for %s failed (attempt %d/%d): %s — retrying in %.1fs",
                series_id,
                attempt + 1,
                _MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)

    raise DataFetchError(
        f"Failed to fetch FRED series {series_id} after {_MAX_RETRIES} retries"
    ) from last_error


class FredDataFetcher:
    """Download macroeconomic series from FRED with local CSV caching.

    This class satisfies the :class:`~loan_pricing.data.protocols.DataFetcher`
    protocol.

    Args:
        config: Project configuration supplying the cache directory and
            the name of the environment variable holding the FRED API key.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        cfg = config or ProjectConfig()
        self._cache_dir = cfg.raw_fred_dir
        self._api_key_env_var = cfg.fred_api_key_env_var
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        series_ids: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch one or more FRED series, using cache when possible.

        Args:
            series_ids: FRED series codes to retrieve.
            start_date: Inclusive start (``YYYY-MM-DD``).
            end_date: Inclusive end (``YYYY-MM-DD``).

        Returns:
            Tidy DataFrame with columns ``[date, series_id, value]``
            where ``date`` values are :class:`datetime.date` objects.

        Raises:
            DataFetchError: If the FRED API is unreachable after retries.
        """
        frames: list[pd.DataFrame] = []

        for sid in series_ids:
            cache = _cache_path_for_series(self._cache_dir, sid)

            if _cache_is_fresh(cache, _CACHE_TTL_SECONDS):
                logger.info("Cache hit for %s", sid)
                frames.append(_read_cache(cache))
                continue

            logger.info("Fetching %s from FRED API", sid)
            api_key = os.environ.get(self._api_key_env_var, "")
            if not api_key:
                raise DataFetchError(
                    f"Environment variable {self._api_key_env_var} is not set"
                )
            df = _fetch_single_series_from_api(sid, start_date, end_date, api_key)
            _write_cache(df, cache)
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["date", "series_id", "value"])

        return pd.concat(frames, ignore_index=True)
