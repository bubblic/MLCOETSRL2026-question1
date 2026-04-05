"""Fetch standardised financial-statement data from SEC EDGAR.

The :class:`SecEdgarFetcher` downloads company-facts JSON from the
XBRL bulk-data endpoint and extracts annual values for the XBRL tags
required by the credit model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from loan_pricing.config import ProjectConfig
from loan_pricing.exceptions import DataFetchError
from loan_pricing.logging_config import get_logger

logger = get_logger(__name__)

_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_SEC_USER_AGENT = "LoanPricingModel/0.1 (academic-research)"

#: XBRL tags extracted for the credit model.
REQUIRED_XBRL_TAGS: tuple[str, ...] = (
    "Revenues",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "RetainedEarningsAccumulatedDeficit",
    "CashAndCashEquivalentsAtCarryingValue",
    "LongTermDebt",
    "InterestExpense",
    "DepreciationAndAmortization",
)

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0


def _cache_path_for_cik(cache_dir: Path, cik: str) -> Path:
    """Return the JSON cache path for a single CIK.

    Args:
        cache_dir: Directory where cached JSON files are stored.
        cik: SEC Central Index Key (zero-padded or plain).

    Returns:
        Absolute path to the cache file.
    """
    return cache_dir / f"CIK{cik}.json"


def _download_company_facts(cik: str) -> dict:
    """Download the full company-facts JSON for one CIK.

    Args:
        cik: SEC Central Index Key.

    Returns:
        Parsed JSON dictionary.

    Raises:
        DataFetchError: After *_MAX_RETRIES* consecutive failures.
    """
    import urllib.request

    url = _COMPANY_FACTS_URL.format(cik=cik)
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": _SEC_USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())  # type: ignore[no-any-return]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait = _BACKOFF_BASE_SECONDS * (2**attempt)
            logger.warning(
                "SEC fetch for CIK %s failed (attempt %d/%d): %s — retrying in %.1fs",
                cik,
                attempt + 1,
                _MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)

    raise DataFetchError(
        f"Failed to fetch SEC data for CIK {cik} after {_MAX_RETRIES} retries"
    ) from last_error


def _write_json_cache(data: dict, path: Path) -> None:
    """Atomically write a JSON payload to *path*.

    Args:
        data: Parsed JSON dictionary.
        path: Target cache file path.
    """
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def _read_json_cache(path: Path) -> dict:
    """Read a cached JSON file.

    Args:
        path: Path written by :func:`_write_json_cache`.

    Returns:
        Parsed JSON dictionary.
    """
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _extract_annual_values(
    facts: dict,
    cik: str,
    ticker: str,
) -> list[dict[str, object]]:
    """Pull annual values for each required XBRL tag from a company-facts blob.

    Args:
        facts: Raw company-facts JSON (``facts`` top-level key).
        cik: Company CIK for the output rows.
        ticker: Company ticker symbol for the output rows.

    Returns:
        List of row dicts with keys
        ``[cik, ticker, fiscal_year, tag, value]``.
    """
    us_gaap: dict = facts.get("facts", {}).get("us-gaap", {})
    rows: list[dict[str, object]] = []

    for tag in REQUIRED_XBRL_TAGS:
        tag_data = us_gaap.get(tag, {})
        units = tag_data.get("units", {})

        # Financial values are typically in USD; try USD first, then pure.
        observations: list[dict] = units.get("USD", units.get("pure", []))

        # Keep only 10-K (annual) filings.
        annual_obs = [
            obs
            for obs in observations
            if obs.get("form") == "10-K"
        ]

        # Group by fiscal year and take the most-recent filing per year.
        yearly: dict[int, float] = {}
        for obs in annual_obs:
            fy = obs.get("fy")
            val = obs.get("val")
            if fy is not None and val is not None:
                yearly[int(fy)] = float(val)

        if yearly:
            for fy, val in sorted(yearly.items()):
                rows.append(
                    {
                        "cik": cik,
                        "ticker": ticker,
                        "fiscal_year": fy,
                        "tag": tag,
                        "value": val,
                    }
                )
        else:
            # Tag absent — emit a single NaN sentinel so downstream
            # code can distinguish "missing for all years" from "tag
            # not requested".
            rows.append(
                {
                    "cik": cik,
                    "ticker": ticker,
                    "fiscal_year": 0,
                    "tag": tag,
                    "value": np.nan,
                }
            )

    return rows


class SecEdgarFetcher:
    """Download XBRL financial-statement data from SEC EDGAR.

    This class satisfies the :class:`~loan_pricing.data.protocols.DataFetcher`
    protocol.

    Args:
        config: Project configuration supplying the cache directory.
        ticker_map: Mapping from CIK strings to ticker symbols, used to
            populate the ``ticker`` column in the output.
    """

    def __init__(
        self,
        config: ProjectConfig | None = None,
        ticker_map: dict[str, str] | None = None,
    ) -> None:
        cfg = config or ProjectConfig()
        self._cache_dir = cfg.raw_sec_dir
        self._ticker_map = ticker_map or {}
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        series_ids: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch XBRL financials for a list of CIKs.

        Args:
            series_ids: CIK strings identifying the companies.
            start_date: Not used directly (annual data is keyed by
                fiscal year), but accepted for protocol compatibility.
            end_date: Not used directly; see *start_date*.

        Returns:
            Tidy DataFrame with columns
            ``[cik, ticker, fiscal_year, tag, value]``.

        Raises:
            DataFetchError: If the SEC endpoint is unreachable after
                retries for any CIK.
        """
        all_rows: list[dict[str, object]] = []

        for cik in series_ids:
            cache = _cache_path_for_cik(self._cache_dir, cik)
            ticker = self._ticker_map.get(cik, "")

            if cache.exists():
                logger.info("Cache hit for CIK %s", cik)
                facts = _read_json_cache(cache)
            else:
                logger.info("Fetching CIK %s from SEC EDGAR", cik)
                facts = _download_company_facts(cik)
                _write_json_cache(facts, cache)

            all_rows.extend(_extract_annual_values(facts, cik, ticker))

        if not all_rows:
            return pd.DataFrame(
                columns=["cik", "ticker", "fiscal_year", "tag", "value"]
            )

        df = pd.DataFrame(all_rows)

        # Filter to requested date window (by fiscal year).
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        df = df[
            (df["fiscal_year"] >= start_year) & (df["fiscal_year"] <= end_year)
            | (df["fiscal_year"] == 0)  # keep NaN sentinels
        ]

        return df.reset_index(drop=True)
