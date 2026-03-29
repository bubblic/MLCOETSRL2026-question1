"""Company-agnostic data loader for the forecast pipeline.

Usage::

    # Base model (no tax anomalies, no inflation):
    data = HistoricalDataLoader("aapl")

    # With inflation only:
    data = HistoricalDataLoader("aapl", include_inflation=True)

    # Full enhancement (inflation + tax anomalies from extracted JSON):
    data = HistoricalDataLoader("aapl", include_inflation=True,
                                tax_anomaly_dir="./extracted_json/tax_anomalies/aapl")

    model = TrainableFinancialModel(opex_module=BayesianOpEx(),
                                    tax_module=TaxWithAnomalies(data.tax_onetime_payments))

To add a new company, create ``financial_forecast/data/<ticker>/`` with:

- ``financial_statements.py`` containing ``get_financial_statements() -> dict``
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf


class HistoricalDataLoader:
    """Loads company-specific historical financial data.

    Dynamically imports from ``financial_forecast.data.<company>/`` based
    on the ticker.  Tax anomaly data is loaded from extracted JSON files
    produced by :class:`TaxAnomalyExtractor`.

    Args:
        company: Company ticker, case-insensitive (e.g. ``"aapl"``).
        include_inflation: Whether to load inflation data.
        tax_anomaly_dir: Directory containing
            ``*.tax-anomalies.llm.json`` files.
            If provided, tax anomaly data is loaded from these files.
    """

    def __init__(
        self,
        company: str,
        include_inflation: bool = False,
        tax_anomaly_dir: Optional[str] = None,
    ):
        self.company = company.lower()
        self.include_inflation = include_inflation
        self.tax_anomaly_dir = tax_anomaly_dir

    @property
    def financial_statements(self) -> Dict[str, tf.Tensor]:
        """Historical financial data as a dict of tensors."""
        return self._load_historical()

    @property
    def inflation(self) -> Optional[tf.Tensor]:
        """Inflation rates, or ``None`` if not included."""
        if not self.include_inflation:
            return None
        return self._load_inflation()

    @property
    def tax_onetime_payments(self) -> Optional[Dict[int, float]]:
        """One-time tax anomaly data, or ``None`` if no dir provided."""
        if self.tax_anomaly_dir is None:
            return None
        return self._load_tax_from_extracted_json()

    def _import_company_module(self, module_name: str) -> Any:
        """Import ``financial_forecast.data.<company>.<module_name>``."""
        fqn = f"financial_forecast.data.{self.company}.{module_name}"
        try:
            return importlib.import_module(fqn)
        except ModuleNotFoundError:
            raise FileNotFoundError(
                f"No {module_name} module found for company "
                f"{self.company!r} (expected {fqn})"
            ) from None

    def _load_historical(self) -> Dict[str, tf.Tensor]:
        mod = self._import_company_module("financial_statements")
        return mod.get_financial_statements()

    def _load_inflation(self) -> tf.Tensor:
        from financial_forecast.data.inflation import get_us_inflation

        return get_us_inflation()

    def _load_tax_from_extracted_json(self) -> Dict[int, float]:
        """Load tax data from extracted JSON files.

        Reads all ``*.tax-anomalies.llm.json`` files in
        :attr:`tax_anomaly_dir`, extracts non-null ``tax_onetime_amount``
        values, and returns them as ``{year: amount_usd}``.
        """
        tax_dir = Path(self.tax_anomaly_dir)
        if not tax_dir.exists():
            raise FileNotFoundError(f"Tax anomaly directory not found: {tax_dir}")

        json_files = sorted(tax_dir.glob("*.tax-anomalies.llm.json"))
        if not json_files:
            raise FileNotFoundError(f"No tax anomaly JSON files found in: {tax_dir}")

        result: Dict[int, float] = {}
        for path in json_files:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            extraction = payload.get("extraction", {})
            scale = extraction.get("amount_scale")
            amount = extraction.get("tax_onetime_amount")
            year = extraction.get("current_tax_year")
            if amount is not None and scale is not None and year is not None:
                # Extracted amounts are in billions; convert to USD
                result[int(year)] = float(amount) * float(scale)

        return result
