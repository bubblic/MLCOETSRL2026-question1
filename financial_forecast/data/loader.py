"""Company-agnostic data loader for the forecast pipeline.

Usage::

    # Base model (no tax anomalies, no inflation):
    data = HistoricalDataLoader("aapl")

    # With inflation only:
    data = HistoricalDataLoader("aapl", include_inflation=True)

    # Full enhancement (inflation + tax anomalies):
    data = HistoricalDataLoader("aapl", include_inflation=True,
                                include_tax_onetime=True)

    model = TrainableFinancialModel(opex_module=BayesianOpEx(),
                                    tax_anomalies=data.tax_onetime_payments)
    ForecastPipeline(model=model, data=data, ...).run()

To add a new company, create ``financial_forecast/data/<ticker>/`` with:

- ``financial_statements.py`` containing ``get_financial_statements() -> dict``
- (optional) ``tax_onetime_payments.py`` containing
  ``get_tax_onetime_payments() -> tf.Tensor``

No changes to this loader are needed.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import tensorflow as tf


class HistoricalDataLoader:
    """Loads company-specific historical financial data.

    Dynamically imports from ``financial_forecast.data.<company>/`` based
    on the ticker.  Each company package must expose a standard interface:

    - ``financial_statements.get_financial_statements() -> Dict[str, tf.Tensor]``
    - ``tax_onetime_payments.get_tax_onetime_payments() -> tf.Tensor``
      (optional)

    Args:
        company: Company ticker, case-insensitive (e.g. ``"aapl"``).
        include_inflation: Whether to load inflation data.  When ``False``,
            :attr:`inflation` returns ``None`` and the pipeline trains
            without inflation adjustments.
        include_tax_onetime: Whether to load one-time tax anomaly data.
            When ``False``, :attr:`tax_onetime_payments` returns ``None``
            and the pipeline trains without tax adjustments.
    """

    def __init__(
        self,
        company: str,
        include_inflation: bool = False,
        include_tax_onetime: bool = False,
    ):
        self.company = company.lower()
        self.include_inflation = include_inflation
        self.include_tax_onetime = include_tax_onetime

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
    def tax_onetime_payments(self) -> Optional[tf.Tensor]:
        """One-time tax anomaly data, or ``None`` if not included."""
        if not self.include_tax_onetime:
            return None
        return self._load_tax()

    def _import_company_module(self, module_name: str):
        """Import ``financial_forecast.data.<company>.<module_name>``."""
        fqn = f"financial_forecast.data.{self.company}.{module_name}"
        try:
            return importlib.import_module(fqn)
        except ModuleNotFoundError:
            raise FileNotFoundError(
                f"No {module_name} module found for company {self.company!r} "
                f"(expected {fqn})"
            ) from None

    def _load_historical(self) -> Dict[str, tf.Tensor]:
        mod = self._import_company_module("financial_statements")
        return mod.get_financial_statements()

    def _load_inflation(self) -> tf.Tensor:
        from financial_forecast.data.inflation import get_us_inflation
        return get_us_inflation()

    def _load_tax(self) -> tf.Tensor:
        mod = self._import_company_module("tax_onetime_payments")
        return mod.get_tax_onetime_payments()
