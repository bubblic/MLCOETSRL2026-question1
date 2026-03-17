"""Financial forecasting model implementations.

Provides three model variants that all inherit from
:class:`BaseFinancialModel`:

- :class:`SimpleFinancialModel` — deterministic, constant parameters.
- :class:`TrainableFinancialModel` — trainable ``tf.Variable`` parameters.
- :class:`BayesianFinancialModel` — Bayesian variational inference with
  ``TransformedVariable`` and probabilistic OpEx.
"""

from financial_forecast.models.base import BaseFinancialModel
from financial_forecast.models.simple_model import SimpleFinancialModel
from financial_forecast.models.trainable_model import TrainableFinancialModel
from financial_forecast.models.bayesian_model import BayesianFinancialModel
from financial_forecast.models.llm_forecaster import AzureReasoningBalanceSheetForecaster

__all__ = [
    "BaseFinancialModel",
    "SimpleFinancialModel",
    "TrainableFinancialModel",
    "BayesianFinancialModel",
    "AzureReasoningBalanceSheetForecaster",
]
