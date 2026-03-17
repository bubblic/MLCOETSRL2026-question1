"""Financial forecasting package using TensorFlow.

This package provides deterministic, trainable, and Bayesian financial
forecasting models based on the Cash Budget construction logic (Pareja, 2009).
It uses TensorFlow for automatic differentiation, GPU acceleration, and
graph-mode optimization via ``tf.function``.

Modules:
    types: Shared dataclasses (FinancialState, EconomicInputs).
    models: Deterministic, trainable, and Bayesian model implementations.
    layers: Custom TensorFlow Keras layers for forecast computations.
    training: Training pipelines and orchestration.
    inference: Monte Carlo forecast execution and summary.
    extraction: LLM-powered financial statement extraction from PDFs.
    clients: Azure LLM client wrapper.
    data: Historical financial data loaders.
"""

from financial_forecast.types import EconomicInputs, FinancialState

__all__ = [
    "FinancialState",
    "EconomicInputs",
]
