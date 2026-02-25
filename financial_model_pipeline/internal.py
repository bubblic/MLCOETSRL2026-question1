"""Compatibility shim for legacy imports.

This module re-exports symbols from the split modules.
"""

from .forecast import run_monte_carlo_forecast
from .io_utils import TRAINING_RESULTS_DIR, _get_training_results_path
from .model import TrainableFinancialModel
from .pipeline import run_training_and_forecast
from .plotting import plot_historical_and_forecast, plot_opex_fit_with_aleatoric_noise

__all__ = [
    "TRAINING_RESULTS_DIR",
    "_get_training_results_path",
    "TrainableFinancialModel",
    "run_monte_carlo_forecast",
    "plot_opex_fit_with_aleatoric_noise",
    "plot_historical_and_forecast",
    "run_training_and_forecast",
]
