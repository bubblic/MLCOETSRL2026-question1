"""Public package interface for the trainable financial model pipeline."""

from .forecast import run_monte_carlo_forecast
from .model import TrainableFinancialModel
from .pipeline import run_training_and_forecast
from .plotting import plot_historical_and_forecast, plot_opex_fit_with_aleatoric_noise

__all__ = [
    "TrainableFinancialModel",
    "run_monte_carlo_forecast",
    "plot_opex_fit_with_aleatoric_noise",
    "plot_historical_and_forecast",
    "run_training_and_forecast",
]
