"""Abstract base class for financial model trainers.

Trainers encapsulate optimisation loops that update a model's parameters
in-place.  Concrete subclasses implement :meth:`train` for a specific
parameter group (policy, structural, etc.).

The dependency flow is one-directional:

    trainers  ->  models/base  ->  types
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from financial_forecast.models.base import BaseFinancialModel
    from financial_forecast.types import HistoricalTrainingData


class BaseTrainer(ABC):
    """Abstract interface for training financial model parameters.

    Subclasses must implement :meth:`train`, which receives a model
    instance and historical data, then updates the model's parameters
    via gradient descent.
    """

    @abstractmethod
    def train(
        self,
        model: BaseFinancialModel,
        data: HistoricalTrainingData,
        loss_scale_mode: str = "std",
        show_plot: bool = False,
    ) -> None:
        """Train the model's parameters in-place.

        Args:
            model: A :class:`BaseFinancialModel` whose trainable
                attributes will be updated.
            data: Historical time series (already scaled to model
                units).
            loss_scale_mode: ``"std"`` (normalize by historical std)
                or ``"none"`` (no normalization).
            show_plot: Whether to display diagnostic plots.
        """
