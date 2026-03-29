"""Abstract base class for financial model trainers.

Trainers encapsulate optimisation loops that update a model's parameters
in-place.  Concrete subclasses implement :meth:`train` for a specific
parameter group (policy, structural, etc.).

The dependency flow is one-directional:

    trainers  ->  models/base  ->  types
"""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Abstract interface for training financial model parameters.

    Subclasses must implement :meth:`train`, which receives a model
    instance and historical data, then updates the model's parameters
    via gradient descent.
    """

    @abstractmethod
    def train(self, model: object, **kwargs) -> None:
        """Train the model's parameters in-place.

        Args:
            model: A financial model whose trainable attributes will be
                updated.
            **kwargs: Training configuration and historical data arrays.
        """
