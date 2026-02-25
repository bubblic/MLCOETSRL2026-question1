"""Training entry points for the trainable financial model."""

from typing import Any

from .model import TrainableFinancialModel


def train_simple_policies(
    model: TrainableFinancialModel, *args: Any, **kwargs: Any
) -> None:
    """Train simple policy parameters on a model instance."""
    model.train_simple_policies(*args, **kwargs)


def train_structural_parameters(
    model: TrainableFinancialModel, *args: Any, **kwargs: Any
) -> None:
    """Train structural parameters on a model instance."""
    model.train_structural_parameters(*args, **kwargs)


__all__ = ["train_simple_policies", "train_structural_parameters"]
